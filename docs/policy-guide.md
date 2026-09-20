# The policy guide

Every routing decision jev-route makes comes out of one YAML file. This document is the
complete reference to that file: every key, every default, every validation error, the exact
expression grammar, and five complete deployment examples.

Read it in order the first time. After that, the
[key reference](#2-every-key) and the [expression variables](#4-expression-variables) are the
two sections you will keep coming back to.

Related documents: [architecture.md](architecture.md) for the 8-step chain,
[privacy.md](privacy.md) for exactly what leaves the process, and
[graduation.md](graduation.md) for the export → train → evaluate → package → graduate pipeline.

---

## Contents

1. [What a policy file is](#1-what-a-policy-file-is)
2. [Every key](#2-every-key)
3. [The expression grammar](#3-the-expression-grammar)
4. [Expression variables](#4-expression-variables)
5. [Tiers and `tier_order`](#5-tiers-and-tier_order)
6. [Rule ordering](#6-rule-ordering)
7. [`on_uncertain`: the calibration payoff](#7-on_uncertain-the-calibration-payoff)
8. [`on_backend_down`: outages and the invariant](#8-on_backend_down-outages-and-the-invariant)
9. [`gate`: configuring the local hard gate](#9-gate-configuring-the-local-hard-gate)
10. [`cache`](#10-cache)
11. [`logging`](#11-logging)
12. [`shadow`](#12-shadow)
13. [Five complete examples](#13-five-complete-examples)
14. [Validation checklist](#14-validation-checklist)

---

## 1. What a policy file is

A policy file is the operator's surface. It is the only part of jev-route you are expected to
edit, and it is the only part you *need* to edit.

Concretely it is a YAML mapping that `jev_route.policy.Policy` loads, validates, and then
evaluates on every request:

```python
from jev_route import Policy, Router

policy = Policy.from_file("policies/default.yaml")  # validates, raises PolicyError on a bad file
router = Router.from_policy_file("policies/default.yaml")  # also builds the backend it names
```

Three properties follow from that, and they are the reason the file exists in this form:

**It is reviewed like code.** A policy change is a pull request. A reviewer can read
`sensitivity_confidence_below: 0.8` and know precisely what it does, which is more than can be
said for a threshold buried in a Python module. The `reason:` string on every rule exists so
the decision log carries a human sentence, not just a rule id.

**It is loaded once and evaluated often.** `Policy.from_file` parses every rule expression into
a compiled AST at load time (`compile_expression` memoizes them process-wide). A malformed
policy fails at startup, not at request time under load. That is the difference between a
deployment that does not come up and a deployment that silently misroutes one request in ten
thousand.

**`version: 1` is checked at load.** `POLICY_VERSION` is `1`. `Policy.from_dict` reads
`version` (defaulting to `1` when the key is absent) and refuses anything else:

```text
PolicyError: unsupported policy version 2; this build understands version 1
```

The check exists so that a policy written for a future schema cannot be half-understood by an
older build. An absent `version` is accepted and treated as `1`; a *wrong* one is not.

Two more load-time behaviours worth knowing before you write a file:

* `Policy.from_file` raises `PolicyError: policy file not found: <path>` if the path does not
  exist. There is no silent fallback inside `Policy` itself.
* `Policy.from_yaml` raises `PolicyError: policy YAML must be a mapping at the top level` if
  the document parses to a list or a scalar.

Where the file comes from is up to your deployment. The CLI looks at `--policy`, then
`$JEV_ROUTE_POLICY`, then `policies/default.yaml`.

---

## 2. Every key

### How to read this section

Each entry gives the **type**, the **default**, **what it does**, and the **error** a bad value
produces. Two conventions matter:

* **"read by"** names the code that consumes the key. `Policy.from_dict` only *stores* the
  `backend`, `gate`, `cache`, `logging`, and `shadow` sections as plain dicts, with two
  deliberate exceptions: `gate.semantic` and `gate.blocked_metadata` are parsed and validated
  *inside* `from_dict` (a `mode: enfoce` typo fails the load, not the request). For the rest,
  the real reading happens later — in `Router.__init__`, `build_cache`, `build_sink`,
  `build_backend`, and `_gate_from_policy`. So a bad value in those sections usually fails when
  the **Router is constructed**, not when the policy is parsed.
* **Unknown keys are ignored, in most sections.** There is no schema validator that rejects a
  key it does not know. `on_uncertain` and `on_backend_down` are filtered down to their
  dataclass fields, and the rest are read with `cfg.get(...)`. A typo like `logging.excerptmode`
  does not error; it silently leaves the default in place. That is the single most common way a
  policy file lies to its author, and it is why [section 14](#14-validation-checklist) ends
  with "prove the behaviour, do not trust the file". The two `gate` sub-sections above are the
  exception: they reject unknown keys, because in a gate a typo that does nothing is worse than
  a typo that fails the deploy.

### Top level

| key | type | default | what it does |
| --- | --- | --- | --- |
| `version` | int | `1` | Schema version. Must equal `POLICY_VERSION` (`1`). |
| `backend` | mapping | `{}` → `name: mock` | Which decision backend answers the four questions. |
| `gate` | mapping | `{}` | Local hard-gate behaviour. The gate always runs. |
| `tiers` | mapping[str, str \| list[str]] | **required** | Tier name → LiteLLM model name(s). |
| `tier_order` | list[str] | `["local", "cheap", "strong"]` (`schema.TIERS`) | Ascending capability. |
| `pii_threshold` | float | `0.5` | Probability at or above which PII counts as present. |
| `rules` | list[mapping] | **required** | Ordered condition → outcome pairs. First match wins. |
| `on_uncertain` | mapping | see below | Confidence floors and how far to escalate. |
| `on_backend_down` | mapping | see below | What to do when the backend cannot answer. |
| `cache` | mapping | `{}` → enabled, in-memory | Judgement cache in front of the backend. |
| `logging` | mapping | `{}` → **no sink** | Decision log. This is the training dataset. |
| `shadow` | mapping | `{}` → disabled | Router-level side-run of a second backend. |

Errors at this level:

```text
unsupported policy version 2; this build understands version 1
policy must define at least one tier under `tiers:`
policy YAML must be a mapping at the top level
policy YAML failed to parse: <yaml error>
policy file not found: <path>
```

Note that `logging` has no working default: `build_sink` returns a `NullSink` when `path` is
missing. A policy with no `logging:` section routes fine and records nothing — which means it
accumulates no training data. Set `logging.path` explicitly.

### `backend`

Read by `jev_route.backends.build_backend(policy)`, which dispatches on `backend.name`. The
policy loader never validates `backend.name`; an unknown or unavailable name fails when the
Router builds the backend.

| key | type | default | what it does |
| --- | --- | --- | --- |
| `name` | str | `mock` | `mock` \| `jev` \| `distilled` \| `shadow`. Lower-cased before dispatch. |
| `api_key` | str | *(none)* | Literal API key. **Do not put one in a policy file.** |
| `api_key_env` | str | `TYPESAFE_API_KEY` | Env var to read the key from. Only consulted when `api_key` is absent or empty. |
| `api_url` | str | `https://api.typesafe.ai/v1/systemone` | Endpoint the Jev backend POSTs to. |
| `model` | str | `jev-latest` | System One model alias. Also the initial `model_version` until the first response reports one. |
| `timeout_seconds` | float | `5.0` | Per-attempt budget. This call is on the critical path of every request. |
| `max_retries` | int | `2` | Retries *after* the first attempt, clamped to `>= 0`. |
| `include_domain` | bool | `true` | Whether to ask the fourth question (`domain`). |
| `artifact` | str | `./artifacts/jev-route-distilled` | `distilled` only: path to the trained artifact. |
| `feature_mode` | bool | `false` | `distilled` only: train/serve on features rather than text. |
| `primary` | mapping | **required for `shadow`** | `shadow` only: the backend whose decision is served. |
| `shadow` | mapping | **required for `shadow`** | `shadow` only: the backend run alongside it. |
| `log_disagreements` | bool | `true` | `shadow` only: record where the two backends differ. |
| `shadow_timeout_seconds` | float | `5.0` | `shadow` only: budget for the side-run. |
| `temperature` | float | `0.55` | `mock` only: softmax temperature of the mock's distributions. |

`api_key` and `api_key_env` are combined as:

```python
api_key = cfg.get("api_key") or os.environ.get(str(cfg.get("api_key_env", "TYPESAFE_API_KEY")), "")
```

So a literal key wins, and the env var name is itself configurable. Use `api_key_env` and keep
the secret in your secret store. A policy file is committed to a repository; a key in it is a
key in your git history forever.

Errors:

```text
BackendError: unknown backend 'gold'; expected one of: mock, jev, distilled, shadow
BackendError: backend.name: shadow requires primary, shadow side(s); each needs at least 'name'. Got: ['name']
ValueError: JevBackend requires an API key: pass api_key= or set TYPESAFE_API_KEY.
            Use MockBackend for offline development.
```

The `JevBackend` refusal is deliberate. A router that quietly stops classifying is worse than
one that will not start: it would answer every request at maximum uncertainty and fail closed
forever, and you would find out from a cost report instead of a stack trace.

`include_domain: false` is a token saving with a real consequence. When the domain question is
not asked, `_parse_answers` fills `domain` with `ChoiceAnswer.uniform(DOMAINS)` — choice
`"analysis"` (the middle of the ladder), confidence `0.0`, `confidence_reported: false`. So:

* any rule reading `domain` becomes dead — it will only ever see `"analysis"`;
* `domain_confidence` will only ever be `0.0`;
* your decision log will carry a uniform domain distribution, which is useless as a distillation
  target for that question.

Set it `false` only when no rule reads `domain` *and* you do not intend to distill a domain head.

The `distilled` and `shadow` branches of `build_backend` import their modules lazily
(`jev_route.backends.distilled`, `jev_route.backends.shadow`). The policy file cannot tell you
whether those modules are present in your install; the failure is an `ImportError` when the
Router is constructed. `backend.name: mock` and `backend.name: jev` need nothing beyond the
core dependencies.

### `gate`

Read by `Router.__init__` (`on_force_local`) and by `_gate_from_policy` (`disabled_detectors`,
`placeholder_domains_as_pii`).

| key | type | default | what it does |
| --- | --- | --- | --- |
| `on_force_local` | str | `skip_backend` | Whether a gate-forced request still pays for a classification. |
| `disabled_detectors` | list[str] | `[]` | Detector names to silence. Unknown names raise. |
| `placeholder_domains_as_pii` | bool | `false` | Treat RFC 2606 addresses (`user@example.com`) as personal data. |
| `semantic` | mapping | `mode: shadow`, everything else off | Layer 2 of the gate: the local distilled semantic scorer. **Validated at parse time.** |
| `blocked_metadata` | mapping | `enabled: true`, `path: null` | The refusal stream: what a gate block is recorded as, and where. **Validated at parse time.** |

`on_force_local` is lower-cased, then validated:

```text
ValueError: gate.on_force_local must be skip_backend|still_classify, got 'bogus'
```

```text
ValueError: unknown detector names in disabled_detectors: ['nope']
```

Both are raised when the `Router` is constructed, not when the policy is parsed. See
[section 9](#9-gate-configuring-the-local-hard-gate) for what each value actually costs.

`semantic` and `blocked_metadata` are the two `gate` sub-sections validated at **parse** time,
and the only ones that reject unknown keys:

```text
PolicyError: gate.semantic has unknown keys ['bogus']. Known: ['artifact', 'enforce_requires',
            'level', 'mode', 'shadow_metrics', 'threshold']. Failing the deploy is the point:
            an ignored knob here is a gate in a mode its operator did not ask for.
```

`semantic` keys: `mode` (`off` | `shadow` | `enforce`, default `shadow` — an absent artifact
makes the layer *inert*, not an error), `artifact` (path to the trained scorer), `threshold`
(default `0.5`; the artifact's own threshold wins once loaded), `level` (the level a firing
layer asserts, default `confidential`), `shadow_metrics` (path to the live shadow observation
JSON), and `enforce_requires` (the promotion criteria; defaults `min_recall: 0.99`,
`max_false_positive_rate: 0.02`, `max_disagreement_rate: 0.05`, `max_semantic_miss_rate: 0.02`,
`min_positive_examples: 200`, `min_negative_examples: 500`, `min_shadow_examples: 1000`, with a
hard floor of `0.90` below which `min_recall` is rejected outright). `blocked_metadata` keys:
`enabled` and `path` (default `gate-blocks.jsonl` beside the decision log — a separate file on
purpose, because the refusal stream feeds rule improvement and the decision log feeds model
training). The full semantics of both, including shadow mode and the promotion path, is in
[docs/gate-layers.md](gate-layers.md).

There is no key that turns the **deterministic** gate off. `Router.__init__` always has a
`HardGate`; there is no configuration in which unscanned text is routed. Individual detectors
can be silenced, the gate cannot. (`gate.semantic.mode: off` turns off *layer 2* — the optional
semantic layer — which is a different component, and the default `shadow` is the measured,
cannot-block state of that layer, not a way to disable the gate.)

### `tiers`, `tier_order`, `pii_threshold`

```yaml
tiers:
  local:
    - qwen38            # a one-element list
  cheap: qwen3.8-flash  # a bare string is accepted and becomes a one-element tuple
  strong:
    - qwen3.8-max
tier_order: [cheap, strong]
pii_threshold: 0.5
```

| key | type | default | what it does |
| --- | --- | --- | --- |
| `tiers.<name>` | str or list[str] | **required**, at least one | Model names as they appear in your LiteLLM `model_list`. |
| `tier_order` | list[str] | `schema.TIERS` = `["local", "cheap", "strong"]` | Ascending capability order. |
| `pii_threshold` | float | `0.5` | `pii >= pii_threshold` → `pii_present` is true. |

Tier values are stripped and coerced to strings; empty entries are dropped. Errors:

```text
policy must define at least one tier under `tiers:`
tier names must be non-empty
tier 'gold' must map to a model name or a list of model names
tier 'gold' lists no models
tier_order names tiers that are not defined: ['gold']
tier_order contains duplicates
```

Two defaults here bite people, and both are load-time errors rather than surprises at runtime,
which is the point:

**Omitting `tier_order` gives you `["local", "cheap", "strong"]`, not your tier names.** If you
rename your tiers — `small`, `medium`, `large` — and omit `tier_order`, the load fails:

```text
PolicyError: tier_order names tiers that are not defined: ['local', 'cheap', 'strong']
```

If you rename your tiers, declare `tier_order` explicitly.

**The default failure tiers must exist.** `on_backend_down` defaults to
`fail_closed_tier: local` and `fail_open_tier: strong`, and both are validated against `tiers:`.
A policy with tiers `small`/`medium`/`large` and no `on_backend_down` section fails with:

```text
PolicyError: failure tier 'local' is not defined under `tiers:`
```

Either name your tiers `local`/`cheap`/`strong`, or always write `on_backend_down` explicitly.
Writing it explicitly is the better habit: it is the one section whose default you do not want
to inherit by accident.

`pii_threshold` is `float(...)`-coerced and not range-checked. A value above `1.0` means
`pii_present` can never be set by the probability alone; a value below `0.0` means it always is.
Both are allowed because both are occasionally what an operator wants in a test.

### `rules[]`

| key | type | default | what it does |
| --- | --- | --- | --- |
| `id` | str | `rule-<index>`, or `rule-<index>-default` for the default rule | Recorded on every decision as `decision.rule_id`. |
| `if` | str (expression) | *(none)* | Condition. Omit it to write the default rule. |
| `then` | mapping or str | **required** | A tier name, or a mapping with `tier` and/or `model`. |
| `then.tier` | str | *(none)* | Must name a tier defined under `tiers:`. |
| `then.model` | str | *(none)* | Pins one exact deployment and bypasses round-robin. |
| `reason` | str | `matched rule <id>` / `default rule <id>` | Human sentence written to the log. |

`then: cheap` is shorthand for `then: {tier: cheap}`. Errors, in the order `_parse_rule` and
`from_dict` can raise them:

```text
rule #0 must be a mapping with `if:`/`then:` keys
rule #0: `then:` must be a tier name or a mapping
rule #0: `then:` must set at least one of tier or model
rule #0: tier 'gold' is not defined under `tiers:` (['cheap', 'local', 'strong'])
rule #0: `then.model` must be non-empty
rule expression is empty
rule expression 'pii_present and' is not valid syntax: invalid syntax
disallowed syntax 'Call' in rule expression 'len(metadata) > 3'. Allowed: ...
unknown variable 'tenant' in rule expression 'tenant == "acme"'. Available: ...
only simple names may be subscripted, in "metadata['a']['b'] == 1"
policy defines no rules
at most one rule may omit `if:` (the default rule)
the default rule (no `if:`) must be last
policy must end with a default rule that has no `if:`
```

The last three are the ordering contract; [section 6](#6-rule-ordering) explains why the default
rule is mandatory rather than optional.

One honest edge case: a rule that sets `then.model` and no `then.tier` records an **empty
string** as its tier. `Policy.evaluate` returns `(rule, rule.tier or "", rule.model)` for a
model-pinning rule, so `decision.tier` is `""` and `decision.model` is the pinned name. If your
dashboards group by tier, give model-pinning rules a `tier:` too — it is recorded for
provenance and selects nothing.

### `on_uncertain`

Read into `UncertaintyPolicy`. Keys that are not dataclass fields are dropped silently.

| key | type | default | what it does |
| --- | --- | --- | --- |
| `sensitivity_confidence_below` | float or null | `0.8` | Below this reported confidence, escalate sensitivity. `null` disables the bump. |
| `sensitivity_bump_levels` | int | `1` | How many rungs up `SENSITIVITY_LEVELS` to move. |
| `complexity_confidence_below` | float or null | `0.7` | Below this reported confidence, escalate complexity. `null` disables the bump. |
| `complexity_bump_levels` | int | `1` | How many rungs up `COMPLEXITY_LEVELS` to move. |
| `pii_uncertain_threshold` | float | `0.35` | Bottom of the "not a no" band. |
| `pii_uncertain_counts_as_present` | bool | `true` | Treat the uncertain band as PII present. |
| `force_local_confidence_below` | float or null | `0.25` | Read **certainty**, not top probability: below this, the request is "too uncertain to egress" — sensitivity is forced to `regulated` and the tier to `local`, whatever the backend said. `null` disables it. This is the highest-impact knob in the section: a backend answering `public` at confidence `0.10` becomes `regulated` → `local`. |

Positive bumps are clamped by the escalation helpers rather than rejected, because a bump past
the top of a ladder is a legitimate way to write "always the strictest level". The section does
have validation errors, in two shapes: a non-mapping section, and a *negative* bump — a negative
`*_bump_levels` is not a milder setting, it is the inversion of the knob's purpose (de-escalating
an uncertain judgement is how "confidential, not sure" becomes "internal, send it"), so it fails
the deploy:

```text
PolicyError: on_uncertain must be a mapping, got list
PolicyError: on_uncertain.sensitivity_bump_levels is -2; bumps only ever go stricter. Set it to
            0 to disable the bump, or to a positive count to escalate that many levels. A
            negative value would move an uncertain judgement DOWN the ladder, which is exactly
            the inversion this knob exists to prevent.
```

`0` loads and means "no bump".

See [section 7](#7-on_uncertain-the-calibration-payoff) for the guards that decide *when* these
fire.

### `on_backend_down`

Read into `FailurePolicy`. Keys other than the three below are dropped silently.

| key | type | default | what it does |
| --- | --- | --- | --- |
| `mode` | str | `fail_closed` | `fail_closed` (safest tier) or `fail_open` (most capable tier). |
| `fail_closed_tier` | str | `local` | Tier used in `fail_closed`, and the tier `fail_open` falls back to when the gate overrides it. |
| `fail_open_tier` | str | `strong` | Tier used in `fail_open` when the gate does not override. |

```text
PolicyError: on_backend_down.mode must be fail_closed or fail_open, got 'fail_sideways'
PolicyError: failure tier 'gold' is not defined under `tiers:`
PolicyError: on_backend_down must be a mapping or a mode string, got list
```

Both tiers are validated whether or not the mode uses them, so a `fail_closed` deployment still
has to name a real `fail_open_tier`. That is intentional: the mode is a one-line change during
an incident, and it should not be the moment you discover the tier does not exist.

A bare string is accepted as shorthand for the mode:

```yaml
on_backend_down: fail_open
```

`from_dict` checks `isinstance(..., str)` first and turns it into `{"mode": "fail_open"}`, so the
two failure tiers keep their defaults and are still validated. Anything that is neither a mapping
nor a string is rejected:

```text
PolicyError: on_backend_down must be a mapping or a mode string, got list
```

Prefer the mapping form in a real file. `on_backend_down` is the section an operator edits
during an incident, and `mode: fail_open` next to two explicit tier names says more than a bare
word does.

### `cache`

Read by `jev_route.cache.build_cache(policy.cache)` when the `Router` is constructed.

| key | type | default | what it does |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | `false` → `NullCache`, no caching at all. |
| `kind` | str | `memory` | `memory` → `InMemoryTTLCache`, `redis` → `RedisCache`. |
| `ttl_seconds` | float | `900.0` | Per-entry lifetime, clamped to `>= 0`. `0` disables writes. |
| `max_entries` | int | `8192` | Memory cache only. LRU bound, clamped to `>= 1`. |
| `url` | str or null | `None` | Redis only. Connection URL. |
| `key_prefix` | str | `"jevroute:"` | Redis only. Namespace prefix on every key. |

```text
ValueError: unknown cache kind 'disabled'; expected 'memory' or 'redis'
ValueError: RedisCache needs url= or an injected client=
RuntimeError: RedisCache requires the redis package: pip install 'jev-route[redis]'   # only when redis is not installed
```

`kind: disabled` is **not** a value. Turn the cache off with `enabled: false`. These are
`ValueError`/`RuntimeError`, not `PolicyError`, and they surface at Router construction. The
`RuntimeError` line only occurs in an environment without the `redis` package — the dev venv
here has it, so the branch is unreachable locally; it is `# pragma: no cover` for the same
reason.

### `logging`

Split across two readers: `build_sink(policy.logging)` consumes `enabled`, `path`, `max_bytes`,
`flush_every`; `Router.__init__` consumes `excerpt_mode` and `hash_salt`. Both read the same
dict, which is why the keys live in one section.

| key | type | default | what it does |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | `false` → `NullSink`, records are discarded. |
| `path` | str or null | `None` → `NullSink` | JSONL file path. Parent directories are created at construction. |
| `excerpt_mode` | str | `hash` | `hash` \| `redacted` \| `none`. Lower-cased, then validated. |
| `hash_salt` | str | `""` | Salt mixed into `excerpt_hash`. |
| `max_bytes` | int | `268435456` (256 MiB) | Rotate when the file would exceed this. `0` disables rotation. |
| `flush_every` | int | `1` | `fsync` every N records, clamped to `>= 1`. |

```text
ValueError: logging.excerpt_mode must be hash|redacted|none, got 'bogus'
```

`enabled: true` with no `path` is silently a `NullSink`. The rest of the failure behaviour has
two faces, and they differ:

* **Parent does not exist and cannot be created** — `JsonlSink.__init__` calls
  `mkdir(parents=True, exist_ok=True)`, so this raises `PermissionError` when the Router is
  constructed. Fail fast, before traffic arrives.
* **Parent exists but the file cannot be written** (a read-only directory, a full disk) — the
  handle is opened lazily on the *first write*, the failure is caught and counted, and routing
  continues: the record is simply missing from the log. `write()` swallows `OSError` on purpose,
  because a full disk must not take down the request path; the price is a silent hole in the
  training set. The count is on the sink — `router.route()` exposes it as `out["log"]["write_errors"]`
  — and the only defence is to look at it. Point `path` at a volume the process user can write
  to, and check that number after an incident.

See [section 11](#11-logging) for what each `excerpt_mode` permits you to distill later.

### `shadow`

Read by `Router.__init__` only, and only for two keys.

| key | type | default | what it does |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Allow the router-level side-run of a second backend. |
| `sample_rate` | float | `1.0` | Fraction of classified requests to side-run. Deterministic per excerpt. |

No validation errors. `sample_rate` is `float(...)`-coerced; `_sample_hit` treats `>= 1.0` as
always and `<= 0.0` as never.

`shadow.enabled: true` is **necessary but not sufficient**. `Router.__init__` sets

```python
self._shadow_enabled = bool((policy.shadow or {}).get("enabled")) and shadow_backend is not None
```

so the side-run only happens when a second `DecisionBackend` is passed to the `Router`
constructor. `Router.from_policy_file` does not pass one. See [section 12](#12-shadow) for the
two distinct shadow mechanisms and which config drives each.

### Keys that appear in YAML and are read by nothing

This list is the answer to "does that knob do anything?". Each entry was checked by reading
every consumer of the section.

| key | where it appears | status |
| --- | --- | --- |
| `cache.skip_when_gate_forced` | in older copies of `policies/default.yaml` and `deploy/k8s/policy-cluster.yaml`; removed from both on 2026-09-20 | **Ignored, and now removed.** `build_cache` reads `enabled`, `kind`, `ttl_seconds`, `max_entries`, `url`, `key_prefix` and nothing else; no other module reads it. The behaviour the comment describes is already true for a different reason: a gate-forced request under `on_force_local: skip_backend` never reaches the cache, because the cache is consulted only inside the `else` branch that calls the backend. Under `still_classify` the gate-forced request *is* cached, whatever this key says. |
| `shadow.primary` | commented out in `policies/default.yaml` | **Ignored.** The router-level shadow takes its second backend from the `Router(shadow_backend=...)` constructor argument. The `backend.primary` key (under `backend:`, for `backend.name: shadow`) is the one that is read. |
| `shadow.shadow` | commented out in `policies/default.yaml` | **Ignored.** Same reason; the read key is `backend.shadow`. |
| `shadow.log_disagreements` | commented out in `policies/default.yaml` | **Ignored.** `build_backend` reads `backend.log_disagreements`. The router-level shadow always records disagreements in the record's `shadow` field. |

Because unknown keys are silently ignored, these four are harmless — but they are also the
reason to read this section rather than the comments in a shipped YAML file. If a key is not in
the tables above, it does nothing.

---

## 3. The expression grammar

Rule conditions are Python syntax, but they are **parsed, not `eval`'d as text**.
`Expression.compile` runs `ast.parse(text, mode="eval")`, walks every node, and rejects anything
outside a fixed whitelist:

```python
_ALLOWED_NODES = frozenset(
    {
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.USub,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Subscript,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
    }
)
```

The compiled tree is then evaluated with builtins removed:

```python
eval(code, {"__builtins__": {}}, dict(namespace))
```

### Why it is built this way

A policy file is configuration. It is loaded at startup from wherever operators keep config — a
ConfigMap, a repo, a mounted volume, an S3 bucket with loose permissions. `eval()` on operator
config is not a security control, and "we checked that it looked safe" is not one either. The
whitelist turns the question from *"is this string harmless?"* — which is undecidable in general
— into *"is every node in this tree one of 29 known types?"*, which is a loop.

The second half of the design is **when** it fails. Everything is validated at load time, and
`PolicyError` names the offending node:

```text
PolicyError: disallowed syntax 'Call' in rule expression 'len(metadata) > 3'. Allowed: names,
constants, comparisons, and/or/not, in/not in, literal lists/sets/tuples, subscripting, + - *.
No calls, no attribute access.
```

A policy that will not load does not deploy. That is strictly better than a policy that loads
and then raises on the first request matching a malformed rule, at 3am, under load.

The one exception is a *runtime* failure inside a syntactically valid expression — a missing
metadata key, for example. That cannot be caught at load time, because the namespace does not
exist yet. It surfaces as:

```text
PolicyError: rule expression "metadata['nope'] == 1" failed to evaluate: 'nope'
```

raised out of `route_text` for that one request. See the metadata guidance in
[section 4](#4-expression-variables) for how to write a subscript that cannot fail.

### What is allowed

| construct | example | notes |
| --- | --- | --- |
| name from the allow-list | `pii_present` | Unknown names are rejected at load time. |
| constant | `0.8`, `"frontier"`, `True`, `None` | Strings, numbers, booleans, `None` — but **not dict literals**: `Dict` is not in the node whitelist, so `metadata == {'a': 1}` is a `PolicyError: disallowed syntax 'Dict'` at load time. |
| `==` `!=` `<` `<=` `>` `>=` | `pii >= 0.5` | |
| `in` / `not in` | `sensitivity in ["confidential", "regulated"]` | Works on lists, sets, tuples, and dicts (dict = key membership). |
| `is` / `is not` | `gate_sensitivity_floor is None` | Use it for `None` only. |
| chained comparison | `0.35 <= pii < 0.5` | One `ast.Compare` with several operators. |
| `and` / `or` / `not` | `not pii_present or degraded` | Short-circuits, which is how you guard a subscript. |
| literal list / set / tuple | `["a", "b"]`, `{"a", "b"}`, `("a",)` | |
| subscript on a simple name | `metadata['tenant']` | |
| `+` `-` `*` | `char_len * 2 + word_count > 500` | Binary. |
| unary minus | `-word_count < 0` | `ast.USub`. |
| parentheses | `(a or b) and c` | Grouping only; not a separate node type. |

### What is rejected

Every row below was checked against `compile_expression`. The error names the node type.

| rejected | node | why it is not on the whitelist |
| --- | --- | --- |
| `len(metadata) > 3` | `Call` | No calls. A call is arbitrary code execution. |
| `metadata == {'a': 1}` | `Dict` | No dict literals (or `{**metadata}` splats). Comparison against a literal map is not a routing decision. |
| `metadata.get('t') == 'x'` | `Attribute` + `Call` | No attribute access. `x.y` reaches any object in the namespace. |
| `complexity.upper() == 'FRONTIER'` | `Attribute` + `Call` | Same. Normalise in the code that builds the namespace, not in the rule. |
| `__import__('os').system('ls')` | `Call` | Rejected twice over: no calls, and `__import__` is not a known name. |
| `[c for c in metadata]` | `ListComp` | No comprehensions (or `SetComp`, `DictComp`, `GeneratorExp`). |
| `lambda: True` | `Lambda` | No lambdas. |
| `f'{complexity}'` | `JoinedStr` | No f-strings. They are calls to `format` in disguise. |
| `(x := 1) and True` | `NamedExpr` | No walrus. It would bind a name into the namespace. |
| `[*gate_detectors]` | `Starred` | No unpacking. |
| `2 ** 8 > 1` | `Pow` | `**` is not whitelisted. |
| `char_len % 2 == 0` | `Mod` | `%` is not whitelisted. |
| `char_len // 2 > 1` | `FloorDiv` | `//` is not whitelisted. |
| `char_len / 2 > 1` | `Div` | `/` is not whitelisted either. |
| `metadata['a']['b'] == 1` | nested `Subscript` | "only simple names may be subscripted". |
| `metadata['a':'b']` | `Slice` | Slices are not whitelisted. |
| `pii if pii_present else 0.0` | `IfExp` | No ternary. Write two rules, or use `and`/`or`. |
| `yield 1`, `del x`, `a; b` | — | Not valid in `mode="eval"` at all: a `SyntaxError` becomes a `PolicyError`. |

The arithmetic restriction is worth stating plainly because it surprises people: **only `+`,
`-`, and `*` are allowed.** `ast.BinOp` is whitelisted, but of its operators only `ast.Add`,
`ast.Sub`, and `ast.Mult` are. `**`, `//`, `%`, and `/` are all rejected, so you cannot write a
ratio or a modulo test in a rule. Compute it in the code that builds the namespace instead — a
feature is a better place for arithmetic than a policy file, because a feature is tested.

Note also what the whitelist does *not* need to defend against: `metadata[0]` is accepted (a
constant subscript on a simple name), and so is `True` as a whole expression. Neither is a
security problem, and rejecting them would only make the grammar harder to explain.

### YAML quoting, the practical trap

Expressions contain quotes, so the YAML scalar has to be quoted too. This does not parse:

```yaml
rules:
  - id: topic.minors
    if: "kw_minors" in advisory_topics     # YAML error: scalar followed by content
```

```text
PolicyError: policy YAML failed to parse: while parsing a block mapping ...
expected <block end>, but found '<scalar>'
```

Two forms do work. Wrap the whole expression in single YAML quotes and keep double quotes
inside:

```yaml
    if: '"kw_minors" in advisory_topics'
```

Or wrap it in double YAML quotes and use single quotes inside the expression:

```yaml
    if: "'contract' in metadata and metadata['contract'] == 'public-sector'"
```

Pick one and use it consistently in a file, because a reviewer should be able to scan the `if:`
column without parsing quotes in their head.

---

## 4. Expression variables

`EXPRESSION_VARIABLES` in `policy.py` is the allow-list of names a rule may use.
`_namespace()` in `router.py` is the dict actually passed to `Expression.evaluate()`. They are
the same 31 keys — comparing the two sets gives an empty symmetric difference — so there is no
variable you can name that will not be bound, and none that is bound but unusable.

An unknown name is a load error, not a runtime error:

```text
PolicyError: unknown variable 'tenant' in rule expression 'tenant == "acme"'. Available:
complexity, sensitivity, domain, complexity_confidence, sensitivity_confidence,
domain_confidence, pii, pii_present, gate_force_local, gate_blocks_backend,
gate_sensitivity_floor, gate_detectors, advisory_topics, semantic_fired, semantic_score,
semantic_force_local, semantic_mode, degraded, cached, tier, model,
char_len, word_count, line_count, code_blocks, question_marks, has_stack_trace, lang,
n_prior_turns, requested_model, metadata
```

### The judgement variables

These come from the decision backend, then from the merge and escalation steps.

| variable | runtime type | values | where it comes from |
| --- | --- | --- | --- |
| `complexity` | `str` | `trivial` \| `standard` \| `hard` \| `frontier` | **Effective** value: the backend's choice, possibly escalated by `on_uncertain`. |
| `sensitivity` | `str` | `public` \| `internal` \| `confidential` \| `regulated` | **Effective** value: `max(backend choice, the combined floor of both gate layers)` — layer 1's deterministic floor and, in enforce mode, layer 2's asserted level — then possibly escalated by `on_uncertain`. |
| `domain` | `str` | `code` \| `writing` \| `analysis` \| `chat` \| `data-extraction` | The backend's raw choice. Never merged with the gate, never escalated. |
| `complexity_confidence` | `float` 0..1 | | The backend's **raw** reported confidence for complexity. Escalation changes the choice, not this number. |
| `sensitivity_confidence` | `float` 0..1 | | The backend's **raw** reported confidence for sensitivity. |
| `domain_confidence` | `float` 0..1 | | Raw. `0.0` when `backend.include_domain` is false. |
| `pii` | `float` 0..1 | | Probability PII is present: `max(backend noul, gate pii_floor)`. A gate hit on a PII detector sets the floor to `1.0`, because a checksum-validated identifier is not a probability. |
| `pii_present` | `bool` | | `pii >= pii_threshold`, **or** the `on_uncertain` band fired. |

The distinction between `complexity` and `complexity_confidence` matters when you write a rule.
`sensitivity` has already been bumped by the time you see it; `sensitivity_confidence` is the
number that *caused* the bump. A rule like `sensitivity_confidence < 0.6` therefore tests the
backend's uncertainty, not the effective label, and it is a legitimate thing to want — for
example "route anything the backend is this unsure about to a tier a human reads".

The ladders are fixed in `schema.py` and are not configurable:

```python
COMPLEXITY_LEVELS = ("trivial", "standard", "hard", "frontier")
SENSITIVITY_LEVELS = ("public", "internal", "confidential", "regulated")
DOMAINS = ("code", "writing", "analysis", "chat", "data-extraction")
TIERS = ("local", "cheap", "strong")
```

### The gate variables

From `GateVerdict`, produced by `HardGate.scan()` on the raw excerpt before any model is
consulted.

| variable | runtime type | values | notes |
| --- | --- | --- | --- |
| `gate_force_local` | `bool` | | At least one non-advisory detector with `force_local` matched. |
| `gate_blocks_backend` | `bool` | | A blocking-class detector matched. Nothing was sent anywhere, not even redacted text. |
| `gate_sensitivity_floor` | `str` or `None` | a level name, or `None` | The highest floor from non-advisory findings. `None` when no non-advisory detector matched. |
| `gate_detectors` | `set[str]` | detector names | **A set**, built as `{f.detector for f in verdict.findings}`. Includes advisory detectors. |
| `advisory_topics` | `set[str]` | detector names | Only the advisory (topic-keyword) detectors that matched. |

Because both are sets, `in` is the natural test and it is O(1):

```yaml
    if: '"payment_card" in gate_detectors'
```

`gate_detectors` answers "is sensitive *data* present, or a regulated *topic* mentioned?" and
includes both kinds. `advisory_topics` answers only the second. The distinction is the whole
reason the gate has advisory detectors: `"explain how HIPAA works"` mentions a regulated domain
and contains nothing regulated, and routing every compliance question to an air-gapped GPU is
exactly the failure this project exists to fix.

Detector names you can match on — 22 in the default set, listed in full in
[section 9](#9-gate-configuring-the-local-hard-gate).

### The semantic variables

Rules see the *actionable projection* of the layer-2 assessment, not the raw assessment: the
first three variables are **zeroed unless the layer is enforcing** (`semantic.enforced`). That
is shadow's contract in arithmetic — a rule that read a shadow score and routed on it would let
a layer that may not change anything change a routing decision without anybody deciding to.
`semantic_mode` is always the truth, so a policy can tell the cases apart, and the full
assessment, score included, is on the logged record either way. With the layer `off` or not
present, all four read as if nothing happened (`"off"`). Full semantics:
[docs/gate-layers.md](gate-layers.md).

| variable | runtime type | values | notes |
| --- | --- | --- | --- |
| `semantic_fired` | `bool` | | True when the layer is enforcing and fired (score at or above threshold). Zeroed in shadow. |
| `semantic_score` | `float` 0..1 | | The layer-2 score, when enforcing. `0.0` in shadow, `off`, or when no artifact is loaded. |
| `semantic_force_local` | `bool` | | True when the layer is enforcing and fired; the router then treats it exactly like a layer-1 `force_local` for egress. Zeroed in shadow. |
| `semantic_mode` | `str` | `off` \| `shadow` \| `enforce` | The mode the layer ran in for this request. Always the truth, in every mode. |

### The feature variables

From `RequestFeatures`, computed locally by `prompts.compute_features()` over the **raw**
excerpt. Counts and booleans only: nothing here reproduces the prompt, which is why they are
safe to log even in `excerpt_mode: hash`.

| variable | runtime type | how it is computed |
| --- | --- | --- |
| `char_len` | `int` | `len(raw_excerpt)`. The excerpt is capped at `MAX_EXCERPT_CHARS` (4000). |
| `word_count` | `int` | Whitespace-separated token count. |
| `line_count` | `int` | `text.count("\n") + 1`, or `0` for empty text. |
| `code_blocks` | `int` | Fenced blocks: half the number of triple-backtick fences, integer-divided. |
| `question_marks` | `int` | `text.count("?")`. |
| `has_stack_trace` | `bool` | A stack-frame, Python traceback, or native frame pattern matched. |
| `lang` | `str` | Coarse stopword guess: `en`, `de`, `fr`, `es`, `it`, `pt`, `hu`, `nl`, or `und`. Needs at least 4 words and a real signal; otherwise `und`. |
| `n_prior_turns` | `int` | Count of `user` messages minus one, from the message list. `0` for `route_text`. |

`char_len` is bounded by the excerpt budget, not by your prompt. A 90 KB prompt and a 4000
character prompt both read `char_len == 4000`. If you need to distinguish them, use
`n_prior_turns` or put the real length in `metadata`.

`lang` is a fingerprint, not a language detector. It exists to separate the languages a router
actually sees; it will return `und` on short text and on languages it has no stopword list for.
Do not build a compliance rule on it.

`RequestFeatures` has 24 fields. Only these 8 are exposed to rules. The other 16 —
`sentence_count`, `mean_word_len`, `digit_ratio`, `upper_ratio`, `punct_ratio`,
`non_ascii_ratio`, `inline_code_spans`, `urls`, `exclamations`, `has_diff`, `has_json`,
`n_messages`, `tool_output_present`, `gate_detectors` (the name→count mapping),
`n_gate_findings`, `gate_force_local` — are logged and used as distillation features, but a rule
cannot read them. If you need one in a rule, that is a code change to `_namespace()`, not a
policy change.

### The request variables

| variable | runtime type | values | notes |
| --- | --- | --- | --- |
| `requested_model` | `str` | model name or `""` | The model the client asked for. `""` when the call did not come through a proxy (`route_text`, CLI, evals). |
| `metadata` | `dict[str, Any]` | | `dict(metadata or {})` — **exactly the mapping the caller passed**, unfiltered. |

`metadata` deserves care, because it is the one variable whose contents are not controlled by
this library:

* Rules see the caller's dict as given, including nested dicts and lists. You can write
  `metadata['tags']` and test membership on it.
* **Everything else sees the key-filtered copy, and the key filter runs first.** The six
  prompt-bearing keys — `messages`, `prompt`, `content`, `input`, `raw`, `body`
  (`UNSAFE_METADATA_KEYS`) — are dropped on *both* paths: the decision record and the backend.
  Metadata is caller-supplied and can smuggle unredacted text; those six keys are the smuggling
  routes, and a caller that forwards a whole request body as `metadata={"prompt": ...}` would
  otherwise write the raw prompt into the training dataset, silently defeating
  `excerpt_mode: hash`.
* The **decision record** then applies a second, *type* filter: only scalar values
  (`None`, `str`, `int`, `float`, `bool`) survive into the log. A nested dict is dropped from
  the log but still visible to your rules — so a rule can depend on data that is not in the
  training set. Prefer scalars in metadata if you want the rule and the dataset to agree.

A missing key is a runtime error, not `False`:

```text
PolicyError: rule expression "metadata['nope'] == 1" failed to evaluate: 'nope'
```

Guard it with short-circuit `and`, which is the only safe subscript form the grammar allows:

```yaml
    if: "'plan' in metadata and metadata['plan'] == 'enterprise'"
```

### Four variables that are constant at rule-evaluation time

These are in the allow-list, and they are bound, but they cannot vary. Documenting them as
useful would be a lie, so here is the truth about each.

| variable | value rules always see | why |
| --- | --- | --- |
| `cached` | `False` | `_namespace()` hardcodes `"cached": False`. The cache is consulted *before* the namespace is built, and the hit is recorded on `RoutingDecision.cached` (`backend_name.endswith(":cached")`), not in the namespace. |
| `tier` | `""` | Hardcoded. The tier is the *output* of rule evaluation, so it cannot also be an input. |
| `model` | `""` | Hardcoded, same reason. |
| `degraded` | `False` | `_route` short-circuits before `Policy.evaluate` when the backend came back degraded: `if backend_result.degraded: tier, rule_id, reason = self._failure_tier(...)`. Rules never run on a degraded result, so `degraded` is `False` in every namespace that is ever evaluated. |

All four are kept in the allow-list because they are part of the documented namespace shape and
because a rule author reading a decision record sees fields with those names. Writing a rule
against any of them is a no-op:

```yaml
    if: degraded          # never true; a degraded request never reaches the rules
    if: cached            # never true; use RoutingDecision.cached when reading the log
    if: tier == ""        # always true
```

Use `decision.cached` and `decision.degraded` when you read the log. Do not use them when you
write a policy.

### Example expressions

All eight below use only allow-listed syntax and only variables from the tables above. Each was
compiled and evaluated against a real router namespace.

```yaml
# 1. The gate's hard verdict. Belt-and-braces: the router already merged the floor
#    into `sensitivity`, so this rule is about intent, not about safety.
- id: gate.force-local
  if: gate_force_local
  then: { tier: local }
  reason: the local hard gate matched a structured identifier or credential

# 2. Data protection. Effective sensitivity, so it already includes the gate floor
#    and any uncertainty bump.
- id: data.sensitive
  if: sensitivity in ["confidential", "regulated"] or pii_present
  then: { tier: local }
  reason: sensitive or personal data must not leave the infrastructure

# 3. The PII band the threshold does not cover: a real probability that is not a yes.
- id: data.maybe-personal
  if: pii > 0.2
  then: { tier: local }
  reason: a non-trivial chance of personal data is treated as personal data

# 4. A regulated topic, without regulated data. Advisory only, so it never moved a floor.
- id: topic.health-or-minors
  if: '"kw_health_regulation" in advisory_topics or "kw_minors" in advisory_topics'
  then: { tier: local-strong }
  reason: special-category topic handled on-prem as a matter of policy

# 5. Capability, reached only once data protection is satisfied.
- id: complexity.frontier
  if: complexity == "frontier"
  then: { tier: strong }
  reason: task needs frontier-class reasoning

# 6. Shape-based routing. Arithmetic is limited to + - *, so no ratios here.
- id: long-hard
  if: char_len > 3000 and complexity == "hard"
  then: { tier: strong }
  reason: long hard task; the cheap tier loses the thread

# 7. A credential detector by name, from the set of detectors that fired.
- id: secret.in-prompt
  if: '"provider_api_key" in gate_detectors or "inline_credential" in gate_detectors'
  then: { tier: local }
  reason: a prompt carrying live credentials is answered on self-hosted models

# 8. Caller metadata, guarded so a missing key cannot raise at request time.
- id: tenant.pinned
  if: "'contract' in metadata and metadata['contract'] == 'public-sector'"
  then: { tier: local, model: mistral-eu-selfhosted }
  reason: a public-sector contract pins one EU deployment exactly
```

---

## 5. Tiers and `tier_order`

### A tier is a list of deployments

`tiers` maps a policy-level name to one or more concrete LiteLLM model names — the `model_name`
entries in your LiteLLM proxy `model_list`, not provider identifiers:

```yaml
# config: policies/default.yaml
tiers:
  local:
    - qwen38            # self-hosted: data never leaves the building
  cheap:
    - qwen3.8-flash
  strong:
    - qwen3.8-max
```

`RoutingDecision` keeps the two apart on purpose. `tier` is what the policy reasons about;
`model` is what the provider config owns. That separation is what lets you change a deployment
without touching a policy, and change a policy without touching a deployment.

### Round-robin within a tier

When a tier lists several models, `Policy.pick_model` round-robins them:

```python
def pick_model(self, tier: str) -> str:
    models = self.models_for_tier(tier)
    if len(models) == 1:
        return models[0]
    with self._cursor_lock:
        index = self._cursor_state.get(tier, 0)
        self._cursor_state[tier] = (index + 1) % len(models)
    return models[index]
```

The cursor is in-process state, guarded by a lock, and starts at `0` on every boot. So:

* the spread is even within a process, not across a fleet. Two proxy workers each run their own
  cursor, and both start on the first model.
* a restart resets the cursor. It is deliberately not persisted: a routing decision must not
  depend on durable state it can lose.
* a single-model tier never touches the lock. Most tiers are one model.

List several models in a tier when you want load spread across identical deployments — two
replicas of the same on-prem model behind two LiteLLM entries, for example. Do not list models
of *different* capability in one tier; the tier is the unit your policy reasons about, and a
tier that sometimes means "the 70B model" and sometimes means "the 8B model" is a tier you
cannot write a rule about.

### `then.model` pins exactly

A rule may name a model directly:

```yaml
- id: tenant.pinned
  if: "'contract' in metadata and metadata['contract'] == 'public-sector'"
  then:
    tier: local
    model: mistral-eu-selfhosted
  reason: a public-sector contract pins one EU deployment exactly
```

`Policy.evaluate` checks `rule.model` first:

```python
if rule.model:
    return rule, (rule.tier or ""), rule.model
tier = rule.tier or self.tier_order[-1]
return rule, tier, self.pick_model(tier)
```

A pinned model bypasses round-robin entirely. That is the point: pinning exists for tenancy and
residency commitments, where "one of three equivalent deployments" is not good enough and
"this one, every time" is. Keep `tier:` on the rule anyway — it is recorded for provenance, and
your log analysis groups by tier.

The model name is **not** validated against `tiers:`. Only `then.tier` is. A typo in
`then.model` produces a decision naming a model your LiteLLM config does not have, and the
failure appears in the proxy, not in the policy loader. Check pinned names against your
`model_list` in review.

### `tier_order` is ascending capability

```yaml
tier_order: [cheap, strong]
```

Left is weaker and cheaper, right is stronger and dearer. `Policy.bump_tier` walks it:

```python
def bump_tier(self, tier: str, steps: int = 1) -> str:
    if tier not in self.tier_order:
        return tier
    idx = level_index(tier, self.tier_order)
    return self.tier_order[max(0, min(len(self.tier_order) - 1, idx + steps))]
```

Two facts about that function, both verified against the shipped default policy:

**`bump_tier` on a tier that is not in `tier_order` returns it unchanged.** With
`tier_order: [cheap, strong]`, `bump_tier("local")` is `"local"`, and `bump_tier("local", 3)` is
still `"local"`. There is no error and no wraparound.

**The shipped default omits `local` on purpose.**

```yaml
# config: policies/default.yaml
tier_order: [cheap, strong]   # no `local`
```

`local` is a **data-residency** tier, not a capability rung. Its defining property is "self-
hosted, nothing leaves the building", which is orthogonal to how smart the model is. Putting it
in an ascending capability list would assert that `local` is weaker than `cheap` and that
bumping `cheap` upward could land on `local` — neither of which is a claim the router should
make. Omitting it means:

* `bump_tier("local")` is a no-op, so no escalation path can silently move regulated traffic
  onto a cloud tier or cheap cloud traffic onto your on-prem GPU;
* the order describes only the ladder a capability escalation is allowed to climb;
* an operator who *does* have a capable on-prem deployment adds a distinct tier for it
  (`local-strong`) rather than overloading `local`. Example (a) in
  [section 13](#13-five-complete-examples) does exactly that.

If you omit `tier_order`, you get `schema.TIERS` — `["local", "cheap", "strong"]` — which
*does* include `local`, and which requires all three tier names to exist. See
[section 2](#tiers-tier_order-pii_threshold).

### What `tier_order` actually drives today

Be precise about this, because the honest answer is narrower than the name suggests.

`bump_tier` is a public method on `Policy` and **no code path in the router calls it**. The
uncertainty machinery escalates *labels*, not tiers: `escalate_complexity` and
`escalate_sensitivity` move a request up `COMPLEXITY_LEVELS` / `SENSITIVITY_LEVELS`, and your
rules then map the escalated label to a tier. That is the right shape — a policy should decide
what a stricter label costs — but it means `tier_order` is not on the hot path of a routing
decision.

What `tier_order` does today:

* it is validated at load (defined tiers, no duplicates);
* it is the ladder `bump_tier` walks, for code that calls it — your own escalation logic, tests,
  and the LiteLLM routing plugin, which ranks fallback candidates safest-first using
  `on_backend_down.fail_closed_tier` followed by `tier_order`;
* `Policy.evaluate` has a `tier_order[-1]` fallback for a rule with neither tier nor model,
  which load-time validation makes unreachable.

Write `tier_order` anyway. It documents your capability ladder for the next person reading the
file, it is what the plugin ranks by, and it is what `bump_tier` will walk the day you write a
rule that needs it.

### Label escalation

`escalate_sensitivity` and `escalate_complexity` both delegate to `schema.bump_within`, which
moves `steps` positions up an ascending ladder and clamps at both ends:

```python
def bump_within(level: str, ladder: Sequence[str], steps: int) -> str:
    idx = level_index(level, ladder)
    if idx < 0:
        return level
    return ladder[max(0, min(len(ladder) - 1, idx + steps))]
```

Verified against the shipped defaults:

| call | result |
| --- | --- |
| `escalate_sensitivity("public", 1)` | `internal` |
| `escalate_sensitivity("regulated", 5)` | `regulated` (clamped at the top) |
| `escalate_complexity("trivial", 2)` | `hard` |
| `escalate_complexity("frontier", 2)` | `frontier` (clamped at the top) |
| an unknown level | returned unchanged (`level_index` gives `-1`) |

A positive `steps` is always the conservative direction, because both ladders are declared
ascending. Clamping at the top is why `sensitivity_bump_levels: 99` is a legitimate way to write
"any uncertainty about sensitivity means regulated".

---

## 6. Rule ordering

### First match wins

`Policy.evaluate` walks `rules` in file order and returns on the first match:

```python
for rule in self.rules:
    if rule.is_default or rule.expression.evaluate(namespace):
        ...
        return rule, tier, self.pick_model(tier)
```

There is no scoring, no specificity, no priority field. Position *is* priority. That is a
deliberate limitation: a routing policy you can only understand by running a resolution
algorithm is a policy you cannot review in a pull request. Read top to bottom, first hit wins,
done.

The consequences are worth internalising before you write rules:

* A broad rule early swallows everything below it. `if: sensitivity in ["internal",
  "confidential", "regulated"]` above your capability rules means the capability rules only ever
  see `public`.
* A narrow rule late is dead code. Putting `if: gate_blocks_backend` below
  `if: sensitivity in ["confidential", "regulated"]` still works — every blocking detector sets
  the floor to `regulated` (the six identifier detectors) or `confidential` (the four
  credential detectors; both are in that list) — but the second rule is what fires, and your
  log says so. Order rules so the id in the log is the id that explains the decision.
* Evaluation stops at the first match, so a request never pays for rules below it.

### Exactly one default rule, and it must be last

A default rule is a rule with no `if:`. `from_dict` enforces:

```python
if not rules:
    raise PolicyError("policy defines no rules")
defaults = [r for r in rules if r.is_default]
if len(defaults) > 1:
    raise PolicyError("at most one rule may omit `if:` (the default rule)")
if defaults and defaults[0] is not rules[-1]:
    raise PolicyError("the default rule (no `if:`) must be last")
if not defaults:
    raise PolicyError("policy must end with a default rule that has no `if:`")
```

All three are load-time errors:

```text
policy defines no rules
at most one rule may omit `if:` (the default rule)
the default rule (no `if:`) must be last
policy must end with a default rule that has no `if:`
```

The default rule is **required**, not optional. Without it, a request matching nothing would
have no answer, and the router would have to invent one at request time — under load, in
production, in a code path nobody tested. Failing loudly at load time is strictly better. The
`PolicyError("no rule matched and no default rule is defined")` at the end of `evaluate` is
annotated `# unreachable by validation`, and it is.

A default rule *not* last is also rejected, rather than silently ignored. If it were allowed,
every rule below it would be unreachable, and a reviewer would have to notice that. The loader
notices instead.

### Why data-protection rules come before capability rules

The shipped default policy puts them in this order:

```yaml
rules:
  - id: gate.force-local     # 1. the local gate's hard findings
  - id: data.sensitive       # 2. regulated / confidential / PII present
  - id: complexity.frontier  # 3. capability, only now
  - id: complexity.hard
  - id: default              # 4. everything else, cheapest tier that can do it
```

Because first match wins, order encodes precedence, and precedence encodes what you care about
most. Data protection first means **a hard task on regulated data is still regulated**. Reverse
the two blocks and a frontier-complexity prompt containing a patient record routes to your
strongest *cloud* model — the decision is technically correct on the capability axis and it is a
data-egress incident.

A router that saves money or improves answer quality by leaking a patient record has not saved
anything and has not improved anything. Cost and capability are optimisations; residency is a
constraint. Constraints go first. That is the entire reason rule order is the operator's
responsibility rather than something the engine infers.

Note what the ordering does *not* do. It is not the safety mechanism. Even if you delete both
data-protection rules, `_route` merges the gate's sensitivity floor into `sensitivity`
**before** `Policy.evaluate` runs, so a checksum-validated card number arrives at your rules
already labelled `regulated`. The gate is a floor on all rules, not one rule among them. Rule
order is how you express intent; the merge is what makes the intent un-bypassable.

---

## 7. `on_uncertain`: the calibration payoff

This is the section that makes jev-route a calibrated router instead of a classifier with extra
steps. An argmax cannot say "I am 62% sure this is internal". Jev can, and `on_uncertain` is
what you do with that.

```yaml
# config: policies/default.yaml
on_uncertain:
  sensitivity_confidence_below: 0.8
  sensitivity_bump_levels: 1
  complexity_confidence_below: 0.7
  complexity_bump_levels: 1
  pii_uncertain_threshold: 0.35
  pii_uncertain_counts_as_present: true
```

### Bumps only ever go stricter

Both escalations move *up* an ascending ladder (`SENSITIVITY_LEVELS`,
`COMPLEXITY_LEVELS`) and clamp at the top. There is no key that moves a label down, and no key
that reduces a gate floor. "70% sure it is internal" becomes "treat it as confidential";
nothing in the file can turn it into "treat it as public".

That asymmetry is the whole design. Uncertainty is a cost you pay in capability or money, never
in data protection. If you need a looser reading of an uncertain answer, write a rule against
`sensitivity_confidence` and own the decision explicitly, in a file a reviewer reads.

Set `sensitivity_confidence_below: null` (or `complexity_confidence_below: null`) to switch that
bump off entirely. `_route` guards each block with `is not None`, so `null` disables it rather
than making every request escalate:

```python
if (
    uncertainty.sensitivity_confidence_below is not None
    and backend_result.answers.sensitivity.confidence < uncertainty.sensitivity_confidence_below
    and classified
    and not backend_result.degraded
):
```

Measured against the shipped defaults, with a backend reporting `sensitivity: public` at
confidence `0.50`:

| config | effective sensitivity |
| --- | --- |
| `sensitivity_bump_levels: 1` | `internal` |
| `sensitivity_bump_levels: 2` | `confidential` |
| `sensitivity_confidence_below: null` | `public` (no bump) |
| backend says `regulated` at confidence `0.50`, bump `1` | `regulated` (clamped) |

### Escalation is skipped when nobody judged the request

Two guards sit on both bumps: `classified` and `not backend_result.degraded`.

`classified` is set to `True` only inside the branch that actually calls a backend:

```python
if skip_backend:
    backend_result = _gate_only_result(verdict)
    backend_name = "gate"
else:
    ...
    classified = True
```

So a request the gate refused to send anywhere gets **no confidence escalation**. Why: when the
gate skips the backend, `_gate_only_result` fills in the answers itself — sensitivity at the
gate's floor with confidence `1.0`, and complexity and domain as `ChoiceAnswer.uniform(...)`
with confidence `0.0`. Those uniform answers are maximum-uncertainty placeholders, not
judgements. Applying a confidence floor to them would report a `hard -> frontier` bump that no
model ever made.

`not backend_result.degraded` skips escalation for the same reason on the outage path: a
degraded result is `DecisionAnswers.unknown()`, all four answers uniform, all confidences
`0.0`. Escalating on that would fabricate a judgement out of a timeout.

The code comment states the principle, and it is worth quoting because it explains a decision
that looks like a bug from the outside:

> True only when a backend actually classified this request. Gate-blocked and skipped requests
> have uniform answers by construction, so applying confidence-floor escalation to them would
> report a "hard -> frontier" bump that nobody ever judged. Honest provenance matters more in a
> training log than a tidy-looking one.

This log is your training dataset. A row that says "escalated hard → frontier" teaches the
student that a model made that call. If nobody made it, the row is a lie, and lies in a training
set are expensive: you find them after you have trained on them.

Note that degraded results never reach the rules at all — `_route` short-circuits to
`_failure_tier`. See [section 8](#8-on_backend_down-outages-and-the-invariant).

### `pii_uncertain_counts_as_present`

A noul near `0.5` is a coin flip, not a "no". This is the fail-safe reading:

```python
pii_present = effective_pii >= self.policy.pii_threshold
if (
    uncertainty.pii_uncertain_counts_as_present
    and not pii_present
    and effective_pii >= uncertainty.pii_uncertain_threshold
):
    pii_present = True
    escalations.append(
        f"pii treated as present at p={effective_pii:.2f} (uncertain band >= {uncertainty.pii_uncertain_threshold})"
    )
```

So `pii_present` is true when `pii >= pii_threshold` **or** when `pii` lands in the band
`[pii_uncertain_threshold, pii_threshold)`. With the defaults that band is `[0.35, 0.5)`.

**The band is only meaningful when `pii_uncertain_threshold < pii_threshold.`** Set it to `0.6`
with a `pii_threshold` of `0.5` and the band is empty: any `pii >= 0.6` already set
`pii_present`, so the condition `not pii_present` is never true at the same time. The knob does
nothing and the file still loads. Check the two numbers together.

Unlike the two label bumps, this block has no `classified` or `degraded` guard. It does not need
one: a gate-skipped request that matched a PII detector carries `pii_floor = 1.0`, so
`pii_present` is already true; and a degraded result carries `pii = 0.5`, which is already
`>= pii_threshold` at the default. The band only does work on a real backend answer.

### Every escalation is logged

`escalations` is a list of human-readable strings assembled during the merge and the bumps, and
it lands on `RoutingDecision.escalated`, which lands in the decision record. These shapes
appear:

```text
sensitivity internal->confidential (local gate floor: confidential)
pii 0.20->1.00 (local gate matched an identifier)
complexity hard->frontier (confidence 0.55 < 0.7)
sensitivity public->internal (confidence 0.36 < 0.8)
pii treated as present at p=0.40 (uncertain band >= 0.35)
sensitivity internal->regulated (confidence 0.10 < 0.25: too uncertain to leave the infrastructure)
sensitivity internal->confidential (semantic gate layer 2 asserted confidential at p=0.99)
```

The first two are the gate merge (layer 1's floors, and — when an enforced layer 2 outranks them
— layer 2's, named as the origin). The middle three are the `on_uncertain` bumps and the
uncertain-PII band. The second-to-last is the `force_local_confidence_below` block: the answer
was so uncertain that it is treated as `regulated` outright. The last is layer 2 asserting its
level in enforce mode. The origin string names *which layer* raised the floor, because
"the gate said so" is not an explanation an operator can act on once there are two gates.

An empty `escalated` list means the decision is exactly what the backend said, unmodified. That
is a useful property for distillation: you can filter to rows where the teacher's raw judgement
survived intact, and you can audit every row where it did not. It is also the first place to look
when a routing decision surprises you — `jev-route explain <request_id>` prints the record, and
`escalated` tells you whether the policy or the model moved the answer.

The record also keeps the raw distribution alongside the effective choice, unchanged. `answers`
on the record carries the effective labels; the `probabilities` maps inside them are the
backend's, untouched.

---

## 8. `on_backend_down`: outages and the invariant

```yaml
# config: policies/default.yaml
on_backend_down:
  mode: fail_closed
  fail_closed_tier: local
  fail_open_tier: strong
```

A backend "cannot answer" when it returns `degraded: True` — a timeout, an HTTP error after
retries, an open circuit breaker, an unparseable body. `DecisionBackend` implementations are
required not to raise on an outage; they return maximum-uncertainty answers with `degraded` set
and a `degrade_reason` string. The router then skips the rules entirely:

```python
if backend_result.degraded:
    tier, rule_id, reason = self._failure_tier(
        verdict, backend_result.degrade_reason, semantic=assessment
    )
    model = self.policy.pick_model(tier)
else:
    rule, tier, model = self.policy.evaluate(namespace)
    rule_id, reason = rule.rule_id, rule.reason
```

The rule id is always `backend.down`, and the reason string carries the mode and the underlying
cause:

```text
decision backend unavailable (timeout after 5.0s); fail_closed -> tier local
decision backend unavailable (circuit open (state=open)); fail_open -> tier strong
```

### `fail_closed` vs `fail_open`

**`fail_closed`** routes to the safest tier — for a sensitivity-aware router, the tier whose data
never leaves the building. You choose it when the risk you care about is data egress.

**`fail_open`** routes to the most capable tier. You choose it when the risk you care about is
answer quality: an outage degrades your product instead of your compliance posture.

Both are defensible. Silently picking one for the operator is not, which is why it is explicit
config with a safe default (`fail_closed` → `local`).

Both tier names are validated at load whether or not the mode uses them, so flipping `mode`
during an incident cannot fail. That is the point of validating the unused one.

### The invariant: the gate outranks `fail_open`

`router.py` declares:

```python
#: Sensitivity levels at which the gate's floor outranks a ``fail_open`` config.
_GATE_OVERRIDES_FAIL_OPEN = ("confidential", "regulated")
```

and `_failure_tier` enforces it:

```python
mode = self.policy.failure.mode
tier = self.policy.failure.fail_open_tier if mode == "fail_open" else self.policy.failure.fail_closed_tier
note = f"decision backend unavailable ({reason or 'unknown'}); {mode} -> tier {tier}"
combined_floor = resolve_floor(
    verdict.sensitivity_floor, semantic.sensitivity_floor if semantic is not None else None
)
gate_index = level_index(combined_floor or "", SENSITIVITY_LEVELS)
overrides = gate_index >= level_index(_GATE_OVERRIDES_FAIL_OPEN[0], SENSITIVITY_LEVELS)
if (verdict.force_local or overrides) and mode == "fail_open":
    tier = self.policy.failure.fail_closed_tier
    note += f"; local gate floor took precedence over fail_open -> tier {tier}"
return tier, "backend.down", note
```

The floor that outranks `fail_open` is the *combined* floor of both gate layers, the same
`resolve_floor` maximum the normal path applies. A request whose only sensitivity signal is an
enforced layer-2 assertion is the same egress risk during an outage as one layer 1 judged, and
`fail_open` must not be the path that sends it out. (In **shadow** mode the layer-2 floor is
`None` by construction, so a shadow layer can no more rescue a `fail_open` request than it can
change any other decision.)

In words: **if either local gate layer forced local, or the combined floor of the two layers is
`confidential` or `regulated`, then `fail_open` is overridden and the request goes to
`fail_closed_tier` anyway.** A config knob is not allowed to become a data-egress path. The docstring on `_failure_tier` puts
it exactly that way:

> `fail_open` is honoured for quality risk, but never at the expense of the gate: if either
> local layer found sensitive data, the safest tier wins regardless of how the operator
> configured outages. A config knob is not allowed to become a data-egress path.

Measured against a backend that always degrades, with
`on_backend_down: {mode: fail_open, fail_closed_tier: local, fail_open_tier: strong}`:

| request | gate floor | tier | reason recorded |
| --- | --- | --- | --- |
| `Write a Python function that reads a CSV.` | none | `strong` | `... fail_open -> tier strong` |
| `Explain how HIPAA works.` (advisory only) | none | `strong` | `... fail_open -> tier strong` |
| `internal only: restructure the deploy pipeline` (advisory-only) | none — `kw_internal_only` is advisory | `strong` | `... fail_open -> tier strong` |
| `mail dana.wu@northside-health.org ...`, `on_force_local: still_classify` | `confidential` | `local` | `... fail_open -> tier strong; local gate floor took precedence over fail_open -> tier local` |

An `internal` floor does **not** override `fail_open`; only `confidential` and `regulated` do
(the arithmetic: `level_index("internal") < level_index("confidential")`). `internal` is "ours,
not secret", and routing it to a capable cloud model during an outage is a quality decision, not
a leak. Note that none of the shipped detectors sets a non-advisory `internal` floor, so in
practice this row of the ladder is unreachable — it is stated for the arithmetic's completeness,
and pinned by the `confidential`/`regulated` cases of the test.

The fourth row needs `on_force_local: still_classify` to be reachable at all. Under the default
`skip_backend`, an email address means the backend is never called, so there is no outage to
recover from: the decision comes from the gate at full confidence and `degraded` is false.
`still_classify` is what creates the window where a gate-forced request meets a dead backend,
and the invariant is what closes it.

A blocking-class detector (card number, SSN, IBAN, NHS number, NI number, private key, provider
API key, inline credential, basic-auth URL, medical record number) always sets
`blocks_backend`, which always skips the backend regardless of `on_force_local`. Those requests
never reach `_failure_tier` at all.

---

## 9. `gate`: configuring the local hard gate

```yaml
# config: policies/default.yaml (abridged; the shipped file also configures
# `blocked_metadata` and `semantic`, documented below)
gate:
  on_force_local: skip_backend
  disabled_detectors: []
  placeholder_domains_as_pii: false
```

The gate has **two local layers** in this build. Layer 1 is the deterministic detector set this
section configures; layer 2 is the semantic layer under `gate.semantic`, documented in
[docs/gate-layers.md](gate-layers.md). Layer 1 runs first, in-process, with no network, on the
**raw** excerpt, on every request, and nothing can be routed without it having run: `Router`
always constructs a `HardGate`, and `scan` is called on every request — that is the invariant
the tests pin. What the invariant does *not* say is that a detector must fire: the detector list
can be emptied (see `disabled_detectors` below), in which case the gate runs and finds nothing.
Within what it finds, a finding can only make the outcome stricter, and you cannot talk a
detector out of a match.

### `on_force_local`

What to do when the gate has already decided a request must stay local.

**`skip_backend`** (default). Do not call the decision backend at all. `_route` sets
`skip_backend = verdict.blocks_backend or ((verdict.force_local or assessment.force_local) and
self._on_force_local == "skip_backend")` — an enforced layer-2 firing skips the backend exactly
like a layer-1 force — and the answers come from `_gate_only_result`: sensitivity at the gate's floor
with confidence `1.0` and `confidence_reported: false`, complexity and domain uniform, `pii` at
the gate's floor, `model_version: "gate-1.0.0"`, `backend: "gate"` on the decision.

Nothing leaves the process. You pay nothing. The request is still logged, with the gate's
deterministic verdict, which is a high-confidence training label precisely because a regex and a
checksum produced it.

The cost is coverage: these rows carry no complexity or domain judgement from a real model, and
`classified` is false so no uncertainty escalation applies to them.

**`still_classify`**. Call the backend anyway, on the **redacted** excerpt, so the log carries
complexity and domain labels for these prompts too. Better distillation coverage — your student
model sees the full shape of your traffic instead of a hole wherever the gate fired.

The cost is egress and money. Redaction is best-effort, which is exactly why `blocks_backend`
exists as a separate flag: for the dangerous class the router does not rely on redaction at all.
`still_classify` never overrides `blocks_backend`; it only affects the `force_local`-without-
blocking case (email, phone, date of birth, street address, named individual).

Choose `still_classify` when you are bootstrapping and accumulating a dataset, and when the
residual risk of a redacted excerpt reaching your decision backend is acceptable under your
DPA. Choose `skip_backend` when it is not. `still_classify` also widens the window where the
[section 8](#8-on_backend_down-outages-and-the-invariant) invariant does the work.

Invalid values are rejected at Router construction, lower-cased first:

```text
ValueError: gate.on_force_local must be skip_backend|still_classify, got 'bogus'
```

### `disabled_detectors`

Individual detectors may be silenced. Some shops have legitimate high-volume phone numbers in
prompts — an order-number field that trips `phone_number` on every request is a gate that cries
wolf, and a gate that cries wolf gets switched off, which is worse than not having it.

```yaml
gate:
  disabled_detectors: [phone_number, street_address]
```

Unknown names raise at construction, checked against `DEFAULT_DETECTORS`:

```text
ValueError: unknown detector names in disabled_detectors: ['nope']
```

Disabling a detector removes it from `self.detectors`, so it does not scan, does not redact, and
does not appear in `gate_detectors`. **The gate itself always runs** — `scan` is called on every
request and always returns a verdict — but note that there *is* a key that empties the detector
list: `disabled_detectors` naming all 22 names yields a gate that runs and finds nothing. The
tests pin that state as *legal* (the gate object exists and `scan` ran), not as protected. What
empties it is your decision to make, and it deserves the review a firewall-rule deletion gets,
because once layer 1 can fire on nothing, the only local defence left is layer 2 — and in its
default shadow mode, layer 2 can measure but not block.

Disabling a *blocking* detector is a data-egress decision, not a tuning decision, and should be
reviewed as one. Disabling an advisory `kw_*` detector only removes a hint.

### The 22 default detectors

Non-advisory detectors set floors. Advisory detectors never do: they are forwarded to the
backend as `local_gate_topic_hints` and recorded as `advisory_topics`.

**Blocking class — `blocks_backend: true`, never sent anywhere, not even redacted:**

| detector | category | floor | validated by |
| --- | --- | --- | --- |
| `payment_card` | regulated | `regulated` | Luhn checksum, 13–19 digits |
| `us_ssn` | regulated | `regulated` | shape `NNN-NN-NNNN` |
| `iban` | regulated | `regulated` | ISO 13616 mod-97 |
| `uk_nhs_number` | regulated | `regulated` | mod-11 check digit |
| `uk_national_insurance` | regulated | `regulated` | prefix/suffix letter rules |
| `medical_record_number` | regulated | `regulated` | shape |
| `private_key_block` | secret | `confidential` | PEM header |
| `provider_api_key` | secret | `confidential` | provider key shapes |
| `inline_credential` | secret | `confidential` | `password=` / `api_key:` style |
| `basic_auth_url` | secret | `confidential` | `scheme://user:pass@host` |

**Personal identifiers — `force_local: true`, redacted, but a redacted excerpt may still be
classified remotely:**

| detector | category | floor | validated by |
| --- | --- | --- | --- |
| `email_address` | pii | `confidential` | `_real_email` (skips reserved domains) |
| `phone_number` | pii | `confidential` | 9–15 digits after stripping separators |
| `date_of_birth` | pii | `regulated` | labelled-date shape, several languages |
| `street_address` | pii | `confidential` | number + street type + postcode/ZIP |
| `named_individual` | pii | `confidential` | explicit naming construction only |

**Advisory topic keywords — hints, never floors:**

| detector | category | floor | matches |
| --- | --- | --- | --- |
| `kw_health_regulation` | regulated | `regulated` | HIPAA, ePHI, diagnosis, prescription, GDPR, DSGVO |
| `kw_legal_privilege` | regulated | `regulated` | attorney-client, deposition, subpoena, litigation hold |
| `kw_financial_regulation` | regulated | `regulated` | PCI-DSS, cardholder data, CVV, GLBA, `kyc document`, `swift code` |
| `kw_minors` | regulated | `regulated` | COPPA, FERPA, child records, under 13 |
| `kw_confidential_business` | keyword | `confidential` | trade secret, `under nda` / `nda-covered`, layoff plan, salary, board resolution |
| `kw_security_vulnerability` | keyword | `confidential` | zero-day, CVE-YYYY-NNNN, exfiltration, breach response |
| `kw_internal_only` | keyword | `internal` | internal only, restricted distribution, staff only |

The non-advisory detectors set `pii_floor = 1.0` when they fire, because a deterministic
hit is a certainty, not a probability — and `1.0` is what makes the gate un-overridable by a
backend that answers `0.2`. The advisory detectors above never do: a topic mention is a hint,
not an identifier. (The match column shows the phrases as the patterns require them — bare
`KYC` or `NDA` do not fire; the patterns demand `kyc document`, `swift code`, `under nda`.)

`named_individual` is deliberately narrow. A loose person-name regex fires on "customer Call
Transcripts" and "client Library Here", and identifying people is a judgement call rather than a
pattern match. Names the gate misses are the calibrated backend's job.

### `placeholder_domains_as_pii`

RFC 2606 reserves `example.com`, `example.org`, `example.net`, and friends for documentation.
An address at one of those domains is not personal data, and firing on `sales@example.com` is
how a gate earns a reputation for crying wolf. By default `_real_email` rejects them, plus the
other placeholders documentation and test suites use:

```text
example.com  example.org  example.net  example.edu  example.invalid  localhost
test.com  domain.com  email.com  company.com  acme.com  foo.com  bar.com
users.noreply.github.com
```

Subdomains of a reserved domain are reserved too.

Set `placeholder_domains_as_pii: true` to treat them as real. It swaps the email detector's
validator for `_always`. Turn it on in tests and evals that need to exercise the email detector
without inventing a real-looking domain — `evals/data/labeled_prompts.jsonl` uses synthetic
identifiers from published test ranges, and the validator policy is documented in
[evals/data/README.md](../evals/data/README.md). Leave it off in production.

Verified: with the default, `mail sales@example.com` produces no findings and
`mail dana@northside-health.org` produces one `email_address` finding at floor `confidential`
with `force_local`. With `placeholder_domains_as_pii: true`, the first produces an
`email_address` finding too.

### `blocked_metadata`

What a **refusal** is recorded as, and where. When the gate blocks a request, the router writes
a `GateBlockRecord` to its own stream — a second file, `gate-blocks.jsonl`, beside the decision
log unless `path` says otherwise.

```yaml
gate:
  blocked_metadata:
    enabled: true            # default: on
    # path: ./decision-log/gate-blocks.jsonl   # default: sibling of logging.path
```

The record carries exactly `request_id`, `timestamp`, the detector ids that fired, the
deterministic feature vector, and the excerpt hash. No excerpt text under any configuration, no
raw secret, no matched span. The file is separate from the decision log on purpose: this stream
feeds **rule improvement** (which detectors fire, on what shape of request, how often), and the
decision log feeds **model training** — no training export can sweep a blocked prompt up by
accident. `enabled: false` stops the stream; blocked requests are still refused and still land
in the decision log, they simply produce no telemetry.

### `semantic`

Layer 2 of the gate: a local, distilled semantic scorer for the class of sensitivity regex
cannot see. **This sub-section is validated at parse time** — unknown keys and bad values are
`PolicyError`s at load, unlike the other `gate` keys. The full treatment — shadow mode, the
promotion criteria, the bootstrap paradox it exists to solve, and a runnable demonstration — is
[docs/gate-layers.md](gate-layers.md). The shipped default:

```yaml
gate:
  semantic:
    mode: shadow              # off | shadow | enforce; shadow is the default and the safe one
    # artifact: ./artifacts/gate-semantic   # directory holding semantic.json; absent -> inert
    threshold: 0.5
    level: confidential       # the floor asserted when it fires (enforce mode)
    # shadow_metrics: ./artifacts/gate-semantic/shadow-metrics.json
    enforce_requires:
      min_recall: 0.99
      max_false_positive_rate: 0.02
      max_disagreement_rate: 0.05
      max_semantic_miss_rate: 0.02
      min_positive_examples: 200
      min_negative_examples: 500
      min_shadow_examples: 1000
```

`mode: enforce` is necessary and **not sufficient**: the artifact's measured metrics must also
satisfy `enforce_requires` at layer construction, or the router refuses to start and prints the
numbers. It never silently downgrades to shadow. Promotion is run with
`jev-route graduate --track gate`; see the gate-layers reference for what that check covers and
what the cutover policy carries.

---

## 10. `cache`

```yaml
# config: policies/default.yaml (the shipped file omits `kind`, which then
# defaults to memory)
cache:
  enabled: true
  kind: memory        # memory | redis   (there is no "disabled"; use enabled: false)
  ttl_seconds: 900
  max_entries: 8192
```

### The key property: it caches judgements, not decisions

`cache.put` stores a `BackendResult` — the four answers with their full probability
distributions, the model version, the questions sent, the latency. It does **not** store the
tier, the model, the matched rule, or the escalations. Those are recomputed on every request,
from the cached judgement and the *current* policy.

The module docstring states why this is the reason the cache is a separate module:

> if you cache the final tier, a policy change silently does not apply to cached traffic for as
> long as the TTL runs, and an operator who tightens a sensitivity rule will see it appear to
> work while warm requests keep taking the old path.

So: **a policy edit takes effect on the next request, warm cache or not.** You can tighten
`sensitivity_confidence_below` from `0.8` to `0.9`, redeploy, and every cached judgement is
immediately re-reasoned under the new floor. That is the behaviour an operator needs during an
incident, and it is why the TTL is not a correctness parameter.

The cache key is built in `_cache_key`:

```python
payload = "\x00".join([f"v{policy_version}", backend_name, request.redacted_excerpt, ",".join(request.advisory_topics)])
return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
```

Four inputs: the policy **version** (always `1` today), the backend **name**, the **redacted
excerpt**, and the **advisory topics**. Consequences worth knowing:

* Two backends never serve each other's cached answers — switching `backend.name` invalidates
  everything, which is what you want when you graduate from `jev` to `distilled`.
* The key uses the redacted excerpt, so a change in redaction rules produces new keys rather
  than reusing a stale judgement.
* Metadata is **not** in the key. Two requests with identical text and different tenants share a
  judgement. That is correct — the judgement is about the text — and it means a
  `metadata`-based rule still routes them differently, because rules run per request.
* The policy version is in the key, but `version` is pinned to `1` by the loader, so a rule edit
  does not invalidate the cache. It does not need to: cached judgements are still valid
  judgements under new rules.

### `kind: memory`

`InMemoryTTLCache`: a bounded LRU (`OrderedDict`, `move_to_end` on hit, `popitem(last=False)` on
overflow) with a per-entry expiry checked on read. `ttl_seconds` is clamped to `>= 0` and
`max_entries` to `>= 1`; a `ttl_seconds` of `0` makes `put` a no-op, which is a way to keep the
stats object while disabling caching.

It is single-process. In a multi-worker proxy each worker holds its own cache, which is fine —
the hit rate drops, correctness does not. There is no cross-worker coherence to lose, because
the cached value is a judgement about a piece of text and two workers reaching the same
judgement independently is not a conflict.

### `kind: redis`

```yaml
cache:
  enabled: true
  kind: redis
  url: redis://localhost:6379/0
  key_prefix: "jevroute:prod:"
  ttl_seconds: 3600
```

`RedisCache` stores the serialized `BackendResult` as JSON with `ex=ttl_seconds`. Full
probability distributions survive the round trip (`_result_to_json` writes
`answers.to_dict()`), so a cached answer is indistinguishable from a fresh one downstream —
which matters, because the logged record must carry the same soft targets either way.

Requirements and failures:

```bash
pip install 'jev-route[redis]'
```

```text
RuntimeError: RedisCache requires the redis package: pip install 'jev-route[redis]'
ValueError: RedisCache needs url= or an injected client=
```

`key_prefix` defaults to `jevroute:`. Set it per environment. Two environments sharing one Redis
with one prefix will serve each other's judgements, and the key does not contain your policy's
rules — only its version.

`RedisCache.clear()` raises `NotImplementedError` on purpose:

```text
NotImplementedError: refusing to FLUSHDB; delete the key prefix explicitly
```

The sync `get`/`put` shims run the coroutine on a worker thread when an event loop is already
running, so a Redis cache does not deadlock a synchronous caller.

### Degraded results are never cached

```python
def put(self, key: str, result: BackendResult) -> None:
    if self.ttl_seconds <= 0:
        return
    # Never cache a degraded answer: it is maximum-uncertainty noise, and
    # caching it would pin fail-closed behaviour for the whole TTL after a
    # one-second network blip.
    if result.degraded:
        return
```

`RedisCache.aput` has the same guard. Without it, a one-second blip would pin every affected
request to the fail-closed tier for the whole TTL, and the outage would outlast itself by
fifteen minutes.

### Gate-forced requests and the cache

The cache is only consulted in the branch that calls a backend. Under
`on_force_local: skip_backend` a gate-forced request never reaches it — there is nothing to
cache, because nothing was asked. Under `still_classify` it does get cached, like any other
classified request. The `cache.skip_when_gate_forced` key in `policies/default.yaml` is not read
by any code; see [section 2](#keys-that-appear-in-yaml-and-are-read-by-nothing).

### Observing it

`Router.stats()["cache"]` reports `hits`, `misses`, `hit_rate`, `evictions`, `size`,
`max_entries`, `ttl_seconds` for the memory cache; `hits`, `misses`, `hit_rate`, `ttl_seconds`
for Redis; `enabled: false` for `NullCache`. A cached decision is visible in the log as
`decision.backend == "<name>:cached"`, `decision.cached == true`, and
`backend_latency_ms == 0.0`.

---

## 11. `logging`

```yaml
# config: policies/default.yaml
logging:
  enabled: true
  path: ./decision-log/decisions.jsonl
  excerpt_mode: hash
  hash_salt: ""
  max_bytes: 268435456   # 256 MiB; 0 disables rotation
  flush_every: 1
```

This is not an audit trail that happens to be useful. **It is the training set.**
`jev_route.distill` reads these records and produces the local model that eventually replaces
the cloud backend. Hold the log to dataset standards, not logging standards.

### `excerpt_mode`

Validated in `Router.__init__`, lower-cased first:

```text
ValueError: logging.excerpt_mode must be hash|redacted|none, got 'bogus'
```

**`hash`** (default). The record carries `excerpt_hash` — a salted SHA-256 of the raw excerpt,
truncated to 16 hex characters — plus all 24 deterministic features, plus the full soft
distributions. **No prompt text is retained.** Distillation is limited to feature-based models:
the student learns from counts, ratios, and booleans. `excerpt` is `null` on every record.

**`redacted`**. Also stores the redacted excerpt in `excerpt`. This enables text-based
distillation (a ModernBERT-class student) and it is a real increase in what you retain. The
placeholders are typed, not blank: `[payment_card]` tells a complexity classifier "something
regulated was here" without reproducing it. Advisory topic keywords are **never** redacted —
redacting "HIPAA" out of "explain how HIPAA works" would leave a classifier nothing to read.

**`none`**. No excerpt handling difference from `hash` in the record: `excerpt` stays `null`,
`excerpt_hash` is still written. The mode is meaningful as an explicit statement of intent in
the file, and it protects you if a future version adds a text field.

| mode | `excerpt` field | text-based distillation | feature-based distillation |
| --- | --- | --- | --- |
| `hash` | `null` | no | yes |
| `redacted` | redacted excerpt | yes | yes |
| `none` | `null` | no | yes |

`jev-route log-stats` reports which mode you actually collected under, because it decides which
graduation path is open:

```text
DISTILLABLE: 41233/51890 records carry redacted excerpt text.   # example counts
  -> text-mode distillation is available (best accuracy).
```

or

```text
No record carries excerpt text (logging.excerpt_mode is 'hash' or 'none').
  -> features-mode distillation only.
```

Note the denominator in the first line. Records with text are records that were *not*
gate-blocked, so the two modes are not evenly represented in your data.

### Gate-blocked requests are never written as text

```python
excerpt: str | None = None
# Gate-blocked requests are never written as text, whatever the operator
# configured. The same judgement that kept the excerpt out of a cloud API
# keeps it out of the log.
if self.excerpt_mode == "redacted" and not gate_blocked:
    excerpt = redacted_excerpt
```

`gate_blocked` is the router's `skip_backend` flag: true when a blocking detector matched, when
either gate layer forced local under `on_force_local: skip_backend` (layer 2 only in enforce
mode — a shadow firing never skips the backend), or when an enforced layer-2 assertion made the
request stay local. Under `excerpt_mode: redacted`,
those rows still get `excerpt: null`. Verified: a request containing an email address, routed
under `redacted`, writes `excerpt: null`; the next request, with no gate finding, writes the
full redacted text.

There is no key that overrides this. A file on disk in your own infrastructure is a disclosure
surface too — it gets backed up, shipped to a warehouse, read by whoever has log access — and
the gate's judgement that content is too sensitive to send to a cloud API applies with at least
as much force to your own log directory.

What you *do* keep for those rows is everything else: the hash, the features, the gate findings
with their span hashes, the effective labels, the tier, the rule, the reason. A
`GateFinding` stores a SHA-256 of the matched span, not the span, so the log can prove "a
Luhn-valid card number was present" without keeping the card number.

### `hash_salt`

```python
hashlib.sha256((salt + "\x00" + text).encode("utf-8")).hexdigest()[:16]
```

Setting a salt means two deployments sharing a log store cannot trivially cross-reference
prompts by hash. It is read by `Router.__init__` (default `""`), not by `build_sink`.

Read the docstring's own caveat before you rely on it:

> The hash is a *dedup and cache key*, not an anonymization guarantee: a short prompt from a
> known corpus is still guessable. That is documented, not hidden.

Salting also changes which requests the shadow sampler picks, because `_sample_hit` hashes
`excerpt_hash`. Changing the salt mid-life is a change in sampling, not just in the stored
value — and it invalidates nothing else, since the cache key is built from the excerpt, not the
hash.

### `max_bytes` and rotation

Rotation triggers when the current file size plus the new line would exceed `max_bytes`:

```python
if self.max_bytes and self.path.exists() and self.path.stat().st_size + len(line) > self.max_bytes:
    self._rotate_locked()
```

`_rotate_locked` **renames** the current file to `<stem>.<YYYYMMDDTHHMMSS><suffix>`, adding a
numeric suffix if that name is taken, then reopens. It never truncates and never deletes. The
log is append-only: rotation creates a new file, it does not edit the old one. `max_bytes: 0`
disables rotation, which means one file that grows until the disk does.

`iter_all_records(directory, pattern="decisions*.jsonl")` reads rotated siblings oldest-first by
mtime, so a training export can span rotations. Keep rotated files in the same directory as the
live one, or point the exporter at both.

### `flush_every`

`fsync` every N records, clamped to `>= 1`. The default `1` is one syscall per record, which is
the right default for a dataset: a lost tail on crash costs training rows. Raise it only if you
have measured the write path and know what you are trading. The file is opened with
`buffering=1` (line-buffered text append) in addition.

A write is one `write()` call under a lock. On a file opened `O_APPEND` that is effectively
atomic per line, which is why a multi-worker proxy can share one path without interleaving
records.

### `enabled` and `path`

`enabled: false` gives a `NullSink`. So does `enabled: true` with no `path` — silently:

```python
if not cfg.get("enabled", True):
    return NullSink()
path = cfg.get("path")
if not path:
    return NullSink()
```

A `NullSink` still counts records (`stats()["log"]["written"]`), so you can tell the difference
between "logging off" and "nothing routed". If your log is empty, check `path` before you check
anything else. This is the one configuration mistake that costs you the product: routing works,
the decisions look right, and you accumulate no dataset.

`JsonlSink.__init__` creates the parent directory (`mkdir(parents=True, exist_ok=True)`), so the
path must be writable by the process user at startup. An unwritable location raises
`PermissionError` when the `Router` is constructed — fail fast, before traffic arrives.

OSError *during* a write is counted, not raised:

```python
except OSError:
    # A full disk must not take down the request path. Count it and
    # carry on: routing still works, the log has a hole.
    self._errors += 1
```

Watch `stats()["log"]["write_errors"]`. A full disk should be an alert, not a silent hole in
your training data.

### What one record contains

`kind`, `schema_version`, `request_id`, `timestamp`, `requested_model`, `excerpt_hash`,
`excerpt`, all 24 `features`, `questions_sent` (verbatim, so a logged decision can be
reproduced; empty when the gate blocked the call), the whole `decision` (tier, model, rule_id,
reason, effective labels, the four answers with full distributions, the gate verdict, escalations,
degraded flag, latencies, cached), `metadata` (scalars only), `shadow`, and `semantic` — the
layer-2 assessment (mode, score, fired, asserted level, enforced, the layer-1 side, the
disagreement kind; no text), present when layer 2 ran, `null` otherwise.

Records are schema-versioned and additive-only. `schema_version` is stamped on every record, a
field may be added, and it may not be silently renamed or retyped — because the value of the log
is that it accumulates, and old records must stay readable.

---

## 12. `shadow`

```yaml
# config: policies/default.yaml (the shipped file omits `sample_rate`, which
# then defaults to 1.0)
shadow:
  enabled: false
  sample_rate: 1.0
```

Shadow mode runs a second backend alongside the one that serves the decision, and records where
they disagree. This is how graduation earns the cutover: you do not switch from the cloud
teacher to your own distilled student on the strength of an offline eval. You run both on live
traffic, measure disagreement on *your* distribution, and cut over when the number holds.

### Two mechanisms, and which config drives which

This is the most commonly misread part of the file, so it is stated plainly.

**1. The router-level side-run**, driven by the `shadow:` section:

```python
self._shadow_enabled = bool((policy.shadow or {}).get("enabled")) and shadow_backend is not None
self._shadow_sample_rate = float((policy.shadow or {}).get("sample_rate", 1.0))
```

`shadow.enabled: true` is necessary but **not sufficient**. A second `DecisionBackend` must be
passed to the constructor:

```python
from jev_route import Router, Policy
from jev_route.backends import build_backend

# `policies/shadow.yaml` is not shipped; save this section's YAML block as that
# file (plus the backend: block you want to serve from) before running this.
policy = Policy.from_file("policies/shadow.yaml")
router = Router(
    policy,
    build_backend({"backend": {"name": "distilled", "artifact": "./artifacts/v1"}}),
    shadow_backend=build_backend({"backend": {"name": "jev", "api_key_env": "TYPESAFE_API_KEY"}}),
)
```

`Router.from_policy_file` does not inject a shadow backend, so a policy-only
`shadow.enabled: true` is inert on that path. If you want the side-run without writing that
constructor call, use mechanism 2.

**2. The backend-level composition**, driven by `backend.name: shadow`:

```yaml
backend:
  name: shadow
  primary:                               # must be a mapping with a `name` key
    name: distilled
    artifact: ./artifacts/jev-route-distilled-v1
  shadow:
    name: jev
    api_key_env: TYPESAFE_API_KEY
  log_disagreements: true
  shadow_timeout_seconds: 5.0
```

`build_backend` builds both from the nested mappings and hands them to `ShadowBackend`. Note the
shape: `primary` and `shadow` must be **mappings with a `name` key**, not bare strings.
`build_backend` does `{**cfg.get("primary", {}), "name": str(cfg["primary"]["name"])}`, so:

```text
TypeError: 'str' object is not a mapping     # primary: distilled   (scalar shorthand)
KeyError: 'primary'                          # primary omitted entirely
```

The commented example in `policies/default.yaml` shows the scalar shorthand. The loader does not
accept it. The `shadow.primary`, `shadow.shadow`, and `shadow.log_disagreements` keys in that
same commented block are read by nothing — they belong under `backend:`. See
[section 2](#keys-that-appear-in-yaml-and-are-read-by-nothing).

Both branches import their backend module lazily at construction time — the policy loader never
touches `importlib`, so a policy naming `distilled` parses fine and only the Router constructor
builds the module. Both modules ship in the package as of this writing
(`jev_route.backends.shadow`, `jev_route.backends.distilled`); on an install where one is absent
the constructor raises `ModuleNotFoundError` for the missing name, not the loader.

### What `ShadowBackend` actually does

`ShadowBackend` is a drop-in `DecisionBackend`. It **always answers with the primary's
`BackendResult` object itself** — not a copy, not a merge — so provenance (`questions_sent`,
`latency_ms`, `degraded`) is exactly what the serving backend produced. `name` is `"shadow"`,
which is what lands in `decision.backend` and in the cache key, and `model_version` mirrors the
*primary's current* version, because the primary is what answered.

By default it runs the shadow as a **background asyncio task**, bounded by
`shadow_timeout_seconds`. `decide()` returns as soon as the primary answers, so the shadow adds
zero latency to the request. That default is the reason this class exists separately from the
router's shadow: a config you put in front of production traffic at cutover cannot tax every
request for the sake of measuring it.

Telemetry lives in a **bounded in-process ring buffer**, not in the decision log:

```python
backend.reports()  # every retained report, oldest first
backend.disagreements()  # only reports where agrees is False
backend.stats()  # cumulative counters + disagreement_rate + per-field counts
```

`stats()` reports `requests`, `sampled`, `completed`, `timeouts`, `errors`, `degraded`,
`disagreements`, `callback_errors`, `disagreement_rate`, and `fields` (per-field disagreement
counts for `complexity`, `sensitivity`, `domain`, `pii`). `disagreement_rate` is
`disagreements / completed` — the fraction of shadowed traffic where both backends actually
answered and gave different decisions. A shadow that timed out, raised, or degraded carries
`agrees: None` and counts as completed-but-not-disagreeing, so a candidate model that degrades
constantly shows up in `degraded` rather than as a flattering 0% disagreement rate.

`log_disagreements` gates only the **outbound side effect**: with `false`, no `on_disagreement`
callback fires. The ring buffer and `stats()` keep recording regardless — they are the backend's
own introspection API. Do not confuse them with graduation evidence: `graduate` never reads the
ring buffer. It reads shadow evidence from a file (`--shadow-metrics`) or from the decision log
(`--log`), so durability comes from one of those two, not from the buffer.

Four constructor arguments are **not settable from the policy file**, because `build_backend`
passes only `primary`, `shadow`, `log_disagreements`, and `shadow_timeout_seconds`:

| argument | default | effect |
| --- | --- | --- |
| `mode` | `"background"` | `"await"` instead runs both concurrently and returns after both settle. |
| `sample_rate` | `1.0` | The backend-level sampling rate. Shadow every request. |
| `max_reports` | `256` | Ring-buffer size. |
| `on_disagreement` | `None` | Callback for shipping a disagreement somewhere durable. |

That last row is the practical consequence of the two mechanisms differing: **backend-level
shadow telemetry does not survive the process** unless you inject an `on_disagreement` callback
in code, and the policy file has no key for it. Router-level shadow telemetry lands in the
decision log and survives forever.

### Which one to use

| | router-level (`shadow:`) | backend-level (`backend.name: shadow`) |
| --- | --- | --- |
| turns on with | `shadow.enabled: true` **and** `Router(shadow_backend=...)` | `backend.name: shadow` in the policy file alone |
| sampling key | `excerpt_hash` (salted by `logging.hash_salt`) | `request_id`, else the redacted excerpt hash |
| sampling rate | `shadow.sample_rate` | `1.0`, not configurable from YAML |
| added latency | the shadow's, on every sampled request (awaited inline) | none by default (background task) |
| where the comparison lands | the record's `shadow` field, in the training log | an in-process ring buffer |
| survives a restart | yes | no |
| affects the served decision | no | no |

They compose: a router with `shadow.enabled` can wrap a `ShadowBackend`. If you can only have
one, take the router-level one — its telemetry lands in the training log and survives a restart,
where the backend-level buffer dies with the process. Which one you need depends on which
cutover you are earning: the **gate track** (`graduate --track gate`) reads its live shadow
evidence from the decision log (`--log`), where router-level shadow telemetry lives; the
**router track** earns the cutover from the exported dataset's held-out split, not from shadow
rows at all.

### `sample_rate`

```python
def _sample_hit(rate: float, key: str) -> bool:
    """Deterministic sampling from the excerpt hash, so replays are reproducible."""
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < rate
```

Sampling is **deterministic from `excerpt_hash`**, not random. The same prompt is either always
side-run or never side-run, in every process, on every replay. That matters for two reasons: a
reproduced incident reproduces its shadow row, and a repeated prompt does not burn shadow quota
on every occurrence. It also means your sample is a sample of *distinct prompts*, weighted by
nothing — a prompt that appears ten thousand times contributes one shadow comparison, not ten
thousand.

`rate >= 1.0` short-circuits to always and `<= 0.0` to never, so `sample_rate: 0` is a clean way
to keep `enabled: true` in the file while spending nothing.

The shadow only runs on requests that reached a backend. A gate-blocked or gate-skipped request
has no shadow row, because there was no primary judgement to compare against.

### What gets recorded

`_run_shadow` returns a report that `_log` writes to the record's `shadow` field:

```python
{
    "backend": self.shadow_backend.name,
    "model_version": shadow.model_version,
    "answers": shadow.answers.to_dict(),
    "latency_ms": shadow.latency_ms,
    "degraded": shadow.degraded,
    "disagreements": disagreements,
    "agrees": not disagreements,
}
```

`disagreements` is built by comparing the `choice` of `complexity`, `sensitivity`, and `domain`,
and the `pii` value with a fixed tolerance:

```python
if abs(primary.answers.pii.value - shadow.answers.pii.value) >= 0.25:
    disagreements["pii"] = {"primary": ..., "shadow": ...}
```

Each entry records both sides, e.g. `{"sensitivity": {"primary": "internal", "shadow":
"confidential"}}`. `agrees` is the boolean you will aggregate on. The full soft distributions of
the shadow answer are stored too, so a disagreement can be re-examined as "how far apart were
they, and how confident was each side" rather than just "they differed".

A shadow failure is telemetry, not an error:

```python
except Exception as exc:  # a shadow failure is telemetry, not an error
    return {"backend": self.shadow_backend.name, "error": f"{type(exc).__name__}: {exc}"}
```

The shadow never affects the served decision and never fails the request. If your shadow backend
is down, you lose comparison rows and nothing else.

### Using it to graduate

The disagreement rate on live traffic is the number the cutover decision rests on. Filter the log
to rows with a `shadow` field, then look at:

* `agrees` rate overall, and per question — a student that agrees on complexity and disagrees on
  sensitivity is not ready, and the aggregate hides it;
* disagreements weighted by tier consequence. `internal` vs `confidential` changes where the
  request runs; `writing` vs `analysis` changes nothing;
* `disagreements.pii` separately, since it is the one field compared with a tolerance rather
  than for equality;
* whether disagreements cluster on gate-adjacent traffic, which is where the teacher's judgement
  is doing the most work.

`jev-route graduate` reads the same log and compares against thresholds
(`--min-tier-agreement`, `--max-ece`, `--max-latency-ms`, `--min-samples`) before it will write a
cutover policy. The tier-agreement figure comes from the exported dataset's **held-out split**
— `evaluate` scores student against teacher on rows the student never trained on, through the
same policy — so the discipline is in the export: a dataset cut from your real log is what makes
the number about your traffic. The shadow rows in the log are the *continuous* check after the
cutover, not the evidence for it. See [graduation.md](graduation.md).

When the rate holds where you want it, set `backend.name: distilled`, delete the `shadow:`
section's effect, and the egress stops permanently. That is the last config change in the
project's lifecycle, and it is a config change.

---

## 13. Five complete examples

Five deployment shapes, each a complete file that `Policy.from_yaml` accepts. Every tier named
in a rule or in `on_backend_down` is defined under `tiers:`; every `tier_order` entry is a
defined tier; each file has exactly one default rule and it is last; every expression uses only
allow-listed syntax and only variables from [section 4](#4-expression-variables).

Model names are placeholders for entries in your LiteLLM `model_list`. No example contains an
API key: they all use `api_key_env`. None of the example files is shipped under `policies/` —
`Policy.from_file("policies/healthcare.yaml")` and friends raise on a fresh clone until you save
each block under its stated name. That is deliberate: an example is a template to edit, not a
config to run.

To check any of these before you deploy it:

```bash
python -c "from jev_route import Policy; p = Policy.from_file('policies/healthcare.yaml'); print(p.tiers, p.tier_order, [r.rule_id for r in p.rules])"
```

A `PolicyError` here is a file that will not start. That is the cheapest place to find out.

### (a) Healthcare: fail closed, capable on-prem, nothing retained

The shape this project's gate was designed around. Two on-prem tiers, because "keep it local"
and "keep it local *and* answer a hard clinical question well" are different requirements and
collapsing them means either a weak model on regulated data or regulated data on a cloud model.
`local-strong` lists two replicas of the same deployment and round-robins them.

Uncertainty is tuned strict: `sensitivity_confidence_below: 0.9` with `sensitivity_bump_levels: 2`
means one unsure sensitivity read lands on `confidential`, which the rule above sends on-prem.
`pii_uncertain_threshold: 0.2` widens the "not a no" band to `[0.2, 0.5)` — a 20% chance that a
prompt names a patient is treated as a certainty that it might.

`fail_closed`, and `fail_open_tier: local-strong` as well, so there is no mode in this file that
reaches a cloud tier during an outage. `excerpt_mode: hash` retains no prompt text, which is the
trade: your distillation is features-only. The cache TTL is short (300 s) because clinical
traffic repeats less than support traffic does, and a short TTL keeps a policy change from being
masked for long on the small fraction that does repeat.

The cloud tiers are still here, and that is deliberate: a request the backend judges `public` or
`internal` with no PII signal — a scheduling question, a documentation draft — carries nothing
regulated, and routing it on-prem buys nothing. If your DPA says otherwise, delete the two
complexity rules and the default rule's tier, and send everything to `local-strong`.

```yaml
# config: policies/healthcare.yaml
version: 1

backend:
  name: jev
  api_key_env: TYPESAFE_API_KEY   # never a literal key in a committed file
  timeout_seconds: 3.0
  max_retries: 1
  include_domain: false           # no rule below reads `domain`

gate: {on_force_local: skip_backend, disabled_detectors: [], placeholder_domains_as_pii: false}

tiers:
  local:        [phi-onprem-8b]          # smallest on-prem model; the fail-closed floor
  local-strong: [qwen-onprem-72b-a, qwen-onprem-72b-b]   # capable on-prem, round-robined
  cheap:        [cloud-flash-eu]
  strong:       [cloud-max-eu]
tier_order: [cheap, strong]              # `local*` omitted: residency, not capability

pii_threshold: 0.5

rules:
  - id: gate.force-local
    if: gate_force_local
    then: {tier: local-strong}
    reason: the local gate matched a structured identifier or credential material

  - id: data.regulated
    if: sensitivity in ["confidential", "regulated"] or pii_present
    then: {tier: local-strong}
    reason: patient-identifiable or confidential data never leaves the building

  - id: data.minors-topic
    if: '"kw_minors" in advisory_topics'
    then: {tier: local-strong}
    reason: paediatric or student-record topic handled on-prem as a matter of policy

  - id: complexity.frontier
    if: complexity == "frontier"
    then: {tier: strong}
    reason: non-sensitive task that needs frontier-class reasoning

  - id: complexity.hard
    if: complexity == "hard"
    then: {tier: strong}
    reason: non-sensitive task that needs a strong model

  - id: default
    then: {tier: cheap}
    reason: routine, non-sensitive task

on_uncertain:
  sensitivity_confidence_below: 0.9      # tighter than the shipped default
  sensitivity_bump_levels: 2             # one unsure read lands on confidential
  complexity_confidence_below: 0.8
  complexity_bump_levels: 1
  pii_uncertain_threshold: 0.2           # a wide band: 0.2..0.5 all count as present
  pii_uncertain_counts_as_present: true

on_backend_down: {mode: fail_closed, fail_closed_tier: local, fail_open_tier: local-strong}

cache: {enabled: true, kind: memory, ttl_seconds: 300, max_entries: 4096}

logging:
  enabled: true
  path: /var/log/jev-route/decisions.jsonl
  excerpt_mode: hash                     # no prompt text retained, ever
  hash_salt: ""
  max_bytes: 268435456
  flush_every: 1

shadow: {enabled: false}
```

### (b) Cost-optimised startup: cheap by default, strong only when it pays

The opposite priority, and the one most deployments actually start with. The default rule sends
everything to `cheap`, and only `complexity == "frontier"` — plus one shape-based exception for
long hard tasks — reaches `strong`. The `cheap` tier lists two identical deployments so the
round-robin spreads load.

`complexity_confidence_below: null` switches off capability escalation entirely. That is the
honest cost decision: an uncertain complexity read bumping `hard` to `frontier` is exactly the
mechanism that quietly doubles a bill, and this deployment would rather be occasionally
under-powered than permanently over-spent. Sensitivity escalation stays on at the shipped
defaults, because that one protects data rather than quality.

`fail_open` to `strong`. During an outage this deployment loses money rather than capability,
and the gate still outranks the knob — a credential in the prompt goes to `local` whatever
`mode` says.

`excerpt_mode: redacted` because a startup that expects to graduate needs text-mode
distillation, and the cache is Redis with a one-hour TTL and an environment-scoped `key_prefix`
so every proxy worker shares it. Redis needs `pip install 'jev-route[redis]'`.

The one thing to watch: `fail_open` plus a long cache TTL means an outage degrades *upward* and
the degraded results are not cached, so every request during the outage pays for the strong
tier. That is the price of the mode. If it is too high, use `fail_closed` and accept worse
answers during outages.

```yaml
# config: policies/startup.yaml
version: 1

backend:
  name: jev
  api_key_env: TYPESAFE_API_KEY
  timeout_seconds: 2.5                   # routing is on the critical path; keep it tight
  max_retries: 1
  include_domain: false

gate: {on_force_local: skip_backend, placeholder_domains_as_pii: false}

tiers:
  local:  [small-self-hosted]
  cheap:  [flash-a, flash-b]             # two identical deployments, round-robined
  strong: [max-model]
tier_order: [cheap, strong]

pii_threshold: 0.5

rules:
  - id: gate.force-local
    if: gate_force_local
    then: {tier: local}
    reason: local hard gate matched an identifier or credential

  - id: data.sensitive
    if: sensitivity in ["confidential", "regulated"] or pii_present
    then: {tier: local}
    reason: sensitive or personal data stays on self-hosted models

  - id: complexity.frontier
    if: complexity == "frontier"
    then: {tier: strong}
    reason: only frontier-class reasoning pays for the strong tier

  - id: long-hard
    if: char_len > 3000 and complexity == "hard"
    then: {tier: strong}
    reason: long hard task; the cheap tier loses the thread

  - id: default
    then: {tier: cheap}
    reason: routine task, cheapest tier that can do it

on_uncertain:
  sensitivity_confidence_below: 0.8
  sensitivity_bump_levels: 1
  complexity_confidence_below: null      # capability escalation costs money: switched off
  complexity_bump_levels: 1
  pii_uncertain_threshold: 0.35
  pii_uncertain_counts_as_present: true

# fail_open: during an outage the risk we care about is answer quality, not egress.
# The gate still outranks it (see docs/policy-guide.md section 8).
on_backend_down: {mode: fail_open, fail_closed_tier: local, fail_open_tier: strong}

cache:
  enabled: true
  kind: redis                            # shared across proxy workers
  url: redis://localhost:6379/0
  key_prefix: "jevroute:prod:"
  ttl_seconds: 3600

logging:
  enabled: true
  path: ./decision-log/decisions.jsonl
  excerpt_mode: redacted                 # we want text-mode distillation later
  max_bytes: 268435456
  flush_every: 1

shadow: {enabled: false}
```

### (c) EU / GDPR: personal data never leaves the region

The residency tier is the only tier personal data may reach, and the rules enforce it three
different ways, because three different signals can say "this is personal":

* `gate_force_local` — a deterministic identifier matched;
* `pii_present or sensitivity in ["confidential", "regulated"]` — the calibrated judgement;
* `pii > 0.2` — a probability below the uncertain band that is still not negligible.

`pii_uncertain_threshold: 0.45` with `pii_threshold: 0.5` is a narrow band, and it is the highest
of the five examples: under GDPR the cost of a false negative is not symmetric with the cost of a
false positive, so the band starts close to the threshold. Keep the two numbers ordered — a
`pii_uncertain_threshold` at or above `pii_threshold` makes the knob inert
([section 7](#7-on_uncertain-the-calibration-payoff)).

`on_force_local: skip_backend`. When the gate has decided a request is personal, nothing leaves
the region — not even a redacted excerpt for a complexity read. That is the "gate
`still_classify` off" choice, and it costs distillation coverage on exactly the rows you are
least allowed to keep anyway.

Special-category topics are kept in-region even though they are advisory and set no floor. That
is a policy choice, not a technical necessity: `"explain how HIPAA works"` contains no personal
data, and the default policy would send it to a cloud tier. This deployment prefers the
precaution, and says so in the `reason:` string.

`tenant.pinned` shows `then.model` doing its actual job — a contractual commitment to one named
deployment, bypassing round-robin. Note the guard: `'contract' in metadata and ...`, because a
missing key is a request-time `PolicyError`, not `False`.

`fail_open_tier: local` too, so both outage modes stay in-region. `excerpt_mode: hash` retains no
text, and `hash_salt` is set to a deployment-specific value so a shared log store cannot be
cross-referenced by hash. Read the caveat in [section 11](#11-logging): a salt is not
anonymization.

What a policy file cannot do is enforce region. `cheap` and `strong` are EU-region deployments
only because your LiteLLM `model_list` points those names at EU endpoints. The policy names the
tier; the provider config owns where it lives. Review both together.

```yaml
# config: policies/eu-gdpr.yaml
version: 1

backend:
  name: jev
  api_key_env: TYPESAFE_API_KEY
  api_url: https://api.typesafe.ai/v1/systemone
  model: jev-latest
  timeout_seconds: 4.0
  max_retries: 1
  include_domain: true                   # no rule reads `domain`, but the log should carry it

# skip_backend: when the gate has decided, nothing leaves the region -- not even
# a redacted excerpt for classification.
gate: {on_force_local: skip_backend, disabled_detectors: [], placeholder_domains_as_pii: false}

tiers:
  local:  [mistral-eu-selfhosted]        # EU-hosted or self-hosted: personal data only ever lands here
  cheap:  [flash-eu-region]              # EU-region deployments, non-personal traffic only
  strong: [max-eu-region]
tier_order: [cheap, strong]

pii_threshold: 0.5

rules:
  - id: gate.force-local
    if: gate_force_local
    then: {tier: local}
    reason: an identifier or credential matched locally; nothing leaves the region

  - id: data.personal
    if: pii_present or sensitivity in ["confidential", "regulated"]
    then: {tier: local}
    reason: personal data is processed only on EU-hosted or self-hosted models

  - id: data.maybe-personal
    if: pii > 0.2
    then: {tier: local}
    reason: a non-trivial chance of personal data is treated as personal data

  - id: topic.health-or-minors
    if: '"kw_health_regulation" in advisory_topics or "kw_minors" in advisory_topics'
    then: {tier: local}
    reason: special-category topic; kept in-region as a precaution

  - id: tenant.pinned
    if: "'contract' in metadata and metadata['contract'] == 'public-sector'"
    then: {tier: local, model: mistral-eu-selfhosted}
    reason: a public-sector contract pins one EU deployment exactly

  - id: complexity.frontier
    if: complexity == "frontier"
    then: {tier: strong}
    reason: non-personal frontier task, EU-region deployment

  - id: default
    then: {tier: cheap}
    reason: non-personal routine task, EU-region deployment

on_uncertain:
  sensitivity_confidence_below: 0.85
  sensitivity_bump_levels: 1
  complexity_confidence_below: 0.7
  complexity_bump_levels: 1
  pii_uncertain_threshold: 0.45          # a narrow band, but a real one: 0.45..0.5
  pii_uncertain_counts_as_present: true

# fail_open_tier is `local` too: there is no configuration of this deployment in
# which an outage sends anything to a non-EU model.
on_backend_down: {mode: fail_closed, fail_closed_tier: local, fail_open_tier: local}

cache: {enabled: true, kind: memory, ttl_seconds: 600, max_entries: 8192}

logging:
  enabled: true
  path: /var/log/jev-route/decisions.jsonl
  excerpt_mode: hash                     # no prompt text retained
  hash_salt: "eu-prod-7f3a1c"            # deployment-specific; see section 11
  max_bytes: 268435456
  flush_every: 1

shadow: {enabled: false}
```

### (d) Offline air-gapped: the end state

No cloud key anywhere in this file. `backend.name: distilled` reads a trained artifact from
disk, and the only module in the package that can reach `api.typesafe.ai` is never constructed.
Prove it after deploying:

```bash
grep -r "api.typesafe.ai" src/          # one file: src/jev_route/backends/jev.py
grep -rn "api_key" policies/airgap.yaml # nothing
```

Both tiers are on-prem, so `tier_order: [local, strong]` is honest here — unlike the shipped
default, where `local` is deliberately omitted because it is a residency tier rather than a
capability rung. Here `bump_tier("local")` really does mean "the bigger machine in the same
room". Note that no code path in the router calls `bump_tier`; the order still matters for
validation and for the LiteLLM plugin's safety ranking ([section 5](#5-tiers-and-tier_order)).

`feature_mode: true` matches how the log was collected. If the artifact was trained on text,
set it false and make sure `excerpt_mode: redacted` was on while you collected.

The cache is in-memory with a long TTL and a large bound: a local student is fast, but repeated
traffic still costs an inference, and there is no shared store in an air-gapped network.

**Logging stays on.** This is the point people get wrong. Air-gapped does not mean "stop
recording": the log is the dataset that produced this artifact and the dataset that produces the
next one. A router that stopped logging is a router that can never improve. `excerpt_mode:
redacted` here, so the next generation can be text-mode.

`backend.name: distilled` needs `jev_route.backends.distilled` present in your install; the
policy loader does not check that, the Router constructor does.

```yaml
# config: policies/airgap.yaml
version: 1

# No cloud key anywhere in this file. `grep -r api.typesafe.ai src/` returns one
# module, and this policy never names it.
backend:
  name: distilled
  artifact: ./artifacts/jev-route-distilled
  feature_mode: true                     # the log was collected under excerpt_mode: hash

gate: {on_force_local: skip_backend, disabled_detectors: [], placeholder_domains_as_pii: false}

tiers:
  local:  [onprem-small]                 # the workhorse
  strong: [onprem-large]                 # bigger model, same building
tier_order: [local, strong]              # both tiers are on-prem, so both may be rungs here

pii_threshold: 0.5

rules:
  - id: gate.force-local
    if: gate_force_local
    then: {tier: local}
    reason: local hard gate matched an identifier or credential

  - id: data.sensitive
    if: sensitivity in ["confidential", "regulated"] or pii_present
    then: {tier: local}
    reason: sensitive data stays on the smallest self-hosted model that can take it

  - id: complexity.demanding
    if: complexity in ["hard", "frontier"]
    then: {tier: strong}
    reason: demanding task goes to the large on-prem model

  - id: default
    then: {tier: local}
    reason: routine task, small on-prem model

on_uncertain:
  sensitivity_confidence_below: 0.75     # a distilled student reports its own certainty
  sensitivity_bump_levels: 1
  complexity_confidence_below: 0.7
  complexity_bump_levels: 1
  pii_uncertain_threshold: 0.35
  pii_uncertain_counts_as_present: true

on_backend_down: {mode: fail_closed, fail_closed_tier: local, fail_open_tier: strong}

cache: {enabled: true, kind: memory, ttl_seconds: 3600, max_entries: 16384}

# Still on. The log is the dataset that produced this artifact, and it is the
# dataset that produces the next one.
logging:
  enabled: true
  path: /var/log/jev-route/decisions.jsonl
  excerpt_mode: redacted
  max_bytes: 268435456
  flush_every: 1

shadow: {enabled: false}
```

### (e) Shadow-mode graduation: earning the cutover

This is the shape `jev-route graduate --write` produces. Be precise about what the command
does, because it is less than this file: it rewrites **only the `backend:` block** of your
policy (all other sections are inherited unchanged into the new file — `policies/default.yaml`
becomes `policies/default-graduated.yaml`), and that block composes `primary: distilled` with
`shadow:` set to your *current* backend's config. Your distilled model serves traffic; the cloud
teacher runs alongside inside the backend layer.

Know where the disagreement lands before you read the next paragraph: in this shape it goes to
the backend's **in-process ring buffer** and, because `log_disagreements` is on, to whatever
your `on_disagreement` callback ships. It does **not** go to the decision record's `shadow`
field — that field is written by the *router-level* side-run (the `shadow:` section at the
bottom of this file), which only fires when the `Router` is constructed with
`shadow_backend=...`, and `Router.from_policy_file` does not do that. If you want the
disagreement in the training log, construct the router yourself and pass it. You cut over on a
measurement of your own traffic, not on an offline eval — and you cut back the same way, by
editing one file.

Two shadow mechanisms appear here and they are not redundant
([section 12](#12-shadow)):

* `backend.name: shadow` composes the two backends inside the backend layer. `primary` and
  `shadow` must be mappings with a `name` key — the scalar shorthand in the commented block of
  `policies/default.yaml` does not load.
* `shadow: {enabled: true, sample_rate: 0.1}` drives the router-level side-run, and it is only
  live when the `Router` is constructed with `shadow_backend=...`. `Router.from_policy_file`
  does not pass one.

`sample_rate: 0.1` side-runs one distinct prompt in ten, chosen deterministically from
`excerpt_hash` — the same prompt always lands the same way, so replays reproduce and a repeated
prompt does not burn quota. `max_retries: 0` on the teacher because a side-run is telemetry: it
should give up immediately rather than add latency to a decision it does not influence.

Everything else is deliberately unchanged from the policy you graduated *from*. During shadow
mode you are measuring whether the student reproduces the teacher's decisions **under the rules
you actually run**, so changing the rules while measuring invalidates the measurement. Change
them after the cutover, one at a time, with the log to prove what each change did.

How you read the result depends on where you made it durable: from the backend,
`stats()` and `disagreements()` (in-process; inject an `on_disagreement` callback to make it
survive the process); from the log, the record's `shadow` field — if you constructed the router
with `shadow_backend=...`, because a file's `shadow:` section does not do that by itself.
`jev-route log-stats` reports nothing about shadow rows; reading them is a parse of the JSONL.
Look at agreement per question, not just overall, and weight disagreements by tier consequence:
`internal` vs `confidential` changes where a request runs, `writing` vs `analysis` changes
nothing.

```yaml
# config: policies/graduation-shadow.yaml
version: 1

# The shape `jev-route graduate --write` produces: your distilled model serves
# traffic, the cloud teacher runs alongside on a sample, disagreements are logged.
backend:
  name: shadow
  primary:                               # must be a mapping with a `name` key
    name: distilled
    artifact: ./artifacts/jev-route-distilled-v1
    feature_mode: false
  shadow:
    name: jev
    api_key_env: TYPESAFE_API_KEY
    model: jev-latest
    timeout_seconds: 5.0
    max_retries: 0                       # a side-run should not retry; it is telemetry
    include_domain: true
  log_disagreements: true
  shadow_timeout_seconds: 5.0

gate: {on_force_local: skip_backend, placeholder_domains_as_pii: false}

tiers:
  local:  [onprem-small]
  cheap:  [cloud-flash]
  strong: [cloud-max]
tier_order: [cheap, strong]

pii_threshold: 0.5

rules:
  - id: gate.force-local
    if: gate_force_local
    then: {tier: local}
    reason: local hard gate matched an identifier or credential

  - id: data.sensitive
    if: sensitivity in ["confidential", "regulated"] or pii_present
    then: {tier: local}
    reason: sensitive or personal data must not leave the infrastructure

  - id: complexity.frontier
    if: complexity == "frontier"
    then: {tier: strong}
    reason: task needs frontier-class reasoning

  - id: complexity.hard
    if: complexity == "hard"
    then: {tier: strong}
    reason: task needs a strong model

  - id: default
    then: {tier: cheap}
    reason: routine task, no sensitivity signal

# Left at the shipped defaults on purpose: during shadow mode you want the
# decision rules identical to the ones the disagreement rate is measuring.
on_uncertain:
  sensitivity_confidence_below: 0.8
  sensitivity_bump_levels: 1
  complexity_confidence_below: 0.7
  complexity_bump_levels: 1
  pii_uncertain_threshold: 0.35
  pii_uncertain_counts_as_present: true

on_backend_down: {mode: fail_closed, fail_closed_tier: local, fail_open_tier: strong}

cache: {enabled: true, kind: memory, ttl_seconds: 900, max_entries: 8192}

logging:
  enabled: true
  path: ./decision-log/decisions.jsonl
  excerpt_mode: redacted
  max_bytes: 268435456
  flush_every: 1

# The router-level side-run. Only live when the Router is constructed with
# shadow_backend=...; see section 12. 0.1 samples distinct prompts, deterministically.
shadow: {enabled: true, sample_rate: 0.1}
```

---

## 14. Validation checklist

A policy file is config that becomes behaviour. Check the behaviour, not the file — because
unknown keys are ignored silently, a file can be exactly what you wrote and still not be what
runs.

### Before you open the pull request

1. **It loads.** Every structural error is a load error.

   ```bash
   python -c "from jev_route import Policy; Policy.from_file('policies/mine.yaml'); print('ok')"
   ```

2. **Print what the loader understood.** This is the check that catches typos, because it shows
   the parsed objects rather than the YAML text:

   ```bash
   python - <<'PY'
   from jev_route import Policy
   from jev_route.cache import build_cache
   p = Policy.from_file("policies/mine.yaml")
   print("tiers      :", {k: list(v) for k, v in p.tiers.items()})
   print("tier_order :", p.tier_order)
   print("rules      :", [(r.rule_id, r.expression.source if r.expression else "<default>",
                            r.tier, r.model) for r in p.rules])
   print("uncertainty:", p.uncertainty)
   print("failure    :", p.failure)
   print("pii_thresh :", p.pii_threshold)
   print("backend    :", dict(p.backend))
   print("gate       :", dict(p.gate))
   print("cache      :", type(build_cache(p.cache)).__name__, p.cache)
   print("logging    :", dict(p.logging))
   print("shadow     :", dict(p.shadow))
   PY
   ```

   Two caveats on this check. `dict(p.backend)`, `dict(p.gate)`, `dict(p.logging)`,
   `dict(p.shadow)`, and `dict(p.cache)` print the **raw YAML mappings** — every key you wrote
   appears there, known or not, so an unknown key in one of those sections shows up in the print
   and still does nothing. The attribute objects (`p.tiers`, `p.uncertainty`, `p.failure`, the
   rule objects, the parsed `GatePolicy` fields) show what the parser *understood*: if a key you
   wrote is absent from the parsed object, nothing reads it.

3. **Order.** Data-protection rules above capability rules. Exactly one rule with no `if:`, and
   it is last. Read the `if:` column top to bottom and say out loud what each rule swallows.

4. **Names.** Every tier in every `then.tier`, in `tier_order`, and in `fail_closed_tier` /
   `fail_open_tier` is defined under `tiers:`. The loader checks all of them; if you renamed
   tiers, check that you did not inherit a default that points at the old names.

5. **Expressions.** Every name is one of the 31 in
   [section 4](#4-expression-variables) (the four `semantic_*` variables are shadow-mode
   constants — they only carry information under `gate.semantic.mode: enforce`). Arithmetic is `+ - *` only. Every `metadata[...]`
   subscript is guarded by `'key' in metadata and ...`. Quoted expressions use one consistent
   YAML quoting style.

6. **Secrets.** No literal `api_key`. Keys come from `api_key_env`.

   ```bash
   grep -nE "api_key[^_]|sk-|AKIA|ghp_" policies/mine.yaml   # should print nothing
   ```

7. **Retention.** `logging.excerpt_mode` is what your retention policy permits, `path` is on a
   volume the process user can write to, `max_bytes` is not `0` unless you mean unbounded, and
   `hash_salt` is set if the log store is shared. Then read
   [privacy.md](privacy.md) before you turn `excerpt_mode` up.

8. **The three knobs that can move data.** `on_backend_down.mode`, `gate.on_force_local`, and
   `gate.semantic.mode`. The third one moves data *away* from the cloud only (an enforced
   layer-2 firing is a stricter floor, never a looser one), but it changes what the gate does on
   live traffic, which is what a reviewer needs to see. If either changed, the pull request
   description should say why in a sentence a reviewer can argue with.

### Load-testing it offline

`backend.name: mock` needs no API key and no network, and `MockBackend` is deterministic by
contract — same input, same output, in any process, on any machine, no randomness and no clock.
That makes an offline policy test reproducible, which is the property you want from a check that
gates a deploy.

The CLI is the fast path. `--backend` overrides whatever the file says, so you can test a
`jev` policy without a key:

```bash
jev-route doctor --policy policies/mine.yaml --offline
jev-route demo   --policy policies/mine.yaml --backend mock
jev-route route  --policy policies/mine.yaml --backend mock \
    --metadata '{"tenant":"acme"}' -v "Design a sharded write path and prove the invariant"
```

`demo` prints a canned set covering the cases that matter — a greeting, a routine coding task, a
frontier design task, regulated *data* (the gate blocks the cloud call), a regulated *topic* with
no data, a compliance question about an ordinary product (advisory topic, stays on a cloud tier —
the case that best shows a topic mention is not a floor), and credential material. It also warns
you about the one thing that will confuse you:

```text
note   : MockBackend is deliberately underconfident, so you will see more
         [escalated] lines than a calibrated backend produces.
```

Do not tune `on_uncertain` thresholds against the mock. It is a plumbing fixture: it exists so
policy thresholds, confidence floors, gate merging, logging, and caching are all exercised for
free, and it is deliberately less accurate than Jev. Tune thresholds against your real backend
and your real log.

For a fixed regression set, drive the router directly and diff the output. Point `logging.path`
at a throwaway file so the test does not pollute your dataset:

```python
import asyncio
from jev_route import Policy, Router
from jev_route.backends import build_backend

POLICY = "policies/mine.yaml"
CASES = [
    ("greeting", "hi there, thanks!"),
    ("routine code", "Write a Python function that reads a CSV and sums one column."),
    ("frontier", "Design a sharded write path with consensus and prove the safety invariant."),
    ("regulated data", "Patient John Smith, SSN 666-45-1234, MRN: ABC-9931, type 2 diabetes."),
    ("regulated topic", "Explain how HIPAA actually works and who it applies to."),
    ("credential", "Debug this: our deploy uses sk-live-abcdefghijklmnopqrstuvwx and it 401s."),
    ("pii band", "Summarize this thread from dana.wu@northside-health.org about billing."),
]


async def main() -> None:
    policy = Policy.from_file(POLICY).with_overrides(
        backend={"name": "mock"},
        logging={"enabled": True, "path": "/tmp/jev-route-test/decisions.jsonl", "excerpt_mode": "hash"},
    )
    router = Router(policy, build_backend(policy))
    try:
        for label, text in CASES:
            d = await router.route_text(text)
            print(f"{label:16s} tier={d.tier:12s} model={d.model:22s} rule={d.rule_id}")
            print(
                f"{'':16s} cplx={d.effective_complexity} sens={d.effective_sensitivity} "
                f"pii={d.answers.pii.value:.2f} cached={d.cached} backend={d.backend}"
            )
            for note in d.escalated:
                print(f"{'':16s} escalated: {note}")
        # the same prompt again, to prove the cache path
        d = await router.route_text(CASES[1][1])
        print(f"{'repeat':16s} cached={d.cached} backend={d.backend}")
        print("cache:", router.stats()["cache"])
    finally:
        await router.aclose()


asyncio.run(main())
```

Save the output. Run it again after your policy change and diff:

```bash
python check_policy.py > /tmp/before.txt
# edit the policy
python check_policy.py > /tmp/after.txt
diff -u /tmp/before.txt /tmp/after.txt
```

The diff is the review artifact. Every changed line is a request class whose routing you just
changed, and a reviewer can read it without knowing the expression grammar. Commit it to the
pull request.

What the diff will *not* show you, and what you should not claim from it:

* **Accuracy.** The mock is not calibrated. Only your real backend on your real traffic is.
* **Latency or cost.** <TODO: measured by evals>
* **Escalation rates.** The mock is deliberately underconfident, so its escalation share is not
  yours. <TODO: measured by evals>

The labelled dataset in `evals/data/labeled_prompts.jsonl` is the right input for a policy
regression run against a real backend. It contains only synthetic identifiers drawn from
published test ranges — SSN area `666` is never issued, `4111 1111 1111 1111` is Visa's public
test number — and `evals/validate_dataset.py` enforces that, in CI, on every commit:

```bash
python evals/validate_dataset.py
```

### After you deploy

```bash
jev-route log-stats --policy policies/mine.yaml
```

Read these five numbers, and know what each one means before you act on it:

| field | what it tells you |
| --- | --- |
| `rules` | Which rules actually fire. A rule that never fires is dead; a rule that fires on everything is swallowing the ones below it. |
| `escalated` | The share of decisions the policy moved away from the backend's raw answer. This is your uncertainty cost, in rows. |
| `gate_blocked_cloud` | The share the gate refused to send anywhere. Rising means your traffic changed, not that the gate got stricter. |
| `degraded` | The share that hit `on_backend_down`. Anything sustained here is a backend problem, and every one of those rows is fail-closed rather than judged. |
| `records_with_excerpt_text` | Which graduation path is open to you. Zero means features-only distillation. |

Plus two operational ones from `Router.stats()`: `log.write_errors` (should be zero; a non-zero
value means your training data has holes and your disk or mount is the reason) and
`cache.hit_rate` (a low rate on repetitive traffic means the TTL or the key is wrong; a high rate
means the cache is doing the job that keeps routing latency off your critical path).

`jev-route explain <request_id>` prints one full record. When a single decision surprises you,
that is the first command to run, and `decision.escalated` is the first field to read: it tells
you whether the model or the policy moved the answer.

### The short version

* Load it. Print it. Diff its behaviour against a fixed set of prompts on `mock`.
* Data rules before capability rules. Default rule last. Every tier name real.
* No literal keys. No unknown keys you think are doing something.
* `excerpt_mode` and `on_backend_down.mode` are the two keys a reviewer should read twice.
* The log is the product. If `logging.path` is wrong, everything else is a hobby.

---

*This document describes jev-route 0.1.0, policy `version: 1`. The defaults, error messages, and
behaviours above were checked against `src/jev_route/policy.py`, `router.py`, `gate.py`,
`gate_semantic.py`, `cache.py`, `logging_sink.py`, `schema.py`, `prompts.py`, and
`backends/__init__.py` as of the 2026-09-20 two-layer-gate change; a 2026-09-19 line-by-line
audit of this document then corrected the rows that had gone stale (gate disable-ability, the
cache block, the log-write failure modes, the shadow/graduation claims, and the variable
counts). Where a key appears in a shipped YAML file but no code reads it,
[section 2](#keys-that-appear-in-yaml-and-are-read-by-nothing) says so — the last known instance
of that, `cache.skip_when_gate_forced`, was removed from `policies/default.yaml` rather than
documented.*
