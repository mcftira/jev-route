# Architecture

jev-route is one request in, one routing decision out. This document explains the shape of
that path and, more importantly, why it has that shape — because almost every structural
decision here is defending a property rather than organising code.

The four properties being defended:

1. **The local hard gate runs first, always locally, never model-decided, never bypassed.**
2. **Fail closed.** A decisioning outage routes to the safest tier, not the cheapest one.
3. **One egress point.** Exactly one module in the package talks to `api.typesafe.ai`.
4. **The log is the product.** Every decision is persisted with full soft distributions, so
   the router you run today is the training data for the router you own later.

---

## The chain

`Router._route()` in `src/jev_route/router.py` is the whole system. Eight steps, and the order
is not negotiable.

```text
 request (messages or text)
      │
      ▼
 1. EXCERPT      excerpt_messages() / excerpt_from_text()
      │          bounded to MAX_EXCERPT_CHARS = 4000; newest user turn gets the
      │          largest share of the budget; system prompts excluded by default;
      │          tool output counted as a feature, never as text
      ▼
 2. GATE         HardGate.scan(raw_excerpt)
      │          local, deterministic, in-process, no network. Runs on the RAW
      │          excerpt. Produces a GateVerdict: findings, sensitivity_floor,
      │          pii_floor, force_local, blocks_backend, advisory_topics
      ▼
 3. REDACT       redact(raw_excerpt, gate) -> typed placeholders
      │          compute_features(raw_excerpt, ...) -> text-free RequestFeatures
      │          excerpt_hash = sha256(salt + raw_excerpt)[:16]
      ▼
      ├── skip_backend?  (blocks_backend, OR force_local && on_force_local==skip_backend)
      │        │
      │   yes  ▼  _gate_only_result(verdict)      backend = "gate"
      │        │  sensitivity := gate floor @ p=1.00, complexity/domain := uniform,
      │        │  questions_sent := {}   ← NOTHING was sent anywhere
      │        │
      │   no   ▼
 4. DECIDE      cache.get(key) -> hit?  backend = "<name>:cached"
      │                       -> miss? await backend.decide(DecisionRequest)
      │                                cache.put(key, result)
      │          (optional) shadow backend runs alongside, off the decision path
      ▼
 5. MERGE       _merge_gate(answers, verdict)
      │          sensitivity := max(backend, gate floor)
      │          pii          := max(backend, gate floor)
      │          max, never min. No backend can talk its way out of a finding.
      ▼
 6. ESCALATE    on_uncertain
      │          complexity_confidence  < threshold -> bump N levels stricter
      │          sensitivity_confidence < threshold -> bump N levels stricter
      │          pii in the uncertain band          -> treat as present
      │          skipped for gate-only and degraded results (honest provenance)
      ▼
 7. EVALUATE    policy.evaluate(namespace)
      │          first matching rule wins -> (rule, tier, model)
      │          degraded? -> _failure_tier(): fail_closed / fail_open, with the
      │                       gate floor outranking fail_open
      ▼
 8. LOG         sink.write(DecisionRecord)
                 full distributions, features, gate verdict, escalations, latency.
                 Gate-blocked requests are never written as text.
```

### Why the order is the architecture

Steps 1→2→3 are a privacy property, not a pipeline convenience. The gate scans the **raw**
excerpt; redaction is applied to the raw excerpt; and only the redacted form is ever handed to
a backend. If you redacted first and then gated, the gate would be scanning placeholders and
would find nothing.

Step 5 before step 7 is what makes the gate a constraint rather than a rule. The policy sees
the **effective** sensitivity — already floored — so even a policy with every gate rule
deleted cannot route a card number to a cloud model.

Steps 5 and 6 together are what make this a *calibrated* router rather than a classifier with
extra steps. Step 8 is what makes it a router you eventually own.

### Why escalation is skipped when nobody classified

`Router._route` guards both escalation blocks with `classified and not backend_result.degraded`.
A gate-blocked request has uniform complexity and domain **by construction** — nobody judged
them. Applying a confidence floor to a uniform distribution would log
`complexity hard -> frontier (confidence 0.00 < 0.7)` for a request no model ever looked at.
The source comment is the justification: *honest provenance matters more in a training log than
a tidy-looking one.* The same reasoning is why `InMemoryTTLCache.put()` refuses to cache a
degraded result — caching maximum-uncertainty noise would pin fail-closed behaviour for the
whole TTL after a one-second network blip.

---

## `DecisionBackend` is the architectural center

`src/jev_route/backends/base.py` defines a four-attribute protocol:

```python
@runtime_checkable
class DecisionBackend(Protocol):
    name: str
    model_version: str

    async def decide(self, request: DecisionRequest) -> BackendResult: ...
    async def aclose(self) -> None: ...
```

Everything above it — the policy engine, the router, the cache, the sink, the LiteLLM
integrations, the CLI — speaks to that protocol and has no idea where the answer came from.

That indirection is not for tidiness. **It is the mechanism that makes the project's promise
real.** You bootstrap on a calibrated cloud model, your traffic accumulates as labelled
decisions, you distill a local model from them, and then you change one config value and the
cloud dependency is gone. Nothing else in the system changes, because nothing else in the
system ever knew.

Two invariants every implementation must obey:

* **Return the full schema.** A backend that cannot answer a question returns that question at
  maximum uncertainty (`ChoiceAnswer.uniform`, `NoulAnswer.unknown`), never omits it. The
  policy engine and the log format both assume the schema is total.
* **Never raise for an expected outage.** Timeouts, rate limits, DNS failures and circuit-open
  conditions come back as a degraded `BackendResult`. Only programmer errors propagate. The
  router's fail-closed behaviour depends on this — a backend that raises turns a routing
  decision into a 500.

### The implementations

| backend | `name` | network | key | role |
| --- | --- | --- | --- | --- |
| `JevBackend` | `jev` | TypeSafe cloud | `TYPESAFE_API_KEY` | **Day one.** Calibrated, zero training data |
| `MockBackend` | `mock` | none | none | Deterministic offline fixture. Tests, CI, demos, and orgs that cannot send anything out |
| `DistilledBackend` | `distilled` | none | none | **The end state.** Serves an artifact trained on your own log |
| `ShadowBackend` | `shadow` | both | as configured | Runs two backends, serves one, logs disagreements. The cutover path |

`build_backend()` in `backends/__init__.py` is the factory. `distilled` and `shadow` are
imported **lazily** inside their branches: a deployment running on Jev should not pay for the
artifact loader or numpy, and importing `jev_route` must succeed with no numerical stack
installed at all.

`MockBackend` deserves a word, because it looks like a stub and is not one. It computes real
probability distributions from real request features (marker regexes, length, code fences,
stack traces, prior turns, gate topic hints) through a temperature-`0.55` softmax, and it is
deterministic across processes and machines — no randomness, no clock, no counter. That is
what makes the confidence-floor, gate-merging, shadow-mode and distillation code paths testable
against plausible output instead of a hardcoded argmax. It is also *deliberately less accurate
than Jev*: zero cost and zero egress in exchange for a coarse read.

### One egress point

```bash
$ grep -rn --include='*.py' "api.typesafe.ai" src/
src/jev_route/backends/jev.py:4:That is a hard invariant, not a convention: ``grep -r api.typesafe.ai src/`` must
src/jev_route/backends/jev.py:39:DEFAULT_API_URL = "https://api.typesafe.ai/v1/systemone"
src/jev_route/backends/__init__.py:46:            api_url=str(cfg.get("api_url", "https://api.typesafe.ai/v1/systemone")),
```

Two files name the host, and that is the invariant, checkable in one command: `backends/jev.py`
is the only module that performs a request, and `backends/__init__.py` only supplies the default
URL to it. `JevBackend`'s docstring: *you should be able to prove the claim by looking at where
the egress happens, and by deleting one class.*

`JevBackend.__init__` **raises** when no API key is available rather than degrading to "no
key", because a router that quietly stops classifying is worse than one that refuses to start.
If you want no key, ask for `mock`.

### The four questions

One HTTP call, four typed questions, run in parallel server-side:

| question | type | ladder / range |
| --- | --- | --- |
| `complexity` | `choice` | `trivial` → `standard` → `hard` → `frontier` |
| `sensitivity` | `choice` | `public` → `internal` → `confidential` → `regulated` |
| `pii_present` | `noul` | `0.0`–`1.0`, the probability of *yes* |
| `domain` | `choice` | `code` / `writing` / `analysis` / `chat` / `data-extraction` |

`build_questions()` is public and its content is part of the **dataset contract**: changing the
criteria changes what gets logged and therefore what can be distilled. `include_domain: false`
skips the fourth question to save tokens when no policy rule reads `domain`.

Responses are coerced onto our ladders by `_normalize_choice`, which preserves the
distribution, renormalizes it, fills omitted options with `0.0` so the logged distribution
always spans the full ladder, and degrades an unknown option to maximum uncertainty rather
than crashing the router.

---

## The cache stores judgements, not decisions

`src/jev_route/cache.py` caches `BackendResult` — the backend's four answers — and **not** the
`RoutingDecision` the router produced from them.

That distinction is the reason the module exists separately from the policy engine. If you
cache the final tier, a policy change silently does not apply to cached traffic for as long as
the TTL runs. An operator who tightens a sensitivity rule at 14:00 sees it appear to work —
their test prompts are cold — while warm production traffic keeps taking the old path until
14:15. Caching the judgement means a policy edit takes effect on the **next request**, and the
cache only ever saves the network round-trip it was there to save.

The cache key is `sha256(policy_version ‖ backend_name ‖ redacted_excerpt ‖
advisory_topics)[:32]`, so:

* two backends never serve each other's cached answers — important during shadow mode, when a
  `jev` judgement and a `distilled` judgement for the same text are different facts;
* a change in redaction rules produces a new key rather than reusing a stale judgement;
* `policy_version` is in the key even though the cached value is policy-independent, so a
  schema bump cannot be served from a pre-bump cache.

Implementations:

| class | when |
| --- | --- |
| `NullCache` | `cache.enabled: false`, or every request must be classified afresh |
| `InMemoryTTLCache` | default. Bounded LRU + per-entry TTL, single process. In a multi-worker proxy each worker holds its own — the win rate drops, correctness does not |
| `RedisCache` | workers must share. Lazy-imports `redis.asyncio`; raises `pip install 'jev-route[redis]'` when absent. Serializes the full `BackendResult` as JSON, so distributions survive the round trip and a cached answer is indistinguishable from a fresh one downstream |

`RedisCache.clear()` raises `NotImplementedError` rather than `FLUSHDB`-ing a shared cache.

A cache hit is recorded as `backend: "<name>:cached"` and `decision.cached: true`, and its
`latency_ms` is zeroed, so the log distinguishes a real backend call from a replay. Note that
rule expressions see `cached: False` — `_namespace()` hardcodes it, because the namespace is
built before the router knows whether it will matter. Do not write a rule against `cached`.

---

## The policy engine parses, it does not `eval`

Rule conditions in `policies/*.yaml` are compiled from a strictly limited subset of Python's
AST. `_ALLOWED_NODES` in `src/jev_route/policy.py` is the whitelist:

```text
Expression, BoolOp, And, Or, UnaryOp, Not, USub,
Compare, Eq, NotEq, Lt, LtE, Gt, GtE, In, NotIn, Is, IsNot,
Name, Load, Constant, List, Tuple, Set, Subscript, BinOp, Add, Sub, Mult
```

Allowed: names from `EXPRESSION_VARIABLES`, constants, comparisons including `in` / `not in` /
`is` / `is not`, `and` / `or` / `not`, literal list/set/tuple, subscripting **a simple name
only**, binary `+ - *`, unary minus.

Not allowed: calls, attribute access, comprehensions, lambdas, imports, f-strings, the walrus
operator, starred expressions, and `** / // / %` (only `Add`, `Sub` and `Mult` are whitelisted,
so `Pow`, `FloorDiv` and `Mod` are rejected).

The reason is not that `eval` is scary in the abstract. It is that **a policy file is
configuration loaded at startup from wherever operators keep configuration**, and "we checked
it looked safe" is not a control. `_validate_node` walks the tree and raises `PolicyError`
naming the offending node — at **load** time, not at request time. Evaluation then runs
`compile(tree, "<jev-route-policy>", "eval")` with `{"__builtins__": {}}`, and expressions are
memoized in a lock-guarded `_COMPILE_CACHE`, because policies are parsed once and evaluated on
every request.

Three further properties of the engine:

* **The gate is a floor the policy cannot talk itself out of.** Rules *see* `gate_force_local`,
  `gate_blocks_backend` and `gate_sensitivity_floor`, and the shipped policy routes on them
  first. But `_merge_gate` has already applied the floor before rules run. The gate rule in
  `policies/default.yaml` is documented in that file as *belt-and-braces*.
* **Uncertainty is a first-class input.** `on_uncertain` bumps the effective answer *stricter*
  when confidence falls below a threshold. "70% sure it is internal" becomes "treat it as
  confidential". This is the behaviour a regex cannot express and an argmax throws away, and it
  is the single most useful knob in the file.
* **Loading validates everything, loudly.** Unknown policy version, a tier named by a rule that
  is not defined, a `tier_order` naming an undefined tier, more than one default rule, a default
  rule that is not last, no default rule at all, an unknown `on_backend_down.mode`, a failure
  tier that is not defined — all `PolicyError` at load time. A request should never fail because
  a policy was malformed; the process should refuse to start.

`Policy.with_overrides(**changes)` returns a new policy with top-level keys replaced, and
`to_yaml()` / `write_yaml()` round-trip it. This is how `jev-route graduate` performs the
backend swap as a **config change rather than a code change**.

Full key reference: [docs/policy-guide.md](policy-guide.md).

---

## Failure: closed by default, and the gate wins anyway

`FailurePolicy` has three fields: `mode` (`fail_closed`, the default), `fail_closed_tier`
(`local`), `fail_open_tier` (`strong`).

Both modes are defensible. `fail_closed` says the risk you care about is **data egress**: when
you cannot classify a request, send it to the tier whose data never leaves the building.
`fail_open` says the risk you care about is **answer quality**: when you cannot classify, use
the most capable model you have. Silently picking one for the operator is not defensible, so it
is explicit config with a safe default.

The invariant that is not configurable:

```python
_GATE_OVERRIDES_FAIL_OPEN = ("confidential", "regulated")
```

If the local gate found a floor at `confidential` or above, or set `force_local`, then
`fail_open` is overridden and the fail-closed tier wins. `RoutingDecision.reason` records that
it happened:

```text
decision backend unavailable (timeout after 5.0s); fail_open -> tier strong;
local gate floor took precedence over fail_open -> tier local
```

The rule id used for every outage decision is `backend.down`.

### The circuit breaker

`CircuitBreaker` in `backends/base.py` is a minimal three-state breaker:
`closed → open → half-open → closed`, with `failure_threshold` (default 5),
`recovery_seconds` (default 30) and `half_open_max_calls` (default 1).

It exists so a decisioning outage **degrades** the router instead of adding a full timeout to
every single request. Without it, a dead backend turns a 5 ms routing decision into
`N × timeout` and the proxy looks hung rather than fail-closed — which is the worst possible
failure mode for an inline router, because it converts a privacy control into an outage.

It is deliberately **not** thread-safe: it is used from one asyncio event loop. `state` is
recomputed on read so an expired open breaker reports `half-open` without a background task.

Retry policy in `JevBackend`: `max_retries` (default 2) *after* the first attempt, only on
retryable statuses (`429, 500, 502, 503, 504, 529`) and timeouts, with exponential backoff
capped at 2 s and **no jitter** — determinism helps tests, and at this scale thundering herds
are not the concern. Each attempt is wrapped in `asyncio.wait_for(..., timeout_seconds)`
(default 5.0), so a slow call is cancelled cleanly rather than leaking a task.

---

## The decision log is the dataset

`src/jev_route/logging_sink.py` is written to dataset standards rather than logging standards,
and the module docstring says so:

* **Schema-versioned.** Every record carries `schema_version`. A field may be added; it may not
  be silently renamed, retyped, or repurposed. Old records must stay readable, because the value
  of the log is that it accumulates.
* **Soft targets, not just argmax.** Distilling from argmax labels throws away exactly the
  calibration that made the bootstrap worth paying for.
* **Append-only.** Rotation *renames* the current file; it never truncates or edits one.
* **Provable without being leaky.** A `GateFinding` stores a SHA-256 of the matched span, not
  the span.

`JsonlSink` does one `write()` per record under a lock, on a file opened in append mode with
line buffering plus `flush_every` (default 1) `fsync`s. On a regular file with `O_APPEND` that
is effectively atomic per line, which is why a multi-worker proxy can share one path without
interleaving records. An `OSError` — a full disk — is counted and swallowed rather than
propagated: *a full disk must not take down the request path. Routing still works, the log has
a hole.* That is the one place in the codebase where losing data is preferred to losing
availability, and it is the right trade only because the log is a training set and not the
system of record.

Sinks: `NullSink`, `JsonlSink`, `CallbackSink` (ship to a warehouse; a broken callback must not
break routing), `CompositeSink` (fan out; one sink's failure does not affect the others).
Readers: `iter_records(path)` and `iter_all_records(directory)`, both of which skip foreign
lines and tolerate a partial trailing line from an interrupted write — *a log that cannot be
read after a crash is worse than no log.*

`DecisionRecord.from_json` is the migration surface. `iter_records` filters on
`kind == "jev_route.decision"`, so a decision log can share a stream with other event kinds.

Schema and record reference: `src/jev_route/schema.py`. Privacy reference:
[docs/privacy.md](privacy.md).

---

## LiteLLM integration modes

Target: LiteLLM **1.101.0**. Four seams, one shared `Router`.

### Shared plumbing

`src/jev_route/integrations/_shared.py` has **no LiteLLM import** — deliberately, so the router
singleton and the payload helpers are testable without LiteLLM installed and reusable from the
CLI. Only the three `litellm_*` modules touch LiteLLM types, and they are the only place a
version bump can break us.

Two decisions in that module protect against failure modes that look like over-engineering
until you have seen them:

* **The router is built lazily and never raises.** LiteLLM resolves plugin dotted paths at
  proxy startup, and a module-level instance that raises in its constructor takes the whole
  proxy down. A router that cannot find its policy or its API key falls back to a built-in
  policy on `MockBackend` and logs loudly — a booted proxy making coarse decisions beats a proxy
  that will not start, and the fallback is the *safe* direction, not the cheap one. It logs
  under the `LiteLLM` namespace on purpose, because the proxy configures that logger and its
  handlers; a `jev_route.*` logger would propagate to a root a proxy deployment may never have
  configured, and the operator would never see why their traffic was being routed by the
  fallback policy.
* **Metadata is allowlisted, not denylisted.** A denylist of "keys that might contain prompt
  text" has to be complete forever, against a payload LiteLLM owns and changes. An allowlist of
  identity keys has to be correct once.

Environment: `JEV_ROUTE_POLICY`, `JEV_ROUTE_DECISION_LOG` (overrides `logging.path`, because
where a writable volume lives differs between a laptop and a container), `JEV_ROUTE_TIMEOUT_MS`
(default 3000), `JEV_ROUTE_MANAGED_MODELS` (default `*`), `JEV_ROUTE_METADATA_EXTRA_KEYS`,
`JEV_ROUTE_CLASSIFIER_TIERS`.

### Mode 1 — `ClassifierPlugin` (recommended)

`jev_route.integrations.litellm_classifier.JevRouteClassifier`, exposed as the module-level
instance `classifier` for LiteLLM's dotted-path resolver.

Plugs into LiteLLM's **native complexity router**: a `model_list` entry whose
`litellm_params.model` is `auto_router/complexity_router`, with `classifier_type: custom`.
`classify(context)` returns a **tier name** or `None`.

Returning a tier rather than a model is what makes this the most LiteLLM-native of the four:
jev-route supplies the judgement, LiteLLM keeps every consequence of it — cooldowns, fallbacks,
load balancing, health checks. It is also why the plugin returns `None` in three separate
situations (a declined decision, an unknown tier, an internal error) rather than guessing:
`None` hands control to the fallback **the operator chose**, not to a default the class
invented.

The module-level instance is not a convenience. The proxy resolves `classifier_plugin` through
`get_instance_fn`, whose last dotted segment must be a module attribute **holding an instance**,
and `resolve_classifier_plugin` then requires `inspect.iscoroutinefunction(obj.classify)` — so
`classify` is a real coroutine function, not something returning an awaitable.

LiteLLM's own source is the reason this seam exists. From
`litellm/router_strategy/complexity_router/config.py`, describing
`classifier_fallback: default_model`:

> `'default_model'` skips scoring and routes to default_model, which is what a classifier on
> some other taxonomy wants: **a prompt that grades data sensitivity has no use for a
> complexity score**, and scoring one produces a tier unrelated to what the operator configured.

Two constraints when you configure it:

* `classifier_fallback: default_model` and `tier_definitions` are **mutually exclusive** —
  LiteLLM rejects the combination, because with a custom tier set `fallback_tier` *is* that
  knob. jev-route uses `tier_definitions` (`local` / `cheap` / `strong`) plus
  `fallback_tier: local`, so a failed classification lands on the tier whose data never leaves.
* A custom tier set costs you LiteLLM's escalation, adaptive routing, session affinity, tier
  labels and complexity-router plugin pipeline. Deliberate: jev-route does its own escalation
  in the policy engine (`on_uncertain` bumps *stricter* on evidence, not on a retry counter) and
  its own stickiness through the decision cache.

Config: see the README's [LiteLLM integration modes](../README.md#litellm-integration-modes)
and the heavily annotated `examples/litellm-proxy-classifier/config.yaml`. Mode 3's equivalent
is `examples/litellm-proxy-hook/config.yaml`, mode 2's is `examples/litellm-sdk-plugin.py`.

### Mode 2 — `RoutingPlugin`

`jev_route.integrations.litellm_plugin.JevRouteRoutingPlugin`, for
`litellm.Router(plugins=[...])` in the SDK. The router builds a `RoutingContext` and runs it
through every plugin in order; each may mutate it; then the router keeps only the healthy
deployments whose `litellm_params.model` survived in `context.candidate_models`.

Three facts about LiteLLM drive that class's design, and all three are traps:

1. **`candidate_models` holds `litellm_params.model` strings, not the `model_name` alias.** So
   the tier → models mapping must be in the *provider* namespace (`openai/qwen3.8`), not the
   alias namespace (`qwen38`). Get it wrong and every decision falls through to the
   safest-candidate path — which looks like "the plugin works but always picks local" rather
   than like a config error.
2. **An empty candidate list is a 500, not a fallback.** LiteLLM raises
   `ValueError("No deployments left after routing-plugin filtering")` deliberately, treating it
   as a policy decision that must not be bypassed. jev-route has no use for that escape hatch: a
   tier whose models are not among the candidates means the operator's policy and their
   `model_list` disagree, which is a **configuration bug**, and the right response to a
   configuration bug in a safety router is the safest available candidate plus a loud signal —
   not a dropped request. So the plugin never returns an empty list.
3. **Narrowing to a same-length list is silently ignored.** The router only stores the narrowed
   list when `len(context.candidate_models) < len(original)`. Safe for a set-based filter; a
   trap for a plugin that tries to reorder or replace candidates.

"Safest" is not the plugin's opinion. `_safety_ranking()` puts the tier the operator named as
`on_backend_down.fail_closed_tier` first — the place their own policy says data may always go —
then the policy's ascending `tier_order`, then any tier the order does not mention.

The plugin writes a compact JSON summary of the decision into `context.signals["jev_route"]`.

### Mode 3 — `CustomLogger.async_pre_call_hook`

`jev_route.integrations.litellm_hook.JevRoutePreCallHook`. Rewrites `data["model"]` on the way
into the proxy, with no router coupling at all. Use it when you have an existing proxy config
you cannot restructure. It only rewrites models the operator opted in to
(`JEV_ROUTE_MANAGED_MODELS`), and only for completion-shaped call types —
`ROUTABLE_CALL_TYPES` covers `completion`, `text_completion`, `generate_content`,
`anthropic_messages`, `responses` and their async variants. Everything else (embeddings,
reranks, image generation, transcription, file/batch/vector-store endpoints) is ignored,
because routing a text-completion decision onto an embedding call would rewrite a model that
has no chat tier behind it and there is no prompt to judge anyway.

### Mode 4 — plain SDK

```python
from jev_route import Router

router = Router.from_policy_file("policies/default.yaml")
decision = await router.route_messages(messages, metadata={"team_id": "support"})
decision = await router.route_text("...")
decision = router.route_text_sync("...")  # scripts and sync integrations
await router.aclose()
```

No LiteLLM at all: your own gateway, a batch job, an eval harness, or the `jev-route` CLI.
`route_text_sync` detects a running loop and, if there is one, submits to a worker thread with
its own loop rather than deadlocking the caller.

---

## Shadow mode

`backend.name: shadow` (or `shadow.enabled` on the router) runs two backends: one serves, one
observes. `_run_shadow` is off the decision path — its result never affects the tier — and a
shadow failure is **telemetry, not an error**: it is caught and recorded as
`{"backend": ..., "error": "TypeError: ..."}`.

Sampling is deterministic: `_sample_hit(rate, excerpt_hash)` buckets the excerpt hash, so
replays are reproducible and the same prompt is always in or always out.

Disagreement is computed per head — `complexity`, `sensitivity`, `domain` on argmax, and `pii`
when the two probabilities differ by `>= 0.25` — and written to the record's `shadow` field
along with the shadow's full answers, model version, latency and degraded flag. That field is
the evidence base `jev-route graduate` reads.

---

## Reference deployment

The topology this was designed and measured against: a DGX Spark (arm64, GB10) running
k3s v1.36.4 with Rancher, node `spark-b578`.

| tier | deployment | where |
| --- | --- | --- |
| `local` | `qwen38` — llama.cpp serving Qwen3.8-27B-UD-Q4_K_XL with an MTP draft model, on the DGX GPU | in-cluster, `http://qwen.qwen.svc.cluster.local:8010/v1` |
| `cheap` | `qwen3.8-flash` | Alibaba Cloud Model Studio, OpenAI-compatible |
| `strong` | `qwen3.8-max` | same endpoint; measured 2.0–2.3 s for a short completion |
| decision | Jev, `POST https://api.typesafe.ai/v1/systemone`, `jev-latest` (returns `jev-1.13.0`) | TypeSafe cloud |

Measured routing-decision latency for the full four-question call: **~0.67 s from inside the
cluster**, **0.27–0.93 s from a laptop**. The `local` tier is the reason `fail_closed_tier:
local` is a meaningful default rather than a slogan — it is a real, capable, self-hosted model
on the same GPU as the proxy, not a fallback stub.

`deploy/k8s/` holds the manifests for that topology.

---

## Module map

```text
src/jev_route/
  schema.py            DecisionRecord, RoutingDecision, DecisionAnswers, ChoiceAnswer,
                       NoulAnswer, GateVerdict, GateFinding, RequestFeatures.
                       THE CONTRACT. Versioned, stdlib-only, immutable.
  gate.py              HardGate, Detector, DEFAULT_DETECTORS. Checksum validators
                       (Luhn, IBAN mod-97, NHS mod-11, NI shape). Step 2.
  prompts.py           excerpt, redact, hash_text, compute_features, detect_language,
                       build_backend_state. Steps 1 and 3.
  policy.py            Policy, Rule, Expression, UncertaintyPolicy, FailurePolicy,
                       EXPRESSION_VARIABLES, safe-AST validation. Step 7.
  backends/
    base.py            DecisionBackend, DecisionRequest, BackendResult, CircuitBreaker.
    jev.py             JevBackend. THE ONLY egress point.
    mock.py            MockBackend. Deterministic, offline, no key.
    distilled.py       DistilledBackend. Serves a packaged artifact.  [landing]
    shadow.py          ShadowBackend. Two backends, one decision.
                       (both are imported lazily by build_backend, so a deployment
                        running on jev or mock never loads them)
    __init__.py        build_backend(). The factory.
  cache.py             DecisionCache, NullCache, InMemoryTTLCache, RedisCache, build_cache.
  logging_sink.py      DecisionSink, JsonlSink, NullSink, CallbackSink, CompositeSink,
                       build_sink, iter_records, iter_all_records. Step 8.
  router.py            Router. The eight steps.
  cli.py               doctor, demo, route, log-stats, explain, export, train,
                       evaluate, package, graduate.
  integrations/
    _shared.py         Router singleton, payload normalizers, metadata allowlist.
                       No LiteLLM import.
    litellm_classifier.py  JevRouteClassifier  (mode 1)
    litellm_plugin.py      JevRouteRoutingPlugin (mode 2)
    litellm_hook.py        JevRoutePreCallHook   (mode 3)
  distill/
    artifact.py        Vectorizer, StudentModel, DistilledArtifact, save/load. No pickle.
    export.py          inspect_log, resolve_mode, export_dataset. Log -> training set.
```

Dependency rule: **stdlib + PyYAML + httpx in the core.** Anything heavier — numpy,
scikit-learn, torch, transformers, redis, pandas, matplotlib — is a lazy import inside a
function, behind an extra, with an actionable error message:

```text
distillation needs numpy. Install it with:  pip install 'jev-route[distill]'
```

`import jev_route` must succeed with none of them installed, and the router must run end to end
on `mock` with no key. That is what makes the offline promise testable rather than aspirational.

---

## Where to read next

| | |
| --- | --- |
| [docs/privacy.md](privacy.md) | exactly what leaves the process, and when. Read this first |
| [docs/policy-guide.md](policy-guide.md) | every policy key, the expression grammar, five complete examples |
| [docs/graduation.md](graduation.md) | export → train → evaluate → package → graduate |
| [policies/default.yaml](../policies/default.yaml) | the reference policy, heavily commented |
| [evals/data/README.md](../evals/data/README.md) | the 223-row labelled dataset and its synthetic-PII policy |
