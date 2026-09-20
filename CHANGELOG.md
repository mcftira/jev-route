# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Two versioning rules are specific to jev-route, and they exist because the decision log is a
dataset that outlives any single release:

* **`jev_route.schema.SCHEMA_VERSION`** (currently `"1"`) is stamped on every `DecisionRecord`.
  A field may be *added*; it may not be silently renamed, retyped, or repurposed. If the shape
  must change incompatibly, the version is bumped and a migration is added to
  `jev_route.distill.export`. Old logs must stay readable — the value of a decision log is that
  it accumulates.
* **`jev_route.distill.artifact.ARTIFACT_SCHEMA_VERSION`** (currently `"1"`) is stamped on every
  packaged model. An artifact this build cannot read raises `ArtifactError` naming the
  mismatch, rather than mis-predicting quietly.

A change to either is a breaking change to this project's data, and belongs under `Changed`
with the migration named.

---

## [Unreleased]

### Added

- `LayaBackend` (`jev_route.backends.laya`): the fully local, air-gapped decision
  backend. Wraps the open-source Laya System-1 model (convaiinnovations/laya,
  Apache-2.0, arXiv:2503.23303): all four questions in ONE `system_one` forward
  pass with the native 1:1 question mapping, a deterministic head+tail excerpt
  budget (448 tokens default) whose strategy and dropped-token count are written
  onto every decision, and fail-closed `enforce` mode that refuses to construct
  without a fitted calibration artifact. `laya_calibration` adds the temperature
  (golden-section NLL) and isotonic (pure-Python PAV) fitters plus the versioned,
  checksummed artifact format. The local-edition distribution (calibration CLI,
  223-prompt eval, Jev-vs-Laya comparison, gate-head track) ships as laya-route.

- Core routing chain in `jev_route.router.Router`: excerpt → local hard gate → redact → decide
  → merge gate floors → escalate on uncertainty → evaluate policy → log. Entry points
  `route_messages()`, `route_text()`, `route_text_sync()`, and `Router.from_policy_file()`.
- Versioned decision schema in `jev_route.schema`: `DecisionRecord`, `RoutingDecision`,
  `DecisionAnswers`, `ChoiceAnswer` (full distribution + reported/computed confidence),
  `NoulAnswer`, `GateVerdict`, `GateFinding`, `RequestFeatures`.
- Local hard gate in `jev_route.gate`: `HardGate` with 22 deterministic detectors across four
  categories (`pii`, `secret`, `regulated`, `keyword`). Checksum validators for payment cards
  (Luhn), IBANs (ISO 13616 mod-97), UK NHS numbers (mod-11) and UK National Insurance numbers
  (shape rules). Blocking detectors set `blocks_backend`, which prevents the excerpt from
  reaching a cloud backend even redacted. Topic-keyword detectors are `advisory`: forwarded as
  hints, never a floor, never a forced tier.
- Excerpting, redaction and text-free features in `jev_route.prompts`: 4000-character excerpt
  ceiling, typed `[detector_name]` placeholders, salted `excerpt_hash`, and 24 deterministic
  request features so a local model can be trained without retaining prompt text.
- `DecisionBackend` protocol, `DecisionRequest`, `BackendResult` and a three-state
  `CircuitBreaker` in `jev_route.backends.base`.
- `JevBackend`: the only module in the package that calls `api.typesafe.ai`. One POST to
  `/v1/systemone` asking four typed questions (`complexity`, `sensitivity`, `pii_present`,
  `domain`), with retries on retryable statuses, exponential backoff, per-attempt timeout, and
  degrade-instead-of-raise on any outage. A trust-boundary clause is appended to every question
  so prompt text cannot steer its own routing.
- `MockBackend`: deterministic, offline, no API key. Produces real probability distributions
  from real request features so confidence floors, gate merging, shadow mode and distillation
  are all exercisable without a network. Deliberately coarser than Jev.
- `ShadowBackend`: serves one backend's decision while another runs alongside, with
  deterministic hash-based sampling and per-head disagreement capture.
- Policy engine in `jev_route.policy`: safe-AST rule expressions (no `eval`, no calls, no
  attribute access, validated at load time), ordered first-match-wins rules with a mandatory
  default, `on_uncertain` confidence floors, `on_backend_down` fail-closed/fail-open, tier
  round-robin, and `with_overrides()` / `to_yaml()` round-tripping for programmatic policy
  edits.
- Judgement cache in `jev_route.cache`: `NullCache`, `InMemoryTTLCache` (bounded LRU + TTL) and
  `RedisCache` (lazy import, `jev-route[redis]`). Caches backend *judgements*, not routing
  *decisions*, so a policy edit takes effect on warm traffic immediately. Degraded results are
  never cached.
- Decision log in `jev_route.logging_sink`: append-only `JsonlSink` with size-based rotation
  that renames rather than truncates, `NullSink`, `CallbackSink`, `CompositeSink`, and
  crash-tolerant readers `iter_records()` / `iter_all_records()`. Three `excerpt_mode`s —
  `hash` (default, no prompt text retained), `redacted`, `none`. Gate-blocked requests are never
  written as text under any mode.
- CLI `jev-route` with `doctor`, `demo`, `route`, `log-stats`, `explain`, `export`, `train`,
  `evaluate`, `package` and `graduate`.
- LiteLLM 1.101 integrations in `jev_route.integrations`: `JevRouteClassifier`
  (`ClassifierPlugin` for the native complexity router), `JevRouteRoutingPlugin`
  (`RoutingPlugin` for `Router(plugins=[...])`) and a `CustomLogger.async_pre_call_hook` model
  rewrite. Shared plumbing has no LiteLLM import, builds the process-wide `Router` lazily,
  never raises at construction, and allowlists caller metadata to identity keys only.
- Distillation pipeline in `jev_route.distill`: `export` (log inspection, text-vs-features mode
  resolution, per-head masking of gate-blocked rows, `request_id`-hashed reproducible splits,
  stats sidecar), `train` (soft-target KL at temperature `T`, four heads, numpy Adam default,
  optional torch trainer, degeneracy check), and `artifact` (pickle-free `artifact.json` +
  `weights.npz` + `vectorizer.json` envelope with SHA-256 checksums, validated against the live
  schema on load).
- Reference policy `policies/default.yaml`, heavily commented, with data protection ordered
  before cost.
- Evaluation dataset `evals/data/labeled_prompts.jsonl`: 223 labelled prompts covering four
  complexity levels, four sensitivity levels, five domains, 19 negative controls, 12
  prompt-injection rows, 6 gate-boundary cases, Hungarian and German rows, and long pastes. All
  PII is synthetic and drawn from published test ranges. `evals/validate_dataset.py` is a
  stdlib-only guard that enforces the schema, the coverage minimums, the derived `expected_tier`
  and the synthetic-PII safety policy; it runs in CI.
- Kubernetes / k3s deployment manifests in `deploy/k8s/`.
- Documentation: `docs/privacy.md`, `docs/architecture.md`, `docs/policy-guide.md`,
  `docs/graduation.md`.
- Apache-2.0 `LICENSE`, `CONTRIBUTING.md`, `.gitignore` and a GitHub Actions CI workflow
  (Python 3.10/3.11/3.12 on `ubuntu-latest`, an optional non-blocking arm64 matrix, `ruff
  check`, `pytest -m "not network and not slow"`, and the eval-dataset guard).

### Security and privacy notes

- The local hard gate has no off switch. `gate.disabled_detectors` can silence individual
  detectors; `HardGate.scan()` always runs and always returns a verdict.
- A gate sensitivity floor of `confidential` or `regulated`, or any `force_local` finding,
  outranks `on_backend_down.mode: fail_open`. A configuration knob cannot become a data-egress
  path.
- `excerpt_hash` is a dedup and cache key, **not** an anonymization guarantee. A short prompt
  from a known corpus is still guessable. This is documented in `docs/privacy.md` and in the
  docstring of `prompts.hash_text`.
- Caller metadata is stripped of `messages`, `prompt`, `content`, `input`, `raw` and `body`
  before it reaches a backend, filtered to scalar values before it reaches the log, and
  allowlisted to identity keys in the LiteLLM integrations.

### Known gaps

- `evaluate`, `package` and `graduate` are wired into the CLI with a fixed argument contract;
  their `distill` implementations are still landing. `docs/graduation.md` marks what is not yet
  runnable.
- `ruff check .` is not yet clean across `src/`; the CI lint job is expected to be red until a
  one-off `ruff check --fix .` is reviewed and committed.
- `ruff format` is deliberately not enforced in CI: the codebase is not format-clean yet, and a
  permanently red formatting job is worse than no formatting job.
- No cost-savings figure is published. The README carries an explicit `<TODO>` for it rather
  than an estimate.

---

## [0.1.0]

Initial public release. See `[Unreleased]` — the contents are the same until the first tag is
cut.

[Unreleased]: https://github.com/mcftira/jev-route/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mcftira/jev-route/releases/tag/v0.1.0
