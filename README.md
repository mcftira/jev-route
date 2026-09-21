<!--
BADGE NOTE: the PyPI badge below is commented out until the first release is
published. Uncomment it (and delete this comment) once `jev-route` is on PyPI.
-->

![jev-route](docs/banner.jpg)

# jev-route

**Run it. Log it. Distill it. Own it.**

<!-- CI badge: re-enable when this account has GitHub Actions:
[![CI](https://github.com/mcftira/jev-route/actions/workflows/ci.yml/badge.svg)](https://github.com/mcftira/jev-route/actions/workflows/ci.yml)
-->
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
<!-- [![PyPI](https://img.shields.io/pypi/v/jev-route.svg)](https://pypi.org/project/jev-route/) -->

jev-route is a router you run **today** on [TypeSafe Jev](https://typesafe.ai)'s calibrated
cloud decisions — and that turns your own traffic into **your own local routing model**. You
start with one API key and zero training data. Every routing decision is written to a log
with its full probability distribution, so the log *is* a labelled dataset. Once enough of
your traffic has accumulated you distill a local model from it, evaluate it against the
teacher on your own held-out decisions, and graduate to an air-gapped router that is good at
*your* workload and sends nothing anywhere. Cloud-calibrated today, air-gapped tomorrow: the
cloud phase is training wheels, and the graduation pipeline is the product.

It routes on two axes at once — **how hard is this task** and **how sensitive is this data** —
plus a PII judgement and a task domain, and it does so with *calibrated confidence*, so
"I'm 62% sure this is internal" is a routable fact rather than a shrug.

```text
                      ┌───────────────────────────────────────────────┐
   request  ───────►  │ 1. excerpt                                    │
                      │ 2. LOCAL GATE — two local layers:             │
                      │     layer 1 is deterministic and              │
                      │     never bypassed; layer 2 is a              │
                      │     local semantic model, shadow by           │
                      │     default                                   │
                      │ 3. redact    4. decide (jev | mock |          │
                      │               distilled | laya | shadow)      │
                      │ 5. merge gate floors  6. escalate on          │
                      │               uncertainty                     │
                      │ 7. evaluate policy → tier + model             │
                      │ 8. LOG the whole thing, soft probs            │
                      │               included                        │
                      └───────────────────────────────────────────────┘
                                             │
                                             ▼
                              decision-log/decisions.jsonl
                                             │  ← this is your training set
                                             ▼
                          export → train → evaluate → package → graduate
                                             │
                                             ▼
                        backend.name: jev  ────►  backend.name: distilled
```

---

## v0.2: hardening + measured numbers

v0.2 hardens the routing chain with three changes, and each one is a file you can open. The
**tier prefilter** (`src/jev_route/prefilter.py`) is deterministic and runs *before any model
call*: it rules out the tiers that could never serve the request, and when exactly one tier
survives, routing is free — the backend call is skipped and the decision is logged with
`backend: "prefilter"`, so the distill pipeline sees it like any other decision. Pinned-model
routing is a strict opt-in: only with `routing.bypass_on_pinned: true` (default off) does an
explicit model pin bypass the router. And uncertainty is marked, not swallowed: a decision
below `routing.min_confidence` gets `uncertain: true`, and the request is diverted to
`routing.uncertain_fallback` — the decision log doubles as a human review queue.

The measured numbers, each tied to the file that produces it:

| result | number | source |
| --- | --- | --- |
| cost vs always-frontier | **92.3% cheaper** on a 500-request synthetic trace | [`docs/backtest_v0.2.md`](docs/backtest_v0.2.md) |
| injection wrappers + authority framing | **0 leaks** on 120 attack cases | [`evals/injection/RESULTS.md`](evals/injection/RESULTS.md) |
| benign controls | **0 false positives** on 104 cases | [`evals/injection/RESULTS.md`](evals/injection/RESULTS.md) |
| encoding tricks (base64, spaced, typo'd Hungarian PII) | **25 documented leaks**, of 40 cases | [`evals/injection/RESULTS.md`](evals/injection/RESULTS.md) |

The last row is the point, not a bug to hide: a regex-class gate cannot read base64 or a
typo'd identifier, and every one of those 25 leaks is written down in the results file as a
requirement for the distilled semantic layer. The threat model behind the eval:
[docs/privacy.md](docs/privacy.md).

Reproduce — both commands are keyless:

```bash
python -m jev_route.cli backtest --trace traces/demo_500.jsonl
python evals/injection/run.py
```

## Contents

- [v0.2: hardening + measured numbers](#v02-hardening--measured-numbers)
- [Why routing needs calibration](#why-routing-needs-calibration)
- [Architecture at a glance](#architecture-at-a-glance)
- [Bootstrap on Jev](#bootstrap-on-jev)
- [Your traffic becomes your model](#your-traffic-becomes-your-model)
- [Go air-gapped](#go-air-gapped)
- [Quickstart](#quickstart)
- [The local hard gate](#the-local-hard-gate)
- [How the gate learns without ever seeing your secrets](#how-the-gate-learns-without-ever-seeing-your-secrets)
- [LiteLLM integration modes](#litellm-integration-modes)
- [Writing a policy](#writing-a-policy)
- [Honest limitations](#honest-limitations)
- [Inspiration and credit](#inspiration-and-credit)
- [Documentation](#documentation)

## Architecture at a glance

Five diagrams carry the whole story. The colour convention is the security claim:
**teal stays local, oxide red crosses the boundary.** Sources are PlantUML
(`.puml`) in [`docs/diagrams/`](docs/diagrams/); regenerate with
`java -jar plantuml.jar -tsvg docs/diagrams/*.puml`.

### The request path

![The request path — eight steps in a fixed order](docs/diagrams/request-path.svg)

Eight steps, in a fixed order, and the order is the design: excerpt, then the two
local gate layers, *then* the trust boundary, and only then does anything that could
leave the process get touched. Step 8 — the log with full soft distributions — is
what makes the rest of the pipeline possible. Deep dive:
[docs/architecture.md](docs/architecture.md).

### The two-layer gate

![The two-layer local gate](docs/diagrams/gate-two-layer.svg)

Layer 1 is deterministic and never bypassed; layer 2 is a learned scorer that starts
in shadow and is only allowed to enforce on *measured* criteria. Floors merge by
maximum, so the verdict can only ever make routing stricter. The blocked-content
stream carries features and hashes, never text — and it feeds rule improvement,
never model training. Deep dive: [docs/gate-layers.md](docs/gate-layers.md).

### Graduation

![Graduation — from Jev's decisions to your own model](docs/diagrams/graduation-lifecycle.svg)

One bootstrap phase, two independent tracks. The **router track** needs the distilled
model to match the teacher's tiers on held-out data before `graduate --write` is
allowed to touch your policy; the **gate track** needs the semantic layer to hit
measured recall/precision numbers in shadow. Both gates print their numbers and
refuse to write on failure. Deep dive: [docs/graduation.md](docs/graduation.md).

There is a third, local-flavoured variant of the router track: instead of
distilling a student from your own log, start from a pretrained open-source
System-1 model and *calibrate* it on your traffic. That is what
[laya-route](https://github.com/mcftira/laya-route) ships -- the same seam, a
one-line `backend.name: laya` swap, fully air-gapped, $0 per decision, and a
measured Jev-vs-Laya comparison table (the gap is a number, not a feeling).

### Shadow mode

![Shadow mode — earning the cutover](docs/diagrams/shadow-mode.svg)

The distilled model serves; the teacher watches from a background task. Disagreements
are counted, not hidden, and a timeout is recorded as "unknown", never as agreement.
Deep dive: the `backend: shadow` section of [docs/policy-guide.md](docs/policy-guide.md).

### Deployment

![Deployment — DGX Spark cluster (namespace jev-route)](docs/diagrams/cluster-deployment.svg)

One boundary line: everything teal, everything else egress. After graduation the
egress arrows disappear. Manifests: [`deploy/k8s/`](deploy/k8s/).

---

## Why routing needs calibration

An argmax classifier cannot say *"I'm 62% sure this is internal"*. It says `internal` and
throws the 38% away. That is fine when the cost of being wrong is a slightly worse answer. It
is not fine when the cost of being wrong is a patient record leaving your infrastructure, or
every compliance question in your company being routed to your most expensive air-gapped GPU.

Routing on data sensitivity is a **judgement** problem, and judgement problems need a
probability, not a label. A calibrated model gives you one, and a router that reads it can do
the thing no regex and no argmax can: **escalate on uncertainty**. `sensitivity_confidence
< 0.8` becomes "treat it as one level stricter", which is a policy you can review in a pull
request.

Here is real, measured output. The left column is what a keyword/regex-style router does; the
right column is what Jev (`jev-latest`, which returns `jev-1.13.0`) actually answered:

| prompt | keyword-style read | Jev (calibrated) |
| --- | --- | --- |
| `Explain how HIPAA actually works, who it applies to` | `confidential`, `pii=0.87` | `public`, `pii=0.03`, confidence `1.00` |
| `Is a hashed email address still personal data under GDPR?` | `confidential` | `public`, `pii=0.05` |
| `Patient John Smith, SSN 666-45-1234, MRN ABC-9931, type 2 diabetes` | *(local gate blocks the cloud call entirely)* | `regulated`, p=`1.00` |
| `Charge card 4111 1111 1111 1111 … key sk-live-…` | *(local gate blocks the cloud call)* | `regulated`, p=`0.99` |
| `design a sharded write path with consensus and prove the invariant` | `hard` | `frontier`, confidence `0.89` |

<sub>All identifiers above are synthetic and drawn from published test ranges: SSN area `666`
is never issued, `4111 1111 1111 1111` is Visa's public test number, and the API key is
truncated. Confidence and probability values move by a few points between runs — these are a
calibrated model's answers, not constants — but the labels and the ordering do not. In rows 3 and 4 the Jev column is what Jev answers *when asked directly* — in a real
deployment those requests never leave the process, because the local gate matches an SSN and a
medical record number (row 3) and a Luhn-valid card plus a live-shaped credential (row 4),
sets `blocks_backend`, and resolves them locally.</sub>

The point is not that the keyword router is stupid. It is that it answers a different
question. "Does this text *mention* a regulated domain" and "does this text *contain*
regulated data" are not the same question, and conflating them costs you in both directions:
every HIPAA *question* your support team ever asks goes to an air-gapped model and burns the
savings that justified the router, while a confidently-worded prompt that happens to avoid
every keyword sails through to a cloud API.

**jev-route uses both, on purpose.** Deterministic detectors with real checksums (Luhn,
ISO 13616 mod-97, NHS mod-11, NI-number shape rules) handle the identifiers they can *verify*
— those are never bypassed and never model-decided. A calibrated model handles everything
that requires judgement. Regex for what can be proven, calibration for what must be weighed.

This integration point is not a hack, and LiteLLM says so itself. LiteLLM 1.101 ships a native
complexity router, and the source description of its `classifier_fallback: default_model`
option reads, verbatim:

> `'default_model'` skips scoring and routes to default_model, which is what a classifier on
> some other taxonomy wants: **a prompt that grades data sensitivity has no use for a
> complexity score**, and scoring one produces a tier unrelated to what the operator
> configured.
>
> — `litellm/router_strategy/complexity_router/config.py`

jev-route plugs into exactly that seam as a `ClassifierPlugin`.

---

## Bootstrap on Jev

Day one. Zero training data. One API key.

```bash
export TYPESAFE_API_KEY=...        # never committed; .env is gitignored
```

```yaml
# config: policies/production.yaml
backend:
  name: jev
  api_key_env: TYPESAFE_API_KEY
  model: jev-latest
  timeout_seconds: 5.0
  max_retries: 2
  include_domain: true
```

One HTTP call to `POST https://api.typesafe.ai/v1/systemone` asks four typed questions in
parallel and gets back four **distributions**, not four labels:

| question | type | ladder |
| --- | --- | --- |
| `complexity` | choice | `trivial` → `standard` → `hard` → `frontier` |
| `sensitivity` | choice | `public` → `internal` → `confidential` → `regulated` |
| `pii_present` | noul | probability of *yes*, `0.0`–`1.0` |
| `domain` | choice | `code` / `writing` / `analysis` / `chat` / `data-extraction` |

Measured routing-decision latency for the full four-question call, from the shipped 223-prompt
eval against the real Jev backend (`jev-1.13.0`; `evals/results/summary.json`, 186 backend
calls): **p50 318 ms, p95 725 ms, p99 791 ms** — fastest 255 ms, slowest 840 ms. The other 37
prompts were gate-blocked and never made a call at all; those resolve in well under a
millisecond (p50 0.43 ms). A third of a second is a real tax on the request path, which is why
a judgement cache sits in front of the backend and why identical redacted excerpts skip the
call entirely. It is also why the cache stores *judgements* rather than *decisions* — see
[docs/architecture.md](docs/architecture.md).

Two things the bootstrap phase does that a naive integration would not:

* **The excerpt is untrusted input.** Every question carries an explicit trust-boundary
  clause telling the model to ignore any instruction inside the prompt about which tier to
  choose. Model compliance is not a security control, which is why the local hard gate — not
  the prompt — is the thing that actually cannot be talked out of anything. The eval dataset
  ships 12 prompt-injection rows to keep this honest.
* **Outages degrade, they do not raise.** Timeouts, rate limits and a tripped circuit breaker
  come back as maximum-uncertainty answers, and the policy engine turns maximum uncertainty
  into the safest tier. Fail-closed is the default.

Cost per decision is fractions of a cent. `<TODO: measured cost per 1k decisions from
TypeSafe billing — fill from a real invoice, not an estimate>`

---

## Your traffic becomes your model

Every decision is logged — including the ones you never look at — as one JSON line in
`decision-log/decisions.jsonl`. The log is schema-versioned, append-only, and holds the **full
soft distribution behind every answer**, because distilling from argmax labels would throw
away exactly the calibration you paid Jev for.

This is a real record, produced by the quickstart below — the third `jev-route demo` prompt,
copied out of `decision-log/decisions.jsonl` with the `features` block trimmed from 24
fields to six and the key order tidied. Nothing else is edited:

```json
{
  "kind": "jev_route.decision", "schema_version": "1",
  "request_id": "38dab0fe6190472a888de54cd8e97c4d",
  "timestamp": "2026-09-19T12:00:45.177+00:00",
  "excerpt_hash": "ae444bf5d9152b6a",
  "excerpt": null,
  "requested_model": null, "metadata": {}, "questions_sent": {}, "shadow": null,
  "features": { "char_len": 124, "word_count": 20, "code_blocks": 0, "lang": "en", "n_gate_findings": 0, "gate_detectors": {} },
  "decision": {
    "answers": {
      "complexity": {
        "choice": "hard", "confidence": 0.912986, "confidence_reported": false,
        "probabilities": { "frontier": 0.043482, "hard": 0.956468, "standard": 5.1e-05, "trivial": 0.0 }
      },
      "sensitivity": {
        "choice": "internal", "confidence": 0.585179, "confidence_reported": false,
        "probabilities": { "confidential": 0.011753, "internal": 0.201073, "public": 0.786252, "regulated": 0.000922 }
      },
      "domain": {
        "choice": "writing", "confidence": 0.658225, "confidence_reported": false,
        "probabilities": { "analysis": 0.068355, "chat": 0.068355, "code": 0.068355, "data-extraction": 0.068355, "writing": 0.72658 }
      },
      "pii": { "noul": 0.011407 }
    },
    "gate": { "fired": false, "findings": [], "sensitivity_floor": null, "pii_floor": null,
               "force_local": false, "blocks_backend": false, "advisory_topics": [] },
    "tier": "strong", "model": "qwen3.8-max",
    "rule_id": "complexity.hard", "reason": "task needs a strong model",
    "effective_complexity": "hard", "effective_sensitivity": "internal",
    "escalated": [ "sensitivity public->internal (confidence 0.59 < 0.8)" ],
    "backend": "mock", "backend_model_version": "mock-1.0.0",
    "degraded": false, "degrade_reason": null, "cached": false, "latency_ms": 0.218
  },
  "backend_latency_ms": 0.0, "total_latency_ms": 0.218
}
```

Read the `escalated` line: the backend's distribution peaked at `public` (0.786), its confidence
was `0.59` against a `0.8` floor, so the router moved it one level **stricter** and recorded that
it did. Uncertainty is a routable fact, and the audit trail shows it being used.

Note what `choice` means in a logged record. It is the **effective** value — after the gate merge
and the uncertainty bump — which is why `answers.sensitivity.choice` reads `internal` while
`answers.sensitivity.probabilities` still peaks at `public`. The raw distribution is stored
alongside it, unchanged, and that is what `jev-route export` trains on: `_extract_targets()` in
`src/jev_route/distill/export.py` reads `probabilities`, never `choice`, so a student never
learns the policy's adjustments as if they were the teacher's belief.

Three properties make this a dataset rather than a log file:

1. **Soft targets.** Full distributions per head, plus the `pii` noul probability. A student
   trained on these inherits the teacher's uncertainty instead of its overconfidence.
2. **Deterministic features alongside the text.** 24 features per request — counts, ratios and
   booleans, plus one coarse language tag and a gate-detector histogram (length, digit ratio,
   code fences, stack-trace and diff shape, which detectors fired). None of them is prompt
   text. These are computed locally and retained **even when you choose never to store
   prompt text**, which means an operator with the strictest retention policy can still train
   a local model — a feature-based one.
3. **Gate-blocked requests are still labelled.** When the local gate refuses to send an
   excerpt anywhere, the record still carries a `regulated` sensitivity at p=1.00 with the
   detector names that fired. For some detectors that floor rests on a checksum (Luhn, IBAN
   mod-97, NHS mod-11); for the rest on shape alone — either way it was decided by a regex you
   can read, not by a model, so those rows are the highest-confidence labels in the set, and
   they are the rows a cloud teacher could never have labelled for you.

`logging.excerpt_mode` decides how much text you keep. `hash` (the default) keeps no prompt
text at all. See [docs/privacy.md](docs/privacy.md) — that document is the one to read before
you turn it up.

---

## Go air-gapped

The bootstrap phase ends. That is the whole point. The CLI names the lifecycle:

```text
  run on jev  ──►  log fills up  ──►  export  ──►  train  ──►  evaluate  ──►  package
                                                                                 │
     backend.name: shadow { primary: distilled, shadow: jev }  ◄──  graduate  ◄────┘
```

```bash
jev-route log-stats                                    # what have I accumulated?
jev-route export   --out data/ds                       # decision log -> training set
jev-route train    --data data/ds --out artifacts/v1   # soft-target KL distillation
jev-route evaluate --artifact artifacts/v1             # accuracy, ECE, agreement, latency
jev-route package  --model artifacts/v1 --out artifacts/v1-packaged
jev-route graduate --artifact artifacts/v1-packaged --policy policies/default.yaml
```

**Export.** The log becomes a dataset, in one of two modes — and which one you can have is
decided by the `logging.excerpt_mode` you *ran* with, weeks earlier. `text` mode needs the
redacted excerpt, so it needs `excerpt_mode: redacted`; `features` mode needs only the
deterministic feature vector, so it works even under the default `hash` mode where no prompt
text was ever retained. Gate-blocked requests are in neither text export, because they were
never written as text. `--holdout 0.2` splits off the held-out set. The split is a hash of `request_id`, not a
shuffled index, so a row never changes side as the log grows — and `--seed` does not move
rows between sides. It is recorded in the stats sidecar as provenance and nothing more.

**Train.** Soft-target distillation: KL divergence against the teacher's distributions at
temperature `T` (`--temperature 2.0`), over three softmax heads (`complexity`, `sensitivity`,
`domain`) plus one Bernoulli head (`pii`). The default trainer is numpy/scikit-learn and runs
on a CPU. `--trainer torch` buys autodiff over the same linear student and copies the weights back
into numpy arrays, so a torch-trained model still serves without torch installed. There is no
encoder path: `train.py` rejects an `encoder` config outright rather than half-supporting a
transformer student this artifact format could not store.

**Evaluate.** Per-class accuracy and macro-F1 per head, **ECE per head** — a distilled router
that is accurate but miscalibrated has lost the only thing worth distilling — latency
p50/p95/p99, and two agreement numbers against the teacher:

* *argmax agreement* — do the two pick the same label?
* *tier agreement* — do the two, run through **the same policy**, pick the same **tier**?

Tier agreement is the number that matters — the more decision-relevant of the two: it ignores
label errors the policy never reads and counts the ones that change where a request runs. Disagreeing
about `domain` is harmless: no policy worth running routes on it alone. Flipping `sensitivity`
across the `confidential` boundary is not harmless at all — it is the difference between your
self-hosted GPU and a third-party API. Comparing tiers measures the disagreements you actually
care about and ignores the ones you do not.

**Package.** A self-describing artifact directory: `artifact.json`, one `.npy` per weight array
(`W.npy`, `b.npy`), `vectorizer.json`, and after packaging `metrics.json` and `dataset.json` —
with a SHA-256 checksum per file, and **no pickle anywhere**. Arrays are saved and loaded with
`allow_pickle=False`, because a pickle in an artifact directory is arbitrary code execution the
moment somebody loads a file they downloaded. They are `.npy` rather than `.npz` on purpose: an
npz is a zip, zip entries carry timestamps, and a timestamp is the end of byte-reproducible
retraining. On load, the head label sets are validated against `jev_route.schema` and the artifact
schema version against the set this build understands, so an artifact trained against a different
ladder raises instead of quietly mis-routing for a month.

**Graduate.** One command compares your local model to the teacher on held-out decisions
against four thresholds, and only performs the swap if all four pass:

```bash
jev-route graduate --artifact artifacts/v1 --policy policies/default.yaml --write
```

| threshold | default | meaning |
| --- | --- | --- |
| `--min-tier-agreement` | `0.95` | student and teacher, through the same policy, land on the same tier |
| `--max-ece` | `0.10` | expected calibration error per head |
| `--max-latency-ms` | `50.0` | estimated added latency of the student on the request path |
| `--min-samples` | `500` | enough held-out decisions to believe any of the above |

`<TODO: a real graduate report pasted here — run the pipeline on an actual accumulated
decision log. Every number must come from the command's own output, never an estimate.>`

The swap is a **config change, not a code change**: it writes `backend.name: shadow` with
`primary: distilled` and `shadow: jev`, so your local model serves traffic while Jev keeps
running alongside it and every disagreement is logged. When the disagreement rate stays where
you want it, you drop the shadow and the egress stops permanently. It writes a **new** policy
file by default and backs up the original — `--in-place` is opt-in, because a graduation that
silently rewrites your production policy is not a graduation.

Full pipeline reference: [docs/graduation.md](docs/graduation.md).

---

## Quickstart

The whole system runs end to end with the deterministic `MockBackend` — **no API key and no
network**. That is not a demo mode bolted on: it is the same router, the same local hard gate,
the same policy engine and the same decision log, with a cheaper and coarser judgement behind
the `DecisionBackend` interface.

```bash
git clone https://github.com/mcftira/jev-route.git
cd jev-route
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

jev-route doctor --offline     # can this machine run it?
jev-route demo                 # route seven canned prompts; no API key
```

Real output from `jev-route demo`, on a machine with no `TYPESAFE_API_KEY` set:

```text
policy : policies/default.yaml
backend: mock
tiers  : {"local": ["qwen38"], "cheap": ["qwen3.8-flash"], "strong": ["qwen3.8-max"]}
note   : MockBackend is deliberately underconfident, so you will see more
         [escalated] lines than a calibrated backend produces. Escalating on
         uncertainty is the fail-safe; run `--backend jev` to see it calibrated.

greeting [escalated]
   -> tier=cheap model=qwen3.8-flash rule=default 1ms
   complexity=trivial sensitivity=internal pii=0.01

routine single-step coding task [escalated]
   -> tier=cheap model=qwen3.8-flash rule=default 0ms
   complexity=standard sensitivity=internal pii=0.02

multi-step system design with a proof obligation [escalated]
   -> tier=strong model=qwen3.8-max rule=complexity.hard 0ms
   complexity=hard sensitivity=internal pii=0.01

regulated DATA present -> the local gate blocks the cloud call entirely [gate]
   -> tier=local model=qwen38 rule=gate.force-local 0ms
   complexity=hard sensitivity=regulated pii=1.00

regulated TOPIC, no data -> the case keyword routers get wrong [gate]
   -> tier=cheap model=qwen3.8-flash rule=default 0ms
   complexity=standard sensitivity=public pii=0.01

credential material in the prompt [gate]
   -> tier=local model=qwen38 rule=gate.force-local 0ms
   complexity=hard sensitivity=confidential pii=1.00

compliance question about an ordinary product decision [gate] [escalated]
   -> tier=strong model=qwen3.8-max rule=complexity.hard 0ms
   complexity=hard sensitivity=public pii=0.01
```

Rows 4 and 6 are the architecture working: the gate matched an SSN and a medical record
number (row 4) and a Luhn-valid card plus a live-shaped credential (row 6), set
`blocks_backend`, and **nothing was sent anywhere** — the `regulated` / `confidential` label
and `pii=1.00` came from a deterministic detector, not a model. Row 5 is the argument from the
top of this README: a regulated *topic* with no regulated *data* stays on the cheap tier.

And `jev-route doctor --offline`:

```text
policy file      : policies/default.yaml
policy backend   : mock
tiers            : {"local": ["qwen38"], "cheap": ["qwen3.8-flash"], "strong": ["qwen3.8-max"]}
failure mode     : fail_closed -> local
excerpt mode     : hash
log path         : ./decision-log/decisions.jsonl

  [FAIL] TYPESAFE_API_KEY set                        required for backend.name: jev
  [ok ] gate detects a card + email                 2 findings
  [ok ] gate treats a HIPAA *question* as advisory  topics=['kw_health_regulation']
  [ok ] MockBackend usable offline

3/4 checks passed
```

The one `[FAIL]` is the point: no key, and the router still runs.

Every demo decision was also logged. Route one more prompt and read the log back:

```bash
jev-route route "Is a hashed email address still personal data under GDPR?"
jev-route log-stats
```

```text
routed: 'Is a hashed email address still personal data under GDPR?'
  tier      : local
  model     : qwen38
  rule      : data.sensitive -- sensitive or personal data must not leave the infrastructure
  complexity: standard  (confidence 0.78)
  sensitivity: confidential (confidence 0.70)
  pii       : 0.083
  domain    : code
  backend   : mock (mock-1.0.0)
  latency   : 1.2 ms
  escalated : sensitivity internal->confidential (confidence 0.70 < 0.8)
```

That `escalated` line is the calibration paying rent: the backend's argmax said `internal`, its
confidence was `0.70`, the policy's floor is `0.8`, so the router moved it one level **stricter**
and wrote down that it did.

```text
decision records : 8
schema versions  : {'1': 8}
teacher versions : {'mock-1.0.0': 6, 'gate-1.0.0': 2}
tiers            : {'cheap': 3, 'local': 3, 'strong': 2}
backends         : {'mock': 6, 'gate': 2}
rules fired      : {'default': 3, 'complexity.hard': 2, 'gate.force-local': 2, 'data.sensitive': 1}
complexity       : {'hard': 4, 'standard': 3, 'trivial': 1}
sensitivity      : {'internal': 3, 'public': 2, 'confidential': 2, 'regulated': 1}
escalated        : 5 (62.5%)
degraded         : 0 (0.0%)
gate forced local: 2 (25.0%)
gate blocked cloud: 2 (25.0%)
latency ms p50/p95/p99: {'p50': 0.2, 'p95': 1.2, 'p99': 1.2}

No record carries excerpt text (logging.excerpt_mode is 'hash' or 'none').
  -> features-mode distillation only. To enable text mode, set
     logging.excerpt_mode: redacted and collect more traffic.
     Note: gate-blocked requests are never stored as text, by design.
  -> usable training rows (excluding degraded): 8
```

<sub>The latency percentiles move by a tenth of a millisecond between runs. Everything above
them is a count, and counts do not move.</sub>

That last block is the graduation contract stated by the tool itself: with the default
`excerpt_mode: hash` you can still distill, from features. Set it to `redacted` before you
collect the traffic if you want the text path. See [docs/privacy.md](docs/privacy.md).

The same thing from Python — this is what the LiteLLM integrations call:

```python
"""Route five prompts. No API key, no network: policies/default.yaml names `mock`."""

import asyncio

from jev_route import Router

PROMPTS = [
    "hi, thanks for the fix!",
    "Summarize this customer support thread and draft a reply to the customer",
    "Explain how HIPAA actually works and who it applies to",
    "Patient John Smith, SSN 666-45-1234, MRN ABC-9931, type 2 diabetes",
    "Design a sharded write path with consensus and prove the invariant holds",
]


async def main() -> None:
    router = Router.from_policy_file("policies/default.yaml")
    for prompt in PROMPTS:
        d = await router.route_text(prompt)
        print(
            f"{d.tier:<6} {d.model:<14} "
            f"sens={d.effective_sensitivity:<12} cmplx={d.effective_complexity:<9} "
            f"backend={d.backend:<10} | {d.reason}"
        )
    stats = router.stats()
    print()
    print(f"backend calls: {stats['cache']['misses']}  cache hits: {stats['cache']['hits']}")
    print(f"logged: {stats['log']['written']} records to {stats['log']['path']}")
    await router.aclose()


asyncio.run(main())
```

```text
cheap  qwen3.8-flash  sens=internal     cmplx=trivial   backend=mock       | routine task, no sensitivity signal
strong qwen3.8-max    sens=internal     cmplx=hard      backend=mock       | task needs a strong model
cheap  qwen3.8-flash  sens=public       cmplx=standard  backend=mock       | routine task, no sensitivity signal
local  qwen38         sens=regulated    cmplx=hard      backend=gate       | local hard gate matched a structured identifier or credential
strong qwen3.8-max    sens=internal     cmplx=hard      backend=mock       | task needs a strong model

backend calls: 4  cache hits: 0
logged: 5 records to decision-log/decisions.jsonl
```

That output is byte-stable across runs — `MockBackend` has no randomness, no clock and no
counter. And `backend calls: 4` for five prompts is the gate doing its job: one request never
reached a backend, but all five were logged.

One honest caveat about the mock, and the `route` call above is the example. It is a plumbing
fixture, not a cheap substitute for Jev: it is deliberately underconfident, so you will see
escalations everywhere, and it is coarser than a calibrated model. On `Is a hashed email address
still personal data under GDPR?` the mock lands on `confidential` and routes `local`; Jev answers
`public`, `pii=0.05` (the table above). The escalation machinery worked correctly on a wrong
input — which is the honest failure mode, and the reason the mock exists to exercise plumbing
(policy thresholds, confidence floors, gate merging, logging, shadow mode, distillation) for
free and offline rather than to be accurate.

Now point the same router at Jev. One key changes:

```bash
export TYPESAFE_API_KEY=...                            # never committed; .env is gitignored
jev-route route --backend jev "Explain how HIPAA actually works, who it applies to"
```

```yaml
# config: or permanently, in policies/default.yaml
backend:
  name: jev
  api_key_env: TYPESAFE_API_KEY
```

---

## The local hard gate


The privacy architecture in one paragraph: **a deterministic, local, never-bypassed gate runs
before any model is consulted, and it can only make a decision stricter.**

It is the first thing that runs on every request, in-process, with no network. It is never
model-decided, and no policy setting can stop the scan — individual detectors can be silenced
(`gate.disabled_detectors`, for the shop with legitimate high-volume phone numbers in prompts),
but `HardGate.scan()` always runs and always returns a verdict. There is no configuration in
which unscanned text is routed. Silencing *every* detector is allowed and leaves a gate that
answers `clean` to everything; that is an operator control, not a safety property, and it is why
the checklist in [docs/privacy.md](docs/privacy.md) asks you to justify each name in the list.

What it does:

| behaviour | meaning |
| --- | --- |
| **Sets a floor** | `sensitivity_floor` and `pii_floor` are merged over the backend's answer with `max`, never `min`. No backend can talk its way out of a gate finding. |
| **Blocks the cloud** | For credential material and regulated identifiers — Luhn-valid cards, SSNs, IBANs, NHS numbers, NI numbers, private key blocks, provider API keys, inline `password=`/`api_key=` values, basic-auth URLs, medical record numbers — `blocks_backend` is true and the excerpt never reaches a remote API, **not even redacted**. |
| **Forces local** | `force_local` routes to your self-hosted tier. `blocks_backend` implies it: if we refuse to show the text to a remote classifier we are certainly not sending it to a remote model. |
| **Redacts** | Everything else it matches (emails, phones, dates of birth, street addresses, named individuals) is replaced with a typed placeholder — `[email_address]`, `[phone_number]` — and the *redacted* excerpt may still be classified for complexity. |
| **Advises, without flooring** | Topic keywords (`HIPAA`, `PCI-DSS`, `attorney-client`, `trade secret`, `zero-day`, …) are **hints, not verdicts**. They are forwarded to the backend as `local_gate_topic_hints` and recorded in the log, but they never set a floor and never force a tier on their own. |
| **Retains nothing** | A `GateFinding` stores a SHA-256 of the matched span, not the span. The log can prove a card number was present without keeping the card number. |
| **Outranks `fail_open`** | If the backend is down and you configured `fail_open`, a gate floor of `confidential` or `regulated` (or any `force_local`) still wins. A config knob is not allowed to become a data-egress path. |

That fifth row is the design decision the whole project rests on. Redacting "HIPAA" out of
"explain how HIPAA works" would leave a classifier with nothing to read, and flooring every
compliance question to `regulated` would route your entire legal and support organisation to
an air-gapped GPU. So: **advisory for topics, hard for identifiers, checksums where checksums
exist.** A gate that cries wolf gets switched off, which is worse than not having one.

Full disclosure of what leaves the process, and when: [docs/privacy.md](docs/privacy.md). Read
it before you deploy. During the bootstrap phase, redacted excerpts do go to TypeSafe's
cloud; for credential material and regulated identifiers they do not; and the phase is
designed to end.

---

## How the gate learns without ever seeing your secrets

### The bootstrap paradox

The gate decides whether a prompt may be shown to a cloud model. It cannot ask a cloud model
whether the prompt may be shown to a cloud model — by the time the answer comes back, the
question has already been sent. Every design decision below follows from that one sentence.

It is also why the deterministic regex layer is not a legacy leftover waiting to be replaced by
something smarter. A router whose "is this safe to send?" check is itself a remote call does not
have a safety check; it has a race. Whatever answers that question has to be local, has to be
cheap enough to run on every request, and has to be decided before a socket is opened. In
`src/jev_route/router.py` it is step 2 of the chain: `gate.scan()` runs over the **raw** excerpt,
and a blocking verdict sets `skip_backend` before any backend is constructed. Two tests in
`tests/test_invariants.py` pin the ordering (`test_router_calls_gate_scan_before_backend_decide`)
and pin that there is no path around it (`test_no_router_path_can_skip_the_gate`).

### Two layers

**Layer 1 — the deterministic floor. Shipped, and permanent.**

`DEFAULT_DETECTORS` in `src/jev_route/gate.py` is 22 detectors: 15 hard, 7 advisory.

* Hard detectors answer *"is sensitive data present"*. They carry real checksums where a
  checksum exists — Luhn for card numbers, ISO 13616 mod-97 for IBANs, mod-11 for NHS numbers,
  the prefix/suffix exclusion rules for UK National Insurance numbers — and they set
  `sensitivity_floor` and `pii_floor = 1.0`. Ten of them also set `blocks_backend`, which means
  the excerpt never reaches a remote API, **not even redacted**.
* Advisory detectors answer *"is a sensitive topic mentioned"*. Seven keyword lists — HIPAA and
  PHI vocabulary, attorney-client privilege, PCI-DSS and GLBA, minors' records, trade secrets,
  vulnerability disclosure, internal-only markings. They are forwarded to the backend as
  `local_gate_topic_hints` and recorded in the log. **They never set a floor and never force a
  tier.** `Detector.__post_init__` raises if an advisory detector is built with `force_local` or
  `blocks_backend`, and `test_no_advisory_detector_can_set_a_floor_in_a_verdict` pins the runtime
  behaviour. Collapsing the two classes is how regex guardrails end up air-gapping every
  compliance question in the company.
* The scan cannot be stopped; the rules can be silenced. `gate.disabled_detectors` exists for the
  shop with legitimate high-volume phone numbers in its prompts, and `HardGate.scan()` runs and
  returns a verdict whatever that list contains. Be precise about the limit of that guarantee:
  `test_gate_config_cannot_disable_the_gate_itself` puts all 22 names in the list and asserts the
  gate still ran — *and* that it answered `clean`. A fully silenced gate is a no-op, and the test
  pins that as intended rather than forbidding it. Every name you add to the list is a detector you
  have chosen not to run, so `docs/privacy.md` asks you to justify each one in the policy file.
* A `GateFinding` keeps a SHA-256 of the matched span truncated to 16 hex chars, not the span.

**Layer 2 — a local semantic layer. `src/jev_route/gate_semantic.py`, shadow by default.**

Regex finds identifiers. It cannot find *context*, and context is where most real sensitivity
lives. Here is the gap, measured on the eval set that ships in this repo:

```bash
python - <<'PY'
import json
from jev_route.gate import DEFAULT_DETECTORS, default_gate

advisory = {d.name for d in DEFAULT_DETECTORS if d.advisory}
rows = [json.loads(line) for line in open("evals/data/labeled_prompts.jsonl")]
sensitive = [r for r in rows if r["labels"]["sensitivity"] in ("confidential", "regulated")]

hard = only_advisory = silent = 0
for row in sensitive:
    fired = {f.detector for f in default_gate.scan(row["text"]).findings}
    if fired - advisory:
        hard += 1
    elif fired:
        only_advisory += 1
    else:
        silent += 1

print(f"{len(rows)} prompts, {len(sensitive)} labelled confidential/regulated")
print(f"  hard detector fired : {hard}")
print(f"  advisory hint only  : {only_advisory}   (never sets a floor, by design)")
print(f"  gate saw nothing    : {silent}")
PY
```

```text
223 prompts, 87 labelled confidential/regulated
  hard detector fired : 37
  advisory hint only  : 9   (never sets a floor, by design)
  gate saw nothing    : 41
```

Those 50 rows are not typos in the dataset. They are prompts like these two, quoted verbatim
from `evals/data/labeled_prompts.jsonl`:

> `Performance improvement plan for Ethan Cole, employee ID 4471, manager Nadia Petrov. His last
> three reviews are attached and one includes a documented absence for treatment.` — labelled
> `confidential`, `pii: true`. The gate returns **no findings at all**.

> `Privileged and confidential - attorney-client. Client: Harrow & Vane Ltd. Matter 2026-0114.
> Opposing counsel's latest offer is attached.` — labelled `regulated`. The gate fires two
> **advisory** detectors and sets no floor.

An HR disciplinary narrative (`pii-hr-001`), a privileged legal summary (`pii-legal-001`), a
compensation calibration round (`hr-perf-001`), a trade secret written out as an arithmetic
formula (`trade-secret-001`), a client account review naming two people and an account number
(`pii-fin-001`), an incident record with a 1.8 GB exfiltration in it (`long-incident-001`): not
one of them contains an identifier any detector in `gate.py` can see. That is the layer-2
problem.

The eval harness already scores it. Its `gate_only` negative control — route on the gate and
nothing else — puts 37 of those 87 rows on the `local` tier and leaks the other 50, which is
`unsafe_errors: 50` in `evals/results/summary.json`. The full router, with a calibrated backend
behind the gate, leaks 11. The 39-row difference is what a semantic layer would have to earn
back *locally*, without a cloud call, once the bootstrap phase ends.

`src/jev_route/gate_semantic.py` is that layer. It is a sparse logistic regression over a stored
vocabulary, stdlib only — no numpy, no torch, no scikit-learn — because the router runs inline on
every request in containers that do not have the training extras installed, and a gate that only
works when scikit-learn is present is a gate that is off in production. Its artifact is one
`semantic.json`, readable with `json.load`, so a reviewer can see every token the model keys on
without running anything.

Four rules govern it, and each is enforced in code rather than in a review checklist.

* **It may only make routing stricter.** It can raise the sensitivity floor and force the local
  tier; it cannot lower either. `resolve_floor()` is a maximum over the layers' floors, so a
  semantic layer that answers `public` for a prompt containing a Luhn-valid card contributes
  `None`, and the deterministic `regulated` survives. That is not a convention the router happens
  to follow — it is the only arithmetic available. `ASSERTABLE_LEVELS` is
  `("confidential", "regulated")`, and a config asking layer 2 to assert `public` is refused at
  load time, because that would be a request to loosen.
* **Shadow is the default, and shadow never blocks.** `gate.semantic.mode` ships as `shadow`: the
  layer runs, scores, and records what it would have done, and its assessment is invisible to the
  policy engine. A shadow that could change routing would stop being a measurement.
* **Promotion is measured, not configured.** `mode: enforce` is necessary and not sufficient.
  Enforce also requires an artifact whose *held-out* metrics satisfy `EnforceCriteria`.
  `SemanticLayer` runs `check_promotion()` over them at construction and `_refuse_enforce()`
  raises with the measured numbers in the message when they are not met. It does not quietly fall
  back to shadow, because a gate that looks enforced and is not, is worse than one that was never
  turned on. The same refusal fires when no artifact is configured at all: *"refusing to start
  rather than running an unmeasured gate."*
* **Its provenance is checked on load.** See below.

The seven criteria, all of them in `policies/default.yaml` under `gate.semantic.enforce_requires`:

| criterion | default | what it bounds |
| --- | --- | --- |
| `min_recall` | `0.99` | held-out positives it must catch. The one that matters: a miss here is content that left the building |
| `max_false_positive_rate` | `0.02` | real negatives it would have air-gapped |
| `max_disagreement_rate` | `0.05` | how often the two layers contradict each other on **live** traffic, read from the shadow log by `measure_shadow_log()` |
| `max_semantic_miss_rate` | `0.02` | of the examples layer 1 *also* catches, how many layer 2 misses |
| `min_positive_examples` | `200` | held-out positives behind the recall figure |
| `min_negative_examples` | `500` | held-out negatives behind the false-positive rate |
| `min_shadow_examples` | `1000` | real requests observed in shadow before promotion is even considered |

`min_shadow_examples` is the criterion no offline number can substitute for: a model that scores
0.99 on a held-out set of sentences somebody wrote has still never seen your traffic, and the
disagreement rate you are about to bound is a property of *that* traffic. `min_recall` has a floor
of its own — `EnforceCriteria.MIN_ALLOWED_RECALL = 0.90` — so an operator can raise the bar to
0.995 but cannot configure it down to 0.6.

Under the shipped default (`mode: shadow`, no `artifact:`) the layer constructs, records an
`unavailable_reason`, and scores nothing. Shadow is inert until there is a model to shadow with.

Separately, `src/jev_route/backends/shadow.py` does the same job one level up for the *routing*
model: `ShadowBackend` serves the primary's answer unchanged while a second backend runs purely
for telemetry, in the background by default so it adds no latency, sampling deterministically per
request. A shadow that times out or throws records `agrees: None` rather than being counted as
agreement or disagreement, because either would corrupt `stats()["disagreement_rate"]` — the one
number the exercise exists to produce. 42 tests in `tests/distill/test_shadow.py`.

### The training set, and why it is safe

Three kinds of row, and only the third one is yours.

**1. Synthetic positives that are provably fake by reservation, not by luck.** Nobody has to
review the file and decide an identifier looks made up. The ranges are reserved by the bodies
that issue them, and `evals/validate_dataset.py` rejects anything outside them — the check the
CI workflow runs as its `eval dataset guard` job.

| shape | allowed values | where it is enforced |
| --- | --- | --- |
| US SSN | area `000`, `666`, `900`–`999` — the SSA reserves these and will never issue them | `validate_dataset.py:353` |
| payment card | published provider test numbers only: `4111 1111 1111 1111`, `5555 5555 5555 4444`, `4242 4242 4242 4242`, `3782 822463 10005`, `6011 1111 1111 1117`, … | `ALLOWED_CARD_NUMBERS` |
| email | RFC 2606 reserved domains, or an invented organisation domain from `FICTIONAL_ORG_DOMAINS` | `validate_dataset.py:344` |
| phone | the reserved fictional `555-01xx` block, in any national format | `validate_dataset.py:360` |
| IBAN | documented test IBANs only | `ALLOWED_IBANS` |
| API key / password | `ALLOWED_SECRET_TOKENS` only (`AKIAIOSFODNN7EXAMPLE`, `sk-test-000…0`); any other `sk-`/`AKIA`/`ghp_`/JWT-shaped token fails | `validate_dataset.py` |

Reserved does not mean *harmless to train on*. The gate fires on `666-45-6789` on purpose —
`test_ssn_shape_is_caught_including_reserved_ranges` pins it — because the only safe reading of a
reserved-range SSN in a production prompt is that somebody pasted a fixture that came out of real
data. Provably fake is a property of the *dataset*, not of the *detector*.

The eval set is checked after the fact; the **generator** is checked at construction time.
`src/jev_route/distill/synthetic_pii.py` builds the positives layer 2 trains on, and every value
it emits carries the reservation it relies on. `SyntheticValue.__post_init__` calls
`verify_fakeness()`, which raises when the claim does not hold — so the generator cannot emit a
real-shaped value even by accident, because the object it would return cannot be built. The
reservations it may claim:

| reservation | what makes the value fake |
| --- | --- |
| `reserved-range` | a numbering authority published a block it will never assign: SSN areas `000`/`666`/`900`–`999` (SSA), NANP central-office code `555` with station numbers `0100`–`0199` (NANPA), RFC 2606 domains, ISO 3166-1 user-assigned country codes used as IBAN prefixes |
| `published-test-value` | a vendor publishes this exact value as its test value: payment PANs from Stripe's and Authorize.Net's public test-card documentation, the AWS documentation access key id |
| `never-issued` | HMRC publishes NI-number prefixes that are never allocated as a pair: `BG`, `GB`, `KN`, `NK`, `NT`, `TN`, `ZZ` |
| `checksum-broken` | the right shape with a deliberately wrong check digit, so no issuer ever assigned it |
| `self-labelled` | constructed rather than captured, carrying a literal `SYNTHETIC` token so an operator who finds one in a log can tell it is ours |
| `placeholder-roster` | a closed list of documentation placeholder names |
| `impossible-value` | the value cannot exist: a date that is not on the calendar |

Two of those are a deliberate downgrade, and the generator says so out loud. A Luhn-*invalid* card
is provably not a real card but will not fire layer 1, which validates PANs with Luhn. An NHS
number cannot be made provably fake with a *valid* mod-11 check digit, so only checksum-broken ones
are emitted and `uk_nhs_number` never fires on them. Every generated value therefore carries an
`expected_gate` field — what the real `HardGate` does with it — so "which layer catches this" is a
claim checked against the gate rather than trusted from a docstring.

The rule that closes the loop is enforced on the **artifact**, not on the training script.
`SemanticArtifact.load()` reads a `Provenance` object out of `semantic.json` and refuses to load
when:

* `trained_on_blocked_content` is `true`. The field is required, so an artifact that omits it is
  refused too — "we did not say we did" is not provenance.
* `positives` is not one of `synthetic`, `public`, `synthetic+public`. `production` is absent from
  `ALLOWED_POSITIVE_SOURCES` on purpose: a production positive *is* blocked content by definition,
  because it only exists as an example if some request was refused egress.
* `negatives` is not in `ALLOWED_NEGATIVE_SOURCES`, which does include `production`. Real traffic
  the gate let through is the most valuable negative set there is, and using it costs nothing: by
  definition it was already allowed to leave the building.

**2. Contextual positives that fire no hard detector at all.** The 50 rows above —
`pii-hr-001`, `hr-perf-001`, `trade-secret-001`, `sec-vuln-001`, `long-incident-001`,
`pii-chat-001` among them. These are the rows a semantic layer is for, and they are the rows a
regex layer can never be tuned to catch without firing on ordinary prose.

**3. Your own traffic that already passed the gate**, used as negatives. Under
`logging.excerpt_mode: redacted` the router keeps the *redacted* excerpt of every request it
classified, so the negatives are real, from your distribution, and already stripped of every
identifier the gate could find.

And the hard rule: **real blocked content is never training data.** Not by policy document — by
code path, and by test.

The decision log is the only input `jev-route export` reads. A gate-blocked request never has
text written into that log, whatever the operator configured:

```python
# src/jev_route/router.py, _log()
if self.excerpt_mode == "redacted" and not gate_blocked:
    excerpt = redacted_excerpt
```

`tests/test_router.py::test_gate_blocked_record_has_no_text_even_in_redacted_mode` sets
`excerpt_mode: redacted`, routes a prompt containing a card number, and asserts the card appears
nowhere in `record.to_json()` — spaced or unspaced. `tests/distill/synthlog.py:1116` asserts the
same over a whole generated log: every gate-blocked record has `excerpt is None`, every
classified record has one, and no retained excerpt can re-trip the gate.

What survives is the label, not the text: `excerpt_hash`, plus
`features.gate_detectors == {"payment_card": 1}` and `features.gate_force_local is True`. The row
is still trainable and the secret is gone. That is the same judgement that kept the excerpt out
of a cloud API, applied a second time on the way into the dataset.

One more separation, and it is physical rather than procedural. Refusals are still worth
measuring — which detectors fire, on what shape of request, how often — because that is how a rule
set gets tuned by a human reading aggregate counts. So `gate.blocked_metadata` writes them to
`decision-log/gate-blocks.jsonl`, a **sibling** of the decision log rather than a line inside it.
The payload is closed: `request_id`, `timestamp`, the detector ids that fired, the deterministic
feature vector, and the excerpt hash. No text under any configuration, no raw secret, no matched
span. It is a separate file because `jev-route export` walks `decisions*.jsonl` in the log
directory, and a separate file is a stronger guarantee than a filter somebody has to remember to
put in the exporter.

Under the shipped default (`excerpt_mode: hash`) no prompt text is retained at all, for any
request, and distillation runs on the 24-field feature vector instead.

### Why the floor is permanent

A semantic layer is a statistical model. The floor is a deterministic pattern match — a real
checksum where one exists (Luhn for card numbers, ISO 13616 mod-97 for IBANs, mod-11 for NHS
numbers, the NI-number prefix and suffix rules), and a strict shape rule where one does not. Be
precise about this, because it is the kind of thing a reviewer checks: of the 15 hard detectors,
6 carry a validator and 9 are shape-only. `us_ssn` matches a `3-2-4` digit run and nothing more.
Shape-only is still deterministic, still auditable, and still not a probability.

Where a false negative means data left the building, you do not get to retire the deterministic
layer because the model got good. The two failure modes are not the same size: a semantic layer
that misses an HR narrative costs you a routing tier, and a shadow-mode disagreement count tells
you it happened. A retired `payment_card` detector that misses a card number costs you a card
number, and nothing tells you at all.

Three more reasons, all of them load-bearing:

* **The floor is what makes the layer above it trainable.** "Blocked content is never written to
  the log" is implemented by consulting the gate verdict. Remove the deterministic layer and you
  remove the mechanism that kept your real secrets out of your training set — the semantic layer
  would then be trained on exactly the data it was designed never to see.
* **The floor is auditable in a way a weight matrix is not.** 22 regexes and 6 validators in one
  file, readable end to end in ten minutes, with a `span_hash` in the log that proves which one
  fired. A reviewer can read that. Nobody can read a logistic regression's token weights and know
  what it will do next — and layer 2's artifact is deliberately stored as plain JSON so the
  vocabulary at least is inspectable.
* **The floor produces labels, not predictions.** A Luhn-valid card is not a 0.94. `gate.py`'s
  module docstring says this outright: determinism is what makes the gate's verdicts usable as
  high-confidence labels when distilling. `pii_floor = 1.0` because a deterministic hit is a
  certainty, and 1.0 is the only value a backend answering 0.2 cannot talk its way past.

The semantic layer is an *addition* to the floor. There is no version of this project in which
it is a replacement.

### What this does not give you

Stated plainly, because a README that oversells the gate is worse than one that names the gap.

* **The semantic layer cannot be trained on your own real sensitive traffic.** That traffic is
  never written down — see the previous section — and this is the correct trade, not an
  oversight. The consequence is that its recall on *your* data is **inferred**, from synthetic
  positives and public positives plus your clean negatives. It is not measured on your actual
  secrets, because nobody measured anything on your actual secrets.
* **Nobody can hand you a recall figure for data they were never allowed to see.** Any number
  quoted for "catches sensitive prose" is a number about somebody else's corpus. The 50-of-87
  above is measured on a synthetic eval set written by this project's authors; your distribution
  is not that one.
* **So the only evidence that counts is yours.** The disagreement rate a shadow layer produces on
  your own live traffic is the real measurement, and it is the reason shadow mode exists and the
  reason promotion is gated on a measured rate rather than a benchmark. If you are not running
  shadow mode, you are not collecting the only evidence that would justify enforcing.
* **The floor has known misses too**, and they are listed in [Honest limitations](#honest-limitations):
  person names, free-text addresses, paraphrased secrets. `pii-chat-001` in the eval set names a
  person and gives a home address — `12 Fenyves utca` — plus a birthday with no year in it, and
  the gate returns nothing: `utca` is not in the `street_address` suffix list, `date_of_birth`
  needs a `dob`-style marker, and a name regex loose enough to catch `Ilona` would fire on every
  product name in existence.

---

## LiteLLM integration modes

jev-route targets LiteLLM **1.101.0**. Four ways in, from "recommended" to "you have a
reason":

| mode | seam | implementation | use it when |
| --- | --- | --- | --- |
| **1. `ClassifierPlugin`** *(recommended)* | LiteLLM's native complexity router, `classifier_type: custom` | `jev_route.integrations.litellm_classifier.JevRouteClassifier` | You run the LiteLLM proxy and want tier selection to be jev-route's job. This is the seam LiteLLM's own source says a sensitivity classifier wants. |
| **2. `RoutingPlugin`** | `litellm.Router(plugins=[...])` in the SDK | `jev_route.integrations.litellm_plugin.JevRouteRoutingPlugin` | You use the LiteLLM SDK router directly and want to *narrow* `candidate_models` before selection. |
| **3. `CustomLogger.async_pre_call_hook`** | proxy `litellm_settings.callbacks` | `jev_route.integrations.litellm_hook.JevRoutePreCallHook` | You need to rewrite `data["model"]` on the way in without owning the routing strategy — e.g. an existing proxy config you cannot restructure. |
| **4. Plain SDK** | `Router.route_messages(...)` / `route_text(...)` | `jev_route.Router` | No LiteLLM at all: your own gateway, a batch job, an eval harness, or the `jev-route` CLI. |

All three LiteLLM modes share one process-wide `Router` (one cache, one circuit breaker, one
log handle), built lazily so importing the module has no side effects, and never raising at
construction — a plugin that fails to configure itself falls back to the built-in policy on
`MockBackend` and logs loudly, because a booted proxy making coarse decisions beats a proxy
that will not start. Metadata handed to the router and the log is **allowlisted** identity keys
only (`user_id`, `team_id`, `request_id`, …), scalars only, truncated — never a payload.

Mode 1 in a proxy config. Every field name below is a real field of LiteLLM 1.101's
`ComplexityRouterConfig`, and no secret appears in the file — `os.environ/...` is resolved by
LiteLLM:

```yaml
# config: litellm proxy config.yaml  (full annotated version:
#         examples/litellm-proxy-classifier/config.yaml)
model_list:
  - model_name: qwen38                 # tier `local`: self-hosted, data never leaves
    litellm_params:
      model: openai/qwen3.8
      api_base: http://qwen.qwen.svc.cluster.local:8010/v1
      api_key: "EMPTY"                 # llama.cpp ignores it; litellm requires non-empty
  - model_name: qwen3.8-flash          # tier `cheap`
    litellm_params:
      model: openai/qwen3.8-flash
      api_base: https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
      api_key: os.environ/ALIBABA_API_KEY
  - model_name: qwen3.8-max            # tier `strong`
    litellm_params:
      model: openai/qwen3.8-max
      api_base: https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
      api_key: os.environ/ALIBABA_API_KEY

  - model_name: auto                   # clients ask for this name
    litellm_params:
      model: auto_router/complexity_router
      # Two spellings, both needed: this one makes litellm treat the target as a
      # dependency of the deployment (health checks, model graph); the one inside
      # complexity_router_config is what the complexity router actually reads.
      complexity_router_default_model: qwen3.8-flash
      complexity_router_config:
        classifier_type: custom
        # The last dotted segment must be a module attribute holding an INSTANCE
        # with an async classify(context). A sync classify is rejected at startup.
        classifier_plugin: jev_route.integrations.litellm_classifier.classifier
        classifier_plugin_timeout_ms: 3000
        default_model: qwen3.8-flash
        # jev-route's own tier set replaces SIMPLE/MEDIUM/COMPLEX/REASONING.
        # With a custom tier set, fallback_tier IS the "classifier could not
        # answer" knob -- and it points at `local`, because that is exactly the
        # situation in which data should not leave the building.
        fallback_tier: local
        tier_definitions:
          - name: local
            description: Sensitive, personal, regulated or credential-bearing requests, and anything the local hard gate matched; must stay on self-hosted models.
          - name: cheap
            description: Routine work with no sensitivity signal - summaries, rewrites, ordinary questions, everyday code.
          - name: strong
            description: Hard multi-step reasoning, proofs, architecture with real tradeoffs, debugging under incomplete information.
        tiers:                         # tier -> model_name pools from above
          local: [qwen38]
          cheap: [qwen3.8-flash]
          strong: [qwen3.8-max]

litellm_settings:
  drop_params: true
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY   # never inlined either
```

`classify(context)` returns a **tier name** — `local`, `cheap` or `strong` — or `None` to
decline. Returning a tier rather than a model is what makes this the most LiteLLM-native of
the four modes: jev-route supplies the judgement, LiteLLM keeps every consequence of it
(cooldowns, fallbacks, load balancing). `None` hands control to the fallback the *operator*
chose instead of a default the plugin invented.

Two LiteLLM constraints worth knowing before you edit that block:

* `classifier_fallback: default_model` — the knob the quote above describes — and
  `tier_definitions` are **mutually exclusive**; LiteLLM rejects the combination. With a
  custom tier set, `fallback_tier` *is* that knob, and jev-route points it at `local` so a
  failed classification lands on the tier whose data never leaves.
* A custom tier set also costs you LiteLLM's escalation, adaptive routing, session affinity,
  tier labels and complexity-router plugin pipeline. That is deliberate: jev-route does its own
  escalation in the policy engine (`on_uncertain` bumps *stricter* on low confidence —
  escalation on evidence, not on a retry counter) and its own stickiness through the decision
  cache.

Environment variables the integrations read: `JEV_ROUTE_POLICY` (policy file path),
`JEV_ROUTE_DECISION_LOG` (overrides `logging.path` — where writable volumes live differs
between a laptop and a container), `JEV_ROUTE_TIMEOUT_MS` (per-request decision budget, default
`3000`), `JEV_ROUTE_MANAGED_MODELS` (comma-separated model names the pre-call hook may rewrite,
default `*`), `JEV_ROUTE_METADATA_EXTRA_KEYS`, `JEV_ROUTE_CLASSIFIER_TIERS`.

Worked configs ship in the repo: `examples/litellm-proxy-classifier/config.yaml` (mode 1,
heavily annotated), `examples/litellm-proxy-hook/config.yaml` (mode 3) and
`examples/litellm-sdk-plugin.py` (mode 2). Mode 4 — the plain `Router` API — is what the
quickstart runs.

Without `TYPESAFE_API_KEY` the shipped policy runs on the deterministic `MockBackend`: coarse
judgements, no network, no cost, and the pipeline still exercises every stage. That is the
intended offline path, not a degraded one.

---

## Writing a policy

The policy file is the part you own. It is YAML, it is reviewed in a pull request like code,
and changing it does not require touching Python. Full reference with five complete
real-world examples: [docs/policy-guide.md](docs/policy-guide.md).

```yaml
# config: trimmed from policies/default.yaml
version: 1

tiers:
  local:  [qwen38]          # self-hosted: data never leaves the building
  cheap:  [qwen3.8-flash]
  strong: [qwen3.8-max]
tier_order: [cheap, strong]  # ascending capability; on_uncertain walks this

rules:                       # first match wins; data protection before cost
  - id: gate.force-local
    if: gate_force_local
    then: { tier: local }
    reason: local hard gate matched a structured identifier or credential

  - id: data.sensitive
    if: sensitivity in ["confidential", "regulated"] or pii_present
    then: { tier: local }
    reason: sensitive or personal data must not leave the infrastructure

  - id: complexity.frontier
    if: complexity == "frontier"
    then: { tier: strong }
    reason: task needs frontier-class reasoning

  - id: complexity.hard
    if: complexity == "hard"
    then: { tier: strong }
    reason: task needs a strong model

  - id: default
    then: { tier: cheap }
    reason: routine task, no sensitivity signal

on_uncertain:                # uncertainty-as-code: always stricter, never looser
  sensitivity_confidence_below: 0.8
  sensitivity_bump_levels: 1
  complexity_confidence_below: 0.7
  complexity_bump_levels: 1
  pii_uncertain_threshold: 0.35
  pii_uncertain_counts_as_present: true

on_backend_down: { mode: fail_closed, fail_closed_tier: local, fail_open_tier: strong }
```

Rule conditions are **parsed, not `eval`'d**: a strictly limited subset of Python's AST
(names from a fixed allow-list, constants, comparisons including `in`/`not in`, `and`/`or`/
`not`, literal collections, subscripting a simple name, `+ - *`). No calls, no attribute
access, no comprehensions, no imports. Anything else raises `PolicyError` at **load** time,
naming the offending node, rather than at request time.

---

## Honest limitations

This project is only worth reading if this section is real.

* **Routing costs latency and money.** A four-question decision measured p50 318 ms / p95
  725 ms / p99 791 ms on the shipped eval (`evals/results/summary.json`). The cache absorbs repeated traffic, gate-blocked traffic skips the backend
  entirely, and a distilled local model removes the round trip — but until then you are
  paying a per-request tax to save a per-request model. If all your traffic is one tier
  already, do not deploy a router.
* **Layer 1 of the gate is regex, and regex misses things.** It has checksums for the
  identifiers where a checksum exists, and it is deliberately narrow about person names
  (a loose name regex fires on "customer Call Transcripts", and a gate that cries wolf gets
  switched off). Names, free-text addresses, and paraphrased secrets will get past it —
  that is the gap the local semantic layer (layer 2) exists to fill, shadow by default and
  enforceable only on measured criteria. The calibrated backend is a further line, not the
  first, and no single layer is complete.
* **`excerpt_hash` is not anonymization.** It is a dedup and cache key. A short prompt from a
  known corpus is guessable by hashing candidates. We say this in the code, in
  [docs/privacy.md](docs/privacy.md), and here.
* **Redaction is best-effort, and best-effort is why `blocks_backend` exists.** For the
  dangerous class we do not rely on redaction at all. For the rest, a redacted excerpt may
  still leave the process during the bootstrap phase. If that is unacceptable, run
  `backend.name: mock` or a distilled artifact and send nothing anywhere.
* **The distilled student is only as good as your log.** A log from one narrow workload
  distills a narrow model. There is no substitute for volume and variety, and the graduation
  thresholds exist so you find this out in a report rather than in production.
* **The mock is not a cheap substitute for Jev.** It is a plumbing fixture. See quickstart
  row 3.
* **Egress during bootstrap is real.** Redacted excerpts of non-blocked traffic go to
  TypeSafe's cloud. The gate runs first and is never bypassed, and credential material and
  regulated identifiers never leave the process at all — but "we redacted it" is not the same
  as "nothing left". The project's answer is that this phase ends, not that it is harmless.
* **Young project.** `0.1.0`. The decision-record schema is versioned and additive-only, but
  the graduation pipeline and the LiteLLM integrations are still settling. Pin your versions
  and read [CHANGELOG.md](CHANGELOG.md).

---

## Inspiration and credit

The routing decisions come from **Jev**, a System One model from
[TypeSafe](https://typesafe.ai). System One models return typed judgements and calibrated
probabilities instead of prose you have to parse, which is what makes "escalate when
confidence < 0.8" a policy rather than a vibe. All credit for the calibrated decisions here
is theirs; every mistake in the routing layer is ours.

The *shape* of this project — an LLM proposes, a calibrated decider chooses, code executes,
and the cloud part is treated as temporary — matches what practitioners are already building
with agent gateways; Mitko Vasilev's agent-gateway experiments are the clearest public
example of the pattern, and jev-route's graduation pipeline is that pattern taken to its
conclusion: eventually the decider is yours.

Two prior projects deserve a straight comparison, because they are good and they are doing
something adjacent:

* **[RouteLLM](https://github.com/lm-sys/RouteLLM)** routes on **complexity** and does it
  well, with serious evaluation behind it. It needs a classifier trained on preference data,
  it has no data-sensitivity axis, and it does not report calibrated confidence — so "not
  sure" is not a routable state. jev-route adds the sensitivity dimension, starts with zero
  training data, and treats uncertainty as a first-class policy input.
* **[vLLM Semantic Router](https://github.com/vllm-project/semantic-router)** gives you
  semantic routing with safety signals, which is genuinely useful. It asks you to run
  self-hosted classifier/LoRA infrastructure to get there. jev-route's lane is the opposite
  starting point: one API key on day one, and you build the self-hosted model later, out of
  your own traffic, when you have the data to make it good.

jev-route's specific lane: **complexity *and* data sensitivity, zero training data to start,
calibrated confidence, fractions of a cent per decision, and a path to owning the model.**
None of that is a reason not to use the projects above; if complexity is your only axis,
RouteLLM is a stronger choice than this.

The eval dataset (223 labelled prompts, `evals/data/labeled_prompts.jsonl`, guarded by
`evals/validate_dataset.py`, which the CI workflow runs as its `eval dataset guard` job) uses
only synthetic PII: identifiers from published test ranges and reserved blocks — Visa's public
test card numbers, the SSN area numbers the SSA reserves and will never issue, the fictional
`555-01xx` phone block, RFC 2606 documentation domains — plus organisation domains invented for
the dataset and listed in the validator. See [evals/data/README.md](evals/data/README.md) for the
safety policy the validator enforces.

---

## Documentation

| | |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | the 8-step chain, `DecisionBackend`, why the cache stores judgements, the four LiteLLM modes, circuit breaker, fail-closed |
| [docs/architecture.html](docs/architecture.html) | the same architecture as a single-page visual tour |
| [docs/gate-layers.md](docs/gate-layers.md) | the two-layer gate: detectors, the semantic layer, promotion criteria, the blocked stream |
| [docs/diagrams/](docs/diagrams/) | PlantUML sources + rendered SVGs, with re-render instructions |
| `jev-route --help` | the CLI: `doctor`, `demo`, `route`, `log-stats`, `explain`, `export`, `train`, `evaluate`, `package`, `graduate` |
| [docs/privacy.md](docs/privacy.md) | **read this first.** exactly what leaves the process, and when |
| [docs/policy-guide.md](docs/policy-guide.md) | every policy key, the expression grammar, five complete examples |
| [docs/graduation.md](docs/graduation.md) | export → train → evaluate → package → graduate |
| [evals/data/README.md](evals/data/README.md) | the labelled dataset and its synthetic-PII safety policy |
| [CONTRIBUTING.md](CONTRIBUTING.md) | how to work on this repo |
| [CHANGELOG.md](CHANGELOG.md) | what changed |

---

## License

Apache-2.0. See [LICENSE](LICENSE). Copyright The jev-route Authors.
