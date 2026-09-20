# Graduation: from Jev's decisions to your own model

This is the pipeline the rest of jev-route exists to feed. Everything before it — the gate,
the calibrated backend, the policy engine, the soft-distribution log — is preparation for the
moment you stop paying a cloud API to make your routing decisions and start making them
yourself, on your hardware, with a model trained on your traffic.

The lifecycle, and the verb for each stage:

```bash
jev-route route    "..."                    # day one, on Jev
#  ... run your traffic for weeks; the decision log fills up ...
jev-route log-stats                          # what have I accumulated?
jev-route export   --out data/ds             # log      -> training dataset
jev-route train    --data data/ds --out artifacts/v1
jev-route evaluate --artifact artifacts/v1
jev-route package  --artifact artifacts/v1 --out artifacts/v1-packaged
jev-route graduate --artifact artifacts/v1-packaged --policy policies/default.yaml --write
```

> **Status.** All five stages run end to end — `export`, `train`, `evaluate`, `package`,
> `graduate` — and every report format and threshold check in this document is the behaviour the
> CLI actually prints. The lifecycle above was verified by running it against the repo's own
> decision log (the `--write` cutover correctly refused on that log; see the readiness section
> for why).

---

## Why soft targets, and not labels

The objective is `KL(teacher ‖ student)` over the teacher's **probability distributions**, not
cross-entropy against its argmax.

That single choice is why this pipeline exists. A hard label says `internal`. The teacher's
distribution says `internal 0.46, confidential 0.38, public 0.16` — and the second one is what
lets the router's `on_uncertain` rules keep working after the cloud dependency is gone.
Distilling argmax labels produces a confident student that the policy engine can no longer
reason about, which is a regression dressed up as a cost saving.

### Temperature

Teacher probabilities `p` are softened to `normalize(p ** (1/T))`; the student's logits are
divided by the same `T`; the loss is scaled by `T²` (Hinton's scaling, so gradient magnitudes
do not shrink as `T` grows).

The algebra matters, because it is what makes the *served* model honest. Matching
`softmax(z/T)` to `normalize(p^(1/T))` implies `z = log p + const`, so `softmax(z) = p` at
`T = 1`. **Inference therefore runs at temperature 1**, and the student reproduces the
teacher's calibration rather than a flattened version of it. Higher `T` transfers more of the
"dark knowledge" in the tails — the relative ordering of the labels the teacher rejected — at
the cost of a noisier gradient. `--temperature 2.0` is the default.

### Four heads

One shared trunk, four output heads, in a **fixed order** because the order is part of the
artifact file format:

| head | type | ladder |
| --- | --- | --- |
| `complexity` | softmax | `trivial` / `standard` / `hard` / `frontier` |
| `sensitivity` | softmax | `public` / `internal` / `confidential` / `regulated` |
| `domain` | softmax | `code` / `writing` / `analysis` / `chat` / `data-extraction` |
| `pii` | **Bernoulli** | a single probability; there is no separate confidence, because the probability *is* the calibration |

Head weights default to `{"complexity": 1.0, "sensitivity": 1.5, "domain": 1.0, "pii": 1.0}`.
`sensitivity` is weighted highest on purpose, and the reason is worth repeating everywhere in
this project: it is the head the **data-egress rule** reads, so an error there is a privacy
incident rather than a cost incident. Override with `--head-weights '{"sensitivity":2.0}'`.

### Labels are the teacher's *raw* belief

Export takes labels from the teacher's raw answers, **not** the policy-adjusted effective ones.
The router re-applies the gate floor and the `on_uncertain` confidence bump to whatever a
backend returns, so baking those adjustments into the student would apply them **twice** and
destroy the calibration that made the cloud phase worth paying for. The effective values are
still exported, under `teacher`, for evaluation and debugging.

This is the single easiest way to silently ruin a distillation, and it is why export and the
router are separate modules with an explicit contract between them.

---

## Stage 1 — `export`

```bash
jev-route export --out data/ds --mode auto --holdout 0.2 --seed 0
```

Produces `data/ds/dataset.jsonl` plus a `data/ds/dataset.stats.json` sidecar. Two passes over
the log by design: the first **measures** what the log can support, the second writes. Memory
stays flat no matter how big the log gets.

### `text` vs `features`, and why the log decides

| mode | needs | produces | student ceiling |
| --- | --- | --- | --- |
| `text` | stored redacted excerpts, i.e. you ran with `logging.excerpt_mode: redacted` | TF-IDF word/bigram counts over the redacted excerpt, hashed into a stored vocabulary | higher — the student reads the request |
| `features` | only `RequestFeatures`, which are always logged | a fixed-width vector **derived from** the 24 deterministic features (the width is fit from the data: the numeric fields plus a per-dataset language and detector vocabulary) | lower — the student reads the request's *shape* |

`--mode auto` picks `text` when the log has it and `features` otherwise. Asking for `text` from
a text-free log is an **error, not a fallback**:

> text mode needs stored prompt excerpts, and none of the N records in … carry one.
> The decision log was written with `logging.excerpt_mode: hash` (the default), which retains a
> hash and the deterministic request features but no text. There is nothing to train a text
> classifier on, and this export refuses to silently train a weaker model than the one you
> asked for.

That refusal is the whole design of this stage in one sentence. Silently training a weaker
model than the operator asked for is exactly the kind of surprise this project should not have.

**Consequence for operators, and it bites late:** `logging.excerpt_mode` is a decision you make
*before* you collect the traffic. A log written under `hash` can never be exported in `text`
mode. If you might want the text path, turn it on at deployment time and accept the retention
tradeoff described in [docs/privacy.md](privacy.md). `jev-route log-stats` tells you which
position you are in:

```text
No record carries excerpt text (logging.excerpt_mode is 'hash' or 'none').
  -> features-mode distillation only. To enable text mode, set
     logging.excerpt_mode: redacted and collect more traffic.
     Note: gate-blocked requests are never stored as text, by design.
```

### Which rows survive, and why

`skip_reason()` decides. A record is dropped, with the reason counted in the stats sidecar, when:

| reason | why |
| --- | --- |
| `unsupported_schema_version` | the record predates the current `SCHEMA_VERSION`. A migration would go here; there is only one version so far, so the honest behaviour is to refuse the row and say which version it was |
| `degraded` | the backend could not answer, so the "distribution" is uniform. **Training on it teaches the student to be uncertain about everything** |
| `no_text` | `text` mode and this record has no stored excerpt |
| `no_signal` | no head has a usable target |

Plus duplicate `request_id`s (default: skipped).

### Gate-blocked rows are kept, with masked heads

This is subtle and it is the reason the gate is a *label* source and not just a filter.

A gate-blocked request never reached a backend, so its `complexity` and `domain` are uniform
and unusable — but its `sensitivity` label is **deterministic and locally produced**, at
p=1.00, from the gate's floor. For some detectors that floor rests on a checksum
(payment card Luhn, IBAN mod-97, NHS mod-11, NINO rules, real-domain email); for the rest it
rests on shape alone. Either way it was decided by a regex you can read, not by a model.
Those rows are therefore:

* **included**, tagged so you can see how many there are (`gate_rows`, `gate_blocked_rows`);
* **masked per head** — the sensitivity and pii heads train on them, the complexity and domain
  heads do not.

They are also rows a cloud teacher could never have labelled for you, because jev-route refuses
to send them anywhere. Your most privacy-sensitive traffic still contributes to the model that
will one day handle it locally.

### Reproducible splits

The train/holdout assignment is a **hash of `request_id`** (`split_bucket`, salted with
`jev-route-distill-split\x00`), not a shuffled index. So re-running export on a grown log keeps
every previous row on the same side. Nothing that was trained on ever moves into the holdout —
which is what makes the graduation numbers mean something. A random split re-drawn on every
export would quietly leak training rows into your "held-out" evaluation and produce a graduate
report you could not trust.

### The stats sidecar

`ExportStats` is written next to the dataset so you can judge it without re-reading it:
`records_read`, `rows_written`, `train_rows`, `holdout_rows`, `skipped` (by reason),
`gate_rows`, `gate_blocked_rows`, `usable_rows_per_head`, `label_support` (per head, per
label), `teacher_backends`, `teacher_model_versions`, `record_schema_versions`,
`teacher_latency_ms` (percentiles), `contains_prompt_text`, `dataset_sha256`, and `warnings`.

Read `label_support` before you train. A head with 12 examples of `frontier` and 4000 of
`standard` will not learn `frontier`, and no amount of tuning fixes a dataset that does not
contain the class.

---

## Stage 2 — `train`

```bash
jev-route train --data data/ds --out artifacts/v1 \
                --trainer auto --temperature 2.0 --epochs 200 --lr 0.05 --l2 1e-5 --seed 0
```

### The default trainer is numpy, and that is deliberate

A tiny shared-trunk network (linear by default) trained with Adam on mini-batches. numpy is
enough, and preferring it is a considered call rather than a limitation:

* the model has tens of thousands of parameters;
* the input is sparse and small;
* BLAS-backed matmuls finish in well under a second on a laptop.

torch would add a large, version-sensitive dependency to buy little at this size — and it
would put a deep-learning runtime in the **request path** of a router whose whole promise is
that the local end state is boring and auditable. `--trainer torch` exists for operators with
very large logs; it runs the *same* student (the linear/MLP, not an encoder — there is no
encoder path) under torch autodiff and produces the same artifact format.

`numpy` is a **lazy import** everywhere in `distill/`. `import jev_route` succeeds with no
numerical stack installed, and the router runs end to end on `mock` or `jev` without it:

```text
distillation needs numpy. Install it with:  pip install 'jev-route[distill]'
```

### Determinism

Everything random comes from one seeded numpy generator: weight init and epoch shuffling. No
network, no GPU, no clock-derived state. The same dataset, config and numpy version produce the
same weights. That is a requirement, not a nicety — you are going to have to defend this model
to somebody, and "we can rebuild it bit for bit" is the defence.

### Degeneracy is checked, not assumed

`check_degenerate()` runs **before** training, and it exists because the failure it guards
against is silent. Its docstring states the case: *a log of gate-blocked requests only produces
uniform teacher distributions for three of the four heads and a constant for the fourth.
Gradient descent happily converges on "always answer the base rate", the metrics look non-NaN,
and the operator ships a model that never actually reads a prompt.*

It **raises `TrainError`** — training does not proceed — when:

* the training split is empty (export produced no usable rows);
* the vectorizer produced zero features;
* **every** teacher distribution in the dataset is identical, so there is no signal to distill.
  The message names the fix: route real traffic while the backend is healthy, and set
  `gate.on_force_local: still_classify` if you want labels for gate-blocked prompts too.

It **returns warnings** when a single head has no usable rows, or when one head's teacher
distributions are all identical — the student can only learn a constant there, and will stay at
its prior. For an all-gate-blocked log that is expected, and the warning says so.

Read the warnings. A head that "contributes nothing" is not a training bug; it is a dataset
that does not contain that class.

---

## Stage 3 — `evaluate`

```bash
jev-route evaluate --artifact artifacts/v1 --bins 10 [--data data/ds] [--policy policies/default.yaml]
```

Reports, per head:

* **per-class accuracy** — not just the overall number, because a head that is 99% accurate by
  never predicting `regulated` is worse than useless;
* **macro-F1** — the number that does not reward a majority-class predictor;
* **ECE** (expected calibration error, `--bins 10`) — **the most important metric in this
  document.** A distilled router that is accurate but miscalibrated has lost the only thing
  worth distilling, because `on_uncertain` thresholds are written against confidence values. A
  student that answers `internal` correctly 90% of the time at `confidence 0.99` will never trip
  your escalation rule, and your policy silently stops protecting you.

Plus:

* **latency p50/p95/p99** for the student, so `graduate` can compare against
  `--max-latency-ms`;
* **teacher agreement**, in two flavours.

### Argmax agreement vs tier agreement

* **Argmax agreement** — does the student pick the same *label* as the teacher, per head?
* **Tier agreement** — run **both** the teacher's answers and the student's answers through
  **the same `Policy`**, and compare the resulting *tiers*.

**Tier agreement is the number that matters — it is the more decision-relevant of the two.**
It is *not* uniformly stricter: the table below shows two disagreements it ignores that argmax
agreement counts. It is stricter only where the policy's rules read the head that was wrong.
Here is why, concretely:

| disagreement | argmax agreement | tier agreement | does it matter? |
| --- | --- | --- | --- |
| teacher `domain=code`, student `domain=analysis` | ✗ | ✓ | **No.** No policy worth running routes on `domain` alone |
| teacher `complexity=hard`, student `complexity=frontier`, both → `strong` | ✗ | ✓ | **No.** Same tier, same model, same cost |
| teacher `sensitivity=internal`, student `sensitivity=confidential`, `internal`→`cheap` and `confidential`→`local` | ✗ | ✗ | **Yes.** That is the difference between your self-hosted GPU and a third-party API |
| teacher `complexity=standard`→`cheap`, student `complexity=hard`→`strong` | ✗ | ✗ | **Yes, financially.** You are paying for capability you did not need |

Argmax agreement counts all four of those as one failure each. It is dominated by the harmless
ones, because `domain` has five classes and is the easiest head to disagree on. Tier agreement
counts only the two that change what actually happens to the request.

It is also the stricter one **in the direction that matters**: a student can be right about
every label and still land on a different tier, if the policy's rules read a head the student
got wrong. And
it is measured against *your* policy, not a canonical one — which is the only version of this
metric that means anything, because the policy is where your risk appetite actually lives.

Run both. Report both. Gate the cutover on tier agreement.

---

## Stage 4 — `package`

```bash
jev-route package --artifact artifacts/v1 --out artifacts/v1-packaged
```

The artifact is a **directory**:

```text
artifacts/v1-packaged/
  artifact.json      the envelope: schema version, head layout, teacher identity and
                     version, training config, environment info, checksums
  W.npy, b.npy       the student's weights, one raw .npy file per parameter. Raw .npy,
                     deliberately not .npz: an npz is a zip, and zip entries carry the
                     current timestamp, so the same weights would produce different bytes
                     (and a different checksum) on every run
  vectorizer.json    the TF-IDF vocabulary, or the feature layout
  metrics.json       the evaluation report
  dataset.json       the export stats sidecar, so the artifact knows what it was trained on
```

Six files, `numpy.load`-able weights, no pickle anywhere.

Two deliberate choices a sceptical reader will look for:

* **No pickle, anywhere.** `weights.npz` is raw arrays and the rest is JSON. A pickle in an
  artifact directory is arbitrary code execution the moment somebody loads a file they
  downloaded, and it silently rots across library versions. This format can be read by
  `json.load` and `numpy.load` and nothing else. If you are shipping artifacts between teams,
  or loading one somebody else trained, that is the entire argument.
* **Loading validates.** The head label sets are checked against the **live** `jev_route.schema`
  ladders, the artifact schema version against `SUPPORTED_ARTIFACT_SCHEMA_VERSIONS`, and the
  weight checksum against the envelope. An artifact trained against an older schema raises
  `ArtifactError` **naming the mismatch** rather than mis-predicting quietly — which is the
  failure mode that would otherwise route regulated data to a cloud model six months after a
  schema bump.

Self-describing means the artifact travels: teacher identity, teacher model version
(`jev-1.13.0`, or `mock-1.0.0`, or a previous student), training config, dataset checksum,
metrics, and the environment it was built in. `DistilledBackend` catches `ArtifactError` and
**degrades instead of raising**, because a routing decision must never fail because a model file
is bad — it must fail *closed*.

---

## Stage 5 — `graduate`

```bash
# report only
jev-route graduate --artifact artifacts/v1-packaged --policy policies/default.yaml

# actually perform the swap
jev-route graduate --artifact artifacts/v1-packaged --policy policies/default.yaml --write
```

### The report

```text
your local model agrees with Jev on <X>% of held-out decisions,
ECE=<Y>, estimated added latency=<Z>ms

  min_tier_agreement   >= 0.95     [ pass | FAIL ]
  max_ece              <= 0.10     [ pass | FAIL ]
  max_latency_ms       <= 50.0     [ pass | FAIL ]
  min_samples          >= 500      [ pass | FAIL ]
```

`<TODO: a real graduate report pasted here, produced by running the pipeline end to end on an
accumulated decision log. Every number must come from the command's own output.>`

### The thresholds

| flag | default | what it protects against |
| --- | --- | --- |
| `--min-tier-agreement` | `0.95` | a student that routes differently from the teacher it replaced. This is the one that matters — see [argmax vs tier](#argmax-agreement-vs-tier-agreement) |
| `--max-ece` | `0.10` | a student that is accurate but miscalibrated, which silently disables every `on_uncertain` rule in your policy |
| `--max-latency-ms` | `50.0` | a "local" model that is slower than the cloud call it replaced. Graduation is supposed to remove latency, not add it |
| `--min-samples` | `500` | a report computed on 40 held-out decisions. All four numbers above are noise at that sample size |

Tune them to your risk appetite, but do not tune `--min-samples` down. A 0.97 tier agreement on
80 samples is not a 0.97 tier agreement.

`--replay` re-runs the held-out prompts through the **live teacher** instead of comparing
against the labels in the dataset. That needs a Jev key and costs money, and it is the stronger
check: it catches a teacher that has moved since you collected the log (`jev-latest` is an
alias, and TypeSafe ship new versions behind it). It has a hard precondition that bites late:
the dataset must be **text-mode**, because the prompts have to be re-sent — on a `features`-mode
dataset the verb refuses with an error. That is the same retention decision as text-mode
*export* (the `logging.excerpt_mode: redacted` trade-off, above), so if you ever want to be
able to replay, set the log mode before you need it, not after.

### The swap

`graduate --write` performs the cutover as a **config change, not a code change**, using
`Policy.with_overrides` — but only if the readiness checks pass. When they do not, `--write`
prints its reasons and writes nothing (the flags do not bypass the gate: readiness runs first,
and a cutover you had to force is a cutover you cannot defend). When they do pass, the swap is:

```yaml
backend:
  name: shadow
  primary:   { name: distilled, artifact: ./artifacts/v1-packaged }
  shadow:    { name: jev, api_key_env: TYPESAFE_API_KEY }
  log_disagreements: true
  shadow_timeout_seconds: 5.0
```

Your local model serves traffic. Jev runs alongside it, inside the backend layer. Note where
a disagreement lands in this shape: the backend's **in-process ring buffer** and, because
`log_disagreements` is on, the `on_disagreement` callback — which `build_backend` does not set,
so with a policy-built backend the ring buffer is the only record, and it dies with the
process. (The decision record's `shadow` field is written by the *router-level* side-run, which
requires `Router(shadow_backend=...)` in code; nothing on the CLI path passes one.) If you want
the disagreement history to survive, construct the router yourself and wire an
`on_disagreement` callback. Nothing about the router, the gate, the policy or the log changes —
which is the payoff of `DecisionBackend` being the architectural center.

Two safety properties of the write itself:

* **It writes a NEW file by default** (`policies/default.yaml` becomes
  `policies/default-graduated.yaml`). `--in-place` is opt-in, and `--out-policy` names the
  destination. A graduation that silently rewrites your production policy is not a
  graduation. Know what the backup actually is: a `bak-<stamp>` copy is made of the
  **destination, and only if it already exists** — so a first graduation into a fresh name
  creates no backup at all (there is nothing there yet), and the second one backs up the
  first's output. With `--in-place` the backup *is* your original policy.
* **Shadow sampling is deterministic.** `_sample_hit(rate, excerpt_hash)` buckets the excerpt
  hash, so the same prompt is always in or always out and replays are reproducible. Set
  `shadow.sample_rate` below `1.0` to price the shadow phase — remember that **egress continues
  while the shadow runs**, because the shadow backend is Jev.

### Ending the cloud phase

When the disagreement rate has stayed where you want it for as long as you want:

```yaml
backend:
  name: distilled
  artifact: ./artifacts/v1-packaged
```

Delete the Jev key from the environment. `grep -rn --include='*.py' "api.typesafe.ai" src/`
returns exactly one file, `backends/jev.py` — the backend module itself — and after the swap it
is not on your request path any more (the repo's own invariant test pins that it is the only
file). Run `jev-route doctor` and confirm the `TYPESAFE_API_KEY` check now fails — and that
everything else passes.

That is the whole promise: **the egress stops permanently.** Not per-request-configurably, not
"only for sensitive prompts", not behind a flag somebody will flip back. The router is yours.

---

## Keeping the model good

Graduation is not a finish line. Your workload drifts, and a student trained on last quarter's
traffic will be quietly wrong about this quarter's.

* **Keep logging.** The log is still being written under `backend.name: distilled`, and it is
  still the dataset. Now the teacher is your own model, so re-training is self-referential —
  which is exactly what shadow mode is for.
* **Re-run shadow periodically.** `backend.name: shadow` with `primary: distilled`,
  `shadow: jev` on a low `sample_rate` is a drift detector in principle — but read the
  limitation before you build on it: with a policy-built backend the disagreements live in the
  backend's in-process ring buffer only, and there is no CLI verb to read them. To make the
  drift signal durable you must either pass an `on_disagreement` callback (construct the router
  yourself) or use the router-level side-run, whose telemetry lands in the decision log. A
  rising disagreement rate is the earliest signal you will get, and it arrives before your users
  notice anything — provided you have somewhere to see it.
* **Re-export and re-train on a schedule.** `export` is idempotent and its splits are stable, so
  a growing log produces a growing training set with no leakage into the holdout.
* **Watch `label_support`.** A new class of traffic shows up there first, as a label with almost
  no rows.
* **Re-run `graduate --replay` after a Jev version change.** `jev-latest` is an alias. If the
  teacher moves, the student is now imitating an older teacher, and the only way to know is to
  ask the current one.

---

## Reference

| stage | CLI | library |
| --- | --- | --- |
| inspect | `jev-route log-stats` | `distill.export.inspect_log` |
| export | `jev-route export` | `distill.export.export_dataset`, `resolve_mode`, `skip_reason`, `assign_split` |
| train | `jev-route train` | `distill.train.TrainConfig`, `soften_distribution`, `build_tensors`, `check_degenerate` |
| artifact | `jev-route package` | `distill.artifact.save_artifact`, `load_artifact`, `DistilledArtifact`, `StudentModel`, `TextVectorizer`, `FeatureVectorizer` |
| serve | `backend.name: distilled` | `backends.distilled.DistilledBackend` (*`build_backend` resolves the name and imports it lazily; a bad or missing artifact degrades to the policy's fallback tier instead of raising*) |
| shadow | `backend.name: shadow` | `backends.shadow.ShadowBackend` |

Extras:

```bash
pip install 'jev-route[distill]'         # numpy + scikit-learn: the default path
pip install 'jev-route[distill-torch]'   # + torch + transformers: the autodiff trainer (the same linear/MLP student, no encoder path exists)
pip install 'jev-route[eval]'            # + pandas + matplotlib: the eval harness
```

See also [docs/architecture.md](architecture.md) for `DecisionBackend` and the shadow backend,
and [docs/privacy.md](privacy.md) for what `excerpt_mode` costs you.
