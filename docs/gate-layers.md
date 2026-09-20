# The gate: two local layers, and how layer 2 earns the right to block

Technical reference for the privacy gate. [docs/privacy.md](privacy.md) says what
leaves the process; this page says how the component that decides that is built,
and what it would take for the second layer to start blocking.

The gate has two layers. Both run **before any model is consulted**, both run
**locally**, and both can only make a routing decision stricter, never looser.

```
request
   |
   v
+------------------------------------------------------------------+
| Layer 1: HardGate (regex + checksums, deterministic)             |
|   - findings set sensitivity floors                              |
|   - credential/regulated hits: blocks the backend outright       |
|   - permanent; no config, policy, or model can remove it         |
+------------------------------------------------------------------+
   |
   v
+------------------------------------------------------------------+
| Layer 2: semantic layer (local distilled scorer)                 |
|   - catches contextual sensitivity regex cannot see              |
|   - mode: off | shadow (default) | enforce                       |
|   - shadow: scores and logs, cannot change what is served        |
|   - enforce: allowed only after measured promotion               |
+------------------------------------------------------------------+
   |
   v
router (backend decision -> tier -> decision log)
```

## Layer 1: the deterministic floor

`jev_route.gate.HardGate` is a set of regex and checksum detectors over the
redacted excerpt. Two kinds of detector, kept deliberately distinct:

* **Hard detectors** — structured identifiers and credential material (Luhn-valid
  card numbers, SSNs, IBANs, NHS/NI numbers, private key blocks, inline
  `password=`/`api_key=`, provider keys, basic-auth URLs, medical record
  numbers). A hit sets a sensitivity floor and can force the local tier and
  block the cloud backend.
* **Advisory detectors** — topic keywords (e.g. HIPAA, PCI). They are hints for
  the policy rules. They never set a floor, because collapsing the two is how a
  regex guardrail ends up air-gapping every compliance question, not just the
  compliance *data*.

A finding never retains the matched text: it carries a hash of the span
(`GateFinding.span_hash`) so a log can prove a detector fired without keeping
the secret that fired it. Layer 1 cannot be turned off by policy; there is no
configuration in which unscanned text is routed.

## Layer 2: the semantic layer

`jev_route.gate_semantic.SemanticLayer` scores the same excerpt with a local
model and, depending on its mode, either logs what it would have done or acts on
it. The point of layer 2 is the class of prompt layer 1 cannot see: an HR
disciplinary narrative, a clinical note written as prose, a termination letter —
sensitive content that contains no identifier and trips no regex.

### The reference scorer

`LexiconScorer` is a sparse logistic regression over a stored vocabulary, pure
Python, no numpy: it evaluates in the interpreter that is already running the
proxy, so layer 2 costs the router an import nothing. Its ceiling is lower than
a transformer's, and that trade is stated rather than hidden — its job is to
catch the *obvious-in-context* cases, not to out-read a person. The vocabulary
and weights live in the artifact as readable JSON, so a reviewer can open the
file and see every token the model keys on.

A real deployment can train a stronger student anywhere and write the same
artifact format; promotion is decided on the artifact's metrics block, not on
how the weights were produced. `fit_scorer` is the reference trainer, provided
so the pipeline (train → measure → promote → refuse) is exercisable end to end
without a numerical stack. Its artifact's metrics are measured on the
**held-out split only** — a promotion decided on training-set recall is a
promotion decided on a number the model was allowed to memorise.

### Modes

| mode | scores | changes the served decision | writes assessments to the log |
| --- | --- | --- | --- |
| `off` | no | no | no |
| `shadow` (default) | yes | **never** | yes — this is the measurement phase |
| `enforce` | yes | yes, stricter only | yes |

* **Shadow is the default**, and it must be safe even when nothing has been
  trained: an absent artifact makes the layer *inert*, not an error. `NullScorer`
  reports `available = False` so the assessment says *why* nothing happened
  instead of printing a confident 0.0 nobody measured.
* **`off` is an operator decision** not to run layer 2; an inert layer is the
  honest state of a deployment that has not trained a model yet. The two are
  distinguished on purpose.
* In shadow, layer 2 may score a prompt at 0.99 and assert the strictest level;
  the served tier, model, and backend are still exactly what they would have
  been with no layer at all. That invariant is pinned by a router test: if
  shadow can change what is served, it is not a measurement.

### Invariants, and where they are pinned

1. **Layer 2 may make routing stricter, never looser.**
   `resolve_floor` is the arithmetic form: it is a *maximum* over the layers'
   floors, so no input can turn a deterministic `regulated` into `public`. A
   semantic layer that answers "public" contributes `None` and therefore
   contributes nothing. Pinned: a test feeds a semantic layer that asserts
   `confidential` next to a deterministic `regulated` floor and checks the
   combined floor, and the reverse — a semantic `public` can never cancel a
   deterministic `regulated`.
2. **Shadow changes nothing served.** Pinned at router level (above).
3. **Layer 2 is local and light.** No cloud call, no module-scope numpy/torch/
   sklearn import. Pinned by `tests/test_invariants.py::TestHeavyDependenciesAreLazy`.
4. **Disagreements are logged without content.** Each assessment carries the
   score, the asserted level, the mode, and layer 1's detector ids — enough to
   compute the promotion numbers offline, no text, no span, no token list. It is
   written to the decision log next to `shadow`, and "which layer said what" is
   a field, not a formatted string. Two disagreement kinds: `semantic_only`
   (layer 2 fires, layer 1 did not — its job) and `semantic_miss` (layer 1
   fired, layer 2 did not — a model that does not recognise a card number as
   sensitive is not a model you want judging the cases regex cannot see).

## Promotion from shadow to enforce

`mode: enforce` is a config flag, and the flag alone is not enough. The
criteria are *measured gates* checked at layer construction, so a deployment
that asks for enforce without earning it **fails to start** rather than starting
in a mode its operator did not ask for.

`EnforceCriteria` (defaults shown):

| criterion | default | what a violation means |
| --- | --- | --- |
| `min_recall` | 0.99 | **The one that matters.** Held-out positives the layer must catch. A miss is content that left the building. Configured below `MIN_ALLOWED_RECALL` (0.90) the config is rejected outright — you may raise the bar, you may not remove it. |
| `max_false_positive_rate` | 0.02 | Real negatives the layer would have air-gapped. A cost and a credibility problem, not a leak: the failure mode is operators turning the layer off. |
| `max_disagreement_rate` | 0.05 | How often the layers contradict on **live traffic**, read from the shadow observation. High disagreement does not mean layer 2 is wrong — catching what layer 1 misses is the job — but it means the pair is not yet a coherent story, and promoting on top of it makes every future incident ambiguous. |
| `max_semantic_miss_rate` | 0.02 | Of what layer 1 also catches, how much layer 2 misses. |
| `min_positive_examples` / `min_negative_examples` | 200 / 500 | Sample sizes. At 0.99 recall, twenty positives proves nothing: one miss is 0.95 and zero misses is a rounding error. |
| `min_shadow_examples` | 1000 | Live requests observed in shadow. The criterion no offline number substitutes for: a model that scores 0.99 on sentences you wrote has still never seen your traffic. |

Where the numbers come from — and why both are needed:

* **Held-out offline set** → `recall` and `false_positive_rate`. Live traffic
  has no ground-truth labels, so these two cannot be measured live.
* **Live shadow log** (`measure_shadow_log` over the decision log) →
  disagreement and semantic-miss rates, and the sample counts. This is the half
  that cannot be faked with a hand-written eval set: it is what layer 2 actually
  did to real traffic while it was unable to affect anything.

The two are overlaid by `merge_metrics`, which requires matching model
versions — shadow numbers collected against last month's artifact are not
evidence about this one, and silently averaging them is how a promotion gets
approved on data describing a model that is no longer deployed.

The answer is a `PromotionReport`: the measured numbers, pass or fail, because
a refusal that does not say *why* gets answered by lowering a threshold in
config. There is no fallback to shadow: a policy that asks for enforce without
the criteria is a deploy failure with the numbers in the message.

The CLI runs the same check and performs the cutover:

```bash
# Measure: the artifact's holdout metrics, overlaid with the live shadow log.
jev-route graduate --track gate \
    --artifact artifacts/semantic/semantic.json \
    --policy policies/default.yaml \
    --log decision-log/decisions.jsonl

# When every check passes, flip the policy to enforce. The cutover file
# carries its evidence: the shadow metrics are written next to it and
# pointed at by gate.semantic.shadow_metrics, because the enforce layer
# re-checks promotion at construction and the live criteria only exist in
# that observation. Without it the file would pass this check and refuse
# to start.
jev-route graduate --track gate \
    --artifact artifacts/semantic/semantic.json \
    --policy policies/default.yaml \
    --log decision-log/decisions.jsonl \
    --write
```

The verb exits `1` with the measured numbers when not ready and writes nothing,
exits `0` with a verified policy when the flip is earned, and re-verifies the
written file by constructing the enforce layer from it -- a cutover file that
would refuse to start is a cutover that happened to nobody.

## The bootstrap paradox

This is the design constraint the whole shape of the gate obeys:

> The gate cannot ask the cloud whether something is safe to send to the cloud.

Therefore:

* the deterministic floor never asks a model — layer 1 is regex and checksums,
  permanently;
* the semantic model is trained **only** on synthetic positives (format-faithful,
  provably fake — see `jev_route.distill.synthetic_pii`) and public corpus
  positives, plus real production **negatives** — traffic the gate already
  allowed through, which is why it is safe to reuse.
* **never on real blocked content.** A blocked record's text is refused by the
  dataset builder; the metadata-only projection a blocked record may contribute
  is text-free by construction.

So the operative sentence: **the gate never trains on your secrets.** The
enforcement is threefold: the closed source list in the dataset module, a
canary test that a gate-blocked record's stored text appears nowhere in the
dataset output, and an AST walk of the module asserting no function reads the
excerpt. See `tests/distill/test_sensitivity_data.py`.

Blocked requests do produce a record — `GateBlockRecord`, on its own stream
(`gate.blocked_metadata.path`, default next to the decision log). It carries
exactly `request_id`, `timestamp`, the detector ids that fired, the
deterministic feature vector, and the excerpt hash. No excerpt text under any
configuration, no raw secret, no matched span. **This metadata feeds rule
improvement, not model training**: new regex and NER patterns are written by a
person reading aggregate counts; they are not fitted to blocked text. The
streams are kept in separate files on purpose — the refusal stream feeds rule
improvement, the decision log feeds model training, and keeping them apart means
no export path can pick up the wrong one by accident.

## Worked example: shadow mode, running

`examples/gate-shadow-demo.py` trains a real layer-2 artifact with
`fit_scorer` and routes three prompts through one router in shadow mode. No
network, no numpy, no API key. Real output:

```
$ .venv/bin/python examples/gate-shadow-demo.py
trained layer-2 artifact: /var/folders/.../jev-shadow-demo-XXX/semantic.json
  model_version='semantic-logreg-72ae4c8e'  scorer='lexicon-logreg'  threshold=0.5
  holdout: n=24 (pos 12 / neg 12)
  holdout metrics: recall=1.0  fpr=0.0  disagreement=0.5
  mean score: positive=0.992  negative=0.036979

(a) benign, no hits
  prompt:        'how do I center a div in css'
  layer 1:       fired=False detectors={} floor=- blocks_backend=False
  layer 2:       mode=shadow score=0.009 fired=False enforced=False disagreement=-
  served:        tier=cheap model='cheap-model' backend='mock' via rule default
  sensitivity:   internal

(b) real-shaped SSN
  prompt:        'please update my profile, my SSN is 123-45-6789'
  layer 1:       fired=True detectors={'us_ssn': 1} floor=regulated blocks_backend=True
  layer 2:       mode=shadow score=0.303 fired=False enforced=False disagreement=semantic_miss
  served:        tier=local model='local-model' backend='gate' via rule gate.force-local
  sensitivity:   regulated

(c) contextual, no regex hits
  prompt:        'can you summarise the notes from our disciplinary hearing last week'
  layer 1:       fired=False detectors={} floor=- blocks_backend=False
  layer 2:       mode=shadow score=0.991 fired=True enforced=False disagreement=semantic_only
  served:        tier=cheap model='cheap-model' backend='mock' via rule default
  sensitivity:   internal

decision log written to /var/folders/.../jev-shadow-demo-XXX/decisions.jsonl (excerpt_mode=hash: no text stored)
```

Reading it:

* **(a)** passes both layers and is served on the default tier.
* **(b)** is caught by layer 1: `us_ssn` fires, the floor is `regulated`, and
  `blocks_backend` is true, so the backend is `gate` — no call was made — and
  the tier is forced local. Layer 2 scored 0.303, below the 0.5 threshold, so it
  *missed* what layer 1 caught: the disagreement kind `semantic_miss` is logged.
  That is the number `max_semantic_miss_rate` is about.
* **(c)** trips no regex. Layer 2 scores 0.991 — above threshold, so it fires —
  but the mode is shadow, so `enforced` is false and the request is served
  exactly as (a) was. The disagreement kind `semantic_only` is logged. That is
  the layer working as designed and changing nothing.
* The holdout is balanced (12/12) and the artifact's version is
  hash-deterministic, so the printed `model_version` is the same on every run.
  The disagreement rate 0.5 is 12 semantic-only rows over 24: every holdout
  positive is one layer 1 cannot see, which is precisely the set layer 2 exists
  for.

## Policy surface

Everything above is declarative and validated at policy load: a typo fails the
deploy, not the request. Every key in `gate.semantic` is honoured or fatal —
unlike some other sections that drop unknown keys, an ignored `mode: enfoce`
would leave a gate in shadow that its operator believes is enforcing, which is
the one failure mode worse than not having the layer.

```yaml
gate:
  semantic:
    mode: shadow            # off | shadow | enforce
    artifact: ./artifacts/semantic/semantic.json   # absent -> layer inert
    threshold: 0.5          # the artifact's own threshold wins once loaded
    level: regulated        # what a firing layer asserts in enforce mode
    shadow_metrics: ./artifacts/shadow-metrics.json  # live numbers, optional
    enforce_requires:
      min_recall: 0.99
      max_false_positive_rate: 0.02
      max_disagreement_rate: 0.05
      max_semantic_miss_rate: 0.02
      min_positive_examples: 200
      min_negative_examples: 500
      min_shadow_examples: 1000
  blocked_metadata:
    enabled: true           # default; the refusal stream
    path: ./decision-log/gate-blocks.jsonl   # omitted -> next to the decision log
```

Policy rules see layer 2's **actionable projection**, not its raw output: in
shadow mode `semantic_fired`/`semantic_score`/`semantic_force_local` are
zeroed, so a rule that references them behaves as if the layer were off. The
raw assessment is still on the record.

## Where to look

| question | file |
| --- | --- |
| detectors, floors, blocking | `src/jev_route/gate.py` |
| modes, promotion, artifacts, the paradox | `src/jev_route/gate_semantic.py` |
| router integration, the pinned invariants | `src/jev_route/router.py`, `tests/test_gate_semantic.py` |
| refusal telemetry | `GateBlockRecord` in `src/jev_route/schema.py`, `src/jev_route/logging_sink.py` |
| what layer 2 may train on | `src/jev_route/distill/sensitivity_data.py`, `src/jev_route/distill/synthetic_pii.py` |
| the egress table | `docs/privacy.md` |
