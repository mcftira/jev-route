# jev-route evaluation report

Run completed 2026-09-19T11:03:16+00:00 (UTC). Rendered from `summary.json` by `evals/report.py`; the render adds no timestamp of its own, so re-rendering the same summary is byte-identical.

| field | value |
| --- | --- |
| backends evaluated | `jev`, `mock` |
| dataset | `/Users/jalapeno/jev-route/evals/data/labeled_prompts.jsonl` |
| dataset sha256_16 / rows | `26b875f47af4fda5` / 223 |
| policy | `/Users/jalapeno/jev-route/policies/default.yaml` |
| policy sha256_16 / version | `26bf41887e399ca4` / 1 |
| git commit | `b1e98fd` |
| `TYPESAFE_API_KEY` present at run time | yes |
| re-analysed from saved rows (`--reuse`) | no |
| calibration bins | 15 |
| concurrency | 12 |

Command:

```console
python evals/run_eval.py --backend both --concurrency 12
```

Every number in this file is read out of `summary.json` at render time; nothing is hardcoded, so re-running the evaluation regenerates a truthful report. **`n/a` means the summary did not carry that field. It never means zero.**

## Headline

> **UNSAFE** means: a prompt whose dataset label requires the air-gapped `local` tier, but
> which the router sent to a cloud tier (`cheap` or `strong`). The text left the building.
> It is a count of **privacy violations**, not of accuracy errors, and it is the single most
> important number in this file. It is deliberately *not* the complement of tier accuracy:
> a router that never leaves `local` scores zero UNSAFE and is useless, and a router can be
> mostly accurate and still leak. Read the two numbers separately, always.

**The three numbers that decide whether this router is worth running** (backend `jev`):

1. **11 UNSAFE rows** out of 223 scored (4.93% of the run). Privacy violations, as defined above. This is the number to fix first.
2. **72.2% tier accuracy** over 223 scored prompts, with 0 / 0 / 0 rows excluded (errored / degraded / missing tier). A tier accuracy of 72.2% is a measurement of a work in progress, not a product claim: 11 prompts that should never have left the building did.
3. **-64.3% cost versus `always_strong`** on the same prompts. Assumption-derived: the price table, the character-per-token proxy and the assumed output length are all estimates, printed in full in the cost section. The *ratio* between strategies is the load-bearing part; the dollar total is not.

Supporting numbers for the same run: sensitivity ECE 0.1289, decision latency p50/p95/p99 = 318.4 / 724.7 / 791.1 ms.

| backend | tier accuracy (n scored) | UNSAFE | unsafe rate | sensitivity ECE | cost vs `always_strong` | decision latency p50/p95/p99 (ms) | excluded err/degr/miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `jev` | 72.2% (n=223) | 11 | 4.93% | 0.1289 | -64.3% | 318.4 / 724.7 / 791.1 | 0 / 0 / 0 |
| `mock` | 51.1% (n=223) | 38 | 17.04% | 0.3664 | -52.0% | 0.0 / 0.0 / 0.0 | 0 / 0 / 0 |

Two naming traps, because both cost the reader a wrong inference:

* The cost strategy called `jev_route` means "the measured routing decisions of the backend in this row". It is not specific to the real Jev API; the offline mock's decisions are priced under the same strategy name.
* `unsafe rate` is UNSAFE divided by *scored* rows, not by dataset rows. The denominator is printed next to every accuracy and rate in this file.

Before quoting the accuracy column, read [The tradeoff: uncertainty escalation on vs off](#the-tradeoff-uncertainty-escalation-on-vs-off). It shows the accuracy number is what the router *pays* for privacy, not a score it maximises.

## What was measured

Provenance first, so no result below has to be taken on trust.

| field | value |
| --- | --- |
| generated_at (UTC) | 2026-09-19T11:03:16+00:00 |
| git commit | `b1e98fd` |
| dataset | `/Users/jalapeno/jev-route/evals/data/labeled_prompts.jsonl` |
| dataset sha256_16 | `26b875f47af4fda5` |
| dataset rows | 223 |
| policy | `/Users/jalapeno/jev-route/policies/default.yaml` |
| policy sha256_16 | `26bf41887e399ca4` |
| policy version | 1 |
| policy failure mode | `fail_closed` |
| gate `on_force_local` | `skip_backend` |
| `TYPESAFE_API_KEY` present | yes |
| `--reuse` (re-analysed saved rows) | no |
| calibration bins | 15 |
| concurrency | 12 |

A real API key was available to the harness, so a `jev` block below can only have come from the live TypeSafe API. Confirm it against `model_versions_observed`: that field is read off the responses, not off the configuration.

### Per-backend run facts

| field | `jev` | `mock` |
| --- | --- | --- |
| model versions observed | `jev-1.13.0` | `mock-1.0.0` |
| run started | 2026-09-19T11:03:16+00:00 | 2026-09-19T11:03:23+00:00 |
| run finished | 2026-09-19T11:03:23+00:00 | 2026-09-19T11:03:24+00:00 |
| rows in default configuration | 223 | 223 |
| backend API calls | 206 | 206 |
| logical router requests | 764 | 764 |
| memo hits (replayed answers) | 558 | 558 |
| retry attempts | 206 | 206 |
| requests needing a retry | 0 | 0 |
| final model version | jev-1.13.0 | mock-1.0.0 |
| degraded at end of run | 0 | 0 |
| reused saved rows | no | no |
| questions sha256 | 8b78e2c56b64a8ad2409c1b5230cc500923b707c5385d4a0dc38bbb175b81166 | n/a |
| concurrency | 12 | 12 |

Raw per-row evidence (one JSONL per policy configuration):

* `jev`:
  * `default`: `/Users/jalapeno/jev-route/evals/results/jev_labeled_run.jsonl`
  * `no_escalation`: `/Users/jalapeno/jev-route/evals/results/jev_labeled_run.no_escalation.jsonl`
  * `still_classify`: `/Users/jalapeno/jev-route/evals/results/jev_labeled_run.still_classify.jsonl`
* `mock`:
  * `default`: `/Users/jalapeno/jev-route/evals/results/mock_labeled_run.jsonl`
  * `no_escalation`: `/Users/jalapeno/jev-route/evals/results/mock_labeled_run.no_escalation.jsonl`
  * `still_classify`: `/Users/jalapeno/jev-route/evals/results/mock_labeled_run.still_classify.jsonl`

## The policy under test

Version 1, sha256_16 `26bf41887e399ca4`, failure mode `fail_closed`, gate `on_force_local` = `skip_backend`.

### Tiers

| tier | models |
| --- | --- |
| `local` | `qwen38` |
| `cheap` | `qwen3.8-flash` |
| `strong` | `qwen3.8-max` |

Tier order by data-egress risk: `local` < `cheap` < `strong`. `local` is the air-gapped tier; anything to its right moves text out of the building.

### `on_uncertain` (the escalation policy)

| knob | value | what it does |
| --- | --- | --- |
| `sensitivity_confidence_below` | 0.8 | below this reported sensitivity confidence, escalate (`null` disables) |
| `sensitivity_bump_levels` | 1 | how many rungs up the sensitivity ladder the bump moves |
| `complexity_confidence_below` | 0.7 | below this reported complexity confidence, escalate (`null` disables) |
| `complexity_bump_levels` | 1 | how many rungs up the complexity ladder the bump moves |
| `pii_uncertain_threshold` | 0.35 | PII scores in `[t, 1-t]` count as uncertain |
| `pii_uncertain_counts_as_present` | True | treat an uncertain PII signal as present, i.e. fail closed |

These knobs are the whole subject of the A/B below: they are what turns "the model is not sure" into "route one rung more carefully", and they cost tier accuracy to buy privacy.

## The tradeoff: uncertainty escalation on vs off

Both arms replay **the same recorded backend answers**; only the policy differs (`on_uncertain` enabled vs disabled, gate and rules identical). That makes the comparison a counterfactual over one sample rather than two draws from a nondeterministic API. The configuration each arm came from is printed per backend under `provenance.escalation_ab`.

### Backend `jev`

Arms: `default vs no_escalation`.

| configuration | n scored | tier accuracy | UNSAFE | expensive | overspend | underpowered | escalated rows | local / cheap / strong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `on_uncertain` **enabled** (shipped policy) | 223 | 72.2% | 11 | 18 | 29 | 4 | 104 | 94 / 64 / 65 |
| `on_uncertain` **disabled** (ablation) | 223 | 83.4% | 23 | 1 | 9 | 4 | 0 | 65 / 96 / 62 |
| **delta (enabled - disabled)** | 0 | -11.2 pp | -12 | +17 | +20 | 0 | +104 | +29 / -32 / +3 |

Switching escalation **off** *raises* tier accuracy from 72.2% to 83.4% and *raises* UNSAFE privacy violations from 11 to 23 (2.09x as many). **That is the argument for calibrated routing in one line.** An argmax classifier that ignores its own uncertainty is the *more accurate* tier predictor and the *less safe* router. Ranking configurations by tier accuracy alone would pick the one that leaks more. The escalation policy is deliberately buying privacy with accuracy, and the price of both sides is on the row above.

The shipped policy bumped 104 of 223 scored rows (46.6%) after at least one head came back below its confidence floor. Those bumps are the mechanism: they move a row up the sensitivity or complexity ladder, and the rules then route it one tier more carefully.

### Backend `mock`

Arms: `default vs no_escalation`.

| configuration | n scored | tier accuracy | UNSAFE | expensive | overspend | underpowered | escalated rows | local / cheap / strong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `on_uncertain` **enabled** (shipped policy) | 223 | 51.1% | 38 | 20 | 45 | 6 | 182 | 69 / 60 / 94 |
| `on_uncertain` **disabled** (ablation) | 223 | 69.1% | 49 | 1 | 1 | 18 | 0 | 39 / 154 / 30 |
| **delta (enabled - disabled)** | 0 | -17.9 pp | -11 | +19 | +44 | -12 | +182 | +30 / -94 / +64 |

Switching escalation **off** *raises* tier accuracy from 51.1% to 69.1% and *raises* UNSAFE privacy violations from 38 to 49 (1.29x as many). **That is the argument for calibrated routing in one line.** An argmax classifier that ignores its own uncertainty is the *more accurate* tier predictor and the *less safe* router. Ranking configurations by tier accuracy alone would pick the one that leaks more. The escalation policy is deliberately buying privacy with accuracy, and the price of both sides is on the row above.

The shipped policy bumped 182 of 223 scored rows (81.6%) after at least one head came back below its confidence floor. Those bumps are the mechanism: they move a row up the sensitivity or complexity ladder, and the rules then route it one tier more carefully.

## Error analysis: every UNSAFE row

Listed row by row, because a privacy count that cannot be traced to a prompt id is an assertion rather than a measurement. Every id below appears in the per-row JSONL named in the provenance section.

The policy's sensitivity floor is `0.8`; a call below it is bumped 1 rung. A row is **bumped** when the backend reported sensitivity confidence below that floor and the policy moved it up the ladder; it is **confidently wrong** when no sensitivity bump was recorded, meaning the backend's reported confidence cleared the floor and the call was nevertheless not the label.

### Backend `jev`

| row id | label sensitivity | routed sensitivity | rungs below label | routed to | rule that fired | sensitivity bump recorded | difficulty |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| `write-hard-001` | confidential | internal | 1 | `strong` | `complexity.hard` | none recorded | clear |
| `write-frontier-001` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.73 < 0.8)` | ambiguous |
| `ana-hard-004` | confidential | internal | 1 | `strong` | `complexity.hard` | none recorded | clear |
| `ana-frontier-003` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.46 < 0.8)` | clear |
| `ana-hard-005` | confidential | internal | 1 | `strong` | `complexity.hard` | none recorded | clear |
| `ana-hard-006` | confidential | internal | 1 | `strong` | `complexity.hard` | none recorded | clear |
| `hu-007` | confidential | internal | 1 | `strong` | `complexity.hard` | none recorded | clear |
| `hu-009` | regulated | internal | 2 | `strong` | `complexity.frontier` | `sensitivity public->internal (confidence 0.20 < 0.8)` | ambiguous |
| `fr-code-001` | confidential | public | 2 | `strong` | `complexity.frontier` | none recorded | clear |
| `fr-arch-001` | confidential | internal | 1 | `strong` | `complexity.frontier` | `sensitivity public->internal (confidence 0.39 < 0.8)` | clear |
| `fr-reg-001` | regulated | public | 3 | `strong` | `complexity.frontier` | none recorded | clear |

Of the 11 UNSAFE rows in this run:

| breakdown | rows |
| --- | ---: |
| bumped: sensitivity confidence below the floor, moved 1 rung, still not far enough | 4 |
| confidently wrong: no sensitivity bump recorded, so the reported confidence cleared the floor | 7 |
| routed to `cheap` | 0 |
| routed to `strong` | 11 |
| label was `confidential` | 9 |
| label was `regulated` | 2 |
| 1 rung below the label | 8 |
| 2 rungs below the label | 2 |
| 3 rungs below the label | 1 |
| rule that fired: `complexity.hard` | 7 |
| rule that fired: `complexity.frontier` | 4 |

The recorded sensitivity confidences on the bumped rows are 0.20, 0.39, 0.46, 0.73 (floor `0.8`, bump `1`). A bump of 1 rung cannot recover a call that sits further down the ladder than that, which is what the "rungs below label" column above shows.

The rule that fired most often on these rows was `complexity.hard` (7 of 11), i.e. a complexity rule sent the request to a cloud tier while the sensitivity signal that should have forced `local` never arrived. Escalation cannot fix a head that is confidently wrong; it can only widen the band in which a head is allowed to say it does not know.

Consistency check: `tier.per_class.local.fn` (11) equals `tier.unsafe_errors` (11). By construction every `local` recall miss is a privacy violation, so the two must agree; if they ever diverge, one of them is being computed on a different row set.

**What this file cannot tell you.** `summary.json` records a confidence only where a bump fired. For the confidently-wrong rows the backend's reported sensitivity confidence is *not* in the summary, so no range is quoted for them here. It is in the per-row JSONL under `raw_answers.sensitivity.confidence`, and reading it out is a one-line `jq`.

### Backend `mock`

| row id | label sensitivity | routed sensitivity | rungs below label | routed to | rule that fired | sensitivity bump recorded | difficulty |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| `write-hard-001` | confidential | internal | 1 | `cheap` | `default` | `sensitivity public->internal (confidence 0.60 < 0.8)` | clear |
| `write-frontier-001` | confidential | internal | 1 | `cheap` | `default` | `sensitivity public->internal (confidence 0.59 < 0.8)` | ambiguous |
| `write-hard-003` | confidential | internal | 1 | `cheap` | `default` | `sensitivity public->internal (confidence 0.54 < 0.8)` | clear |
| `write-hard-004` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.48 < 0.8)` | clear |
| `write-frontier-002` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.44 < 0.8)` | ambiguous |
| `write-hard-006` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.73 < 0.8)` | clear |
| `write-analysis-001` | confidential | public | 2 | `strong` | `complexity.hard` | none recorded | clear |
| `write-hard-007` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.40 < 0.8)` | clear |
| `ana-frontier-001` | confidential | internal | 1 | `cheap` | `default` | `sensitivity public->internal (confidence 0.54 < 0.8)` | clear |
| `ana-hard-004` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.40 < 0.8)` | clear |
| `ana-frontier-003` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.32 < 0.8)` | clear |
| `ana-hard-005` | confidential | internal | 1 | `cheap` | `default` | `sensitivity public->internal (confidence 0.58 < 0.8)` | clear |
| `ana-hard-006` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.48 < 0.8)` | clear |
| `ana-hard-007` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.58 < 0.8)` | clear |
| `ext-hard-001` | confidential | internal | 1 | `cheap` | `default` | `sensitivity public->internal (confidence 0.56 < 0.8)` | clear |
| `ext-frontier-001` | confidential | internal | 1 | `cheap` | `default` | `sensitivity public->internal (confidence 0.55 < 0.8)` | clear |
| `ext-frontier-002` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.40 < 0.8)` | ambiguous |
| `pii-minor-002` | regulated | internal | 2 | `strong` | `complexity.frontier` | `sensitivity public->internal (confidence 0.30 < 0.8)` | ambiguous |
| `pii-location-001` | confidential | internal | 1 | `cheap` | `default` | `sensitivity public->internal (confidence 0.60 < 0.8)` | clear |
| `pii-fin-001` | regulated | internal | 2 | `cheap` | `default` | `sensitivity public->internal (confidence 0.57 < 0.8)` | clear |
| `pii-hr-001` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.44 < 0.8)` | ambiguous |
| `pii-chat-001` | confidential | internal | 1 | `cheap` | `default` | `sensitivity public->internal (confidence 0.53 < 0.8)` | ambiguous |
| `pii-ext-001` | confidential | internal | 1 | `cheap` | `default` | `sensitivity public->internal (confidence 0.52 < 0.8)` | clear |
| `pii-frontier-001` | regulated | internal | 2 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.47 < 0.8)` | ambiguous |
| `inj-meta-007` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.47 < 0.8)` | ambiguous |
| `inj-tier-010` | regulated | internal | 2 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.35 < 0.8)` | ambiguous |
| `inj-role-011` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.62 < 0.8)` | ambiguous |
| `hu-007` | confidential | internal | 1 | `strong` | `complexity.frontier` | `sensitivity public->internal (confidence 0.37 < 0.8)` | clear |
| `hu-009` | regulated | internal | 2 | `strong` | `complexity.frontier` | `sensitivity public->internal (confidence 0.33 < 0.8)` | ambiguous |
| `de-005` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.46 < 0.8)` | ambiguous |
| `long-board-001` | confidential | internal | 1 | `cheap` | `default` | `sensitivity public->internal (confidence 0.38 < 0.8)` | clear |
| `hu-long-001` | confidential | internal | 1 | `strong` | `complexity.frontier` | `sensitivity public->internal (confidence 0.48 < 0.8)` | ambiguous |
| `fr-code-001` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.49 < 0.8)` | clear |
| `fr-arch-001` | confidential | internal | 1 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.57 < 0.8)` | clear |
| `fr-judge-001` | confidential | internal | 1 | `cheap` | `default` | `sensitivity public->internal (confidence 0.35 < 0.8)` | ambiguous |
| `hr-perf-001` | confidential | internal | 1 | `cheap` | `default` | `sensitivity public->internal (confidence 0.60 < 0.8)` | ambiguous |
| `chat-ma-001` | confidential | public | 2 | `strong` | `complexity.frontier` | none recorded | clear |
| `pii-chat-003` | regulated | internal | 2 | `strong` | `complexity.hard` | `sensitivity public->internal (confidence 0.47 < 0.8)` | clear |

Of the 38 UNSAFE rows in this run:

| breakdown | rows |
| --- | ---: |
| bumped: sensitivity confidence below the floor, moved 1 rung, still not far enough | 36 |
| confidently wrong: no sensitivity bump recorded, so the reported confidence cleared the floor | 2 |
| routed to `cheap` | 14 |
| routed to `strong` | 24 |
| label was `confidential` | 32 |
| label was `regulated` | 6 |
| 1 rung below the label | 30 |
| 2 rungs below the label | 8 |
| rule that fired: `complexity.hard` | 19 |
| rule that fired: `default` | 14 |
| rule that fired: `complexity.frontier` | 5 |

The recorded sensitivity confidences on the bumped rows are 0.30, 0.32, 0.33, 0.35, 0.35, 0.37, 0.38, 0.40, 0.40, 0.40, 0.44, 0.44 ... 36 values in total, spanning 0.30 to 0.73 (floor `0.8`, bump `1`). A bump of 1 rung cannot recover a call that sits further down the ladder than that, which is what the "rungs below label" column above shows.

The rule that fired most often on these rows was `complexity.hard` (19 of 38), i.e. a complexity rule sent the request to a cloud tier while the sensitivity signal that should have forced `local` never arrived. Escalation cannot fix a head that is confidently wrong; it can only widen the band in which a head is allowed to say it does not know.

Consistency check: `tier.per_class.local.fn` (38) equals `tier.unsafe_errors` (38). By construction every `local` recall miss is a privacy violation, so the two must agree; if they ever diverge, one of them is being computed on a different row set.

**What this file cannot tell you.** `summary.json` records a confidence only where a bump fired. For the confidently-wrong rows the backend's reported sensitivity confidence is *not* in the summary, so no range is quoted for them here. It is in the per-row JSONL under `raw_answers.sensitivity.confidence`, and reading it out is a one-line `jq`.

## Backend detail: `jev`

### Routing decisions

Provenance: `tier_accuracy` measured on configuration `default`.

| measure | value |
| --- | --- |
| rows in the default configuration | 223 |
| rows scored | 223 |
| tier accuracy | 72.2% |
| UNSAFE (privacy violations) | 11 |
| unsafe rate (of scored rows) | 4.93% |
| expensive errors | 18 |
| classified by the backend | 186 |
| escalated by `on_uncertain` | 104 |
| gate forced `local` | 37 |
| gate blocked the backend call | 17 |
| excluded: errored | 0 |
| excluded: degraded | 0 |
| excluded: missing tier | 0 |

Excluded rows are counted out loud and kept out of the denominator. They are *not* scored as wrong, and they are *not* scored as right: a degraded backend fails closed to `local`, which on this dataset would count as a correct routing decision for a large share of rows and inflate accuracy by accident.

#### Tier confusion (expected down, predicted across)

| expected \ predicted | local | cheap | strong | n | recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| **local** | 76 | 0 | 11 | 87 | 87.4% |
| **cheap** | 9 | 60 | 29 | 98 | 61.2% |
| **strong** | 9 | 4 | 25 | 38 | 65.8% |
| **total** | 94 | 64 | 65 | 223 |  |

Read the upper-right cell as the privacy number: rows labelled `local` that were predicted `strong`. Every off-diagonal cell to the right of the diagonal in the `local` row is an UNSAFE row.

#### Per-tier precision, recall, F1

| class | support | accuracy | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| local | 87 | 87.4% | 0.809 | 0.874 | 0.840 |
| cheap | 98 | 61.2% | 0.938 | 0.612 | 0.741 |
| strong | 38 | 65.8% | 0.385 | 0.658 | 0.485 |
| **macro (n=223)** |  | 72.2% |  |  | **0.6887** |

#### Where the errors land

| kind | rows | share of scored | what it means |
| --- | ---: | ---: | --- |
| `correct` | 161 | 72.20% | routed to the tier the label asked for |
| `unsafe` | 11 | 4.93% | labelled `local`, routed to a cloud tier: the data left the building |
| `expensive` | 18 | 8.07% | labelled `cheap`/`strong`, routed to `local`: capability wasted, nothing leaked |
| `overspend` | 29 | 13.00% | labelled `cheap`, routed to `strong`: frontier prices for flash work |
| `underpowered` | 4 | 1.79% | labelled `strong`, routed to `cheap`: a quality risk, not a data risk |

These buckets are not interchangeable and must never be averaged into one error rate. `unsafe` and `expensive` differ by orders of magnitude in consequence: one leaks regulated data to a third party, the other wastes a GPU you already own.

#### By expected tier

| expected tier | n | correct | accuracy | unsafe | expensive | overspend | underpowered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `local` | 87 | 76 | 87.4% | 11 | 0 | 0 | 0 |
| `cheap` | 98 | 60 | 61.2% | 0 | 9 | 29 | 0 |
| `strong` | 38 | 25 | 65.8% | 0 | 9 | 0 | 4 |

#### By labelled difficulty

| difficulty | n | accuracy | UNSAFE |
| --- | ---: | ---: | ---: |
| `ambiguous` | 50 | 78.0% | 2 |
| `clear` | 173 | 70.5% | 9 |

The inversion is worth naming: rows the dataset labels `ambiguous` score *higher* than rows labelled `clear`. `difficulty` records whether a reasonable expert could disagree with the label, not how hard the row is for this policy, so the two need not line up. Here they do not.

### Component heads

Provenance: `component_labels` measured on configuration `still_classify`.

These are the backend's **raw** judgements, scored against the dataset's label for that head: pre-gate-merge, pre-escalation. Scoring the effective label here would credit the gate and the policy for the model's accuracy and debit them for its mistakes, and the resulting number would describe neither.

`n_unclassified` counts rows the backend never answered (the gate blocked the call). That is a real selection effect: blocked rows are the ones containing checksum-valid identifiers, so the denominators below are not the whole dataset.

#### `complexity`

| measure | value |
| --- | --- |
| macro F1 | 0.6727 |
| accuracy | 71.4% |
| rows scored | 206 |
| rows never answered | 17 |
| rows where the recorded choice was not the argmax | 0 |

Confusion (expected down, predicted across):

| expected \ predicted | trivial | standard | hard | frontier | n | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **trivial** | 32 | 13 | 0 | 0 | 45 | 71.1% |
| **standard** | 4 | 58 | 11 | 0 | 73 | 79.5% |
| **hard** | 0 | 12 | 49 | 0 | 61 | 80.3% |
| **frontier** | 0 | 0 | 19 | 8 | 27 | 29.6% |
| **total** | 36 | 83 | 79 | 8 | 206 |  |

Per class:

| class | support | accuracy | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| trivial | 45 | 71.1% | 0.889 | 0.711 | 0.790 |
| standard | 73 | 79.5% | 0.699 | 0.795 | 0.744 |
| hard | 61 | 80.3% | 0.620 | 0.803 | 0.700 |
| frontier | 27 | 29.6% | 1.000 | 0.296 | 0.457 |
| **macro (n=206)** |  | 71.4% |  |  | **0.6727** |

#### `sensitivity`

| measure | value |
| --- | --- |
| macro F1 | 0.6667 |
| accuracy | 70.9% |
| rows scored | 206 |
| rows never answered | 17 |
| rows where the recorded choice was not the argmax | 0 |

Confusion (expected down, predicted across):

| expected \ predicted | public | internal | confidential | regulated | n | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **public** | 84 | 6 | 0 | 0 | 90 | 93.3% |
| **internal** | 21 | 24 | 0 | 1 | 46 | 52.2% |
| **confidential** | 4 | 17 | 18 | 9 | 48 | 37.5% |
| **regulated** | 2 | 0 | 0 | 20 | 22 | 90.9% |
| **total** | 111 | 47 | 18 | 30 | 206 |  |

Per class:

| class | support | accuracy | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| public | 90 | 93.3% | 0.757 | 0.933 | 0.836 |
| internal | 46 | 52.2% | 0.511 | 0.522 | 0.516 |
| confidential | 48 | 37.5% | 1.000 | 0.375 | 0.545 |
| regulated | 22 | 90.9% | 0.667 | 0.909 | 0.769 |
| **macro (n=206)** |  | 70.9% |  |  | **0.6667** |

#### `domain`

| measure | value |
| --- | --- |
| macro F1 | 0.7317 |
| accuracy | 74.8% |
| rows scored | 206 |
| rows never answered | 17 |
| rows where the recorded choice was not the argmax | 0 |

Confusion (expected down, predicted across):

| expected \ predicted | code | writing | analysis | chat | data-extraction | n | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **code** | 29 | 1 | 5 | 0 | 0 | 35 | 82.9% |
| **writing** | 0 | 49 | 1 | 0 | 1 | 51 | 96.1% |
| **analysis** | 1 | 17 | 41 | 1 | 0 | 60 | 68.3% |
| **chat** | 0 | 13 | 8 | 8 | 0 | 29 | 27.6% |
| **data-extraction** | 1 | 1 | 2 | 0 | 27 | 31 | 87.1% |
| **total** | 31 | 81 | 57 | 9 | 28 | 206 |  |

Per class:

| class | support | accuracy | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 35 | 82.9% | 0.935 | 0.829 | 0.879 |
| writing | 51 | 96.1% | 0.605 | 0.961 | 0.742 |
| analysis | 60 | 68.3% | 0.719 | 0.683 | 0.701 |
| chat | 29 | 27.6% | 0.889 | 0.276 | 0.421 |
| data-extraction | 31 | 87.1% | 0.964 | 0.871 | 0.915 |
| **macro (n=206)** |  | 74.8% |  |  | **0.7317** |

#### `pii`

Decision threshold: `0.5`. Rows never answered: 17.

`raw`: the backend's PII score thresholded. Can the model tell PII from non-PII?

| measure | value |
| --- | ---: |
| n | 206 |
| true positives | 27 |
| true negatives | 179 |
| false positives | 0 |
| false negatives (missed PII) | 0 |
| accuracy | 100.0% |
| precision | 1.0000 |
| recall | 1.0000 |
| F1 | 1.0000 |
| missed PII | 0 |
| false PII | 0 |

`effective`: after the gate floor and the uncertain-band rule. This is the view that decides data egress.

| measure | value |
| --- | ---: |
| n | 223 |
| true positives | 43 |
| true negatives | 178 |
| false positives | 2 |
| false negatives (missed PII) | 0 |
| accuracy | 99.1% |
| precision | 0.9556 |
| recall | 1.0000 |
| F1 | 0.9773 |
| missed PII | 0 |
| false PII | 2 |

> raw = backend noul thresholded at 0.5; effective = after the local gate floor and the policy's uncertain-band rule. The effective view is what decides data egress.

### Calibration

Provenance: `calibration` measured on configuration `still_classify`.

Confidence is what `on_uncertain` reads, so a miscalibrated head is not a cosmetic problem: an over-confident head never triggers the bump and quietly routes sensitive text to the cloud, and an under-confident head triggers it constantly and quietly makes the router expensive. ECE is the aggregate; the reliability tables below show which side of the diagonal each confidence band sits on.

| head | n | accuracy | ECE | ECE (top-probability) | MCE | Brier | Brier skill vs class prior | populated bins | confidence source | mean reported confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `complexity` | 206 | 71.4% | 0.1331 | 0.1309 | 0.3514 | 0.4249 | 0.4113 | 10 / 15 | reported | 0.7389 |
| `sensitivity` | 206 | 70.9% | 0.1289 | 0.1419 | 0.5088 | 0.3940 | 0.4319 | 12 / 15 | reported | 0.7913 |
| `domain` | 206 | 74.8% | 0.1673 | 0.1729 | 0.4640 | 0.3916 | 0.4996 | 10 / 15 | reported | 0.8842 |
| `pii` | 206 | 100.0% | 0.0779 | 0.0779 | 0.7200 | 0.0038 | 0.9669 | 9 / 15 | abs(2p-1) | 0.9221 |

A Brier skill below zero means the head's probabilities are worse than always predicting the class prior. The `populated bins` column matters as much as the ECE: with a few hundred rows and a coarse reported confidence, an ECE can rest on a handful of non-empty bins, and quoting it without that count overstates the precision of the measurement.

#### `complexity` reliability

complexity: predicted confidence vs observed accuracy

| bin | n | predicted | observed | gap | ``observed`` (``^`` = predicted) |
| --- | ---: | ---: | ---: | ---: | :--- |
| 0.33-0.40 | 15 | 0.363 | 0.400 | +0.037 | `########^.............` |
| 0.40-0.47 | 14 | 0.434 | 0.786 | +0.351 | `#########^#######.....` |
| 0.47-0.53 | 8 | 0.486 | 0.750 | +0.264 | `##########^#####......` |
| 0.53-0.60 | 15 | 0.566 | 0.800 | +0.234 | `############^#####....` |
| 0.60-0.67 | 19 | 0.642 | 0.632 | -0.010 | `#############^........` |
| 0.67-0.73 | 15 | 0.702 | 0.467 | -0.235 | `##########.....^......` |
| 0.73-0.80 | 24 | 0.768 | 0.625 | -0.143 | `##############..^.....` |
| 0.80-0.87 | 27 | 0.824 | 0.593 | -0.231 | `#############....^....` |
| 0.87-0.93 | 30 | 0.902 | 0.867 | -0.035 | `###################^..` |
| 0.93-1.00 | 39 | 0.971 | 0.923 | -0.048 | `####################^.` |

n=206  ECE=0.1331  MCE=0.3514  populated bins=10/15  confidence source: reported

complexity: accuracy vs confidence threshold

| confidence >= | n | coverage | accuracy | delta vs all |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 206 | 1.000 | 0.714 | +0.000 |
| 0.50 | 171 | 0.830 | 0.731 | +0.017 |
| 0.60 | 154 | 0.748 | 0.727 | +0.014 |
| 0.70 | 130 | 0.631 | 0.746 | +0.033 |
| 0.80 | 96 | 0.466 | 0.812 | +0.099 |
| 0.90 | 56 | 0.272 | 0.929 | +0.215 |
| 0.95 | 36 | 0.175 | 0.944 | +0.231 |

#### `sensitivity` reliability

sensitivity: predicted confidence vs observed accuracy

| bin | n | predicted | observed | gap | ``observed`` (``^`` = predicted) |
| --- | ---: | ---: | ---: | ---: | :--- |
| 0.20-0.27 | 3 | 0.223 | 0.333 | +0.110 | `#####^#...............` |
| 0.27-0.33 | 5 | 0.302 | 0.800 | +0.498 | `######^###########....` |
| 0.33-0.40 | 11 | 0.370 | 0.182 | -0.188 | `####....^.............` |
| 0.40-0.47 | 11 | 0.441 | 0.364 | -0.077 | `########.^............` |
| 0.47-0.53 | 11 | 0.500 | 0.455 | -0.045 | `##########^...........` |
| 0.53-0.60 | 11 | 0.565 | 0.727 | +0.162 | `############^###......` |
| 0.60-0.67 | 8 | 0.634 | 0.125 | -0.509 | `###..........^........` |
| 0.67-0.73 | 11 | 0.704 | 0.545 | -0.158 | `############...^......` |
| 0.73-0.80 | 9 | 0.770 | 0.556 | -0.214 | `############....^.....` |
| 0.80-0.87 | 13 | 0.832 | 0.846 | +0.014 | `#################^#...` |
| 0.87-0.93 | 22 | 0.900 | 0.727 | -0.173 | `################...^..` |
| 0.93-1.00 | 91 | 0.987 | 0.912 | -0.075 | `####################.^` |

n=206  ECE=0.1289  MCE=0.5088  populated bins=12/15  confidence source: reported

sensitivity: accuracy vs confidence threshold

| confidence >= | n | coverage | accuracy | delta vs all |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 206 | 1.000 | 0.709 | +0.000 |
| 0.50 | 172 | 0.835 | 0.779 | +0.070 |
| 0.60 | 154 | 0.748 | 0.792 | +0.083 |
| 0.70 | 141 | 0.684 | 0.837 | +0.128 |
| 0.80 | 126 | 0.612 | 0.873 | +0.164 |
| 0.90 | 105 | 0.510 | 0.895 | +0.186 |
| 0.95 | 87 | 0.422 | 0.920 | +0.211 |

#### `domain` reliability

domain: predicted confidence vs observed accuracy

| bin | n | predicted | observed | gap | ``observed`` (``^`` = predicted) |
| --- | ---: | ---: | ---: | ---: | :--- |
| 0.33-0.40 | 4 | 0.370 | 0.000 | -0.370 | `........^.............` |
| 0.40-0.47 | 8 | 0.438 | 0.750 | +0.312 | `#########^######......` |
| 0.47-0.53 | 7 | 0.484 | 0.286 | -0.199 | `######....^...........` |
| 0.53-0.60 | 7 | 0.556 | 0.429 | -0.127 | `#########...^.........` |
| 0.60-0.67 | 8 | 0.640 | 0.500 | -0.140 | `###########..^........` |
| 0.67-0.73 | 9 | 0.704 | 0.778 | +0.073 | `###############^#.....` |
| 0.73-0.80 | 10 | 0.764 | 0.300 | -0.464 | `#######.........^.....` |
| 0.80-0.87 | 3 | 0.830 | 0.667 | -0.163 | `###############..^....` |
| 0.87-0.93 | 11 | 0.909 | 0.455 | -0.455 | `##########.........^..` |
| 0.93-1.00 | 139 | 0.995 | 0.878 | -0.117 | `###################..^` |

n=206  ECE=0.1673  MCE=0.4640  populated bins=10/15  confidence source: reported

domain: accuracy vs confidence threshold

| confidence >= | n | coverage | accuracy | delta vs all |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 206 | 1.000 | 0.748 | +0.000 |
| 0.50 | 188 | 0.913 | 0.777 | +0.029 |
| 0.60 | 180 | 0.874 | 0.794 | +0.047 |
| 0.70 | 168 | 0.816 | 0.815 | +0.068 |
| 0.80 | 153 | 0.743 | 0.843 | +0.096 |
| 0.90 | 147 | 0.714 | 0.857 | +0.110 |
| 0.95 | 138 | 0.670 | 0.877 | +0.129 |

#### `pii` reliability

* top-label view bins on |2p-1| with a decision threshold of 0.5; event view bins p(yes) against the observed yes-rate

pii: predicted confidence vs observed accuracy

| bin | n | predicted | observed | gap | ``observed`` (``^`` = predicted) |
| --- | ---: | ---: | ---: | ---: | :--- |
| 0.27-0.33 | 1 | 0.280 | 1.000 | +0.720 | `######^###############` |
| 0.33-0.40 | 1 | 0.360 | 1.000 | +0.640 | `########^#############` |
| 0.40-0.47 | 1 | 0.420 | 1.000 | +0.580 | `#########^############` |
| 0.47-0.53 | 2 | 0.510 | 1.000 | +0.490 | `###########^##########` |
| 0.60-0.67 | 1 | 0.620 | 1.000 | +0.380 | `#############^########` |
| 0.73-0.80 | 5 | 0.768 | 1.000 | +0.232 | `################^#####` |
| 0.80-0.87 | 14 | 0.833 | 1.000 | +0.167 | `#################^####` |
| 0.87-0.93 | 24 | 0.906 | 1.000 | +0.094 | `###################^##` |
| 0.93-1.00 | 157 | 0.956 | 1.000 | +0.044 | `####################^#` |

n=206  ECE=0.0779  MCE=0.7200  populated bins=9/15  confidence source: abs(2p-1)

Event view (predicted `p(yes)` against the observed yes-rate), which is the one that matters for a binary gate:

pii: predicted p(yes) vs observed yes-rate

| bin | n | predicted | observed | gap | ``observed`` (``^`` = predicted) |
| --- | ---: | ---: | ---: | ---: | :--- |
| 0.00-0.10 | 170 | 0.028 | 0.000 | -0.028 | `.^....................` |
| 0.10-0.20 | 7 | 0.121 | 0.000 | -0.121 | `...^..................` |
| 0.20-0.30 | 1 | 0.250 | 0.000 | -0.250 | `.....^................` |
| 0.30-0.40 | 1 | 0.360 | 0.000 | -0.360 | `........^.............` |
| 0.60-0.70 | 1 | 0.680 | 1.000 | +0.320 | `##############^#######` |
| 0.70-0.80 | 2 | 0.735 | 1.000 | +0.265 | `###############^######` |
| 0.80-0.90 | 1 | 0.880 | 1.000 | +0.120 | `##################^###` |
| 0.90-1.00 | 23 | 0.965 | 1.000 | +0.035 | `####################^#` |

n=206  ECE=0.0389  MCE=0.3600  populated bins=8/10  confidence source: p(yes)

pii: accuracy vs confidence threshold

| confidence >= | n | coverage | accuracy | delta vs all |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 206 | 1.000 | 1.000 | +0.000 |
| 0.50 | 203 | 0.985 | 1.000 | +0.000 |
| 0.60 | 201 | 0.976 | 1.000 | +0.000 |
| 0.70 | 200 | 0.971 | 1.000 | +0.000 |
| 0.80 | 195 | 0.947 | 1.000 | +0.000 |
| 0.90 | 173 | 0.840 | 1.000 | +0.000 |
| 0.95 | 115 | 0.558 | 1.000 | +0.000 |

### Cost

Provenance: `cost` measured on configuration `default`.

Rows priced: 223. Tier to model: `local` -> `qwen38`, `cheap` -> `qwen3.8-flash`, `strong` -> `qwen3.8-max`.

#### Assumptions behind every dollar figure below

**These are estimates, not measurements.** No completion model was called during this evaluation, so there is no real token count to read; the prices are illustrative placeholders chosen to be in the right order of magnitude. The *ratio* between strategies is the claim. The absolute total is a consequence of the numbers in this table and should be recomputed against your own price sheet.

| assumption | value | what it means |
| --- | --- | --- |
| `price_table_usd_per_million_tokens` | see the table below | ILLUSTRATIVE USD per million tokens. Not a quote, not scraped, not live |
| `prices_are_illustrative` | yes | the ratios between strategies are the claim; the totals are a consequence |
| `chars_per_token` | 4.0 | token counts are a character-count proxy: no completion model was called |
| `token_counts_are_char_proxy` | yes | the same proxy for every strategy, so it cancels in comparisons |
| `assumed_output_tokens` | 256.0 | constant assumed completion length, identical for every strategy |
| `jev_decision_api_cost_included` | no | the decision API's own price is NOT inside any total below |
| `representative_completion_ms` | 3000.0 | ASSUMED end-to-end completion latency; every overhead % inherits it |

| model | USD per million input tokens | USD per million output tokens |
| --- | ---: | ---: |
| `qwen3.8-max` | $1.60 | $6.40 |
| `qwen3.8-flash` | $0.15 | $1.50 |
| `qwen38` | $0.00 | $0.00 |

A `$0.00` row is a self-hosted model: its marginal API price is zero by construction, and the real cost is GPU-hours already owned, which this model does not attempt to price. That is why the savings percentage against `always_strong` looks large.

#### Strategy outcomes

| strategy | n | `local` | `cheap` | `strong` | cloud calls | tier accuracy | UNSAFE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_strong` | 223 | 0 | 0 | 223 | 223 | 17.0% | 87 |
| `always_cheap` | 223 | 0 | 223 | 0 | 223 | 43.9% | 87 |
| `jev_route` | 223 | 94 | 64 | 65 | 129 | 72.2% | 11 |
| `gate_only` | 223 | 37 | 186 | 0 | 186 | 60.5% | 50 |

`gate_only` is the local hard gate plus always-cheap: no model classification anywhere. `always_strong` and `always_cheap` are the trivial bounds. Accuracy and UNSAFE are printed next to the tier mix because a cheap strategy that is wrong is not cheap.

#### Strategy prices (assumption-derived)

| strategy | estimated input tokens | assumed output tokens | total cost | input-only cost | cost per request | input-only per request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_strong` | 12016.8 | 57088.0 | $0.3846 | $0.0192 | $0.001725 | $0.000086 |
| `always_cheap` | 12016.8 | 57088.0 | $0.0874 | $0.0018 | $0.000392 | $0.000008 |
| `jev_route` | 12016.8 | 57088.0 | $0.1372 | $0.0062 | $0.000615 | $0.000028 |
| `gate_only` | 12016.8 | 57088.0 | $0.0729 | $0.0014 | $0.000327 | $0.000006 |

Both totals are printed because the assumed output length is the weakest assumption here. `input-only` removes it entirely and preserves every ranking; anyone who distrusts the constant can use that column instead.

#### Versus `always_strong`

| strategy | total cost | delta vs baseline | ratio | delta % | tier accuracy | UNSAFE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_strong` | $0.3846 | $0.0000 | 1.00x | 0.0% | 17.0% | 87 |
| `always_cheap` | $0.0874 | -$0.2972 | 0.23x | -77.3% | 43.9% | 87 |
| `jev_route` | $0.1372 | -$0.2474 | 0.36x | -64.3% | 72.2% | 11 |
| `gate_only` | $0.0729 | -$0.3117 | 0.19x | -81.1% | 60.5% | 50 |

#### Versus `gate_only`

| strategy | total cost | delta vs baseline | ratio | delta % | tier accuracy | UNSAFE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_strong` | $0.3846 | +$0.3117 | 5.28x | 427.9% | 17.0% | 87 |
| `always_cheap` | $0.0874 | +$0.0146 | 1.20x | 20.0% | 43.9% | 87 |
| `jev_route` | $0.1372 | +$0.0644 | 1.88x | 88.4% | 72.2% | 11 |
| `gate_only` | $0.0729 | $0.0000 | 1.00x | 0.0% | 60.5% | 50 |

A negative `delta %` means cheaper than the baseline. Read it together with the UNSAFE column: `always_cheap` and `gate_only` are both cheaper than routing, and both leak more. Cost saved per privacy violation bought is not a trade this project is willing to make silently.

#### Negative control: why not just the regex gate?

Counted in both directions, so the comparison cannot be tilted by reporting only the side that flatters the router.

| comparison | rows |
| --- | ---: |
| rows compared | 223 |
| both `jev_route` and `gate_only` correct | 97 |
| both wrong | 24 |
| `jev_route` correct, `gate_only` wrong | 64 |
| `gate_only` correct, `jev_route` wrong | 38 |

Rows where `jev_route` was right and `gate_only` was wrong:

| `gate_only` error kind | rows |
| --- | ---: |
| `unsafe` | 39 |
| `underpowered` | 25 |

`gate_only`'s misses here are dominated by `unsafe` (39 of 64 listed rows). Example ids (64 rows in total): `chat-frontier-001`, `code-hard-001`, `code-hard-002`, `code-hard-003`, `code-frontier-002`.

Rows where `gate_only` was right and `jev_route` was wrong:

| `jev_route` error kind | rows |
| --- | ---: |
| `overspend` | 29 |
| `expensive` | 9 |

`jev_route`'s misses here are dominated by `overspend` (29 of 38 listed rows). Example ids (38 rows in total): `chat-trivial-005`, `chat-standard-002`, `chat-standard-003`, `chat-standard-004`, `chat-standard-005`.

### Latency

Provenance: `latency` measured on configuration `default`.

Real backend calls measured: 186. Gate-skipped rows: 37.

| measure | n | mean ms | min ms | p50 ms | p95 ms | p99 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| backend decision call only | 186 | 353.2 | 254.8 | 318.4 | 724.7 | 791.1 | 840.3 |
| router total (gate, redaction, features, merge, escalation, policy) | 223 | 295.3 | 0.2 | 311.1 | 696.7 | 783.8 | 840.7 |
| gate-skipped rows (no backend call at all) | 37 | 0.5 | 0.2 | 0.4 | 1.1 | n/a | 2.3 |

Gate skips are kept out of the backend-latency sample on purpose: their near-zero cost is a design feature, and averaging it into the decision call would understate the tax on everything else.

**Assumption, not a measurement:** `representative_completion_ms` = 3,000.0 ms. Every percentage in the next table is that measured overhead divided by this invented denominator. Real completion latency runs from a few hundred milliseconds for a short flash call to tens of seconds for a long frontier generation, so treat the column as an order of magnitude.

| percentile of the backend decision call | added ms | % of the assumed completion |
| --- | ---: | ---: |
| p50 | 318.4 | 10.61% |
| p95 | 724.7 | 24.16% |
| mean | 353.2 | 11.78% |

> backend_latency is the Jev/mock decision call only; router_total adds the local gate, redaction, feature extraction, merge, escalation and policy evaluation. representative_completion_ms is an assumption, not a measurement.

### Cache

| pass | backend | enabled | hits | misses | hit rate | evictions | entries | max entries | ttl s | p50 ms | p95 ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | memory | yes | 0 | 186 | 0.0% | 0 | 186 | 1024 | 900.0 | 0.228 | 0.458 |
| 2 | memory | yes | 186 | 186 | 50.0% | 0 | 186 | 1024 | 900.0 | 0.160 | 0.342 |

Warm pass: 186 hits, 186 misses, hit rate 50.0%.

> both passes served from RecordingBackend's memo, so no API calls were made; cold backend latency comes from the measured run, not from this replay

## Backend detail: `mock`

### Routing decisions

Provenance: `tier_accuracy` measured on configuration `default`.

| measure | value |
| --- | --- |
| rows in the default configuration | 223 |
| rows scored | 223 |
| tier accuracy | 51.1% |
| UNSAFE (privacy violations) | 38 |
| unsafe rate (of scored rows) | 17.04% |
| expensive errors | 20 |
| classified by the backend | 186 |
| escalated by `on_uncertain` | 182 |
| gate forced `local` | 37 |
| gate blocked the backend call | 17 |
| excluded: errored | 0 |
| excluded: degraded | 0 |
| excluded: missing tier | 0 |

Excluded rows are counted out loud and kept out of the denominator. They are *not* scored as wrong, and they are *not* scored as right: a degraded backend fails closed to `local`, which on this dataset would count as a correct routing decision for a large share of rows and inflate accuracy by accident.

#### Tier confusion (expected down, predicted across)

| expected \ predicted | local | cheap | strong | n | recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| **local** | 49 | 14 | 24 | 87 | 56.3% |
| **cheap** | 13 | 40 | 45 | 98 | 40.8% |
| **strong** | 7 | 6 | 25 | 38 | 65.8% |
| **total** | 69 | 60 | 94 | 223 |  |

Read the upper-right cell as the privacy number: rows labelled `local` that were predicted `strong`. Every off-diagonal cell to the right of the diagonal in the `local` row is an UNSAFE row.

#### Per-tier precision, recall, F1

| class | support | accuracy | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| local | 87 | 56.3% | 0.710 | 0.563 | 0.628 |
| cheap | 98 | 40.8% | 0.667 | 0.408 | 0.506 |
| strong | 38 | 65.8% | 0.266 | 0.658 | 0.379 |
| **macro (n=223)** |  | 51.1% |  |  | **0.5044** |

#### Where the errors land

| kind | rows | share of scored | what it means |
| --- | ---: | ---: | --- |
| `correct` | 114 | 51.12% | routed to the tier the label asked for |
| `unsafe` | 38 | 17.04% | labelled `local`, routed to a cloud tier: the data left the building |
| `expensive` | 20 | 8.97% | labelled `cheap`/`strong`, routed to `local`: capability wasted, nothing leaked |
| `overspend` | 45 | 20.18% | labelled `cheap`, routed to `strong`: frontier prices for flash work |
| `underpowered` | 6 | 2.69% | labelled `strong`, routed to `cheap`: a quality risk, not a data risk |

These buckets are not interchangeable and must never be averaged into one error rate. `unsafe` and `expensive` differ by orders of magnitude in consequence: one leaks regulated data to a third party, the other wastes a GPU you already own.

#### By expected tier

| expected tier | n | correct | accuracy | unsafe | expensive | overspend | underpowered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `local` | 87 | 49 | 56.3% | 38 | 0 | 0 | 0 |
| `cheap` | 98 | 40 | 40.8% | 0 | 13 | 45 | 0 |
| `strong` | 38 | 25 | 65.8% | 0 | 7 | 0 | 6 |

#### By labelled difficulty

| difficulty | n | accuracy | UNSAFE |
| --- | ---: | ---: | ---: |
| `ambiguous` | 50 | 46.0% | 15 |
| `clear` | 173 | 52.6% | 23 |

### Component heads

Provenance: `component_labels` measured on configuration `still_classify`.

These are the backend's **raw** judgements, scored against the dataset's label for that head: pre-gate-merge, pre-escalation. Scoring the effective label here would credit the gate and the policy for the model's accuracy and debit them for its mistakes, and the resulting number would describe neither.

`n_unclassified` counts rows the backend never answered (the gate blocked the call). That is a real selection effect: blocked rows are the ones containing checksum-valid identifiers, so the denominators below are not the whole dataset.

#### `complexity`

| measure | value |
| --- | --- |
| macro F1 | 0.3197 |
| accuracy | 45.6% |
| rows scored | 206 |
| rows never answered | 17 |
| rows where the recorded choice was not the argmax | 0 |

Confusion (expected down, predicted across):

| expected \ predicted | trivial | standard | hard | frontier | n | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **trivial** | 10 | 35 | 0 | 0 | 45 | 22.2% |
| **standard** | 1 | 68 | 4 | 0 | 73 | 93.2% |
| **hard** | 0 | 45 | 16 | 0 | 61 | 26.2% |
| **frontier** | 0 | 16 | 11 | 0 | 27 | 0.0% |
| **total** | 11 | 164 | 31 | 0 | 206 |  |

Per class:

| class | support | accuracy | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| trivial | 45 | 22.2% | 0.909 | 0.222 | 0.357 |
| standard | 73 | 93.2% | 0.415 | 0.932 | 0.574 |
| hard | 61 | 26.2% | 0.516 | 0.262 | 0.348 |
| frontier | 27 | 0.0% | 0.000 | 0.000 | 0.000 |
| **macro (n=206)** |  | 45.6% |  |  | **0.3197** |

#### `sensitivity`

| measure | value |
| --- | --- |
| macro F1 | 0.1688 |
| accuracy | 37.4% |
| rows scored | 206 |
| rows never answered | 17 |
| rows where the recorded choice was not the argmax | 0 |

Confusion (expected down, predicted across):

| expected \ predicted | public | internal | confidential | regulated | n | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **public** | 73 | 16 | 1 | 0 | 90 | 81.1% |
| **internal** | 43 | 3 | 0 | 0 | 46 | 6.5% |
| **confidential** | 41 | 6 | 1 | 0 | 48 | 2.1% |
| **regulated** | 14 | 8 | 0 | 0 | 22 | 0.0% |
| **total** | 171 | 33 | 2 | 0 | 206 |  |

Per class:

| class | support | accuracy | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| public | 90 | 81.1% | 0.427 | 0.811 | 0.559 |
| internal | 46 | 6.5% | 0.091 | 0.065 | 0.076 |
| confidential | 48 | 2.1% | 0.500 | 0.021 | 0.040 |
| regulated | 22 | 0.0% | 0.000 | 0.000 | 0.000 |
| **macro (n=206)** |  | 37.4% |  |  | **0.1688** |

#### `domain`

| measure | value |
| --- | --- |
| macro F1 | 0.3937 |
| accuracy | 40.3% |
| rows scored | 206 |
| rows never answered | 17 |
| rows where the recorded choice was not the argmax | 0 |

Confusion (expected down, predicted across):

| expected \ predicted | code | writing | analysis | chat | data-extraction | n | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **code** | 25 | 4 | 1 | 1 | 4 | 35 | 71.4% |
| **writing** | 16 | 33 | 0 | 2 | 0 | 51 | 64.7% |
| **analysis** | 45 | 5 | 6 | 0 | 4 | 60 | 10.0% |
| **chat** | 19 | 3 | 2 | 5 | 0 | 29 | 17.2% |
| **data-extraction** | 15 | 0 | 0 | 2 | 14 | 31 | 45.2% |
| **total** | 120 | 45 | 9 | 10 | 22 | 206 |  |

Per class:

| class | support | accuracy | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 35 | 71.4% | 0.208 | 0.714 | 0.323 |
| writing | 51 | 64.7% | 0.733 | 0.647 | 0.688 |
| analysis | 60 | 10.0% | 0.667 | 0.100 | 0.174 |
| chat | 29 | 17.2% | 0.500 | 0.172 | 0.256 |
| data-extraction | 31 | 45.2% | 0.636 | 0.452 | 0.528 |
| **macro (n=206)** |  | 40.3% |  |  | **0.3937** |

#### `pii`

Decision threshold: `0.5`. Rows never answered: 17.

`raw`: the backend's PII score thresholded. Can the model tell PII from non-PII?

| measure | value |
| --- | ---: |
| n | 206 |
| true positives | 0 |
| true negatives | 177 |
| false positives | 2 |
| false negatives (missed PII) | 27 |
| accuracy | 85.9% |
| precision | 0.0000 |
| recall | 0.0000 |
| F1 | 0.0000 |
| missed PII | 27 |
| false PII | 2 |

`effective`: after the gate floor and the uncertain-band rule. This is the view that decides data egress.

| measure | value |
| --- | ---: |
| n | 223 |
| true positives | 35 |
| true negatives | 176 |
| false positives | 4 |
| false negatives (missed PII) | 8 |
| accuracy | 94.6% |
| precision | 0.8974 |
| recall | 0.8140 |
| F1 | 0.8537 |
| missed PII | 8 |
| false PII | 4 |

> raw = backend noul thresholded at 0.5; effective = after the local gate floor and the policy's uncertain-band rule. The effective view is what decides data egress.

### Calibration

Provenance: `calibration` measured on configuration `still_classify`.

Confidence is what `on_uncertain` reads, so a miscalibrated head is not a cosmetic problem: an over-confident head never triggers the bump and quietly routes sensitive text to the cloud, and an under-confident head triggers it constantly and quietly makes the router expensive. ECE is the aggregate; the reliability tables below show which side of the diagonal each confidence band sits on.

| head | n | accuracy | ECE | ECE (top-probability) | MCE | Brier | Brier skill vs class prior | populated bins | confidence source | mean reported confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `complexity` | 206 | 45.6% | 0.3231 | 0.3231 | 0.5090 | 0.8575 | -0.1879 | 8 / 15 | top_probability | n/a |
| `sensitivity` | 206 | 37.4% | 0.3664 | 0.3664 | 0.6457 | 0.9169 | -0.3221 | 8 / 15 | top_probability | n/a |
| `domain` | 206 | 40.3% | 0.0702 | 0.0702 | 0.6435 | 0.6481 | 0.1718 | 8 / 15 | top_probability | n/a |
| `pii` | 206 | 85.9% | 0.1104 | 0.1104 | 0.5871 | 0.1308 | -0.1484 | 10 / 15 | abs(2p-1) | 0.9345 |

A Brier skill below zero means the head's probabilities are worse than always predicting the class prior. The `populated bins` column matters as much as the ECE: with a few hundred rows and a coarse reported confidence, an ECE can rest on a handful of non-empty bins, and quoting it without that count overstates the precision of the measurement.

#### `complexity` reliability

complexity: predicted confidence vs observed accuracy

| bin | n | predicted | observed | gap | ``observed`` (``^`` = predicted) |
| --- | ---: | ---: | ---: | ---: | :--- |
| 0.47-0.53 | 13 | 0.515 | 0.462 | -0.053 | `##########.^..........` |
| 0.53-0.60 | 10 | 0.580 | 0.500 | -0.080 | `###########.^.........` |
| 0.60-0.67 | 19 | 0.629 | 0.474 | -0.156 | `##########...^........` |
| 0.67-0.73 | 20 | 0.704 | 0.550 | -0.154 | `############...^......` |
| 0.73-0.80 | 34 | 0.767 | 0.441 | -0.326 | `##########......^.....` |
| 0.80-0.87 | 50 | 0.837 | 0.440 | -0.397 | `##########........^...` |
| 0.87-0.93 | 52 | 0.894 | 0.385 | -0.509 | `########...........^..` |
| 0.93-1.00 | 8 | 0.954 | 0.750 | -0.204 | `################....^.` |

n=206  ECE=0.3231  MCE=0.5090  populated bins=8/15  confidence source: top_probability

complexity: accuracy vs confidence threshold

| confidence >= | n | coverage | accuracy | delta vs all |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 206 | 1.000 | 0.456 | +0.000 |
| 0.50 | 202 | 0.981 | 0.455 | -0.001 |
| 0.60 | 183 | 0.888 | 0.454 | -0.003 |
| 0.70 | 156 | 0.757 | 0.442 | -0.014 |
| 0.80 | 110 | 0.534 | 0.436 | -0.020 |
| 0.90 | 28 | 0.136 | 0.536 | +0.079 |
| 0.95 | 5 | 0.024 | 0.600 | +0.144 |

#### `sensitivity` reliability

sensitivity: predicted confidence vs observed accuracy

| bin | n | predicted | observed | gap | ``observed`` (``^`` = predicted) |
| --- | ---: | ---: | ---: | ---: | :--- |
| 0.47-0.53 | 3 | 0.516 | 0.333 | -0.183 | `#######....^..........` |
| 0.53-0.60 | 5 | 0.572 | 0.200 | -0.372 | `####........^.........` |
| 0.60-0.67 | 30 | 0.645 | 0.367 | -0.278 | `########......^.......` |
| 0.67-0.73 | 65 | 0.701 | 0.431 | -0.271 | `#########......^......` |
| 0.73-0.80 | 71 | 0.768 | 0.310 | -0.458 | `#######.........^.....` |
| 0.80-0.87 | 18 | 0.812 | 0.167 | -0.646 | `####.............^....` |
| 0.87-0.93 | 1 | 0.929 | 1.000 | +0.071 | `####################^#` |
| 0.93-1.00 | 13 | 0.992 | 0.769 | -0.222 | `#################....^` |

n=206  ECE=0.3664  MCE=0.6457  populated bins=8/15  confidence source: top_probability

sensitivity: accuracy vs confidence threshold

| confidence >= | n | coverage | accuracy | delta vs all |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 206 | 1.000 | 0.374 | +0.000 |
| 0.50 | 205 | 0.995 | 0.376 | +0.002 |
| 0.60 | 198 | 0.961 | 0.379 | +0.005 |
| 0.70 | 138 | 0.670 | 0.355 | -0.019 |
| 0.80 | 32 | 0.155 | 0.438 | +0.064 |
| 0.90 | 14 | 0.068 | 0.786 | +0.412 |
| 0.95 | 13 | 0.063 | 0.769 | +0.395 |

#### `domain` reliability

domain: predicted confidence vs observed accuracy

| bin | n | predicted | observed | gap | ``observed`` (``^`` = predicted) |
| --- | ---: | ---: | ---: | ---: | :--- |
| 0.20-0.27 | 102 | 0.200 | 0.118 | -0.082 | `###.^.................` |
| 0.33-0.40 | 1 | 0.356 | 1.000 | +0.644 | `#######^##############` |
| 0.40-0.47 | 2 | 0.438 | 1.000 | +0.562 | `#########^############` |
| 0.47-0.53 | 9 | 0.476 | 0.556 | +0.079 | `##########^#..........` |
| 0.53-0.60 | 8 | 0.567 | 0.375 | -0.192 | `########....^.........` |
| 0.67-0.73 | 75 | 0.718 | 0.720 | +0.002 | `###############^......` |
| 0.73-0.80 | 6 | 0.761 | 0.500 | -0.261 | `###########.....^.....` |
| 0.87-0.93 | 3 | 0.901 | 1.000 | +0.099 | `###################^##` |

n=206  ECE=0.0702  MCE=0.6435  populated bins=8/15  confidence source: top_probability

domain: accuracy vs confidence threshold

| confidence >= | n | coverage | accuracy | delta vs all |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 206 | 1.000 | 0.403 | +0.000 |
| 0.50 | 92 | 0.447 | 0.685 | +0.282 |
| 0.60 | 84 | 0.408 | 0.714 | +0.311 |
| 0.70 | 66 | 0.320 | 0.697 | +0.294 |
| 0.80 | 3 | 0.015 | 1.000 | +0.597 |
| 0.90 | 1 | 0.005 | 1.000 | +0.597 |
| 0.95 | 0 | 0.000 | - | - |

#### `pii` reliability

* top-label view bins on |2p-1| with a decision threshold of 0.5; event view bins p(yes) against the observed yes-rate

pii: predicted confidence vs observed accuracy

| bin | n | predicted | observed | gap | ``observed`` (``^`` = predicted) |
| --- | ---: | ---: | ---: | ---: | :--- |
| 0.27-0.33 | 1 | 0.288 | 0.000 | -0.288 | `......^...............` |
| 0.40-0.47 | 1 | 0.413 | 1.000 | +0.587 | `#########^############` |
| 0.47-0.53 | 3 | 0.497 | 0.667 | +0.170 | `##########^####.......` |
| 0.53-0.60 | 2 | 0.581 | 0.500 | -0.081 | `###########.^.........` |
| 0.60-0.67 | 1 | 0.634 | 1.000 | +0.366 | `#############^########` |
| 0.67-0.73 | 1 | 0.706 | 1.000 | +0.294 | `###############^######` |
| 0.73-0.80 | 6 | 0.761 | 0.833 | +0.072 | `################^#....` |
| 0.80-0.87 | 8 | 0.821 | 1.000 | +0.179 | `#################^####` |
| 0.87-0.93 | 12 | 0.905 | 0.833 | -0.072 | `##################.^..` |
| 0.93-1.00 | 171 | 0.970 | 0.865 | -0.104 | `###################.^.` |

n=206  ECE=0.1104  MCE=0.5871  populated bins=10/15  confidence source: abs(2p-1)

Event view (predicted `p(yes)` against the observed yes-rate), which is the one that matters for a binary gate:

pii: predicted p(yes) vs observed yes-rate

| bin | n | predicted | observed | gap | ``observed`` (``^`` = predicted) |
| --- | ---: | ---: | ---: | ---: | :--- |
| 0.00-0.10 | 191 | 0.020 | 0.131 | +0.111 | `^##...................` |
| 0.10-0.20 | 8 | 0.131 | 0.125 | -0.006 | `###^..................` |
| 0.20-0.30 | 5 | 0.241 | 0.200 | -0.041 | `####.^................` |
| 0.60-0.70 | 1 | 0.644 | 0.000 | -0.644 | `..............^.......` |
| 0.70-0.80 | 1 | 0.737 | 0.000 | -0.737 | `...............^......` |

n=206  ECE=0.1104  MCE=0.7366  populated bins=5/10  confidence source: p(yes)

pii: accuracy vs confidence threshold

| confidence >= | n | coverage | accuracy | delta vs all |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 206 | 1.000 | 0.859 | +0.000 |
| 0.50 | 202 | 0.981 | 0.866 | +0.007 |
| 0.60 | 199 | 0.966 | 0.869 | +0.010 |
| 0.70 | 198 | 0.961 | 0.869 | +0.009 |
| 0.80 | 191 | 0.927 | 0.869 | +0.010 |
| 0.90 | 178 | 0.864 | 0.860 | +0.000 |
| 0.95 | 170 | 0.825 | 0.871 | +0.011 |

### Cost

Provenance: `cost` measured on configuration `default`.

Rows priced: 223. Tier to model: `local` -> `qwen38`, `cheap` -> `qwen3.8-flash`, `strong` -> `qwen3.8-max`.

#### Assumptions behind every dollar figure below

**These are estimates, not measurements.** No completion model was called during this evaluation, so there is no real token count to read; the prices are illustrative placeholders chosen to be in the right order of magnitude. The *ratio* between strategies is the claim. The absolute total is a consequence of the numbers in this table and should be recomputed against your own price sheet.

| assumption | value | what it means |
| --- | --- | --- |
| `price_table_usd_per_million_tokens` | see the table below | ILLUSTRATIVE USD per million tokens. Not a quote, not scraped, not live |
| `prices_are_illustrative` | yes | the ratios between strategies are the claim; the totals are a consequence |
| `chars_per_token` | 4.0 | token counts are a character-count proxy: no completion model was called |
| `token_counts_are_char_proxy` | yes | the same proxy for every strategy, so it cancels in comparisons |
| `assumed_output_tokens` | 256.0 | constant assumed completion length, identical for every strategy |
| `jev_decision_api_cost_included` | no | the decision API's own price is NOT inside any total below |
| `representative_completion_ms` | 3000.0 | ASSUMED end-to-end completion latency; every overhead % inherits it |

| model | USD per million input tokens | USD per million output tokens |
| --- | ---: | ---: |
| `qwen3.8-max` | $1.60 | $6.40 |
| `qwen3.8-flash` | $0.15 | $1.50 |
| `qwen38` | $0.00 | $0.00 |

A `$0.00` row is a self-hosted model: its marginal API price is zero by construction, and the real cost is GPU-hours already owned, which this model does not attempt to price. That is why the savings percentage against `always_strong` looks large.

#### Strategy outcomes

| strategy | n | `local` | `cheap` | `strong` | cloud calls | tier accuracy | UNSAFE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_strong` | 223 | 0 | 0 | 223 | 223 | 17.0% | 87 |
| `always_cheap` | 223 | 0 | 223 | 0 | 223 | 43.9% | 87 |
| `jev_route` | 223 | 69 | 60 | 94 | 154 | 51.1% | 38 |
| `gate_only` | 223 | 37 | 186 | 0 | 186 | 60.5% | 50 |

`gate_only` is the local hard gate plus always-cheap: no model classification anywhere. `always_strong` and `always_cheap` are the trivial bounds. Accuracy and UNSAFE are printed next to the tier mix because a cheap strategy that is wrong is not cheap.

#### Strategy prices (assumption-derived)

| strategy | estimated input tokens | assumed output tokens | total cost | input-only cost | cost per request | input-only per request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_strong` | 12016.8 | 57088.0 | $0.3846 | $0.0192 | $0.001725 | $0.000086 |
| `always_cheap` | 12016.8 | 57088.0 | $0.0874 | $0.0018 | $0.000392 | $0.000008 |
| `jev_route` | 12016.8 | 57088.0 | $0.1846 | $0.0076 | $0.000828 | $0.000034 |
| `gate_only` | 12016.8 | 57088.0 | $0.0729 | $0.0014 | $0.000327 | $0.000006 |

Both totals are printed because the assumed output length is the weakest assumption here. `input-only` removes it entirely and preserves every ranking; anyone who distrusts the constant can use that column instead.

#### Versus `always_strong`

| strategy | total cost | delta vs baseline | ratio | delta % | tier accuracy | UNSAFE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_strong` | $0.3846 | $0.0000 | 1.00x | 0.0% | 17.0% | 87 |
| `always_cheap` | $0.0874 | -$0.2972 | 0.23x | -77.3% | 43.9% | 87 |
| `jev_route` | $0.1846 | -$0.1999 | 0.48x | -52.0% | 51.1% | 38 |
| `gate_only` | $0.0729 | -$0.3117 | 0.19x | -81.1% | 60.5% | 50 |

#### Versus `gate_only`

| strategy | total cost | delta vs baseline | ratio | delta % | tier accuracy | UNSAFE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_strong` | $0.3846 | +$0.3117 | 5.28x | 427.9% | 17.0% | 87 |
| `always_cheap` | $0.0874 | +$0.0146 | 1.20x | 20.0% | 43.9% | 87 |
| `jev_route` | $0.1846 | +$0.1118 | 2.53x | 153.4% | 51.1% | 38 |
| `gate_only` | $0.0729 | $0.0000 | 1.00x | 0.0% | 60.5% | 50 |

A negative `delta %` means cheaper than the baseline. Read it together with the UNSAFE column: `always_cheap` and `gate_only` are both cheaper than routing, and both leak more. Cost saved per privacy violation bought is not a trade this project is willing to make silently.

#### Negative control: why not just the regex gate?

Counted in both directions, so the comparison cannot be tilted by reporting only the side that flatters the router.

| comparison | rows |
| --- | ---: |
| rows compared | 223 |
| both `jev_route` and `gate_only` correct | 77 |
| both wrong | 51 |
| `jev_route` correct, `gate_only` wrong | 37 |
| `gate_only` correct, `jev_route` wrong | 58 |

Rows where `jev_route` was right and `gate_only` was wrong:

| `gate_only` error kind | rows |
| --- | ---: |
| `underpowered` | 25 |
| `unsafe` | 12 |

`gate_only`'s misses here are dominated by `underpowered` (25 of 37 listed rows). Example ids (37 rows in total): `chat-hard-001`, `chat-frontier-001`, `chat-internal-003`, `code-hard-001`, `code-hard-002`.

Rows where `gate_only` was right and `jev_route` was wrong:

| `jev_route` error kind | rows |
| --- | ---: |
| `overspend` | 45 |
| `expensive` | 13 |

`jev_route`'s misses here are dominated by `overspend` (45 of 58 listed rows). Example ids (58 rows in total): `chat-trivial-007`, `chat-standard-001`, `chat-standard-002`, `chat-standard-003`, `chat-standard-004`.

### Latency

Provenance: `latency` measured on configuration `default`.

Real backend calls measured: 186. Gate-skipped rows: 37.

| measure | n | mean ms | min ms | p50 ms | p95 ms | p99 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| backend decision call only | 186 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| router total (gate, redaction, features, merge, escalation, policy) | 223 | 0.3 | 0.1 | 0.2 | 0.5 | 1.5 | 1.6 |
| gate-skipped rows (no backend call at all) | 37 | 0.2 | 0.1 | 0.2 | 0.4 | n/a | 1.1 |

Gate skips are kept out of the backend-latency sample on purpose: their near-zero cost is a design feature, and averaging it into the decision call would understate the tax on everything else.

**Assumption, not a measurement:** `representative_completion_ms` = 3,000.0 ms. Every percentage in the next table is that measured overhead divided by this invented denominator. Real completion latency runs from a few hundred milliseconds for a short flash call to tens of seconds for a long frontier generation, so treat the column as an order of magnitude.

| percentile of the backend decision call | added ms | % of the assumed completion |
| --- | ---: | ---: |
| p50 | 0.0 | 0.00% |
| p95 | 0.0 | 0.00% |
| mean | 0.0 | 0.00% |

Every value in the backend-latency row is 0.0 ms because this backend is in-process and performs no I/O: the row measures nothing and must not be compared with a real backend's. The router-total row is still a real measurement of everything the router does around the call.

> backend_latency is the Jev/mock decision call only; router_total adds the local gate, redaction, feature extraction, merge, escalation and policy evaluation. representative_completion_ms is an assumption, not a measurement.

### Cache

| pass | backend | enabled | hits | misses | hit rate | evictions | entries | max entries | ttl s | p50 ms | p95 ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | memory | yes | 0 | 186 | 0.0% | 0 | 186 | 1024 | 900.0 | 0.159 | 0.322 |
| 2 | memory | yes | 186 | 186 | 50.0% | 0 | 186 | 1024 | 900.0 | 0.153 | 0.319 |

Warm pass: 186 hits, 186 misses, hit rate 50.0%.

> both passes served from RecordingBackend's memo, so no API calls were made; cold backend latency comes from the measured run, not from this replay

## What the cloud bootstrap buys: real backend vs offline mock

The offline `MockBackend` is a deterministic keyword-and-shape reader with no model behind it. It is the floor, not a competitor: if the real backend cannot beat it, the dataset is measuring keyword matching and the whole exercise is measuring nothing.

| metric | real backend (`jev`) | offline mock (`mock`) | gap (jev - mock) |
| --- | ---: | ---: | ---: |
| tier accuracy | 72.2% | 51.1% | +21.1 pp |
| UNSAFE (privacy violations) | 11 | 38 | -27 |
| PII F1 (raw noul) | 1.0000 | 0.0000 | +100.0 pp |
| `complexity` macro F1 | 0.6727 | 0.3197 | +35.3 pp |
| `complexity` accuracy | 71.4% | 45.6% | +25.7 pp |
| `complexity` ECE | 0.1331 | 0.3231 | -0.1900 |
| `sensitivity` macro F1 | 0.6667 | 0.1688 | +49.8 pp |
| `sensitivity` accuracy | 70.9% | 37.4% | +33.5 pp |
| `sensitivity` ECE | 0.1289 | 0.3664 | -0.2374 |
| `domain` macro F1 | 0.7317 | 0.3937 | +33.8 pp |
| `domain` accuracy | 74.8% | 40.3% | +34.5 pp |
| `domain` ECE | 0.1673 | 0.0702 | 0.0971 |

The gap column is `jev` minus `mock`. For accuracy and F1 a positive gap is the point of the project. For UNSAFE and for ECE, lower is better, so a *negative* gap is the good direction. Units: accuracy and F1 gaps are percentage points; ECE gaps are plain differences on the 0-1 error scale; the UNSAFE gap is a row count.

One result here is easy to misread:

* On `domain` the offline mock has the **lower** ECE (0.0702 vs 0.1673) while its accuracy is also lower (40.3% vs 74.8%). That is not a better model; it is a head that is accurately unsure. Calibration measures whether a stated confidence matches an observed rate, and a reader that says "I am 40% sure" and is right 40% of the time is perfectly calibrated and perfectly useless. This is why every ECE in this report is printed next to accuracy and Brier skill.

## Reproducing this report

```console
# the run that produced these numbers
python evals/run_eval.py --backend both --concurrency 12

# re-analyse the persisted per-row JSONL and re-render, at zero API cost
python evals/run_eval.py --backend both --reuse
```

`--reuse` is the idempotence switch: it reads the JSONL already in the results directory instead of calling any backend, so the report can be re-rendered forever without spending anything. It needs no API key. Two consequences are visible above: `backend_stats` and the `cache` block are properties of a live run and show as `n/a` on a reused one.

Rows are reproduced from the same recorded answers, not re-sampled: the A/B arms are an exact counterfactual over one pass. The question set fingerprint is per backend in the provenance table (`questions_sha256`).

The evaluation itself is deterministic given the same recorded answers, but the real backend is not: a fresh run against the live API re-samples every judgement, so the numbers will move. The dataset sha256_16 and the policy sha256_16 in the provenance table are what make two runs comparable.

## What this evaluation does not measure

* **No completions were generated.** The dataset is prompts without reference completions, so no completion model was called and no real token count exists. Every cost is prompt-side plus a constant assumed output of 256.0 tokens. Answer quality per tier is therefore *assumed* by the `expected_tier` label, not measured.
* **The decision API's own price is excluded** (`jev_decision_api_cost_included` = no). It is not published, and inventing it would be exactly the failure the assumption tables exist to prevent. The call count is reported instead, so anyone with a price sheet can add the term.
* **Token counts are a character-count proxy** at 4.0 characters per token. That is the conventional English approximation and is wrong for code and for Hungarian or German text, both of which this dataset contains. It is wrong by a similar factor for every strategy, so it cancels in comparisons and not in totals.
* **Prices are illustrative** (`prices_are_illustrative` = yes). They are not a quote and not live data.
* **`representative_completion_ms` = 3,000.0 ms is an assumption.** Every overhead percentage inherits it.
* **`expected_tier` is an author-assigned label**, one per prompt. A different labeller would move tier accuracy, and the ambiguous subset in the by-difficulty table is where that judgement is weakest. Accuracy here measures agreement with a documented labelling policy, not agreement with the world.
* **Denominators differ between sections by design.** Tier accuracy, cost and latency come from the shipped policy configuration; component labels and calibration come from the configuration that lets the backend answer the most rows. Each section names its own configuration and its own `n`.
* **Excluded rows are counted, not scored.** Errored and degraded rows sit outside the accuracy denominator and are printed in the headline table. A run that quietly scored fewer rows than it fetched is how benchmark numbers become fiction.
* **UNSAFE is a dataset-labelled notion of "must not leave the building".** It is derived from the `expected_tier` label, not from a legal review of your jurisdiction, your contracts or your residency obligations.
* **This is a single sample of a nondeterministic backend.** No confidence interval is reported, because the harness records one pass. Differences smaller than a few rows between two live runs should be treated as noise.

---

Rendered by `evals/report.py` from `summary.json`. Every figure in this file is read from that summary at render time; the renderer holds no results of its own.
