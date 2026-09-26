# v0.5 Phase 3: calibration-teacher GEPA -- CUT (per the work order's own escape hatch)

The plan was a second GEPA run with TEACHABILITY as the metric (candidate teacher
questions -> quick-train a student -> score = student leak count + ECE), adopted
only if the student beat the v1 shadow head.

## Why it is cut

The escape hatch said: "if the quick-train loop proves flaky, cut this phase and
ship v0.5 without it." It proved flaky before this phase started:

- the v0.4 compile run needed **2 hours for 2 iterations** on the local DGX
  (each reflection is a ~50k-token single-slot generation), wedged the endpoint
  twice, and needed a pod restart;
- a teachability loop multiplies that by a training run per candidate --
  hours per candidate on the same box.

The honest read, with v0.4's own negative result in hand: compiled questions
improved ROUTING (+21.7 pts, adopted) but *worsened* the sensitivity student
(v2 vs v1: 23/40 vs 1/40 encoding leaks -- docs/shadow-v2-comparison.md). A
teachability search over question wording would most plausibly land in the same
place. The v1 shadow head stays the shadow; the contrastive gate (v0.5 Phase 4)
is the promotion bar it must clear on live traffic.

If a future phase wants it back: the harness exists (`optimize/questions.py`
evaluator seam takes any metric; a student-training metric is a one-line swap).
