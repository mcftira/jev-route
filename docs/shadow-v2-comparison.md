# Shadow v2 comparison: compiled-question teacher labels (a negative result)

The v0.4 plan: retrain the shadow sensitivity head on teacher labels produced
by the COMPILED questions (the ones that won the routing adoption gate at
+21.7 pts accuracy), ship it as shadow v2, keep the better one.

## The measurement (same held-out eval, both heads, threshold 0.5)

| head | training labels | cat3 caught (40) | benign FP (104) |
|---|---|---|---|
| v1 | generator hard labels (sensitive=1/0) | **39 (1 leak)** | 52 |
| v2 | Jev + compiled questions' sensitivity choice | 17 (23 leaks) | 39 |

## Verdict: keep v1. v2 is NOT shipped.

## Why, honestly

The compiled questions made the ROUTER better by being sharper about what does
NOT count: fake/sample/synthetic data is explicitly non-PII, technical content
is explicitly not sensitive. That sharpness is correct for routing. As a
sensitivity TEACHER for the obfuscation class it backfires: the compiled
backend relabelled ~30% of the generator's encoding positives (base64 blobs,
spaced digits, typo'd identifiers) as non-sensitive, and the student learned
the laxer boundary. On the eval it misses 23 of 40 encodings v1 catches.

The two questions are different jobs: "route to the right tier" and "spot
disguised sensitive data". Better wording for the first does not transfer to
the second. This is the publishable negative result the work order allowed
for, and it is recorded here rather than smoothed over.

Reproduce: `evals/injection/gen_train_variants.py` (v1 dataset), the teacher
relabel script in the v0.4 run (`/tmp/label_v2.py` + `/tmp/train_v04.py` at
run time; v2 checkpoint at `/tmp/v04-laya-head`, unshipped).
