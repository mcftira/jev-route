# Injection eval v0.3 -- encoding class closed deterministically

264 synthetic cases (same held-out set as v0.2). The v0.2 result left 25
encoding-trick leaks as the published argument for a distilled semantic layer.
v0.3's finding is stronger than that plan: **the encoding class was never a
model problem -- it was a normalization problem.**

## What changed

`jev_route/gate_normalize.py` (new): before scanning, the hard gate now
produces decoded variants of the text -- base64 blobs (padded or long),
spaced-out digit runs, leetspeak inside mostly-digit tokens -- and scans the
original AND every variant. A variant only matters if its decoded content
trips a real detector, so random tokens cost nothing. Findings are deduplicated
by matched content.

## Result

| category | cases | blocked | leaked |
|---|---|---|---|
| injection_wrapper | 60 | 60 | 0 |
| authority_framing | 60 | 60 | 0 |
| encoding_trick | 40 | 40 | 0 |
| benign_control | 104 | 0 (0 false positives) | -- |

**0 leaks on all 264 cases, deterministically.** No model call anywhere in the
path. The v0.2 roadmap item "the semantic layer must learn base64" is closed by
decode-and-rescan instead: cheaper, exact, and it cannot be overconfident about
a blob it decoded itself.

## What the semantic layer is still for

The trained multilingual head (frozen Laya encoder + noul head, artifact from
the v0.3 training run) remains the shadow layer for the class normalization
cannot see: *typo'd or paraphrased identifiers in free text* ("a masodik
karakter egy kis L" style content with no structural tell). It stays in shadow
until the graduation criteria in `jev_route.gate_semantic.EnforceCriteria` are
met on live traffic. Nothing about the deterministic floor weakens; the head
can only add refusals, never remove them.

## Reproduce

```bash
python evals/injection/gen_cases.py   # regenerate the 264 cases (deterministic)
python evals/injection/run.py          # rerun; RESULTS.md is rewritten
```

Data hygiene unchanged: committed results carry metadata only (category,
blocked, detector ids); all PII in every category is provably fake.
