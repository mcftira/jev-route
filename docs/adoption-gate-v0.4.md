# v0.4 adoption gate -- compiled questions

| gate condition | result | bar |
|---|---|---|
| valset outcome accuracy | 0.667 -> 0.883 (**+21.7 pts**) | >= +2 pts or equal at lower cost |
| injection eval with compiled wording | 0 benign fires, 0 attack leaks | 0 leaks / 0 FP |
| human review of question diff | PASS | readable, non-adversarial |

**Verdict: PASSED.** `policies/compiled_questions.yaml` is adopted-eligible;
`questions: compiled` in a policy selects it (`build_backend` loads it into
JevBackend.question_overrides). Default stays `handwritten` until an operator
flips a policy.

What the compile changed (human review summary): the tier definitions got
tighter in the direction the trace demanded -- routine code generation,
single-file changes, translation, localization, support tone and
non-English wording are explicitly `standard` (the handwritten text let
"mentions files" and non-English phrasing inflate difficulty, the source of
most of the baseline's misroutes); fake/sample/synthetic data is explicitly
non-PII; technical content (code, file names, architecture) is explicitly
not "sensitive data" by itself; and the untrusted-content injection-hardening
sentence survived evolution in all three questions.

Reproduce: `jev-route optimize-questions` (manual, keyed; never in CI).
