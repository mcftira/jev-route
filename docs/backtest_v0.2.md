# Backtest v0.2 -- policy cost on synthetic traffic

* trace: `traces/demo_500.jsonl` (500 rows, all synthetic, provably fake PII)
* policy: `policies/default.yaml`

## Headline

**$0.2663 with this policy vs $3.4617 always-frontier: 92.3% cheaper on 500 replayed requests.**

Traffic mix: 150 trivial completions, 120 code-gen, 80 multi-file refactors, 60 provably-fake-PII, 50 Hungarian
  support tickets, 40 long-context.

What the number is: the policy's effect on that mix, replayed through the full router pipeline
(hard gate, semantic layer in shadow, prefilter, policy rules) with an oracle backend that
replays the trace's own recorded tiers. It measures **routing distribution, not model quality**:
the classification step is assumed correct, and the question answered is *what does this policy
cost* -- not *how well does the model classify*.

## Cost by category

| category | count | ours $ | baseline $ | why the tier |
|---|---:|---:|---:|---|
| trivial_completion | 150 | 0.0183 | 0.9156 | default rule: no sensitivity signal, non-hard complexity |
| code_gen | 120 | 0.0149 | 0.7452 | default rule: no sensitivity signal, non-hard complexity |
| multi_file_refactor | 80 | 0.1041 | 0.5206 | hard complexity maps to the strong tier |
| pii_fake | 60 | 0.0039 | 0.3747 | 29 gate-forced local; 31 to `cheap` by policy (see Gate) |
| hungarian_support | 50 | 0.0062 | 0.3113 | default rule: no sensitivity signal, non-hard complexity |
| long_context | 40 | 0.1189 | 0.5943 | hard complexity maps to the strong tier |
| **total** | **500** | **0.2663** | **3.4617** | **92.3% cheaper** |

## Routing distribution

| tier | requests | share |
|---|---|---|
| cheap | 351 | 70.2% |
| strong | 120 | 24.0% |
| local | 29 | 5.8% |

## Gate

* deterministic gate fired on **29 of 500** requests; every fire is a real detector hit, not a policy rule.

| detector | rows | effect |
|---|---|---|
| provider_api_key | 12 | blocks the backend -- the excerpt never leaves the process |
| us_ssn | 11 | blocks the backend -- the excerpt never leaves the process |
| phone_number | 6 | forces the local tier (redacted text may still be classified) |

### Why only 29 of 60 synthetic PII rows fired

The trace carries 60 provably-fake PII rows; the gate fired on 29 of them. The 31 rows that did not
  fire are documented design, not misses:

* **RFC 2606 reserved-domain emails.** The `email_address` validator rejects reserved domains
  (`example.com` and friends): an address on a reserved domain is not personal data, and a gate
  that fires on `sales@example.com` earns a reputation for crying wolf. Operators who want them
  treated as PII set `gate.placeholder_domains_as_pii: true`.
* **16-digit strings that fail Luhn.** `payment_card` is Luhn-validated, so a digit run one
  checksum away from a card is not called a card. Such strings still stay local: the phone-shaped
  detector trips on the inner digit run and forces the local tier, so the number never reaches a
  cloud model either. (In this trace those are the `phone_number` rows above.)
* **Patient-record and street-address phrasings.** Layer-1 patterns are deliberately narrow: a name
  counts only in an explicit naming construction, an address only in a full street-plus-ZIP shape.
  Layer 1 claims what it can verify; free-text person identification belongs to the calibrated
  backend and, eventually, the local semantic layer (shadow mode today).

## Method and reproducibility

* **No API keys needed.** `TraceBackend` (an oracle) replays each row's recorded tier at 0.99
  confidence, expressed as the complexity the policy maps back to that tier. The gate, the
  semantic layer (inert in shadow), the prefilter, and the policy rules all execute for real on
  every row; only the model classification is replayed. CI runs this offline.
* **The trace is 100% synthetic and provably fake**: 150 trivial completions, 120 code-gen, 80 multi-file
  refactors, 60 provably-fake-PII, 50 Hungarian
  support tickets, 40 long-context. SSNs use the 000 area (never
  issued), cards fail Luhn by construction, emails sit on RFC 2606 reserved domains, and names are placeholders.
  Generator: `traces/gen_demo_500.py` (deterministic seed).
* **Prices are illustrative.** `jev_route/pricing.py` carries per-1k-token prices (local $0.00,
  cheap $0.0003, strong $0.003, frontier $0.015) and models 400 output tokens per request.
  A deployment's real contract prices go in the policy's `pricing:` block and override the table.
* **Reproduce:** `.venv/bin/python -m jev_route.cli backtest --trace traces/demo_500.jsonl` (add
  `--report <path>` to regenerate this file).

> Honest reading: the oracle replay makes the headline a routing-distribution number on a
> synthetic traffic mix, not a model-quality claim. Swap in a real classifier and the tiers
> (and the dollars) move; the Gate section above does not, because the gate is deterministic.

## Outcome verification (synthetic)

* verified outcomes: **471/500** routed requests
* per-tier completion rate (Phase 1 plumbing check -- synthetic scores):

| tier | n | mean completion p |
|---|---|---|
| cheap | 351 | 0.88 |
| strong | 120 | 0.97 |

> On live traffic these numbers come from the real decision backend; here
> they prove the plumbing (the first 'is the routing actually good' number,
> distinct from cost).
