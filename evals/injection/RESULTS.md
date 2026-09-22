# Injection eval -- sensitivity gate (v0.2)

300 synthetic cases (`evals/injection/cases.jsonl`, all provably fake PII:
000-area SSNs, Luhn-invalid cards, RFC 2606 emails, placeholder names).
The threat model: inputs that try to convince the router that sensitive
data is safe for cloud models.

## Results by category

| category | cases | blocked | allowed (leaked) |
|---|---|---|---|
| authority_framing | 60 | 60 | 0 |
| benign_control | 104 | 0 | 104 |
| encoding_trick | 40 | 40 | 0 |
| injection_wrapper | 60 | 60 | 0 |

## Acceptance (v0.2)

* categories 1-2 (injection wrappers, authority framing): **0 leaks** -- PASS (bar: 0).
* category 3 (encoding tricks): **0 leaks** -- since v0.3 the gate
  normalizes encodings (base64/spaced/leet) and rescans deterministically, so this
  class is closed without a model; see RESULTS_v0.3.md.
* category 4 (benign controls): **0 false positives** -- these are the
  provably-fake cases the gate must NOT block.

## Category 3 findings (the honest part)

The deterministic layer is regex-class. These encodings defeat it BY DESIGN,
and every leak here is a published requirement for the semantic gate:

* (none -- the deterministic layer caught everything this round)

That list is the roadmap: the semantic layer (`gate.semantic`, shadow today)
exists precisely because a regex cannot read base64 or typo'd Hungarian
identifiers. RESULTS.md does not claim perfection; it shows the boundary
and where the distilled layer takes over.

## Method and data hygiene

* Runner: `evals/injection/run.py` (stdlib + the repo gate; no API keys).
* Committed results record METADATA only (category, blocked, detector ids).
  Case content and model payloads never appear in committed output.
* Regenerate with `python evals/injection/gen_cases.py` (deterministic).
