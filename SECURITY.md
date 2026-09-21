# Security

This document describes the threat model of the local gate, the published
injection-eval results, and how to reproduce them. For what leaves the
process on every request kind, read [docs/privacy.md](docs/privacy.md); for
how the gate is built and how layer 2 earns the right to block, read
[docs/gate-layers.md](docs/gate-layers.md).

## The gate's threat model

The gate is a local, deterministic floor. It runs on every request before any
model is consulted, before redaction, before cache lookup -- in-process, with
no network. No policy rule, backend answer, or configuration can relax a gate
finding; a finding can only make routing stricter.

The deterministic layer (layer 1) has two detector classes:

* **The blocking class -- structured regulated identifiers and credential
  material.** A hit sets the sensitivity floor to `regulated`/`confidential`
  and sets `blocks_backend`: the excerpt is refused to any remote backend,
  redacted or not. The deterministic layer blocks:
  * **SSN-shaped strings** (`us_ssn`: `\d{3}-\d{2}-\d{4}`; there is no
    SSN checksum, so shape is the contract -- a gate that waits for a
    "plausible" SSN misses the ones that are real),
  * **Luhn-valid card numbers** (`payment_card`: 13-19 digits, Luhn-validated,
    optionally spaced or dashed),
  * **provider API keys** (`provider_api_key`: OpenAI/Anthropic `sk-...`,
    AWS `AKIA...`, GitHub PAT, Slack, Google, Hugging Face shapes),
  * and the same class: IBAN (mod-97 validated), UK NHS number (mod-11),
    UK National Insurance number, private key blocks, inline credentials
    (`password=`, `api_key=`, `secret_token=...`), basic-auth URLs, and
    medical record numbers.
* **The non-blocking (advisory) class -- direct personal identifiers.**
  **Emails are advisory**: `email_address` (with `phone_number`,
  `date_of_birth`, `street_address`, `named_individual`) never blocks the
  backend. It raises the sensitivity floor, forces the local tier, and the
  matched span is replaced by a typed placeholder. Under the default
  `gate.on_force_local: skip_backend`, nothing crosses the process boundary
  at all. RFC 2606 reserved domains (`example.com` and friends) are not
  treated as personal data by default: an address on a reserved domain is not
  anybody's email, and a gate that fires on `sales@example.com` earns a
  reputation for crying wolf. Set `gate.placeholder_domains_as_pii: true` to
  treat them as PII.

**16-digit strings stay local by design.** The card detector is
Luhn-validated, so a 16-digit run one checksum away from a card is never
reported as a card -- but it still never reaches a cloud model: the
phone-shaped digit-run detector trips on the inner group and forces the local
tier. The promise is that the Luhn-validated detector does not fire, not that
the gate is blind (pinned in `tests/test_gate.py::test_luhn_invalid_card_is_not_reported_as_a_card`).

Topic keywords (HIPAA, PCI-DSS, ...) are advisory in the strictest sense:
they never set a floor and are forwarded to the backend as hints, because
"mentions a regulated domain" and "contains regulated data" are different
questions.

Layer 2 -- the local semantic scorer (`gate.semantic`) -- is where the
deterministic layer's boundary ends. It runs locally, and in the default
`shadow` mode it scores and logs but can never change what is served. The
leaks listed below are the published argument for it.

## Injection eval (v0.2)

264 synthetic cases (`evals/injection/cases.jsonl`), all provably fake. The
threat model under test: inputs that try to convince the router that
sensitive data is safe for cloud models -- wrapper instructions, authority
framing, and encoding tricks around identifiers the gate is designed to
catch.

Results by category (`evals/injection/RESULTS.md`):

| category | cases | blocked | allowed (leaked) |
|---|---|---|---|
| authority_framing | 60 | 60 | 0 |
| benign_control | 104 | 0 | 104 |
| encoding_trick | 40 | 15 | 25 |
| injection_wrapper | 60 | 60 | 0 |

Acceptance:

* categories 1-2 (injection wrappers, authority framing): **0 leaks** -- PASS (bar: 0).
* category 3 (encoding tricks): **25 leaks** -- documented below, not a
  blocker: this is the published argument for the distilled semantic layer.
* category 4 (benign controls): **0 false positives** -- these are the
  provably-fake cases the gate must NOT block.

### The 25 category-3 leaks

The deterministic layer is regex-class. These encodings defeat it BY DESIGN,
and every leak here is a published requirement for the semantic gate:

* `cat3-000` -- not detected by any deterministic detector
* `cat3-001` -- not detected by any deterministic detector
* `cat3-002` -- not detected by any deterministic detector
* `cat3-003` -- not detected by any deterministic detector
* `cat3-006` -- not detected by any deterministic detector
* `cat3-008` -- not detected by any deterministic detector
* `cat3-009` -- not detected by any deterministic detector
* `cat3-010` -- not detected by any deterministic detector
* `cat3-011` -- not detected by any deterministic detector
* `cat3-014` -- not detected by any deterministic detector
* `cat3-016` -- not detected by any deterministic detector
* `cat3-017` -- not detected by any deterministic detector
* `cat3-018` -- not detected by any deterministic detector
* `cat3-019` -- not detected by any deterministic detector
* `cat3-022` -- not detected by any deterministic detector
* `cat3-024` -- not detected by any deterministic detector
* `cat3-025` -- not detected by any deterministic detector
* `cat3-026` -- not detected by any deterministic detector
* `cat3-027` -- not detected by any deterministic detector
* `cat3-030` -- not detected by any deterministic detector
* `cat3-032` -- not detected by any deterministic detector
* `cat3-033` -- not detected by any deterministic detector
* `cat3-034` -- not detected by any deterministic detector
* `cat3-035` -- not detected by any deterministic detector
* `cat3-038` -- not detected by any deterministic detector

Base64 payloads, digit-by-digit card readings, and typo'd Hungarian
identifiers carry real-shaped PII that no regex can see. That list is the
roadmap: the semantic layer (`gate.semantic`, shadow today) exists precisely
because a regex cannot read base64 or typo'd identifiers. Promotion from
shadow to `enforce` is gated on measured criteria (held-out recall >= 0.99
and the shadow-log rates in `docs/gate-layers.md`); until then, these cases
route exactly as if the layer were absent -- the eval does not hide the gap,
it publishes it.

## Reproduce

```bash
cd jev-route
.venv/bin/python evals/injection/gen_cases.py   # deterministic; rewrites cases.jsonl
.venv/bin/python evals/injection/run.py         # runs the gate, rewrites RESULTS.md
```

No API keys, no network: `run.py` is stdlib plus the repo's gate.

## Data hygiene

* **All PII is provably fake.** SSNs use the 000 area (never issued), card
  numbers fail Luhn by construction (the Luhn-valid rows are the published
  canonical test numbers), emails sit on RFC 2606 reserved domains, and names
  are placeholders. No real user data appears anywhere in the eval or in the
  committed trace (`traces/demo_500.jsonl` is generated by
  `traces/gen_demo_500.py` with a fixed seed).
* **Committed results are metadata only.** The eval records category,
  blocked/allowed, and detector ids; case content and model payloads never
  appear in committed output.
* **Blocked requests are never stored as text.** A gate-blocked request
  writes a `GateBlockRecord` carrying the detector ids, the deterministic
  feature vector, and an excerpt hash -- never the matched span, under any
  configuration. The refusal stream feeds rule improvement, not model
  training, and the gate never trains on blocked content.

## Report a vulnerability

Report security issues via the repository's issue tracker with a
`security` label; do not open a public issue with a working exploit.
