# jev-route evaluation dataset

`labeled_prompts.jsonl` is the ground-truth set for the jev-route router. Each line is one
user prompt with the four labels a Jev System One model is asked to produce, the tier the
default policy should select for it, and a note defending the labelling.

The router is only trustworthy if this file is honest, so the dataset ships with a strict
validator (`evals/validate_dataset.py`) that must pass before any change is merged.

* **Rows:** 223
* **Format:** JSONL, UTF-8, LF line endings, one object per line, exactly one trailing newline
* **Companion:** `evals/validate_dataset.py` (stdlib only, Python 3.9+)

## Schema

Every line has exactly these six top-level keys, in this order (`labels` holds the four
judgements the router asks Jev for):

| key | type | meaning |
| --- | --- | --- |
| `id` | string | stable lowercase hyphenated slug, unique across the file (`pii-ssn-001`) |
| `text` | string | the prompt exactly as a client would send it; newlines are JSON-escaped |
| `labels.complexity` | enum | `trivial` \| `standard` \| `hard` \| `frontier` |
| `labels.sensitivity` | enum | `public` \| `internal` \| `confidential` \| `regulated` |
| `labels.pii` | boolean | does the text actually carry personal data about an identifiable person |
| `labels.domain` | enum | `code` \| `writing` \| `analysis` \| `chat` \| `data-extraction` |
| `expected_tier` | enum | `local` \| `cheap` \| `strong` — derived, never hand-written |
| `difficulty` | enum | `clear` \| `ambiguous` |
| `notes` | string | one sentence saying *why* the labels are what they are |

`expected_tier` is computed from `labels` by the default policy below. It is never edited by
hand: the validator re-derives it and fails on any disagreement.

## Default policy (the thing under test)

Applied in order, first match wins:

1. `sensitivity` in {`confidential`, `regulated`} **or** `pii == true` → **`local`**
2. else `complexity == frontier` → **`strong`**
3. else `complexity == hard` → **`strong`**
4. else → **`cheap`**

Rules 2 and 3 are kept distinct in reasoning (novel-edge work versus ordinary multi-step
work) but both yield `strong`. Note the ordering consequence the dataset deliberately
exercises: rule 1 beats complexity, so a `frontier` prompt that carries PII is `local`,
and a `trivial` prompt that carries PII is also `local`.

## Labelling rubric (condensed)

**complexity**

| level | test |
| --- | --- |
| `trivial` | greeting, thanks, one-line lookup, formatting tweak |
| `standard` | ordinary single-step task a small model handles well |
| `hard` | multi-step reasoning, non-trivial code/architecture, analysis needing care |
| `frontier` | genuinely at the edge: novel research synthesis, hard maths/proofs, large-scale system design, subtle judgement calls where a weaker model produces wrong output |

**sensitivity**

| level | test |
| --- | --- |
| `public` | nothing private at all |
| `internal` | ordinary workplace/business content; no personal data, no regulated domain |
| `confidential` | trade secrets, unreleased product plans, credentials, M&A, security vulnerabilities, employee performance, personal data about an identifiable person |
| `regulated` | data covered by law: health (HIPAA), card numbers (PCI-DSS), government identifiers, legal privilege, financial account data (GLBA), data about minors (COPPA/FERPA), EU personal data (GDPR) |

**pii** — `true` only when the text *actually contains or clearly asks about* personally
identifiable information of a real-looking person: name plus contact details, a national
identifier, a health record of a named person, an account number tied to a person, a precise
location of a person. "Users" and "customers" in the abstract are `false`. Populations and
aggregates are `false` even in a regulated context.

**domain** — the single best of `code`, `writing`, `analysis`, `chat`, `data-extraction`.
When two fit, the label records the deliverable's shape (a patch → `code`, prose →
`writing`, an inference → `analysis`, conversation → `chat`, structured output →
`data-extraction`) and the row is usually marked `ambiguous`.

**difficulty** — `ambiguous` when a reasonable expert could disagree, or when the label
depends on subtle context. 50 of 223 rows (22.4%) are ambiguous on purpose; those rows are what
test calibration rather than mere accuracy.

## Coverage (computed by the validator)

`complexity` — minimum 25 each

| value | rows | share | min | status |
| --- | --- | --- | --- | --- |
| `trivial` | 46 | 20.6% | 25 | ok |
| `standard` | 81 | 36.3% | 25 | ok |
| `hard` | 69 | 30.9% | 25 | ok |
| `frontier` | 27 | 12.1% | 25 | ok |

`sensitivity` — minimum 25 each

| value | rows | share | min | status |
| --- | --- | --- | --- | --- |
| `public` | 90 | 40.4% | 25 | ok |
| `internal` | 46 | 20.6% | 25 | ok |
| `confidential` | 52 | 23.3% | 25 | ok |
| `regulated` | 35 | 15.7% | 25 | ok |

`domain` — minimum 25 each

| value | rows | share | min | status |
| --- | --- | --- | --- | --- |
| `code` | 39 | 17.5% | 25 | ok |
| `writing` | 54 | 24.2% | 25 | ok |
| `analysis` | 65 | 29.1% | 25 | ok |
| `chat` | 29 | 13.0% | 25 | ok |
| `data-extraction` | 36 | 16.1% | 25 | ok |

`expected_tier` (derived)

| value | rows | share |
| --- | --- | --- |
| `local` | 87 | 39.0% |
| `cheap` | 98 | 43.9% |
| `strong` | 38 | 17.0% |

Special groups

| group | rows | required | what it tests |
| --- | --- | --- | --- |
| `pii == true` | 43 | >= 40 | the local gate fires on real shapes |
| `pii == false` and `sensitivity` confidential/regulated | 44 | >= 40 | sensitivity drives `local` without a PII hit |
| negative controls (`neg-*`) | 19 | >= 15 | sensitive *topic*, no sensitive *data* — must not route `local` |
| injection / adversarial (`inj-*`) | 12 | >= 8 | text that tries to relabel itself |
| gate-boundary rows (`boundary-*`) | 6 | 6 | reserved-domain vs fictional-domain emails, trivial-but-PII, topic-only |
| Hungarian (`hu-*`) | 12 | >= 10 | multilingual routing |
| German (`de-*`) | 6 | >= 5 | multilingual routing |
| long prompts (>= 800 chars) | 10 | >= 5 | the router excerpts text before sending it to Jev |
| `difficulty == ambiguous` | 50 | 10-40% of rows | calibration headroom |

Cross-tab of `sensitivity` x `pii`: public 90 (all `pii=false`),
internal 46 (all `pii=false`), confidential 52
(18 with `pii=true`, 34 without), regulated 35
(25 with `pii=true`, 10 without).

## How the data was produced

1. Hand-authored in batches by category — chat, code, writing, analysis, data-extraction,
   then PII shapes, negative controls, adversarial rows, multilingual rows, long pastes,
   and a frontier top-up.
2. Every row was written as a Python record; `expected_tier` was derived by the same policy
   function the validator implements, never typed by hand.
3. The validator was run after each batch. Coverage gaps drove the next batch rather than
   guesses about what was missing.
4. Realism was checked by re-reading rows as a client would send them: terse and chatty
   voices, typos, pasted code and stack traces, multi-turn-looking follow-ups that quote
   earlier context.

## Synthetic-PII safety policy

No real person's data appears in this file. Every identifier is either from a published test
range or invented for this dataset, and the validator enforces the policy rather than
trusting authors.

| shape | allowed values |
| --- | --- |
| card numbers | published test numbers only: `4111 1111 1111 1111`, `5555 5555 5555 4444`, `4242 4242 4242 4242`, `3782 822463 10005`, `6011 1111 1111 1117`, and the other entries of `ALLOWED_CARD_NUMBERS` |
| US SSNs | reserved / never-issued areas only: `000-xx-xxxx`, `666-xx-xxxx`, `9xx-xx-xxxx` |
| phone numbers | fictional `555-01xx` range, in any national format (`415-555-0132`, `+49 30 555 0188`, `+36 30 555 0142`) |
| IBANs | documented test IBANs only (`GB33BUKB20201555555555`, `DE89370400440532013000`, ...) |
| API keys / passwords | the `ALLOWED_SECRET_TOKENS` list only (`AKIAIOSFODNN7EXAMPLE`, `sk-test-000...0`); any other `sk-`/`AKIA`/`ghp_`/JWT-shaped token fails |
| email addresses | RFC 2606 reserved domains, or the invented domains below |
| names | invented; no real customer, employee, patient or public figure |

**Email domains are load-bearing.** An address at `example.com` / `example.org` /
`example.edu` / `example.net` is reserved for documentation, so it is *not* personal data and
a correct PII gate must not fire on it. Those rows are the false-positive controls. Rows where
the address is meant to be real PII use invented organisation or consumer-ISP domains:

`northside-health.org`, `hartmann-klinik.de`, `drkovacs-praxis.hu`, `northwind-logistics.io`,
`acme-logistics.io`, `nordkapp-logistics.fi`, `vireobank.com`, `falconridge-capital.com`,
`kestrel-analytics.com`, `brightpath-schools.org`, `keller-legal.com`, `trevallyn-press.co.uk`,
`lindqvist-bygg.se`, `volanynet.hu`, `postboxmail.com`, `schnellpost.de`, `levelezo.hu`,
`postafiok.hu`, `pureunmail.kr`

All of these are invented for this dataset; any resemblance to a real organisation is
coincidental. Never add a domain belonging to a real company, mail provider, hospital or bank.

The validator couples domains to labels in both directions:

* a row with `pii: true` must carry at least one real personal-data signal (a non-reserved
  address, a `555-01xx` phone, a reserved-range SSN, a test card or IBAN, a birth date, a
  medical/record identifier, an address, an account number, or compensation data). Resting
  `pii: true` on an `example.com` address alone is a failure, because the gate can never fire
  on it and the row would claim coverage the dataset does not have.
* a row with `pii: false` must not contain an address on an invented organisation domain.

## Adding a new prompt

1. Pick an `id` slug from the prefix convention: `chat-`, `code-`, `write-`, `ana-`, `ext-`,
   `pii-`, `neg-` (negative control), `inj-` (adversarial), `boundary-`, `hu-`, `de-`,
   `long-`, `fr-` (frontier).
2. Append one JSON object to `evals/data/labeled_prompts.jsonl` with the six keys above and
   no trailing blank line. Do not write `expected_tier` from memory — derive it from the
   policy, or copy it from the validator's complaint.
3. Write a `notes` sentence that defends the labels. For an `ambiguous` row, say what the
   disagreement is.
4. Keep PII inside the safety policy above. If you need a new fictional organisation domain,
   add it to `FICTIONAL_ORG_DOMAINS` in the validator in the same change.
5. Run the validator and make it pass:

```bash
python3 evals/validate_dataset.py
```

## Validator behaviour

`evals/validate_dataset.py` exits `0` on success and `1` on any failure, printing a coverage
table either way. It fails loudly on:

* malformed JSON lines, blank lines, CRLF endings, a BOM, a missing final newline
* duplicate `id` values, and duplicate prompt texts
* missing or extra keys, at both the row level and inside `labels`
* invalid enum values for `complexity`, `sensitivity`, `domain`, `difficulty`, `expected_tier`
* `pii` that is not a JSON boolean
* `expected_tier` that disagrees with the default policy applied to the labels
* text shorter than 8 characters or longer than 6000, or containing a carriage return
* `notes` shorter than 20 characters
* any PII shape outside the synthetic safety policy (unknown email domain, real-range SSN,
  non-test card number, non-`555-01xx` phone, unknown IBAN, unlisted credential token)
* `pii: true` with no grounding signal, or `pii: false` with an invented-domain address
* `hu-*` / `de-*` rows whose `notes` do not name the language
* coverage below any minimum, or an `ambiguous` share outside 10-40%

Pass a different path to validate a candidate file without touching the shipped one:

```bash
python3 evals/validate_dataset.py path/to/candidate.jsonl
```
