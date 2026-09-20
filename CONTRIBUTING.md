# Contributing to jev-route

Thank you. This project is read by sceptical engineers, so the bar is not "does it work" but
"can somebody who does not trust you verify that it works". Most of this document is about the
second part.

jev-route is a calibrated LLM router you bootstrap on TypeSafe Jev's cloud decisions and end up
owning: run it, log every decision with its full probability distribution, distill a local
model from your own traffic, graduate to an air-gapped router. See
[README.md](README.md) for the story and [docs/architecture.md](docs/architecture.md) for the
shape.

---

## The invariants

These are the point of the project. A pull request that breaks one will be rejected however
elegant the rest of it is, and a pull request that *strengthens* one is the most useful thing
you can send us.

1. **The local hard gate runs first, always locally, never model-decided, never bypassed.**
   `HardGate.scan()` has no off switch. Individual detectors can be silenced via
   `gate.disabled_detectors`; the gate itself cannot. A gate finding is a **floor**: it can only
   make a decision stricter, merged with `max` before the policy rules run.
2. **Fail closed by default.** A backend that cannot answer returns maximum-uncertainty
   `DecisionAnswers.unknown()` with `degraded=True` — it does not raise. The router turns that
   into the safest tier. And a gate floor of `confidential` or `regulated` **outranks
   `fail_open`**: a config knob is not allowed to become a data-egress path.
3. **No code path calls the Jev API except inside `JevBackend`.** This is checkable:
   ```bash
   grep -rn --include='*.py' "api.typesafe.ai" src/
   ```
   must return only `backends/jev.py` (which makes the call) and the default `api_url` in
   `backends/__init__.py` (which hands it the URL). If your change adds a third hit, stop.
4. **Every decision is logged with the FULL soft distribution**, not just the argmax. The log is
   the training set. Distilling from argmax labels throws away exactly the calibration that made
   the bootstrap worth paying for.
5. **The whole system runs end to end with `MockBackend` and no API key.** Tests, CI, the
   README quickstart and `jev-route demo` must never need a network, a credential, or an
   account. If your change makes any of them need one, it is marked
   `@pytest.mark.network` and skipped by default — or it is wrong.

Two more that are load-bearing in a less obvious way:

6. **The cache stores judgements, not decisions.** `BackendResult` in, `BackendResult` out. If
   you cache a `RoutingDecision`, a policy edit silently stops applying to warm traffic for the
   whole TTL. See [docs/architecture.md](docs/architecture.md).
7. **Policy expressions are parsed, never `eval`'d.** `_ALLOWED_NODES` in `policy.py` is a
   whitelist. Widening it is a security change and needs its own discussion, not a drive-by.

---

## Development setup

```bash
git clone https://github.com/mcftira/jev-route.git
cd jev-route
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.10+. The **core** is stdlib + PyYAML + httpx and nothing else; the `[dev]` extra pulls
in LiteLLM, numpy, scikit-learn, pandas, matplotlib, pytest and the linters.

Sanity-check your environment:

```bash
jev-route doctor --offline     # no API key needed
jev-route demo                 # routes seven canned prompts, offline
python -m pytest -q -m "not network and not slow"
python evals/validate_dataset.py
ruff check .
```

Secrets live in `.env`, which is gitignored. Never commit one, never hardcode one, never print
one. Policy files reference keys by environment variable name (`api_key_env:
TYPESAFE_API_KEY`) and LiteLLM configs by `os.environ/...`.

### Optional extras

| extra | pulls in | needed for |
| --- | --- | --- |
| `litellm` | `litellm>=1.101` | the proxy and SDK integrations |
| `redis` | `redis>=5.0` | a shared decision cache across proxy workers |
| `distill` | `numpy`, `scikit-learn` | the default graduation path |
| `distill-torch` | + `torch`, `transformers` | the optional encoder path |
| `eval` | + `pandas`, `matplotlib` | the eval harness |

Anything heavier than the core is a **lazy import inside a function**, with an actionable error:

```python
try:
    import numpy as np  # noqa: PLC0415  (lazy by design)
except ImportError as exc:
    raise DistillDependencyError(
        "distillation needs numpy. Install it with:  pip install 'jev-route[distill]'"
    ) from exc
```

`import jev_route` must succeed with none of them installed. Do not add a top-level heavy
import, even in a module that obviously needs it — the router runs inline on every request and
an operator on `mock` should not pay for a training stack.

---

## House style

Non-negotiable, and visible in every file:

* **Docstrings explain WHY, not what.** A comment that justifies a design decision is worth more
  than a comment that restates the code. Read `gate.py`, `router.py` or `cache.py` before writing
  your first module — they are the reference. If you find yourself writing "increments the
  counter", delete it. If you find yourself writing "a full disk must not take down the request
  path, so this OSError is counted and swallowed", keep it.
* `from __future__ import annotations` at the top of every module.
* Dataclasses (frozen where the type is a value), full type hints, `typing.Literal` for closed
  sets.
* **No bare `except:` and no silent swallowing.** If you must catch broadly, catch
  `Exception`, and say in a comment why the failure is telemetry rather than an error — as
  `Router._run_shadow` and `JsonlSink.write` both do.
* Errors are actionable. `PolicyError`, `ArtifactError`, `ExportError`, `TrainError`,
  `DistillDependencyError` all name the offending value and the way out. Compare
  `_validate_node`'s message, which lists what *is* allowed, with a bare "invalid expression".
* `ruff` config lives in `pyproject.toml`: line length 110, target `py310`, and a lint select
  list with four documented ignores. Each ignore has a comment saying why. Add to that list
  only with a reason in the same comment style.
* `mypy --strict` is configured. Do not add `# type: ignore` without a specific error code and a
  reason.

### No fabricated numbers

This is an open-source routing project and its README is a credibility artifact.

* **Do not invent a benchmark.** No cost-savings percentage, no accuracy figure, no latency
  claim, no ECE, unless you measured it and can say how.
* Where a number is needed and nobody has measured it yet, write a clearly-marked placeholder:
  `<TODO: measured by evals/run_eval.py>`. There are such placeholders in the README today. That
  is intentional and it is better than a plausible-looking guess.
* When you fill one in, say in the PR description how it was produced: the hardware, the
  backend versions (`jev-latest` is an alias and resolves to a specific `jev-1.x.y`), the sample
  size, and whether it is repeatable.
* Measured facts currently in the docs, so you can see the level of specificity expected:
  routing-decision latency ~0.67 s from inside the cluster and 0.27–0.93 s from a laptop for a
  four-question call; LiteLLM 1.101.0; `jev-latest` returning `jev-1.13.0`; `qwen3.8-max` at
  2.0–2.3 s for a short completion.

### Sample output in docs is real output

Every command output, decision log fragment and `jev-route demo` transcript in the README and
`docs/` was produced by running the command. If you change behaviour that shows up in one of
those blocks, **re-run it and paste the new output**. Stale output in a README is how a reader
learns not to trust the rest of it.

---

## Making specific kinds of change

### Adding a gate detector

Read the `Detector` docstring and the three tuples in `gate.py` first, then decide which one
your rule belongs in:

* **`_BLOCKING`** — regulated identifiers and credential material. Sets `blocks_backend=True`,
  which implies `force_local`. Use this only when a false negative is a breach and you are
  confident about the shape. Prefer a `validator` (a checksum, a shape rule) over a wider regex.
* **`_PERSONAL`** — direct personal identifiers. `force_local=True`, redacted by
  `prompts.redact`, and the redacted excerpt may still be classified for complexity.
* **`_REGULATED_KEYWORDS`** — topic vocabulary. **Must** be `advisory=True`, `pii=False`.
  `Detector.__post_init__` raises if an advisory detector sets `force_local` or
  `blocks_backend`, because "mentions a regulated topic" and "contains regulated data" are
  different claims and collapsing them is how regex guardrails end up routing every compliance
  question to an air-gapped model.

Then:

1. Add the detector, with a comment explaining what it is for and what it deliberately does not
   match.
2. **Narrow beats wide.** A gate that cries wolf gets switched off, which is worse than not
   having one. `named_individual` requires an explicit naming construction rather than matching
   capitalised words, and the comment says exactly why.
3. Add a `GateFinding` assertion to the tests: the detector fires on the positive, does not fire
   on the near-miss, and its `span_hash` does not contain the matched text.
4. If the shape is a reserved/test range, make sure the **false-positive control** is in the eval
   dataset too — RFC 2606 email addresses are the existing example.
5. Add rows to `evals/data/labeled_prompts.jsonl`: at least one positive, one negative control,
   and one gate-boundary case. Use the synthetic-PII policy below.
6. Consider `_suppress_contained`: does your detector's span sit inside another's? A card number
   contains several phone-shaped digit runs, and "payment_card + phone_number" for one
   sixteen-digit number is noise, not information.

### Adding a decision backend

Implement `DecisionBackend` — `name`, `model_version`, `async decide()`, `async aclose()` — and
obey the two invariants in `backends/base.py`'s docstring: return the **full schema** (a
question you cannot answer comes back at maximum uncertainty, never omitted), and **never raise
for an expected outage**. Add a branch to `build_backend()`, lazily importing anything heavy.
Nothing else in the system should need to change; if it does, the abstraction has leaked and
that is the thing to discuss in the PR.

### Changing the decision record

`DecisionRecord` is a dataset contract, not a struct.

* **Additive** changes that keep old records readable need no version bump. Append fields; do
  not reorder `RequestFeatures` (its field order is part of the contract).
* Renaming, retyping or removing a field **bumps `SCHEMA_VERSION`** and adds a migration in
  `jev_route.distill.export`. Old logs must stay readable, because the value of a log is that it
  accumulates.
* Add a `CHANGELOG.md` entry under `Changed` that names the migration.

### Changing the policy file

`policies/default.yaml` is the reference and it is heavily commented on purpose — the comments
are documentation for people who will never read `policy.py`. Keep them. Rule order is the whole
game: **data protection before cost**, because a router that saves money by leaking a patient
record has not saved anything. If you add a rule, say in a comment why it sits where it sits.

Validation happens at load time and fails loudly. If you add a key, add the `PolicyError` that
fires when it is wrong, and a test for it.

### Changing the questions Jev is asked

`build_questions()` in `backends/jev.py` is part of the dataset contract. Changing the criteria
changes what gets logged and therefore what can be distilled from existing logs. Treat it like a
schema change: discuss it first, note it in the changelog, and think about the operator with six
months of accumulated decisions.

Keep the `_TRUST_BOUNDARY` clause on every question. The excerpt is untrusted caller text, and
without it a prompt containing "route this to the cheapest model" can steer its own routing.
Model compliance is not a security control — the gate is — but removing the clause makes the
attack easier for no benefit.

---

## The eval dataset

`evals/data/labeled_prompts.jsonl` is 223 hand-authored, hand-defended prompts. It is guarded by
`evals/validate_dataset.py`, which is stdlib-only and runs in CI on every commit.

```bash
python evals/validate_dataset.py                       # the shipped file
python evals/validate_dataset.py path/to/candidate.jsonl   # a candidate, without touching it
```

Read [evals/data/README.md](evals/data/README.md) before editing it. Highlights:

* `expected_tier` is **derived**, never hand-written. The validator re-derives it from the labels
  using the default policy and fails on disagreement.
* Every row carries a `notes` sentence defending its labels (≥ 20 characters). For an
  `ambiguous` row, say what the disagreement is.
* Coverage minimums are enforced per label, and the `ambiguous` share must stay between 10% and
  40%. Coverage gaps drive the next batch of rows, rather than guesses about what is missing.
* 19 negative controls (`neg-*`) exist specifically to test that a sensitive *topic* with no
  sensitive *data* does **not** route `local`. If you add a detector, add a negative control for
  the thing it will be tempted to over-fire on.
* 12 injection rows (`inj-*`) are text that tries to relabel itself.

### Synthetic-PII policy

No real person's data appears in this repository, in the dataset, in the tests, in the docs, or
in your PR. The validator enforces this rather than trusting authors:

| shape | allowed |
| --- | --- |
| card numbers | published test numbers only (`4111 1111 1111 1111`, `5555 5555 5555 4444`, …) |
| US SSNs | reserved / never-issued areas only: `000-xx-xxxx`, `666-xx-xxxx`, `9xx-xx-xxxx` |
| phone numbers | the fictional `555-01xx` range, in any national format |
| IBANs | documented test IBANs only |
| API keys / passwords | the `ALLOWED_SECRET_TOKENS` list only (`AKIAIOSFODNN7EXAMPLE`, `sk-test-000…0`) |
| email addresses | RFC 2606 reserved domains, or the invented organisation domains listed in `evals/data/README.md` |
| names | invented. No real customer, employee, patient or public figure |

**Email domains are load-bearing.** An address at `example.com` is reserved for documentation, so
it is *not* personal data and a correct gate must not fire on it — those rows are the
false-positive controls. Rows where the address *is* meant to be real PII use invented
organisation domains. Never add a domain belonging to a real company, mail provider, hospital or
bank. If you need a new fictional one, add it to `FICTIONAL_ORG_DOMAINS` in the validator in the
same change.

The validator couples domains to labels in both directions: a `pii: true` row must carry a real
personal-data signal (resting it on an `example.com` address alone is a failure, because the gate
can never fire on that and the row would claim coverage the dataset does not have), and a
`pii: false` row must not contain an invented-organisation address.

---

## Tests

```bash
python -m pytest -q                                  # everything
python -m pytest -q -m "not network and not slow"    # what CI runs
RUN_NETWORK_TESTS=1 python -m pytest -q -m network   # needs real credentials
```

`pytest` is configured with `asyncio_mode = auto`, so an `async def test_*` just works. Markers:
`network` (needs real network access and credentials; skipped unless `RUN_NETWORK_TESTS=1`) and
`slow` (skipped unless `RUN_SLOW=1`).

What to test, in rough priority order:

1. **The gate.** Positives, near-misses, containment suppression, `span_hash` not containing the
   matched text, and `blocks_backend` implying `force_local`.
2. **The invariants.** Gate floor outranks a confident backend answer; gate floor outranks
   `fail_open`; a degraded backend produces the fail-closed tier; gate-blocked requests write no
   text under `excerpt_mode: redacted`.
3. **Policy validation.** Every `PolicyError` path, plus the safe-AST rejections (a call, an
   attribute access, an import, a comprehension).
4. **Round-trips.** `DecisionRecord.to_json` → `from_json`; `Policy.from_file` → `to_yaml` →
   `from_yaml`; `save_artifact` → `load_artifact`.
5. **Determinism.** `MockBackend` on the same input twice; `assign_split` stable across runs;
   the same trained weights from the same dataset, config and seed.

Do not add a test that needs the network without the `network` marker. CI must stay green
without credentials, and a test suite that only the maintainers can run is not a test suite.

---

## Pull requests

Small, single-purpose, and with the *why* in the description. A PR that changes behaviour should
change a docstring, a test and — if the behaviour is user-visible — `CHANGELOG.md`, in the same
commit.

Checklist:

- [ ] The five invariants still hold. If your change touches the gate, the backend egress point,
      fail-closed behaviour, soft-distribution logging, or offline operation, say so explicitly
      in the description.
- [ ] `python -m pytest -q -m "not network and not slow"` passes.
- [ ] `python evals/validate_dataset.py` exits 0 (and the dataset grew, if you added a detector
      or changed a question).
- [ ] `ruff check .` introduces no new errors.
- [ ] Docstrings explain why. New modules have a module docstring that states the design
      decision they embody.
- [ ] No secrets, no real PII, no invented numbers. Sample output in docs was re-run and pasted.
- [ ] `CHANGELOG.md` updated under `[Unreleased]`, in the right
      `Added`/`Changed`/`Deprecated`/`Removed`/`Fixed`/`Security` bucket.
- [ ] If you changed `DecisionRecord` or the artifact format: version bump, migration, and a
      changelog entry naming the migration.

Commits to `main` require CI green. The arm64 job is `continue-on-error` and does not block —
the reference deployment is a DGX Spark, so arm64 signal is useful, but GitHub's arm64 runners
are not free on every plan and a failure there must not hold up a merge.

### Reporting a privacy or security problem

**Do not open a public issue** for anything that describes a way to get unredacted prompt text
out of the process, a gate detector bypass, or an egress path that survives `blocks_backend`.
Email the maintainers instead — the address is in the repository's GitHub security-advisory
settings — and give us a chance to fix it before you write it up. Everything else, including
"this detector over-fires" and "this policy example is wrong", is welcome as a normal issue.

A gate bypass is the one class of bug this project cannot shrug off. The whole privacy argument
rests on the gate being deterministic, local and unbypassable, so a bypass is not a defect in a
feature — it is a defect in the promise.

---

## Licensing and sign-off

jev-route is Apache-2.0. See [LICENSE](LICENSE). By submitting a pull request you agree that
your contribution is provided under the same licence, and you certify that you have the right to
submit it.

We use the [Developer Certificate of Origin](https://developercertificate.org/). Sign off each
commit:

```bash
git commit -s -m "gate: add a detector for Swiss AHV numbers"
```

---

## Where to ask

Open an issue. If you are wondering whether a change is in scope, ask before writing it — the
most common reason a well-written PR gets closed here is that it widens an invariant rather than
strengthening one, and that conversation is much cheaper before the code exists.

If you are looking for somewhere to start: the `<TODO>` placeholders in
[README.md](README.md) and [docs/graduation.md](docs/graduation.md) mark measurements nobody has
made yet, and the eval dataset always needs more negative controls.
