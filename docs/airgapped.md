# Air-gapped: point the router at your own System One server

The bootstrap phase ends without a single line of Python changing. This is the
fast path to air-gapped: instead of waiting for your decision log to fill up and
distilling a student (that is [docs/graduation.md](graduation.md)), you serve an
off-the-shelf open System One model on your own machine and point the existing
router at it. Two config lines do the swap:

```yaml
backend:
  name: jev
  airgapped: true
  api_url: http://127.0.0.1:8009/v1
```

## Why one config swap is enough

The whole cloud dependency is **one HTTP call to one endpoint**:
`POST /v1/systemone` with the redacted state and the four typed questions,
answered with distributions. The endpoint's host is operator-supplied
(`api_url`); the literal `api.typesafe.ai` appears in exactly one module
(`backends/jev.py` — the invariant the test suite enforces by grep). So the
swap is:

* **`api_url` may be a base URL.** A URL that already ends in the wire-contract
  path is used verbatim (the cloud default is unchanged); any other base URL
  gets the path appended — `http://127.0.0.1:8009/v1` serves
  `POST http://127.0.0.1:8009/v1/systemone`.
* **`airgapped: true` changes the credential rule.** The backend no longer reads
  or requires `TYPESAFE_API_KEY`. The `Authorization` header still carries
  something — the placeholder `Bearer local` — because a request with no
  Authorization header at all trips naive reverse proxies. That placeholder is
  not a coincidence: Kev's own SDK example logs in with `api_key="local"`.
  An explicit `api_key` is still honoured when a fronting proxy wants one.
  Without the flag, a missing key keeps failing loud at construction: a router
  that quietly stops classifying is worse than one that refuses to start.

Everything above the backend — gate, policy engine, cache, decision log,
distillation, shadow window — cannot tell the modes apart. That is the point.

## Hardware

Kev (the model family this page uses) runs on **CUDA, ROCm, or Apple Silicon
(MLX)**; the 4B and 9B checkpoints fit a 32 GB Mac. The GPU-required step is
exactly one: the serve command in step 1 (its first run downloads the base
model and adapter). Everything else — the jev-route install, the policy edit,
the routing demo — runs on a laptop with no accelerator. CPU-only machines
should use the 0.8B checkpoint and expect slow requests; this page is written
for the 4B.

## 1. Serve Kev-4B locally

**[GPU or Apple Silicon required]** Kev needs Python 3.12 or 3.13 and
[uv](https://docs.astral.sh/uv/) — an independent environment from jev-route's:

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev
uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

The first run downloads the adapter and the Qwen3.5-4B base model, then serves
in bf16 (`KEV_DTYPE=fp32` selects the exact path the published evaluations
use). `--run` also accepts a local checkpoint directory or a Hub revision.

Smoke-test the endpoint — this is the same wire contract the router speaks:

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size.",
  "model": "kev-latest",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                "billing": "Charges, invoices, payment problems"}},
    "escalate":   {"type": "noul", "instructions": "Does this need urgent human attention?"}
  }}'
```

A JSON body with `answers.department.choice`, a probability distribution, and
`answers.escalate.noul` means the server is ready.

## 2. Point the router at it, with zero credentials

Fresh clone of jev-route, in a separate terminal:

```bash
git clone https://github.com/mcftira/jev-route.git && cd jev-route
python -m venv .venv && source .venv/bin/activate
pip install -e .
unset TYPESAFE_API_KEY   # the point: not needed, not sent
```

Copy the shipped policy and replace its `backend:` block:

```bash
cp policies/default.yaml policies/airgapped.yaml
```

```yaml
# policies/airgapped.yaml -- the backend block only; tiers and rules unchanged
backend:
  name: jev
  airgapped: true
  api_url: http://127.0.0.1:8009/v1   # the server's /v1 base; the path is appended
  model: kev-latest
  timeout_seconds: 15.0               # the cloud default of 5.0 is tuned to hosted latency
  max_retries: 2
  include_domain: true
```

The zero-credential routing demo:

```bash
jev-route route "Summarize the three main arguments of this essay about urban cycling infrastructure." \
  --policy policies/airgapped.yaml
jev-route route "hi there, thanks!" --policy policies/airgapped.yaml --json
```

You should get a tier decision from the local model — `backend: jev`,
`backend_model_version` from Kev's response — with no credential anywhere in
the process. The egress boundary has moved to your machine: the only module
that can open a socket is still `backends/jev.py`, and its URL is now
`127.0.0.1`. The redacted excerpt never leaves it, the local hard gate still
runs first and can still block the call before it is made, and the decision
log, cache and distillation pipeline work unchanged.

## The honest quality diff

The numbers below are Kev's published ones (Kev README, "New Sources" columns:
datasets and policy rule types the checkpoint was **not** trained on; each cell
is development / test; lower Brier is better):

| model | accuracy: new sources (dev / test) | Brier: new sources (dev / test) |
| --- | --- | --- |
| Kev-0.8B (Qwen3.5-0.8B base) | 0.652 / 0.684 | 0.499 / 0.460 |
| Kev-4B (Qwen3.5-4B base) | 0.797 / 0.837 | 0.299 / 0.255 |
| Kev-9B (Qwen3.5-9B base) | 0.822 / 0.852 | 0.286 / 0.237 |
| Jev (hosted, `jev-1.13.0`) | 0.857 / – | 0.211 / – |

Read them with their caveats, which Kev states itself: the Jev row is
development-set only — Jev has not been run on the frozen test set, and nobody
knows which datasets it trained on, so this is not a controlled comparison.
Kev-9B trails Jev by 3.5 points on the new-source development set. The Brier
numbers are for raw logits; each checkpoint ships a fitted temperature (about
2.1–2.4) that sharpens the probabilities it serves.

**The accuracy gap is the easy part; the calibration gap is why
`confidence_adjust` exists.** The policy engine reasons about confidence, not
labels: floors like `sensitivity_confidence_below: 0.8` mean "if the backend is
less sure than this, go stricter." Out-of-distribution, a Kev 0.9 is not a Jev
0.9 — the probabilities sit on a different scale, so the same floor can
systematically under- or over-trigger. Rather than retune every floor for every
backend, the operator applies one multiplicative correction to the backend's
reported confidence, in config.

## `backend.profile.confidence_adjust`

```yaml
backend:
  name: jev
  airgapped: true
  api_url: http://127.0.0.1:8009/v1
  profile:
    confidence_adjust: 0.8   # multiply the backend's choice confidences by this
```

What it does, precisely:

* **Multiplicative, on the three choice answers only** — `complexity`,
  `sensitivity`, `domain` — applied to every classified, non-degraded result,
  including cache hits, and clamped to `[0, 1]`.
* **Not the PII noul.** That is a probability, not a confidence, and it feeds
  the PII rules through a different mechanism; scaling it would be a
  category error.
* **Not the logged distributions.** The decision log and the distillation
  pipeline still see exactly what the backend said. The correction is a routing
  decision, not a rewrite of the data.
* **No mutation of the cache.** Adjusted results are new objects; the cached
  result stays raw, so a profile change takes effect on the next read with no
  cache flush.
* **Validated at construction.** A profile that is not a mapping, a factor that
  is a bool, a string, `NaN`, `±inf` or negative is refused at router startup
  with a named error — a silently mis-routed fleet is worse than one that will
  not boot.

**How to pick the number.** Do not guess it. Run Kev behind the router on real
traffic, then measure: on a held-out slice of your own decision log (or a
hand-labelled set), bucket requests by Kev's reported confidence and compare
each bucket's empirical correct-tier rate against its confidence. The factor is
the ratio that aligns them — if Kev reports 0.9 where 80% of those calls turn
out right, that is `0.8`, and the policy floors mean the same thing again.
Re-check afterwards: the graduation gate track's live shadow window already
computes a 15-bin ECE per window, and a corrected backend should show a flatter
reliability curve, not a shifted one.

## What an air-gapped deployment keeps

* **The quota tandem stays local.** `providers:` entries are base URLs, so an
  air-gapped deployment can list several local servers (two Kev GPUs, a 4B and
  a 9B); a 429 flips to the alternate and the flip preserves `airgapped`, so a
  local outage stays inside the building.
* **Per-backend shadow partitioning.** The shadow window's `record()` accepts an
  optional `backend_id` tag, and `window_status_by_backend()` computes the same
  status — agreement, 15-bin ECE, baseline, drift — per tag. Records without a
  tag partition under the window's configured label (default `"jev"`), which is
  also the on-disk default: files written before the tag existed read back as
  one `"jev"` window, and the whole-window `window_status()` — with the
  demotion logic built on it — is byte-for-byte the old behaviour. A
  Jev-then-Kev deployment is thus graded on each backend's own agreement and
  its own baseline, not on the other backend's history.

## When this is not enough

If your workload sits far outside Kev's training distribution, no multiplier
fixes the gap — the distributions themselves, not just their scale, are wrong.
That is the graduation pipeline in [docs/graduation.md](graduation.md): weeks of
your own decisions, distilled into a student that is calibrated on *your*
traffic, behind the same gate, policy engine and log this page did not touch.
