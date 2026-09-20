"""Synthetic decision-log fixture for the distillation tests.

WHY THIS MODULE EXISTS
----------------------
``jev_route.distill`` needs a decision log to work on: export reads it, train
learns from it, evaluate scores the student against the teacher, package ships
the artifact. Every one of those tests needs *hundreds* of records, and asking a
human to hand-write hundreds of log lines is how fixtures rot. So this module
generates them the honest way: it routes a varied prompt corpus through the real
:class:`~jev_route.router.Router`, with the offline
:class:`~jev_route.backends.mock.MockBackend` and a real
:class:`~jev_route.logging_sink.JsonlSink`, and hands back the path.

That means the fixture exercises the parts a hand-written JSONL file would fake:
the local hard gate (findings, floors, ``blocks_backend``), gate/answer merging,
confidence-floor escalation, the decision cache, policy rule evaluation, and the
record serializer. A distillation test that trains on this log is training on
records the production code path could actually have produced.

Two properties are load-bearing for downstream tests:

* **No network, no API key, no egress.** The mock backend is deterministic and
  offline, so CI can run the whole graduation pipeline in a sandbox.
* **Byte-reproducible.** The same ``(n, seed, repeats, excerpt_mode)`` produces
  an identical file, because ``request_id`` is derived from a hash of
  ``(index, seed)`` rather than ``uuid4()`` -- a train/holdout split keyed on
  ``request_id`` has to mean the same thing on the second run -- and because the
  two wall-clock-derived field groups (``timestamp`` and the latency counters)
  are rewritten from a deterministic clock after the router has written the log.
  See :func:`_normalize_volatile_fields`.

The corpus is synthetic. Every identifier in it is either RFC 2606 reserved
documentation space (``example.com``, ``*.test``, ``*.invalid``), a published
vendor test value (the Visa test card ``4111 1111 1111 1111``, the AWS
documentation key ``AKIAIOSFODNN7EXAMPLE``), or obviously fake. Nothing here is
anybody's data.

KNOWN GAPS -- what this fixture does NOT give you, so nobody assumes otherwise:

* **No cache hits.** The corpus is duplicate-free and every extra pass gets a
  variant suffix, so ``decision.cached`` is always false and ``backend`` is
  ``mock`` or ``gate``, never ``mock:cached``. That trade is deliberate: two
  records with identical text and different ``request_id`` would leak across a
  train/holdout split. A test that needs a cached decision should route the same
  prompt twice through its own :class:`~jev_route.router.Router`.
* **No degraded records.** :class:`~jev_route.backends.mock.MockBackend` never
  fails. Pass ``backend=`` something that returns ``degraded=True`` to exercise
  the fail-closed path; :func:`build_log` counts those in ``LogStats.degraded``.
* **``requested_model`` is null everywhere**, because the corpus arrives as bare
  text rather than through the LiteLLM proxy.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from jev_route.backends.base import DecisionBackend
from jev_route.backends.mock import MockBackend
from jev_route.gate import default_gate
from jev_route.logging_sink import JsonlSink, iter_records
from jev_route.policy import Policy
from jev_route.prompts import excerpt_from_text
from jev_route.router import Router
from jev_route.schema import (
    COMPLEXITY_LEVELS,
    DOMAINS,
    SENSITIVITY_LEVELS,
    TIERS,
    DecisionRecord,
)

#: Repo root, derived from this file's location so the fixture works no matter
#: which directory pytest is invoked from.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: The reference policy every synthetic log is routed under. Tests that need a
#: different one pass their own :class:`~jev_route.policy.Policy` to
#: :func:`build_log`.
POLICY_PATH: Path = REPO_ROOT / "policies" / "default.yaml"

#: Suffix appended to a prompt when the corpus is walked more than once, so that
#: a 300-record log holds 300 *distinct* excerpts. Distinctness matters: two
#: records with identical text but different ``request_id`` would leak across a
#: train/holdout split and quietly inflate every metric downstream.
VARIANT_SUFFIX = " (variant {n})"

#: Deterministic epoch for the synthetic clock. A fixture log with 2026 wall-clock
#: timestamps in it is a fingerprint of the machine that built it; a fixed epoch
#: makes the file reproducible and makes the ordering obvious to a human reader.
_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Multi-line prompts. Kept as flush-left module constants so that the string a
# test sees is exactly the string below: an indented triple-quoted literal would
# smuggle leading whitespace into every continuation line and change the features
# (char_len, line_count, indent-sensitive regexes) the router computes from it.
# --------------------------------------------------------------------------- #
_PROMPT_STACK_TRACE = """\
Our nightly batch job died at 03:12 and I cannot tell whether it is our bug or the database. Where do I start?

```
Traceback (most recent call last):
  File "/srv/app/worker.py", line 88, in run
    flush(session)
  File "/srv/app/db.py", line 214, in flush
    session.commit()
  File "/srv/.venv/lib/python3.12/site-packages/sqlalchemy/orm/session.py", line 1230, in commit
    self._transaction.commit()
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) server closed the connection unexpectedly
```

It only happens on the worker that also serves proxy traffic, and only after the connection pool change."""

_PROMPT_DIFF = """\
Does this diff look safe to merge, or did I break the public API? Review it line by line.

```diff
--- a/src/jev_route/schema.py
+++ b/src/jev_route/schema.py
@@ -41,7 +41,8 @@ SCHEMA_VERSION = "1"
 COMPLEXITY_LEVELS = ("trivial", "standard", "hard", "frontier")
-SENSITIVITY_LEVELS = ("public", "internal", "confidential")
+SENSITIVITY_LEVELS = ("public", "internal", "confidential", "regulated")
 DOMAINS = ("code", "writing", "analysis", "chat", "data-extraction")
```

I am mostly worried about readers that stored the old ladder in a config file."""

_PROMPT_SQL = """\
This report query takes four seconds on a 20M row table. What index would you add, and what would it cost on writes?

```sql
SELECT region, DATE_TRUNC('day', created_at) AS day, COUNT(*) AS n
FROM events
WHERE created_at >= NOW() - INTERVAL '30 days'
  AND kind IN ('click', 'purchase')
GROUP BY region, day
ORDER BY n DESC
LIMIT 50;
```"""

_PROMPT_TYPESCRIPT = """\
How do I type this helper so the return type follows the key I pass in, without a cast at the call site?

```ts
type Config = { retries: number; label: string; enabled: boolean };

function pick(config: Config, key: string) {
  return config[key];
}
```

I want `pick(cfg, "retries")` to be a `number` and `pick(cfg, "label")` to be a `string`."""

_PROMPT_DOCKERFILE = """\
Optimize this Dockerfile for layer caching and image size. It rebuilds from scratch on every commit.

```dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y curl postgresql-client
EXPOSE 8080
CMD ["python", "-m", "app"]
```"""

_PROMPT_K8S = """\
Review this deployment manifest for mistakes before I apply it to the staging cluster.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: router
spec:
  replicas: 3
  selector:
    matchLabels:
      app: router
  template:
    metadata:
      labels:
        app: router
    spec:
      containers:
        - name: router
          image: registry.example.com/router:latest
          ports:
            - containerPort: 8080
```"""

_PROMPT_JSON_BLOB = """\
{
  "request_id": "demo-0001",
  "model": "qwen3.8-flash",
  "messages": [
    {"role": "user", "content": "summarize the incident timeline"},
    {"role": "assistant", "content": "three causes, two of them ours"}
  ],
  "usage": {"prompt_tokens": 184, "completion_tokens": 97, "total_tokens": 281},
  "decision": {"tier": "cheap", "rule_id": "default", "cached": false}
}"""

_PROMPT_NGINX_LOGS = """\
Here are a few lines from the access log. Extract the paths with the highest 5xx rate and give me a table.

```
10.0.0.7 - - [04/Jun/2026:09:14:02 +0000] "GET /v1/search HTTP/1.1" 503 0 "-" "curl/8.4.0" 4213
10.0.0.7 - - [04/Jun/2026:09:14:03 +0000] "GET /v1/search HTTP/1.1" 503 0 "-" "curl/8.4.0" 4188
10.0.0.9 - - [04/Jun/2026:09:14:05 +0000] "POST /v1/route HTTP/1.1" 200 411 "-" "python-httpx/0.27" 38
10.0.0.9 - - [04/Jun/2026:09:14:06 +0000] "GET /healthz HTTP/1.1" 200 2 "-" "kube-probe/1.31" 1
```"""

_PROMPT_RESEARCH = (
    "Survey the published work on routing natural-language requests to models of differing capability, "
    "then evaluate which results would still hold for a self-hosted deployment with strict data residency. "
    "Specifically: derive the conditions under which a calibrated probability distribution over task "
    "difficulty beats a hard classifier when the cost ratio between tiers is unknown at training time, "
    "prove why the confidence floor cannot be replaced by a temperature setting, and compare and contrast "
    "the failure modes of cascade routers versus single-shot predictors under distribution shift. "
    "Design the experiment set that would falsify your own conclusions, including the sample size needed "
    "to detect a two-point calibration error at 95% confidence, and finish with the architecture you "
    "would defend in a design review: components, interfaces, observability, and the migration path from "
    "a cloud bootstrap model to a distilled local one. Cite the trade-offs you are making explicitly, "
    "because the reviewers will ask about latency budget, cost per thousand requests, and what happens "
    "when the teacher model is deprecated mid-flight."
)

_PROMPT_SHARDING = (
    "Architect a sharding scheme for a 12 TB Postgres cluster that must keep EU customer rows in the EU "
    "region, survive the loss of one availability zone without dropping writes, and rebalance shards "
    "without a maintenance window. Walk through the consistency trade-offs of the routing layer, derive "
    "the rebalance cost as a function of shard count, and explain how you would prove the residency "
    "invariant holds after a split-brain event rather than asserting it."
)

# The prior turns a multi-turn record is built from. Deliberately bland and free
# of digits, identifiers, and sensitive vocabulary: they must never change the
# gate verdict of the prompt they are glued in front of, or GATED_PROMPTS would
# stop describing the corpus.
_PRIOR_TURNS: tuple[tuple[dict[str, str], ...], ...] = (
    (
        {"role": "user", "content": "Quick context from earlier: how do I run the test suite for this repo?"},
        {"role": "assistant", "content": "Run pytest from the repo root. Add -x to stop at the first failure."},
    ),
    (
        {"role": "user", "content": "Following up on my last question: does the config file support overrides?"},
        {
            "role": "assistant",
            "content": "Yes. Every key can be overridden, and the policy file stays the source of truth.",
        },
    ),
    (
        # A tool turn. ``excerpt_messages`` leaves tool content out of the excerpt
        # by design, but it still sets ``tool_output_present`` -- without this
        # variant that feature column is constant across the whole log, and a
        # trainer cannot be tested on a column that never varies.
        {
            "role": "user",
            "content": "Earlier you asked me to run the suite; here is what came back.",
        },
        {"role": "tool", "content": "exit code 0; 42 tests passed, 0 failed"},
        {
            "role": "assistant",
            "content": "Thanks -- a green run, so the regression must be in the deploy step.",
        },
    ),
)

#: Every ``n``-th record is routed as a short conversation instead of a bare
#: string. WHY: without it, ``n_messages``, ``n_prior_turns`` and
#: ``tool_output_present`` are constant columns in the exported feature matrix,
#: and a trainer cannot tell whether it is using them or ignoring them.
MESSAGE_EVERY = 9

#: Offset of the first message-routed record, chosen so it is not also the first
#: record in the log.
MESSAGE_OFFSET = 4


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #
#: Hand-written prompt corpus. Roughly 160 requests covering the five domains the
#: router knows (``code``, ``writing``, ``analysis``, ``chat``,
#: ``data-extraction``), the whole complexity ladder from "hi" to a multi-paragraph
#: research brief, the whole sensitivity ladder from a public blog post to a
#: regulated identifier, three non-English languages, and the shapes that make
#: feature extraction interesting: code fences, a traceback, a unified diff, a
#: bare JSON blob, and one-liners.
#:
#: Ordering is by category, not by label, so that truncating to ``n`` records
#: still yields a varied sample rather than fourteen greetings.
PROMPTS: tuple[str, ...] = (
    # -- chat, and the trivial end of the complexity ladder ----------------- #
    "hi",
    "thanks",
    "hey, you there?",
    "ok, got it",
    "good morning! ready when you are",
    "lol, that is a good one",
    "yep, ship it",
    "brb, coffee",
    "ping",
    "no worries, take your time",
    "\U0001f44d looks good to me",
    "goodnight, see you tomorrow",
    "what is the weather usually like in Lisbon in June?",
    "tell me a joke about load balancers",
    "any quick recipe for a tomato soup with canned tomatoes?",
    "ok",
    "sure, go ahead",
    "haha, nice one",
    "can you repeat that?",
    "bye for now!",
    "Define idempotency in one sentence.",
    "Name three Python web frameworks.",
    "List the seven OSI layers, top down.",
    # -- code and debugging ------------------------------------------------ #
    "Why does `list.sort()` return None instead of the sorted list in Python?",
    "Write a regex that matches a semver string such as 1.2.3 or 1.2.3-rc.1",
    "I get `TypeError: cannot unpack non-iterable NoneType object` -- what usually causes that?",
    "How do I make a pytest fixture run once per session but still be parametrized?",
    (
        "I rebased onto main and now my branch has duplicate commits. How do I clean that up "
        "without losing the review history?"
    ),
    (
        "Refactor this handler to use a dispatch table instead of the if/elif chain, and keep the "
        "error message for unknown kinds:\n\n"
        "```python\n"
        "def handle(kind, payload):\n"
        "    if kind == 'create':\n"
        "        return create(payload)\n"
        "    elif kind == 'update':\n"
        "        return update(payload)\n"
        "    elif kind == 'delete':\n"
        "        return delete(payload)\n"
        "    raise ValueError(kind)\n"
        "```"
    ),
    (
        "This regex should match ISO 8601 timestamps but it also matches plain dates. What is wrong "
        "with it?\n\n`^(\\d{4})-(\\d{2})-(\\d{2})`"
    ),
    "Explain the difference between a mutex and a semaphore to a junior engineer, with a Python example.",
    "What does `EXPLAIN ANALYZE` actually measure, and why can the plan differ between two runs?",
    "Write a failing test first, then the implementation: a function that parses `1h30m` into seconds.",
    "How do I benchmark a Python function without warm-up effects fooling me?",
    "Give me a code review checklist for a pull request that touches authentication.",
    "Explain what a foreign key constraint does, using a two-table example.",
    (
        "The container gets OOMKilled roughly every 36 hours. How would you prove that it is a memory "
        "leak rather than a traffic spike?"
    ),
    (
        "We have an intermittent race condition: two workers claim the same job id about once a day. "
        "What is the most likely root cause, and how would you confirm it from the logs?"
    ),
    (
        "Our p95 latency on the /search endpoint doubled after we added the cache layer. Walk me through "
        "how you would find the bottleneck before you touch any code."
    ),
    (
        "Our ORM emits an N+1 query pattern on the report page. Compare and contrast eager loading with "
        "batched fetching, and say which one you would pick for a read-heavy dashboard."
    ),
    (
        "Design a token-bucket rate limiter that works across eight stateless API workers sharing a "
        "single Redis instance, and explain what happens during a Redis failover."
    ),
    (
        "We are migrating forty services from REST to gRPC. Compare and contrast a big-bang cutover with "
        "a strangler-fig migration, and evaluate which one survives a hiring freeze."
    ),
    (
        "Name the two hard problems in computer science, then design a cache invalidation strategy for a "
        "product catalog that is updated by three different writers."
    ),
    (
        "Prove that Dijkstra's algorithm settles nodes in non-decreasing distance order when every edge "
        "weight is non-negative, and show by counterexample why it breaks with a negative edge."
    ),
    (
        "Derive the maximum throughput of a three-stage pipeline with stage latencies of 20 ms, 35 ms and "
        "12 ms, then prove that adding a second worker to the middle stage raises it and adding one to "
        "the last stage does not."
    ),
    (
        "Survey the literature on consensus under partial synchrony and evaluate which of those results "
        "still hold for a geo-replicated deployment where one region can be partitioned for minutes."
    ),
    (
        "Here is the before and the after. Which one reads better, and is the timeout worth the extra "
        "argument?\n\n"
        "```python\nresult = client.fetch(url)\n```\n\n"
        "```python\nresult = client.fetch(url, timeout=5)\n```"
    ),
    (
        "Sign-off questions before Friday: is the retention window defensible? is the redaction "
        "complete? is the tier choice auditable? and can we explain all of it to a customer?"
    ),
    _PROMPT_SHARDING,
    _PROMPT_RESEARCH,
    _PROMPT_STACK_TRACE,
    _PROMPT_DIFF,
    _PROMPT_SQL,
    _PROMPT_TYPESCRIPT,
    _PROMPT_DOCKERFILE,
    _PROMPT_K8S,
    _PROMPT_NGINX_LOGS,
    # -- writing ----------------------------------------------------------- #
    "Write a 200-word release note for version 2.4 of an open source command-line tool.",
    "Write a haiku about a router that never sleeps.",
    "Write a short public marketing description for a mechanical keyboard.",
    "Fix the typos and grammar in this paragraph, but do not change the tone.",
    "Write the CONTRIBUTING.md section that explains how to run the test suite.",
    "Write a docstring for a function that routes a request to a model tier.",
    "Summarize this 3000-word blog post into one tweet-length sentence and a three-bullet TL;DR.",
    "Give me three opening lines for a conference talk about calibrated routing.",
    "Write a children's poem about a lighthouse that is afraid of the dark.",
    "Turn these bullet points into one paragraph for the quarterly newsletter.",
    (
        "Draft a polite email to a customer explaining that their feature request is on the roadmap but "
        "not scheduled for this quarter."
    ),
    (
        "Rewrite this paragraph in a friendlier tone: 'Your submission was rejected because it did not "
        "contain the required fields. Resubmit when complete.'"
    ),
    "Give me five headline options for a public blog post about decision logs.",
    "Translate this changelog entry into German, leaving code identifiers untouched.",
    (
        "Write an apology post for our public status page about yesterday's four-hour outage. No "
        "excuses, and no promise we cannot keep."
    ),
    (
        "Rewrite this error message so a non-technical user understands it: 'upstream connect error or "
        "disconnect/reset before headers'."
    ),
    ("Help me phrase a firm but professional message to a vendor about repeated SLA misses this quarter."),
    (
        "Write a cover letter for a data engineer role. My details: Jane Q. Example, "
        "jane@example.com, portfolio at https://example.com/portfolio."
    ),
    (
        "Draft a job description for a staff backend engineer. Leave the compensation band as a "
        "placeholder, I will fill it in after the review."
    ),
    "Draft the internal announcement for the platform reorg. Internal only, do not distribute yet.",
    # -- analysis ---------------------------------------------------------- #
    "Our churn went from 2.1% to 3.4% month over month. What are the first three things you would check?",
    "Explain Simpson's paradox using a routing example.",
    "Our error budget is 0.5% and we are at 0.42% with nine days left. What are the realistic options?",
    (
        "Estimate the monthly compute cost of serving two million requests if 30% of them route to a "
        "frontier model that costs eight times as much per token."
    ),
    "Compare and contrast per-request pricing with provisioned throughput for a spiky workload.",
    (
        "An A/B test ran nine days with 4200 users per arm and conversion of 3.1% versus 3.6%. Is that "
        "significant, and what would you do next?"
    ),
    "Forecast next quarter's support ticket volume from these monthly counts: 410, 455, 480, 520, 610.",
    "Evaluate the trade-offs of building our own router versus adopting an open source gateway.",
    "Why does p99 latency stay high after we cached 90% of incoming requests?",
    "Analyze this cohort retention table and tell me where the leak is.",
    "Calculate the break-even point between two vendors given a 10k free tier and per-token pricing.",
    (
        "What is the base rate fallacy, and how does it apply to a personal-data detector with 99% "
        "precision running on a corpus where 0.1% of documents actually contain it?"
    ),
    "Give me a decision matrix for choosing a vector database, weighted for operational burden.",
    "Estimate the GPU-hours needed to fine-tune a 150M parameter classifier on 50k labelled rows.",
    "Analyze how sensitive this pricing model is to a 20% increase in input token cost.",
    (
        "Interpret these regression coefficients for me: intercept 0.42, prompt length 0.0031, code "
        "blocks 0.18, prior turns -0.07, and the interaction term 0.0004."
    ),
    (
        "Which single metric should a routing product optimize -- cost per request, answer quality, or "
        "data residency compliance? Argue for one and steelman the other two."
    ),
    (
        "Root cause analysis: the deploy succeeded in every region except eu-central-1. Here is the "
        "timeline: 09:02 rollout starts, 09:07 health checks fail in eu-central-1, 09:11 rollback, "
        "09:20 traffic restored. What do you need next?"
    ),
    (
        "Estimate the storage growth of a JSONL decision log at 4000 records per day and 1.2 kB per "
        "record over three years, with monthly rotation and gzip at 8x."
    ),
    (
        "We have two candidate models with calibration curves that cross at 0.6. Which one do you trust "
        "more for a router that escalates below a 0.7 confidence floor, and why?"
    ),
    # -- data extraction --------------------------------------------------- #
    "Extract every URL from this text and return them as a JSON array, preserving order.",
    (
        "Compare the guidance on these three pages and give me one paragraph: "
        "https://example.com/retention, https://example.org/log-format and https://example.net/faq"
    ),
    "Turn this YAML into a JSON object without changing the key order.",
    "Extract the SKU codes from this product dump and deduplicate them.",
    (
        "Convert this CSV into JSON with one object per row:\n\n"
        "```\nid,name,region\nEX-1,Widget A,eu\nEX-2,Widget B,us\nEX-3,Widget C,eu\n```"
    ),
    (
        "Parse this access log line into named fields: "
        '10.0.0.7 - - [04/Jun/2026:09:14:02 +0000] "GET /v1/search HTTP/1.1" 503 0'
    ),
    (
        "Scrape this HTML table into a markdown table:\n\n"
        "```html\n<table><tr><th>tier</th><th>model</th></tr>"
        "<tr><td>local</td><td>qwen38</td></tr></table>\n```"
    ),
    (
        "Here is an invoice as text. Extract vendor, invoice number, due date and total into JSON. "
        "Vendor: Example Supplies Ltd. Invoice EX-2291. Due 2026-07-01. Questions to "
        "billing@example.com."
    ),
    (
        "Normalize these messy address strings into street, city and postal code fields. Here are the "
        "first five as examples of the input shape."
    ),
    "Extract the function names and their line numbers from this Python file and return a table.",
    "Convert this list of UTC timestamps to Europe/Budapest and give me a CSV with both columns.",
    (
        "Parse this stack trace into a table of frame number, file, line and function name:\n\n"
        "```\n"
        "Traceback (most recent call last):\n"
        '  File "/srv/app/cli.py", line 41, in main\n'
        "    export(path)\n"
        '  File "/srv/app/export.py", line 17, in export\n'
        "    raise ValueError('empty log')\n"
        "ValueError: empty log\n"
        "```"
    ),
    "Map these 40 log level strings onto a normalized severity enum and return the counts per level.",
    (
        "Pull the error codes out of this payload and sort them by frequency:\n\n"
        '```json\n{"errors": [{"code": "E1001", "n": 4}, {"code": "E1002", "n": 11}]}\n```'
    ),
    "Convert this OpenAPI fragment into a table of endpoints, methods and required parameters.",
    "Extract the diff hunks from this patch file and list every file it touches.",
    _PROMPT_JSON_BLOB,
    # -- sensitivity ladder: public -> internal -> confidential -> regulated  #
    (
        "Is our decision log GDPR compliant if we only store a hash of the prompt and the derived "
        "features, never the text?"
    ),
    "Is a hashed prompt still personal data under GDPR? Argue both sides and then pick one.",
    "Explain how HIPAA applies to a small clinic that wants to use a hosted LLM API for intake notes.",
    (
        "We have to decide whether protected health information may go to a third-party classifier. "
        "What does the business associate agreement have to cover?"
    ),
    "Draft the data-processing addendum wording for a subprocessor, DSGVO-flavoured, two paragraphs.",
    "Which fields of a clinical note are normally redacted before the note is used for training?",
    ("PCI-DSS question: does a truncated card number written to an application log still count as cardholder data?"),
    "We need a COPPA and FERPA review of the student records feature before we allow under-13 signups.",
    "This memo is attorney-client privileged. Summarize it for the general counsel in five bullets.",
    (
        "Strictly confidential: the unreleased pricing tiers for Q4. Turn them into a comparison table "
        "for the leadership offsite."
    ),
    ("What is our exposure if the restructuring plan leaks before the board resolution is signed? Internal only."),
    ("Explain how the salary bands for level four engineers are set and how compensation is reviewed each cycle."),
    (
        "Our data residency policy says EU customer data may not leave the region. Which tier satisfies "
        "that, and what do we lose by choosing it?"
    ),
    (
        "The security review found an unpatched CVE-2024-1234 in a transitive dependency. What is the "
        "patch plan and who signs off on it?"
    ),
    (
        "We found an unpatched CVE-2024-12345 in a transitive dependency and need the breach response "
        "plan drafted tonight."
    ),
    # -- gate-firing identifiers: obviously fake, checksum-valid ------------ #
    ("The refund failed for card 4111 1111 1111 1111 and the gateway says 'do not honor'. What are my options?"),
    (
        # Unspaced PAN, and deliberately NOT followed by a full stop: payment_card
        # guards its match with `(?![\d.])` so that dotted version numbers cannot
        # trip it, which means "row: 4111111111111111." would sail through. A
        # corpus that silently loses its own gate case is worse than no corpus.
        "Our payment page logs the full PAN -- example row 4111111111111111 from the sandbox. Tell me "
        "exactly what to change so we can pass PCI compliance."
    ),
    (
        "The test fixture contains the SSA-reserved example SSN 000-00-0000. How do I scrub values like "
        "that from the export before it leaves the building?"
    ),
    "Please refund invoice EX-2291 to IBAN DE89 3704 0044 0532 0130 00 and confirm the SWIFT code.",
    "Look up the appointment for NHS number 943 476 5919 and tell me what the clinic should prepare.",
    "The employee's NI number is AB123456C. Add them to the payroll test run before Friday.",
    (
        "Here is the case identifier, MRN: 4829173. Summarize the treatment plan and flag anything the "
        "on-call doctor should know."
    ),
    ("Patient DOB: 1988-03-14, intake form attached. Summarize the risk factors for the clinical note."),
    "The sandbox key is sk-FAKE0000TESTKEY0000DONOTUSE00 and the client still returns 401. Why?",
    "I accidentally committed AKIAIOSFODNN7EXAMPLE to the repository. What is the rotation procedure?",
    (
        "This PEM file starts with -----BEGIN PRIVATE KEY----- and the service refuses to load it. "
        "Which format does it actually want?"
    ),
    "Our staging DSN is postgres://svc:secret123@db.example.com:5432/app and the migration times out.",
    "The CI config has api_key = supersecretvalue123 hardcoded. Rewrite it to read from the environment.",
    "Email the finished export to analyst@widgets.test and copy the audit mailbox at audit@example.org.",
    "The customer asked us to call them on +1 555 0100 123 before Friday to confirm the delivery slot.",
    "Ship the replacement unit to 1234 Example Avenue, Springfield 90210 and bill the same address.",
    "The client named Jordan Example Smith wants their contract amended to add the EU residency clause.",
    # -- non-English ------------------------------------------------------- #
    "Warum gibt `list.sort()` in Python None zurück und warum ist die Liste danach nicht die alte?",
    "Entwirf eine Rate-Limiter-Architektur für acht zustandslose API-Worker mit einem gemeinsamen Redis.",
    "Ist das DSGVO-konform, wenn wir nur einen Hash des Prompts und die Merkmale speichern?",
    "Schreib mir bitte eine kurze öffentliche Erklärung für das README, mit einem Beispiel und den Schritten.",
    (
        "Écris un article de blog public de 300 mots sur les journaux de décisions, avec une "
        "introduction et des exemples pour l'équipe."
    ),
    "Pourquoi notre latence p95 a-t-elle doublé après l'ajout de la couche de cache ?",
    "Bonjour, peux-tu relire ce courriel pour l'équipe et me dire si le ton est trop direct ?",
    "Rédige une note interne confidentielle sur la migration, sans données personnelles.",
    "Szia! Köszönöm a segítséget, már működik.",
    "Írj egy rövid nyilvános blogbejegyzést a döntési naplókról, példa adatok nélkül.",
    "Miért dob hibát a rendszer, ha bekapcsoljuk a cache réteget?",
    "GDPR szempontból rendben van, ha csak a prompt hash-ét és a jellemzőket tároljuk?",
)

#: The prompts the local gate is expected to fire on, derived by running the real
#: :class:`~jev_route.gate.HardGate` over the real excerpt at import time.
#:
#: WHY derived and not hand-listed: a hand-written list drifts the moment a
#: detector regex or a prompt changes, and a fixture that lies about its own
#: expectations is worse than no fixture. Note that "fired" includes *advisory*
#: topic hits (``gdpr``, ``hipaa``, ``under nda``), which set no floor and block
#: nothing -- see ``Detector.advisory``. Also note that an ``example.com`` address
#: is deliberately absent from this list: RFC 2606 domains are documentation
#: space, not personal data, and the default gate is configured not to cry wolf
#: about them. Prompts that must exercise the email detector use a reserved
#: ``.test`` domain instead.
GATED_PROMPTS: tuple[str, ...] = tuple(
    prompt for prompt in PROMPTS if default_gate.scan(excerpt_from_text(prompt)).fired
)

#: Lookup form of :data:`GATED_PROMPTS` for tests that check membership per record.
GATED_PROMPT_SET: frozenset[str] = frozenset(GATED_PROMPTS)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
#: The three values ``logging.excerpt_mode`` accepts. Mirrored from the router so
#: a typo fails here, at fixture-construction time, instead of deep inside a test.
EXCERPT_MODES: tuple[str, ...] = ("hash", "redacted", "none")


@dataclass(frozen=True)
class LogStats:
    """What a synthetic log contains, as counts a test can assert on.

    WHY a dataclass rather than a dict: the distillation tests assert on these
    numbers constantly (``assert stats.gate_fired > 0``), and a frozen dataclass
    gives them attribute access, equality, and a readable repr without inventing
    a key-naming convention per test file.

    Field notes:

    ``gate_blocked``
        Records where the gate prevented *any* backend call, identified by
        ``decision.backend == "gate"`` in the log. That is a stricter thing than
        ``gate_fired``: advisory topic hits (``gdpr``, ``under nda``) fire the gate
        without blocking, and a ``force_local`` hit only blocks under the default
        ``gate.on_force_local: skip_backend``.
    ``with_excerpt``
        Records carrying prompt text. Always 0 in ``hash`` mode; in ``redacted``
        mode it is ``records - gate_blocked``, because the router never writes a
        gate-blocked request as text regardless of configuration.
    ``complexity`` / ``sensitivity``
        The *effective* labels -- after gate merging and confidence-floor
        escalation -- because those are what the policy reasoned about and what a
        student model is trained to reproduce.
    """

    path: str
    records: int
    gate_fired: int
    gate_blocked: int
    degraded: int
    with_excerpt: int
    excerpt_mode: str
    #: tier -> count, in ``TIERS`` order.
    tiers: dict[str, int]
    #: complexity label -> count, in ladder order.
    complexity: dict[str, int]
    #: sensitivity label -> count, in ladder order.
    sensitivity: dict[str, int]
    #: domain label -> count, in ``DOMAINS`` order.
    domain: dict[str, int]
    #: ``backend_model_version`` of the model that actually classified these
    #: records. Gate-only records report ``gate-1.0.0`` and are excluded from the
    #: vote when anything else classified, so this names the teacher, not the gate.
    teacher_model_version: str


def default_policy(*, excerpt_mode: str = "hash", log_path: str | Path | None = None) -> Policy:
    """The reference policy, pinned to ``excerpt_mode``/``log_path`` and the mock backend.

    Loads ``policies/default.yaml`` through :meth:`Policy.from_file` so the
    fixture is routed by exactly the rules an operator would ship, then rewrites
    only the ``logging`` keys the caller cares about (path, excerpt_mode,
    enabled). Everything else --
    gate config, tiers, rules, uncertainty floors -- is left untouched, because a
    fixture that quietly relaxes the policy is not testing the policy.

    ``backend.name`` is forced to ``mock`` on purpose: this module is the offline
    path, and a fixture that can end up needing an API key is a fixture that
    fails on a laptop with no network.
    """
    if excerpt_mode not in EXCERPT_MODES:
        raise ValueError(f"excerpt_mode must be one of {EXCERPT_MODES}, got {excerpt_mode!r}")
    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"reference policy not found at {POLICY_PATH}")

    base = Policy.from_file(POLICY_PATH)
    logging_cfg: dict[str, Any] = dict(base.logging)
    logging_cfg["enabled"] = True
    logging_cfg["excerpt_mode"] = excerpt_mode
    if log_path is not None:
        logging_cfg["path"] = str(log_path)
    backend_cfg: dict[str, Any] = {**base.backend, "name": "mock"}
    return base.with_overrides(logging=logging_cfg, backend=backend_cfg)


def read_log(path: str | Path) -> list[DecisionRecord]:
    """Every :class:`~jev_route.schema.DecisionRecord` in a synthetic log, in order."""
    return list(iter_records(path))


def build_log(
    path: str | Path,
    *,
    n: int | None = None,
    excerpt_mode: str = "hash",
    policy: Policy | None = None,
    seed: int = 0,
    backend: DecisionBackend | None = None,
    repeats: int = 1,
) -> LogStats:
    """Route the corpus through a real router and write the decision log to ``path``.

    Args:
        path: the JSONL file to write. It is truncated first: a fixture that
            appends to whatever a previous test left behind is not reproducible.
        n: exact record count. When omitted, the corpus is walked ``repeats``
            times. When given, it wins over ``repeats`` and the corpus is cycled
            or truncated to fit.
        excerpt_mode: ``hash`` (default, no prompt text retained), ``redacted``
            (store the redacted excerpt), or ``none``. Passed to the router
            explicitly, so it overrides whatever ``policy`` says -- the argument
            is the more specific request.
        policy: routing policy. Defaults to :func:`default_policy`.
        seed: salt for the ``request_id`` derivation. Changing it changes the ids
            and nothing else, which is what makes a train/holdout split shufflable
            without changing the labels.
        backend: decision backend. Defaults to a fresh
            :class:`~jev_route.backends.mock.MockBackend`. Inject a failing
            backend to exercise the degraded path.
        repeats: how many passes over the corpus to make when ``n`` is None.

    Returns:
        :class:`LogStats` summarizing the file that was just written.

    Determinism: identical ``(n, seed, repeats, excerpt_mode, policy, backend)``
    produces byte-identical files. Three things make that true, and all three are
    deliberate:

    1. ``request_id`` is ``sha256(seed, index)`` truncated to 32 hex chars -- the
       same shape as the ``uuid4().hex`` the router would otherwise generate, so
       downstream code that assumes the shape keeps working, but reproducible.
    2. Records are routed strictly sequentially, so the decision cache sees the
       same hit/miss order every time (a concurrent ``gather`` would not). The
       cache comes from the policy -- a 900 s TTL by default -- so a build has to
       finish inside one TTL for the ``cached`` flags to be reproducible; 300
       records take about 0.1 s.
    3. The wall-clock fields the router stamps -- ``timestamp``,
       ``decision.latency_ms``, ``total_latency_ms``, ``backend_latency_ms`` --
       are rewritten from a deterministic clock in
       :func:`_normalize_volatile_fields`. They are properties of the machine that
       built the log, not of the routing decision, and leaving them in would make
       "the same input gives the same file" false.

    Safe to call from a sync test *and* from inside a running event loop; see
    :func:`_run_coroutine`.
    """
    if excerpt_mode not in EXCERPT_MODES:
        raise ValueError(f"excerpt_mode must be one of {EXCERPT_MODES}, got {excerpt_mode!r}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if n is not None and n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    target = Path(path)
    resolved = policy if policy is not None else default_policy(excerpt_mode=excerpt_mode, log_path=target)
    plan = _plan(n=n, repeats=repeats)

    # Truncate rather than append. JsonlSink opens with "a" because a production
    # log accumulates; a fixture log must not inherit the previous run.
    if target.exists():
        target.unlink()

    router = Router(
        resolved,
        backend if backend is not None else MockBackend(),
        sink=JsonlSink(target),
        excerpt_mode=excerpt_mode,
    )
    _run_coroutine(_route_all(router, plan, seed=seed))
    _normalize_volatile_fields(target, seed=seed, expected=len(plan))
    return _summarize(target, excerpt_mode=excerpt_mode)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _plan(*, n: int | None, repeats: int) -> list[tuple[int, str]]:
    """The ``(index, text)`` pairs to route, in order.

    ``index`` is the position in the log and the only input to ``request_id``
    besides the seed, so it must be stable: the same call must produce the same
    plan, forever.
    """
    limit = len(PROMPTS) * repeats if n is None else n
    plan: list[tuple[int, str]] = []
    for index in range(limit):
        pass_number, offset = divmod(index, len(PROMPTS))
        text = PROMPTS[offset]
        if pass_number:
            text += VARIANT_SUFFIX.format(n=pass_number + 1)
        plan.append((index, text))
    return plan


def _request_id(index: int, seed: int) -> str:
    """Reproducible 32-hex-char id, shaped like the ``uuid4().hex`` it replaces."""
    payload = f"jev-route.synthlog\x00{seed}\x00{index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _conversation(index: int, text: str) -> list[dict[str, str]]:
    """Wrap ``text`` in a short, gate-clean conversation for the multi-turn records."""
    prior = _PRIOR_TURNS[(index // MESSAGE_EVERY) % len(_PRIOR_TURNS)]
    return [*prior, {"role": "user", "content": text}]


async def _route_all(router: Router, plan: Sequence[tuple[int, str]], *, seed: int) -> None:
    """Route every planned request, sequentially, then close the router.

    Sequential on purpose. The router caches on the redacted excerpt, so running
    these concurrently would make the ``cached`` flags and the backend names
    depend on task interleaving -- nondeterminism in a fixture that exists to
    remove it.
    """
    try:
        for index, text in plan:
            request_id = _request_id(index, seed)
            # Scalars only: the router drops anything else, and a fixture should
            # not rely on undocumented filtering.
            metadata = {"synthlog_seed": seed, "synthlog_index": index}
            if index % MESSAGE_EVERY == MESSAGE_OFFSET:
                await router.route_messages(_conversation(index, text), metadata=metadata, request_id=request_id)
            else:
                await router.route_text(text, metadata=metadata, request_id=request_id)
    finally:
        # Closes the sink (flush) and the backend. Without it the last line of the
        # log may not be on disk when the caller starts reading.
        await router.aclose()


def _run_coroutine(coro: Any) -> Any:
    """Run ``coro`` whether or not the caller is already inside an event loop.

    Same pattern as ``jev_route.router._run_coroutine``, copied rather than
    imported because it is private there: ``asyncio.run`` raises when a loop is
    already running, which is the normal state inside a ``pytest-asyncio`` test,
    so the work moves to a worker thread with its own loop instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _synthetic_latency_ms(record: DecisionRecord) -> float:
    """A stable, plausible latency derived from the record's own features.

    Real latencies are a measurement of the machine that built the log. These are
    a function of the record, so two builds agree byte for byte while a reader
    still sees longer requests taking longer.
    """
    features = record.features
    return round(
        0.2 + (features.char_len % 23) * 0.05 + 0.15 * min(features.code_blocks, 4) + 0.1 * features.n_prior_turns,
        3,
    )


def _normalize_volatile_fields(path: Path, *, seed: int, expected: int) -> None:
    """Rewrite the four wall-clock-derived fields with deterministic values.

    Runs *after* the router and its :class:`JsonlSink` have written the file, so
    the records themselves -- gate findings, merges, escalations, rule ids,
    probabilities -- are exactly what the production code path produced. Only the
    fields that encode "when and how fast on this machine" are replaced. The file
    is re-serialized through :meth:`DecisionRecord.to_json`, the same serializer
    the sink used, so nothing else about the bytes changes.
    """
    records = list(iter_records(path))
    if len(records) != expected:
        raise RuntimeError(
            f"sink wrote {len(records)} records to {path}, expected {expected}; "
            "JsonlSink swallows OSError, so check disk space and permissions"
        )

    lines: list[str] = []
    for index, record in enumerate(records):
        total_ms = _synthetic_latency_ms(record)
        # A decision that never reached a backend -- refused by the gate, or served
        # from the cache -- has no backend time to report. Keeping that true here
        # preserves both the total >= backend invariant and the shape of a real log.
        reached_backend = record.decision.backend != "gate" and not record.decision.cached
        backend_ms = round(total_ms * 0.6, 3) if reached_backend else 0.0
        timestamp = (_EPOCH + timedelta(seconds=index, milliseconds=(seed * 137) % 1000)).isoformat(
            timespec="milliseconds"
        )
        normalized = replace(
            record,
            timestamp=timestamp,
            total_latency_ms=total_ms,
            backend_latency_ms=backend_ms,
            decision=replace(record.decision, latency_ms=total_ms),
        )
        lines.append(normalized.to_json() + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def _teacher_version(records: Sequence[DecisionRecord]) -> str:
    """The model version that actually labelled this log.

    Gate-only records report ``gate-1.0.0``. That is honest provenance, but it is
    not the teacher: when any backend classified anything, its version wins. Ties
    break alphabetically so the answer cannot depend on iteration order.
    """
    counts: Counter[str] = Counter(r.decision.backend_model_version for r in records)
    classified = {v: c for v, c in counts.items() if not v.startswith("gate-")}
    pool = classified or dict(counts)
    if not pool:
        return ""
    return sorted(pool.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _ordered_counts(counts: Counter[str], ladder: Sequence[str]) -> dict[str, int]:
    """Counts in ladder order, dropping labels that never occurred.

    Only observed labels appear, so ``len(stats.complexity)`` is the number of
    distinct labels in the log -- the thing a corpus-quality assertion wants.
    Anything outside the ladder (a custom policy can invent tiers) is appended in
    sorted order rather than silently dropped.
    """
    ordered = {label: counts[label] for label in ladder if counts.get(label)}
    order = set(ladder)
    ordered.update({label: n for label, n in sorted(counts.items()) if label not in order and n})
    return ordered


def _summarize(path: Path, *, excerpt_mode: str) -> LogStats:
    """Read the finished log back and count what is in it."""
    records = read_log(path)
    tiers: Counter[str] = Counter(r.decision.tier for r in records)
    complexity: Counter[str] = Counter(r.decision.effective_complexity for r in records)
    sensitivity: Counter[str] = Counter(r.decision.effective_sensitivity for r in records)
    domain: Counter[str] = Counter(r.decision.answers.domain.choice for r in records)
    return LogStats(
        path=str(path),
        records=len(records),
        gate_fired=sum(1 for r in records if r.decision.gate.fired),
        # `backend == "gate"` is how the router records "the gate refused to let
        # this reach any backend"; see Router._route's skip_backend branch.
        gate_blocked=sum(1 for r in records if r.decision.backend == "gate"),
        degraded=sum(1 for r in records if r.decision.degraded),
        with_excerpt=sum(1 for r in records if r.excerpt is not None),
        excerpt_mode=excerpt_mode,
        tiers=_ordered_counts(tiers, TIERS),
        complexity=_ordered_counts(complexity, COMPLEXITY_LEVELS),
        sensitivity=_ordered_counts(sensitivity, SENSITIVITY_LEVELS),
        domain=_ordered_counts(domain, DOMAINS),
        teacher_model_version=_teacher_version(records),
    )


__all__ = [
    "GATED_PROMPTS",
    "GATED_PROMPT_SET",
    "POLICY_PATH",
    "PROMPTS",
    "REPO_ROOT",
    "LogStats",
    "build_log",
    "default_policy",
    "read_log",
]


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
def _print_stats(stats: LogStats, *, header: str = "synthetic decision log") -> None:
    """Human-readable dump of a :class:`LogStats`, in ladder order per dimension."""
    print(f"{header}: {stats.path}")
    print(f"  records              {stats.records}")
    print(f"  gate_fired           {stats.gate_fired}   (gate_blocked {stats.gate_blocked}, degraded {stats.degraded})")
    print(f"  with_excerpt         {stats.with_excerpt}   (excerpt_mode {stats.excerpt_mode!r})")
    print(f"  teacher_model        {stats.teacher_model_version}")
    for label, counts in (
        ("tiers", stats.tiers),
        ("complexity", stats.complexity),
        ("sensitivity", stats.sensitivity),
        ("domain", stats.domain),
    ):
        pretty = "  ".join(f"{key}={value}" for key, value in counts.items())
        print(f"  {label:<19}{pretty}")
    print(f"  repr                 {stats!r}")


def _check_corpus() -> None:
    """The corpus is the asset; these are the properties that make it worth having."""
    assert len(PROMPTS) >= 120, f"corpus too small to train on: {len(PROMPTS)} prompts"
    assert len(set(PROMPTS)) == len(PROMPTS), "corpus contains duplicate prompts"
    assert all(prompt.strip() for prompt in PROMPTS), "corpus contains an empty prompt"
    # GATED_PROMPTS must be exactly the default gate's opinion of the corpus, or
    # every test reasoning about gate coverage is reasoning about a stale list.
    for prompt in PROMPTS:
        fired = default_gate.scan(excerpt_from_text(prompt)).fired
        assert fired == (prompt in GATED_PROMPT_SET), f"GATED_PROMPTS disagrees with the gate: {prompt[:60]!r}"
    assert len(GATED_PROMPTS) >= 8, f"too few gate-firing prompts to be useful: {len(GATED_PROMPTS)}"
    # Every detector must be reachable from the corpus, so a test that filters on
    # `gate_detectors` is never filtering on a column that is always empty.
    covered = {name for prompt in GATED_PROMPTS for name in default_gate.scan(excerpt_from_text(prompt)).detectors()}
    missing = {detector.name for detector in default_gate.detectors} - covered
    assert not missing, f"corpus never trips these detectors: {sorted(missing)}"
    assert any("4111 1111 1111 1111" in p for p in GATED_PROMPTS), "corpus lost the Visa test card prompt"
    assert any("@example.com" in p for p in PROMPTS), "corpus lost the RFC 2606 email contrast case"


def _check_hash_log(stats: LogStats, records: list[DecisionRecord], *, expected_fired: int) -> None:
    """Invariants of the default (hash) log: no text retained, real gate, varied labels."""
    assert stats.records == len(records) == 300, f"expected 300 records, got {stats.records}"
    assert stats.excerpt_mode == "hash"
    ids = [record.request_id for record in records]
    assert len(set(ids)) == len(ids), "request_id collided; a split keyed on it would be ambiguous"
    assert all(len(i) == 32 and all(c in "0123456789abcdef" for c in i) for i in ids), (
        "request_id must keep the 32-hex-char shape the router's uuid4().hex had"
    )
    assert all(record.schema_version == "1" for record in records)
    assert all(record.kind == "jev_route.decision" for record in records)
    assert all(record.total_latency_ms >= record.backend_latency_ms for record in records)

    # The point of the fixture: the gate must be exercised, and the labels varied
    # enough that a training test can actually fail when the trainer is broken.
    assert stats.gate_fired > 0, "no record made the gate fire; the corpus is useless for gate tests"
    assert stats.gate_blocked > 0, "no record was blocked from reaching a backend"
    assert stats.gate_blocked <= stats.gate_fired
    assert len(stats.complexity) >= 3, f"need >=3 complexity labels, got {sorted(stats.complexity)}"
    assert len(stats.sensitivity) >= 3, f"need >=3 sensitivity labels, got {sorted(stats.sensitivity)}"
    assert len(stats.tiers) >= 2, f"need >=2 tiers, got {sorted(stats.tiers)}"
    assert len(stats.domain) == len(DOMAINS), f"corpus does not cover every domain: {stats.domain}"
    assert stats.degraded == 0, "MockBackend never degrades; inject a failing backend to test that path"
    assert stats.teacher_model_version == MockBackend.model_version

    # The gate count must follow from the corpus and the plan -- not from whatever
    # the router happened to log. This is what proves the multi-turn records did
    # not smuggle a gate hit in through their prior turns.
    assert stats.gate_fired == expected_fired, (
        f"log has {stats.gate_fired} gate-fired records, the corpus implies {expected_fired}"
    )
    for record in records:
        blocked = record.decision.backend == "gate"
        assert blocked == (record.decision.gate.blocks_backend or record.decision.gate.force_local)
        assert record.decision.gate.fired == bool(record.decision.gate.findings)
        assert record.decision.tier, "every record must resolve to a tier"

    # hash mode retains no text at all. That is the privacy contract, not a detail.
    assert stats.with_excerpt == 0, "hash mode must not retain prompt text"
    assert all(record.excerpt is None for record in records)


def _check_redacted_log(
    stats_hash: LogStats, stats_red: LogStats, records_red: list[DecisionRecord], path: Path
) -> None:
    """Redacted mode keeps text -- but never for a request the gate blocked."""
    assert stats_red.with_excerpt == stats_red.records - stats_red.gate_blocked
    for record in records_red:
        if record.decision.backend == "gate":
            assert record.excerpt is None, "a gate-blocked request must never be written as text"
            continue
        assert record.excerpt, "redacted mode should retain the excerpt of every classified request"
        # Anything the gate would force local must have been redacted away, so the
        # retained text can never re-trip the identifier detectors.
        assert not default_gate.scan(record.excerpt).force_local, (
            f"retained excerpt still trips the gate: {record.excerpt[:80]!r}"
        )
        assert "4111 1111 1111 1111" not in record.excerpt
        assert "4111111111111111" not in record.excerpt
    # Same corpus, same seed: the two modes must agree on every label. Only the
    # retained text may differ.
    assert replace(stats_hash, path="") == replace(stats_red, path="", excerpt_mode="hash", with_excerpt=0)
    # Serialization is stable round-trip: what is on disk parses back to itself.
    for line, record in zip(path.read_text(encoding="utf-8").splitlines(), records_red, strict=True):
        assert DecisionRecord.from_json(line) == record


def _check_determinism(tmp: Path) -> None:
    """Same arguments -> identical bytes; different seed -> identical labels."""
    first = tmp / "synthlog-determinism-a.jsonl"
    second = tmp / "synthlog-determinism-b.jsonl"
    stats_a = build_log(first, n=155, seed=3, excerpt_mode="redacted")
    stats_b = build_log(second, n=155, seed=3, excerpt_mode="redacted")
    assert first.read_bytes() == second.read_bytes(), "two builds with the same arguments must be identical"
    assert replace(stats_a, path="") == replace(stats_b, path="")

    # A different seed re-keys the log without changing a single label, which is
    # what makes it safe to reshuffle a train/holdout split between runs.
    third = tmp / "synthlog-seed-c.jsonl"
    stats_c = build_log(third, n=155, seed=99, excerpt_mode="redacted")
    ids_a = {record.request_id for record in read_log(first)}
    ids_c = {record.request_id for record in read_log(third)}
    assert ids_a != ids_c and ids_a.isdisjoint(ids_c), "seed must change the request ids"
    assert replace(stats_a, path="") == replace(stats_c, path=""), "seed must not change any label"

    # repeats must produce distinct excerpts, or a split would leak text from the
    # training half into the holdout half and inflate every metric downstream.
    repeated = tmp / "synthlog-repeats.jsonl"
    stats_rep = build_log(repeated, repeats=2, excerpt_mode="redacted", seed=1)
    assert stats_rep.records == 2 * len(PROMPTS)
    hashes = [record.excerpt_hash for record in read_log(repeated)]
    assert len(set(hashes)) == len(hashes), "variant suffixes failed; repeated excerpts would leak across a split"


def _check_edge_cases(tmp: Path) -> None:
    """Empty logs and running event loops must not be a special case for callers."""
    empty = tmp / "synthlog-empty.jsonl"
    stats_empty = build_log(empty, n=0)
    assert stats_empty.records == 0 and stats_empty.teacher_model_version == ""
    assert stats_empty.complexity == {} and read_log(empty) == []

    # pytest-asyncio tests call this fixture from inside a running loop, where
    # asyncio.run would raise. Prove the thread fallback works.
    async def _inside_loop() -> LogStats:
        return build_log(tmp / "synthlog-async.jsonl", n=20, seed=5)

    stats_async = asyncio.run(_inside_loop())
    assert stats_async.records == 20


def _self_check() -> int:
    """Build the demo logs, assert every invariant above, print the stats.

    Run as ``python tests/distill/synthlog.py``. This is the fixture's own test
    suite, kept in ``__main__`` so the module is self-checking: anyone editing the
    corpus or the normalizer immediately learns whether the log it produces is
    still varied, still gated where it should be, and still byte-reproducible.
    """
    tmp = Path("/tmp")
    _check_corpus()

    demo = tmp / "synthlog-demo.jsonl"
    stats = build_log(demo, n=300, seed=0)
    records = read_log(demo)
    _print_stats(stats)
    expected_fired = sum(1 for index, _ in _plan(n=300, repeats=1) if PROMPTS[index % len(PROMPTS)] in GATED_PROMPT_SET)
    _check_hash_log(stats, records, expected_fired=expected_fired)

    redacted = tmp / "synthlog-demo-redacted.jsonl"
    stats_red = build_log(redacted, n=300, seed=0, excerpt_mode="redacted")
    records_red = read_log(redacted)
    print()
    _print_stats(stats_red, header="synthetic decision log (redacted)")
    _check_redacted_log(stats, stats_red, records_red, redacted)

    _check_determinism(tmp)
    _check_edge_cases(tmp)

    print()
    verdicts = [default_gate.scan(excerpt_from_text(p)) for p in GATED_PROMPTS]
    print(
        f"corpus: {len(PROMPTS)} prompts; {len(GATED_PROMPTS)} gate-firing "
        f"({sum(v.blocks_backend for v in verdicts)} blocking, "
        f"{sum(v.force_local and not v.blocks_backend for v in verdicts)} force-local only, "
        f"{sum(not v.force_local for v in verdicts)} advisory only)"
    )
    print("self-check passed: determinism, gate coverage, label spread and the privacy contract all hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
