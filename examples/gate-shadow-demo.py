"""Layer-2 shadow-mode demonstration: the real artifact, running, able to change nothing.

Run it (no network, no numpy, no API key):

    .venv/bin/python examples/gate-shadow-demo.py

What it shows, with one router in ``gate.semantic.mode: shadow``:

  (a) a benign prompt                      -> passes both layers, served normally
  (b) a prompt with a real-shaped SSN      -> caught by layer 1; the backend is blocked
  (c) a contextually sensitive prompt with
      no regex hits                        -> flagged by layer 2 (score above
      threshold), disagreement logged, and
      still served: shadow cannot change the decision

The layer-2 artifact is trained by ``fit_scorer``, the pure-Python reference
trainer, on contextual-sensitive positives (no identifiers in them -- exactly
what regex cannot see) and benign production negatives. Nothing here is a stub;
the trained weights are a hash-deterministic function of the corpus, so the
printed model version is the same on every run.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from jev_route import MockBackend, Router
from jev_route.cache import NullCache
from jev_route.gate_semantic import SemanticArtifact, fit_scorer
from jev_route.logging_sink import CallbackSink, CompositeSink, build_sink
from jev_route.policy import Policy

#: Contextual-sensitive prose: sensitive while containing no identifier at all.
POSITIVE_FRAGMENTS = [
    "the disciplinary hearing notes from last week",
    "our disciplinary hearing went badly and the employee may resign",
    "minutes from the disciplinary hearing are being circulated",
    "the patient reports persistent pain after the procedure",
    "the patient reports new symptoms since the last visit",
    "her clinical notes describe the diagnosis in detail",
    "his clinical notes were reviewed by the board",
    "the hr file contains the review outcome and salary",
    "the hr file lists the grievances raised by the team",
    "the performance review cites the incident from march",
    "the performance review recommends a formal warning",
    "the termination letter states the effective date",
    "the termination letter references the final interview",
]
#: Benign traffic. Real production negatives come from the decision log after the
#: gate has cleared them; these stand in for the demo.
NEGATIVE_FRAGMENTS = [
    "how do I center a div in css",
    "write a python function that sorts a list of dictionaries",
    "what is the best way to store configuration in a container",
    "explain the difference between a stack and a queue",
    "how do i deploy a flask app to heroku",
    "why does my sql query return duplicate rows",
    "what is the idiomatic way to parse dates in rust",
    "how do I make a table responsive in tailwind",
    "explain what a virtual machine monitor does",
    "how do I add pagination to a rest endpoint",
    "what is the difference between an index and a scan",
    "how do I name a git branch for a bugfix",
]


async def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="jev-shadow-demo-"))

    # Four variants per fragment: the holdout split is keyed on content hash, and
    # a handful of unique rows can land on one side of the split with nothing on
    # the other.
    positives = [f" {f} (variant {v + 1})" for v in range(4) for f in POSITIVE_FRAGMENTS]
    negatives = [f" {f} (variant {v + 1})" for v in range(4) for f in NEGATIVE_FRAGMENTS]

    # Train a real layer-2 artifact (pure Python; no numpy, no network).
    artifact: SemanticArtifact = fit_scorer(positives, negatives, epochs=200)
    artifact.save(out)
    m = artifact.metrics
    print(f"trained layer-2 artifact: {out / 'semantic.json'}")
    print(f"  model_version={m.get('model_version')!r}  scorer={m.get('scorer')!r}  threshold={artifact.threshold}")
    print(f"  holdout: n={m['n_examples']} (pos {m['n_positives']} / neg {m['n_negatives']})")
    print(
        f"  holdout metrics: recall={m['recall']}  fpr={m['false_positive_rate']}  "
        f"disagreement={m['disagreement_rate']}"
    )
    print(f"  mean score: positive={m['mean_score_positive']:.3f}  negative={m['max_score_negative']}")

    records: list = []
    policy = Policy.from_dict(
        {
            "version": 1,
            "backend": {"name": "mock"},
            "tiers": {"local": ["local-model"], "cheap": ["cheap-model"], "strong": ["strong-model"]},
            "tier_order": ["cheap", "strong", "local"],
            "gate": {"on_force_local": "skip_backend", "semantic": {"mode": "shadow", "artifact": str(out)}},
            "cache": {"enabled": False},
            "logging": {"enabled": True, "path": str(out / "decisions.jsonl"), "excerpt_mode": "hash"},
            "rules": [
                {"id": "gate.force-local", "if": "gate_force_local", "then": {"tier": "local"}},
                {
                    "id": "data.sensitive",
                    "if": 'sensitivity in ["confidential", "regulated"] or pii_present',
                    "then": {"tier": "local"},
                },
                {"id": "default", "then": {"tier": "cheap"}},
            ],
        }
    )
    # Two sinks: the in-memory list (so the demo can print each record) and the
    # JSONL file the policy names. A single CallbackSink would make the closing
    # "decision log written to" line a lie.
    router = Router(
        policy,
        MockBackend(),
        cache=NullCache(),
        sink=CompositeSink([CallbackSink(records.append), build_sink(policy.logging)]),
    )

    prompts = {
        "(a) benign, no hits": "how do I center a div in css",
        "(b) real-shaped SSN": "please update my profile, my SSN is 123-45-6789",
        "(c) contextual, no regex hits": "can you summarise the notes from our disciplinary hearing last week",
    }
    for label, prompt in prompts.items():
        decision = await router.route_text(prompt)
        record = records[-1]
        sem = record.semantic or {}
        v = decision.gate
        print()
        print(label)
        print(f"  prompt:        {prompt!r}")
        print(
            f"  layer 1:       fired={v.fired} detectors={v.detectors()} "
            f"floor={v.sensitivity_floor or '-'} blocks_backend={v.blocks_backend}"
        )
        print(
            f"  layer 2:       mode={sem.get('mode')} score={sem.get('score'):.3f} "
            f"fired={sem.get('fired')} enforced={sem.get('enforced')} "
            f"disagreement={sem.get('disagreement') or '-'}"
        )
        print(
            f"  served:        tier={decision.tier} model={decision.model!r} backend={decision.backend!r} "
            f"via rule {decision.rule_id}"
        )
        print(f"  sensitivity:   {decision.effective_sensitivity}")

    print()
    print(f"decision log written to {out / 'decisions.jsonl'} (excerpt_mode=hash: no text stored)")


if __name__ == "__main__":
    asyncio.run(main())
