"""Turning a request into something a decision backend is allowed to see.

Three jobs, in the order they must happen:

1. **Excerpt.** Pull the text that actually carries the routing signal out of a
   chat-completions message list, bounded by a character budget. Routing does not
   need the whole conversation and should not pay for it.
2. **Redact.** Replace anything the local gate matched with a typed placeholder.
   Redaction is what makes it acceptable to ask a *remote* backend about
   complexity for a prompt that mentions a person's email address.
3. **Feature.** Compute a deterministic, text-free description of the request.
   These features are logged with every decision, which is what lets a distilled
   model be trained even by an operator who never stores prompt text.

The ordering is a privacy property, not a style choice: the gate scans the *raw*
excerpt, redaction is applied to the raw excerpt, and only the redacted form is
ever handed to a backend. See ``docs/privacy.md``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .gate import HardGate, default_gate
from .schema import RequestFeatures

#: Roles whose text carries routing signal. ``system`` prompts are usually
#: boilerplate the operator wrote once and would only dilute the read.
ROUTING_RELEVANT_ROLES = ("user", "assistant")

#: Hard ceiling on what leaves the process. Even before truncation this bounds the
#: blast radius of a misconfigured policy.
MAX_EXCERPT_CHARS = 4000

_CODE_FENCE = re.compile(r"```")
_INLINE_CODE = re.compile(r"`[^`\n]{1,}`")
_URL = re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s<>\"']+", re.IGNORECASE)
# Java/JS frames, Python tracebacks, and native backtraces. Three shapes, one
# boolean: "is there a stack trace in here", which is a strong complexity signal.
_STACK_FRAME = re.compile(
    r"(?:at\s+[\w.$<>]+\s*\([^)]*\)"
    r"|File\s+\"[^\"]+\",\s+line\s+\d+"
    r"|^\s*#\d+\s+0x[0-9a-f]+)",
    re.MULTILINE,
)
_DIFF_LINE = re.compile(r"^(?:[+-]{3}\s|@@\s|[-+](?!---|\+\+))", re.MULTILINE)
_JSONISH = re.compile(r"^\s*[\[{].*[\]}]\s*$", re.DOTALL)
_SENTENCE_END = re.compile(r"[.!?]+\s")

#: Tiny stopword fingerprints. Enough to tell the languages a router actually
#: sees apart; not a language detector, and does not claim to be one.
_LANG_STOPWORDS: dict[str, tuple[str, ...]] = {
    "en": ("the", "and", "is", "of", "to", "you", "that", "it", "for", "with"),
    "de": ("der", "die", "und", "ist", "nicht", "sie", "mit", "auf", "für", "ein"),
    "fr": ("le", "la", "et", "est", "des", "une", "pour", "avec", "dans", "vous"),
    "es": ("el", "la", "que", "de", "y", "no", "un", "por", "con", "para"),
    "it": ("il", "che", "di", "e", "la", "non", "per", "un", "con", "sono"),
    "pt": ("o", "que", "de", "e", "não", "um", "para", "com", "os", "por"),
    "hu": ("a", "az", "és", "hogy", "nem", "egy", "meg", "van", "már", "kell"),
    "nl": ("de", "het", "en", "een", "van", "is", "dat", "niet", "voor", "met"),
}


def flatten_content(content: Any) -> str:
    """Normalize OpenAI-style message content to plain text.

    Handles the three shapes that actually appear on the wire: a bare string, a
    list of typed parts (``{"type": "text" | "image_url" | ...}``), and ``None``.
    Non-text parts collapse to a short marker -- routing on an image is out of
    scope, and pretending otherwise would be a lie in the audit log.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        if content.get("type") == "text" and isinstance(content.get("text"), str):
            return content["text"]
        return f"[{content.get('type', 'non-text')}]"
    if isinstance(content, Sequence):
        return "\n".join(part for part in (flatten_content(item) for item in content) if part)
    return str(content)


def excerpt_messages(
    messages: Sequence[Mapping[str, Any]] | None,
    *,
    max_chars: int = MAX_EXCERPT_CHARS,
    include_prior_turns: int = 2,
    include_system: bool = False,
) -> str:
    """Build the routing excerpt from a chat-completions message list.

    The newest user turn gets the largest share of the budget, because that is
    where the routing signal lives. Earlier turns are included only as far as the
    budget allows and are prepended oldest-first, so a truncated excerpt is still
    readable in order.

    ``include_prior_turns`` counts *user* turns, not messages.
    """
    if not messages:
        return ""
    if max_chars <= 0:
        return ""

    allowed = set(ROUTING_RELEVANT_ROLES)
    if include_system:
        allowed.add("system")

    # Newest-first walk so the budget is spent on the most relevant text.
    picked: list[str] = []
    user_turns = 0
    budget = max_chars
    for message in reversed(list(messages)):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", ""))
        if role not in allowed:
            # Tool output can dwarf a request and is almost never routing
            # signal; it is counted as a feature instead.
            continue
        text = flatten_content(message.get("content")).strip()
        if not text:
            continue
        if role == "user":
            user_turns += 1
            if user_turns > include_prior_turns + 1:
                break
        take = min(len(text), budget)
        if take <= 0:
            break
        picked.append(text[:take])
        budget -= take
        if budget <= 0:
            break

    picked.reverse()
    excerpt = "\n\n".join(picked)
    return excerpt[:max_chars]


def excerpt_from_text(text: str, *, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    """Excerpt a bare string (used by the CLI, evals, and text-completion calls)."""
    if not text:
        return ""
    return text.strip()[:max_chars]


def redact(text: str, gate: HardGate | None = None, *, placeholder: str = "[{name}]") -> tuple[str, int]:
    """Replace every gate-matched span with a typed placeholder.

    Returns ``(redacted_text, n_replacements)``. Overlapping spans are collapsed
    and applied right-to-left so earlier offsets stay valid.

    The placeholder names the detector, not the value: ``[payment_card]`` tells a
    complexity classifier "something regulated was here" -- which is the signal
    that matters -- without reproducing it.
    """
    if not text:
        return text, 0
    active = gate or default_gate
    spans: list[tuple[int, int, str]] = []
    for name, matched in active.redaction_map(text).items():
        for start, end in matched:
            spans.append((start, end, name))
    if not spans:
        return text, 0

    spans.sort(key=lambda s: (s[0], -s[1]))
    merged: list[tuple[int, int, str]] = []
    for start, end, name in spans:
        if merged and start <= merged[-1][1]:
            prev_start, prev_end, prev_name = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), f"{prev_name}+{name}")
        else:
            merged.append((start, end, name))

    out = text
    for start, end, name in reversed(merged):
        out = out[:start] + placeholder.format(name=name) + out[end:]
    return out, len(merged)


def hash_text(text: str, *, salt: str = "") -> str:
    """Stable 16-hex-char digest used as ``excerpt_hash`` in the decision log.

    Salted by default from config so that two deployments sharing a log store
    cannot trivially cross-reference prompts. The hash is a *dedup and cache key*,
    not an anonymization guarantee: a short prompt from a known corpus is still
    guessable. That is documented, not hidden.
    """
    return hashlib.sha256((salt + "\x00" + text).encode("utf-8")).hexdigest()[:16]


def detect_language(text: str) -> str:
    """Coarse language guess from stopword overlap. ``und`` when unsure."""
    words = re.findall(r"[^\W\d_]+", text.lower())
    if len(words) < 4:
        return "und"
    counts = {lang: sum(1 for w in words if w in stops) for lang, stops in _LANG_STOPWORDS.items()}
    best = max(counts.items(), key=lambda kv: kv[1])
    # Require a real signal; two stopwords out of a thousand is noise.
    return best[0] if best[1] >= max(2, len(words) // 40) else "und"


def compute_features(
    text: str,
    *,
    messages: Sequence[Mapping[str, Any]] | None = None,
    gate_detectors: Mapping[str, int] | None = None,
    n_gate_findings: int = 0,
    gate_force_local: bool = False,
) -> RequestFeatures:
    """Deterministic, text-free description of a request.

    Nothing here reproduces the prompt: only counts, ratios, and booleans. This is
    the feature vector a distilled model can be trained on when an operator has
    chosen never to store prompt text.
    """
    words = re.findall(r"\S+", text)
    letters = [c for c in text if c.isalpha()]
    n_chars = len(text)
    n_words = len(words)
    n_letters = len(letters) or 1

    n_messages = len(messages) if messages else (1 if text else 0)
    prior_turns = 0
    tool_output = False
    if messages:
        prior_turns = max(0, sum(1 for m in messages if isinstance(m, Mapping) and m.get("role") == "user") - 1)
        tool_output = any(isinstance(m, Mapping) and str(m.get("role", "")) in ("tool", "function") for m in messages)

    return RequestFeatures(
        char_len=n_chars,
        word_count=n_words,
        line_count=text.count("\n") + 1 if text else 0,
        sentence_count=len([s for s in _SENTENCE_END.split(text) if s.strip()]) if text else 0,
        mean_word_len=round(sum(len(w) for w in words) / n_words, 3) if n_words else 0.0,
        digit_ratio=round(sum(c.isdigit() for c in text) / n_chars, 4) if n_chars else 0.0,
        upper_ratio=round(sum(c.isupper() for c in letters) / n_letters, 4),
        punct_ratio=round(sum(not c.isalnum() and not c.isspace() for c in text) / n_chars, 4) if n_chars else 0.0,
        non_ascii_ratio=round(sum(ord(c) > 127 for c in text) / n_chars, 4) if n_chars else 0.0,
        code_blocks=len(_CODE_FENCE.findall(text)) // 2,
        inline_code_spans=len(_INLINE_CODE.findall(text)),
        urls=len(_URL.findall(text)),
        question_marks=text.count("?"),
        exclamations=text.count("!"),
        has_stack_trace=bool(_STACK_FRAME.search(text)),
        has_diff=bool(_DIFF_LINE.search(text)),
        has_json=bool(_JSONISH.match(text.strip())) if text.strip() else False,
        lang=detect_language(text),
        n_messages=n_messages,
        n_prior_turns=prior_turns,
        tool_output_present=tool_output,
        gate_detectors=dict(gate_detectors or {}),
        n_gate_findings=n_gate_findings,
        gate_force_local=gate_force_local,
    )


#: Metadata keys that can carry the prompt itself.
#:
#: Callers routinely forward a whole request body as metadata, so ``metadata`` is
#: treated as untrusted input on every path that leaves this process. This list is
#: shared by the cloud path (:func:`build_backend_state`) and the decision log
#: (``Router._log``) on purpose: filtering it in one place and not the other is how
#: a router that promises ``excerpt_mode: hash`` ends up writing raw prompt text
#: into its own training dataset.
UNSAFE_METADATA_KEYS: tuple[str, ...] = (
    "messages",
    "prompt",
    "content",
    "input",
    "raw",
    "body",
)


def safe_caller_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy caller metadata with the prompt-bearing keys removed.

    Key-based, not type-based. A scalar ``{"prompt": "..."}`` is the dangerous
    case: it passes any "is this JSON-serialisable" check and still leaks the
    text. Structural filtering belongs to the caller.
    """
    meta = dict(metadata or {})
    for unsafe in UNSAFE_METADATA_KEYS:
        meta.pop(unsafe, None)
    return meta


def build_backend_state(
    redacted_excerpt: str,
    *,
    features: RequestFeatures,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``state`` object handed to a decision backend.

    Named fields rather than one blob, because a System One model reads structure:
    the excerpt is the thing being judged, the features are context about its
    shape, and the metadata is who is asking. Never includes raw text.
    """
    meta = safe_caller_metadata(metadata)
    return {
        "prompt_excerpt": redacted_excerpt,
        "request_features": features.to_dict(),
        "caller_metadata": meta,
    }


__all__ = [
    "MAX_EXCERPT_CHARS",
    "ROUTING_RELEVANT_ROLES",
    "UNSAFE_METADATA_KEYS",
    "build_backend_state",
    "compute_features",
    "detect_language",
    "excerpt_from_text",
    "excerpt_messages",
    "flatten_content",
    "hash_text",
    "redact",
    "replace",
    "safe_caller_metadata",
]
