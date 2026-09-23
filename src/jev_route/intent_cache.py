"""Intent cache: a near-duplicate shortcut checked BEFORE routing.

This is a second, different cache. :mod:`jev_route.cache` caches the backend\'s
*judgement* behind the router (same redacted excerpt -> same answers, so the
network round-trip is skipped but the policy still runs). This one caches the
*decision* (tier + model + why) in front of the router, for text that is
identical or near-identical to something already seen. On a hit, routing is
skipped entirely -- no gate pass, no backend call, no log record -- which is
what makes it useful for the repetitive traffic a production proxy mostly is
(the same support macro, the same onboarding prompt, the same batch of
summaries re-phrased one token at a time).

The trade is stated plainly, because it is the reason this cache is opt-in and
bounded: a cached *decision* goes stale when the policy changes, for as long
as the TTL runs. A policy edit applies immediately to every cache miss and to
every cached request once its entry expires. Set ``ttl_s`` short when you
change policies often.

**Matching, in order, and why:**

1. **Exact, after normalization.** NFKC + casefold + whitespace collapse, so
   ``"  How do I optimize THE p99 latency? "`` is the same intent as
   ``"how do i optimize the p99 latency"``. A dict hit: O(1).
2. **Token-set Jaccard, at or above :data:`DEFAULT_JACCARD_THRESHOLD`.**
   ``"a b c d e"`` vs ``"a b c d f"`` is 4/5 = 0.8 -- same shape of request,
   one different word. Sets, not bags: multiplicity is not what makes two
   prompts the same intent. O(n) over at most ``max_entries`` small sets --
   cheap enough for the hot path at the default 256.
3. **The semantic seam.** When the policy says ``intent_cache.semantic:
   true`` and a seam is wired, an exact/Jaccard miss asks ONE batched noul
   over the top-K (K <= 5) most-similar cached summaries: *same intent and
   same constraints?* The answer is a probability; at or above
   :data:`DEFAULT_SEMANTIC_THRESHOLD` the most similar candidate is replayed
   and the result carries ``hit_kind`` ``"semantic"``. The model call comes
   in through an injectable ``noul_fn(prompt_state) -> float`` (see
   :func:`semantic_match_from_noul`) or a raw ``semantic_match`` callable;
   this module makes no model call of its own and owns no client. A semantic
   hit whose token overlap with the replayed summary is below
   :data:`DEFAULT_SEMANTIC_OVERLAP_FLOOR` is replayed anyway but counted in
   ``semantic_suspicious`` -- a false replay must be visible, not hidden.

**The gate contract.** A request the local sensitivity gate fired on is
*never* cached and *never* served from the cache. ``store`` and ``lookup``
both take ``gate_fired: bool`` and no-op when it is true. The reasoning: the
gate exists so credential-bearing and regulated text leaves no durable trace,
and a decision cache that replays answers for such text is a retention path
for exactly the content the gate refuses to hold. A deterministic gate makes
this safe in the reverse direction too: the same text always fires the same
way, so a text that was clean when stored is clean now.

Dependency-free on purpose: stdlib only, single process, one lock. It is
checked in front of the router by whoever owns the request -- the integrations
and the CLI -- not inside it, because the router is the shared core and this
cache is a deployment choice.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .schema import RoutingDecision

LOGGER = logging.getLogger(__name__)

#: Jaccard score at or above which two token sets are the same intent.
#: 0.9 means "shares 90% of its tokens with a cached request" -- close enough
#: that the same tier/model is almost certainly still the right one, far
#: enough from 1.0 that a genuinely different ask does not collide.
DEFAULT_JACCARD_THRESHOLD = 0.9

_DEFAULT_MAX_ENTRIES = 256
_DEFAULT_TTL_S = 3600.0

#: The model-based matching seam. ``text`` is the normalized incoming text,
#: ``candidates`` the live cached entries; return one of them (or any
#: :class:`CachedAnswer`) on a match, ``None`` otherwise. Must not raise;
#: exceptions are caught and counted as a miss. See ``IntentCache.lookup``.
SemanticMatch = Callable[[str, Sequence["CachedAnswer"]], "CachedAnswer | None"]

#: An injectable noul: ONE probability in ``[0, 1]`` for the batched question
#: "same intent and same constraints?" over a ``prompt_state`` mapping (the
#: incoming normalized text plus the top-K candidate summaries). The cache
#: never calls a model; the integration supplies this callable.
NoulFn = Callable[[Mapping[str, Any]], float]

#: Probability at or above which the batched noul's answer replays a cached
#: decision. 0.7 sits where the rest of this codebase treats a noul as a
#: "yes" -- above coin-flip territory, below overconfidence.
DEFAULT_SEMANTIC_THRESHOLD = 0.7

#: How many of the most-similar cached summaries ride in the batched question.
DEFAULT_SEMANTIC_TOP_K = 5

#: Hard cap on the batch, whatever the caller asks for: the question stays one
#: small call no matter how full the cache is.
MAX_SEMANTIC_TOP_K = 5

#: A semantic hit that replays a summary whose token overlap with the incoming
#: text is below this floor is counted as *suspicious* -- replayed anyway (the
#: model said "same intent and same constraints"), but flagged in
#: ``stats()["semantic_suspicious"]`` and logged, because a confident noul over
#: barely-overlapping tokens is exactly what a false replay looks like.
DEFAULT_SEMANTIC_OVERLAP_FLOOR = 0.3

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFKC + casefold + whitespace collapse: the cache key for "exact".

    NFKC folds compatibility forms (fullwidth letters, ligatures, a
    superscript-9 ``p\u20799``) onto the same characters a plain ASCII prompt
    uses, which is where near-duplicates in real traffic mostly hide.
    """
    if not isinstance(text, str):
        raise TypeError(f"normalize_text expects str, got {type(text).__name__}")
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def token_set(text: str) -> frozenset[str]:
    """The word tokens of ``text``, as a set. Empty for empty/whitespace text."""
    if not isinstance(text, str):
        raise TypeError(f"token_set expects str, got {type(text).__name__}")
    return frozenset(_TOKEN_RE.findall(normalize_text(text)))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Token-set Jaccard similarity in ``[0, 1]``. ``0.0`` when either set is empty."""
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


@dataclass(frozen=True)
class CachedAnswer:
    """One cached decision, plus the identity of the text it was stored under.

    ``stored_at`` is on the cache\'s clock (``time.monotonic`` by default), so
    it is for the cache\'s own expiry arithmetic, not for display.
    ``matched`` records how a lookup found this entry: ``"exact"``,
    ``"near"`` (token-set Jaccard) or ``"semantic"`` (the model-based seam).
    """

    decision: RoutingDecision
    normalized_text: str
    stored_at: float
    matched: str = "exact"
    similarity: float = 1.0

    @property
    def hit_kind(self) -> str:
        """How the lookup found this entry: ``"exact"``, ``"near"`` or ``"semantic"``.

        A miss produces no :class:`CachedAnswer` at all (``lookup`` returns
        ``None``), so the miss side of the ``"exact" / "near" / "semantic" /
        None`` partition is the absence of a result.
        """
        return self.matched


def semantic_match_from_noul(
    noul_fn: NoulFn,
    *,
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    top_k: int = DEFAULT_SEMANTIC_TOP_K,
) -> SemanticMatch:
    """Build the :data:`SemanticMatch` seam from an injectable noul callable.

    This is the model-based matching pass, implemented as ONE batched noul:
    the top-K (K <= :data:`MAX_SEMANTIC_TOP_K`) most token-similar cached
    summaries ride in a single ``prompt_state``, and ``noul_fn`` answers one
    question -- *same intent and same constraints?* -- with a probability.
    At or above ``threshold`` the most similar candidate is replayed; below
    it the miss stands. This module makes no model call and owns no client:
    ``noul_fn`` is the integration's seam (a System One ``noul`` bound to the
    deployment's backend, or a deterministic double in tests).

    The ``prompt_state`` mapping the callable receives::

        {"question": "same intent and same constraints?",
         "incoming": "<normalized incoming text>",
         "candidates": [{"summary": "<normalized cached text>",
                          "token_overlap": 0.42}, ...]}

    candidates are ordered most-similar first. A non-finite probability is a
    miss; one outside ``[0, 1]`` is clamped. An exception from ``noul_fn``
    propagates to :meth:`IntentCache._semantic_pass`, where it is caught,
    logged, and treated as a miss -- a raising noul may cost a round-trip,
    never a request.
    """
    k = max(1, min(int(top_k), MAX_SEMANTIC_TOP_K))
    t = float(threshold)

    def match(norm: str, candidates: Sequence[CachedAnswer]) -> CachedAnswer | None:
        if not candidates:
            return None
        query_tokens = token_set(norm)
        scored = sorted(
            ((c, jaccard(query_tokens, token_set(c.normalized_text))) for c in candidates),
            key=lambda pair: pair[1],
            reverse=True,
        )[:k]
        prompt_state: Mapping[str, Any] = {
            "question": "same intent and same constraints?",
            "incoming": norm,
            "candidates": [
                {"summary": c.normalized_text, "token_overlap": round(score, 6)} for c, score in scored
            ],
        }
        p = float(noul_fn(prompt_state))
        if math.isnan(p):
            return None
        if min(1.0, max(0.0, p)) < t:
            return None
        best, best_score = scored[0]
        return replace(best, matched="semantic", similarity=round(best_score, 6))

    return match


class IntentCache:
    """Bounded, TTL\'d, near-duplicate decision cache. Checked before routing.

    Args:
        max_entries: LRU bound. Eviction is least-recently-used, like
            :class:`jev_route.cache.InMemoryTTLCache`.
        ttl_s: per-entry time-to-live in seconds. ``0`` disables storage.
        jaccard_threshold: minimum token-set Jaccard for a near-duplicate hit.
        clock: monotonic clock; injectable so tests do not sleep.
        semantic_match: the model-based matching seam (see the module
            docstring). ``None`` by default: deterministic matching only.
        noul_fn: injectable ``noul_fn(prompt_state) -> float``; wires the
            seam through :func:`semantic_match_from_noul` with
            ``semantic_threshold`` / ``semantic_top_k``. Mutually exclusive
            with ``semantic_match``; implies ``semantic=True``.
        semantic: the policy flag (``intent_cache.semantic: true``). The
            model-based pass runs only when this is on AND a seam is wired.
            ``True`` for direct construction (wiring is an explicit act);
            :func:`build_intent_cache` passes the policy's value, default
            ``False``.
        semantic_threshold: noul probability at or above which a semantic
            miss becomes a replay (used by ``noul_fn`` wiring).
        semantic_top_k: candidate summaries in the batched question, hard
            capped at :data:`MAX_SEMANTIC_TOP_K` (used by ``noul_fn`` wiring).
        semantic_overlap_floor: a semantic hit whose token overlap with the
            replayed summary is below this floor is replayed but counted in
            ``stats()["semantic_suspicious"]`` and logged (a false replay
            must be visible).

    Thread-safe: one lock, because a proxy calls ``lookup`` from the request
    path and ``store`` from the same path right after the router returns.
    """

    def __init__(
        self,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        ttl_s: float = _DEFAULT_TTL_S,
        *,
        jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
        clock: Callable[[], float] = time.monotonic,
        semantic_match: SemanticMatch | None = None,
        noul_fn: NoulFn | None = None,
        semantic: bool = True,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
        semantic_top_k: int = DEFAULT_SEMANTIC_TOP_K,
        semantic_overlap_floor: float = DEFAULT_SEMANTIC_OVERLAP_FLOOR,
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self.ttl_s = max(0.0, float(ttl_s))
        self.jaccard_threshold = float(jaccard_threshold)
        self._clock = clock
        self._data: OrderedDict[str, tuple[RoutingDecision, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._exact_hits = 0
        self._near_hits = 0
        self._semantic_hits = 0
        self._semantic_suspicious = 0
        self._gate_skips = 0
        self._expired = 0
        self._evictions = 0

        if not isinstance(semantic, bool):
            raise ValueError(f"semantic must be a boolean, got {semantic!r}")
        if not 0.0 <= float(semantic_overlap_floor) <= 1.0:
            raise ValueError(f"semantic_overlap_floor must be in [0, 1], got {semantic_overlap_floor!r}")
        self.semantic_overlap_floor = float(semantic_overlap_floor)

        # Two ways to wire the seam, not both: semantic_match is the raw v0.3
        # seam (any callable returning a CachedAnswer), noul_fn is the batched
        # noul the work order asks for. Passing both is a config that does not
        # know which of the two it is running.
        if semantic_match is not None and noul_fn is not None:
            raise ValueError("pass either semantic_match or noul_fn, not both")
        if noul_fn is not None:
            if not semantic:
                raise ValueError("noul_fn wires the seam, which implies semantic=True")
            semantic_match = semantic_match_from_noul(noul_fn, threshold=semantic_threshold, top_k=semantic_top_k)
            # Passing a noul is an explicit act of wiring: the pass is on, the
            # same as assigning the seam by hand in v0.3.
            semantic = True
        self._noul_fn = noul_fn
        self.semantic = semantic

        # ------------------------------------------------------------------ #
        # SEMANTIC SEAM -- model-based near-duplicates (v0.4).
        #
        # ``lookup`` consults the seam only after the exact and Jaccard passes
        # have both missed, and only when the policy flag ``semantic`` is on
        # AND a seam is wired. The standard seam is :func:`
        # semantic_match_from_noul` (or the ``noul_fn=`` sugar above): ONE
        # batched noul over the top-K most-similar cached summaries, "same
        # intent and same constraints?". That call belongs in an integration,
        # not in this module -- this file is stdlib-only and sits in the hot
        # path, and a router whose cache can phone home has a new outage mode.
        # A wired seam returns an entry stamped ``matched="semantic"`` (the
        # result's ``hit_kind``), or ``None`` for a miss. ``None`` (the
        # default) means the seam is closed and matching is fully
        # deterministic. Exceptions from a hook are caught, logged, and
        # treated as a miss: the cache may cost a round-trip, never a request.
        # ------------------------------------------------------------------ #
        self.semantic_match = semantic_match

    # -- the API ------------------------------------------------------------ #
    def lookup(self, text: str, *, gate_fired: bool = False) -> CachedAnswer | None:
        """The cached decision for ``text``, or ``None``.

        Order: exact-normalized, then token-set Jaccard >= threshold, then the
        ``semantic_match`` seam when one is assigned. ``gate_fired=True``
        returns ``None`` unconditionally: a request the sensitivity gate fired
        on is never served from the cache, whatever is in it.
        """
        if gate_fired:
            self._gate_skips += 1
            return None
        norm = normalize_text(text)
        if not norm:
            self._misses += 1
            return None
        hit, live = self._scan(norm)
        if hit is not None:
            return hit
        # The model-based pass needs BOTH: the policy flag on and a seam
        # wired. A flag without a seam is an unrun deployment; a seam without
        # the flag is a policy that has not opted in.
        if self.semantic and self.semantic_match is not None and live:
            hit = self._semantic_pass(norm, live)
            if hit is not None:
                return hit
        self._misses += 1
        return None

    def _scan(self, norm: str) -> tuple[CachedAnswer | None, list[CachedAnswer]]:
        """Exact then near-duplicate pass over the live entries, under the lock.

        Returns the hit (or ``None``) plus the live entries as
        :class:`CachedAnswer` objects, for the semantic seam to consider.
        Expired entries are dropped on the way past.
        """
        query_tokens = token_set(norm)
        best_key: str | None = None
        best_score = 0.0
        live: list[CachedAnswer] = []
        with self._lock:
            entry = self._data.get(norm)
            if entry is not None:
                decision, expires_at = entry
                if expires_at <= self._clock():
                    del self._data[norm]
                    self._expired += 1
                else:
                    self._data.move_to_end(norm)
                    self._hits += 1
                    self._exact_hits += 1
                    return (
                        CachedAnswer(
                            decision=decision,
                            normalized_text=norm,
                            stored_at=expires_at - self.ttl_s,
                            matched="exact",
                            similarity=1.0,
                        ),
                        [],
                    )

            for key, (decision, expires_at) in list(self._data.items()):
                if expires_at <= self._clock():
                    del self._data[key]
                    self._expired += 1
                    continue
                live.append(CachedAnswer(decision=decision, normalized_text=key, stored_at=expires_at - self.ttl_s))
                if query_tokens:
                    score = jaccard(query_tokens, token_set(key))
                    if score > best_score:
                        best_score = score
                        best_key = key

            if best_key is not None and best_score >= self.jaccard_threshold:
                self._data.move_to_end(best_key)
                self._hits += 1
                self._near_hits += 1
                decision, expires_at = self._data[best_key]
                return (
                    CachedAnswer(
                        decision=decision,
                        normalized_text=best_key,
                        stored_at=expires_at - self.ttl_s,
                        matched="near",
                        similarity=round(best_score, 6),
                    ),
                    [],
                )
            return None, live

    def _semantic_pass(self, norm: str, live: list[CachedAnswer]) -> CachedAnswer | None:
        """The model-based seam. Runs OUTSIDE the lock: a model call under the
        cache lock would stall every other request on this process."""
        hook = self.semantic_match
        if hook is None:
            return None
        try:
            hit = hook(norm, live)
        except Exception as exc:  # broad except, deliberately: a raising hook must not fail the request
            LOGGER.warning(
                "jev-route: semantic_match hook failed (%s: %s); treating as a miss.", type(exc).__name__, exc
            )
            return None
        if hit is None:
            return None
        if not isinstance(hit, CachedAnswer):
            LOGGER.warning(
                "jev-route: semantic_match hook returned %s, not a CachedAnswer; treating as a miss.",
                type(hit).__name__,
            )
            return None
        # A false replay must be visible: the seam said "same intent and same
        # constraints", but the tokens do not back it up. Replay still happens
        # (the model's yes is the decision) -- but the hit is counted in
        # semantic_suspicious and logged, so an overconfident noul shows up in
        # the report instead of hiding inside the hit rate.
        overlap = jaccard(token_set(norm), token_set(hit.normalized_text))
        if overlap < self.semantic_overlap_floor:
            self._semantic_suspicious += 1
            LOGGER.warning(
                "jev-route: suspicious semantic hit: replaying a summary with token overlap %.3f "
                "(below the %.2f floor) although the seam answered 'same intent and same constraints'; "
                "counted in semantic_suspicious for review.",
                overlap,
                self.semantic_overlap_floor,
            )
        self._hits += 1
        self._semantic_hits += 1
        return replace(hit, matched="semantic", similarity=round(overlap, 6))

    def store(self, text: str, decision: RoutingDecision, *, gate_fired: bool = False) -> None:
        """Cache ``decision`` under ``text``\'s normalized form.

        A no-op when ``gate_fired`` is true (the gate contract, module
        docstring), when the text normalizes to nothing, or when the TTL is
        zero. A second store for the same normalized text replaces the entry.
        """
        if gate_fired:
            self._gate_skips += 1
            return
        if not isinstance(decision, RoutingDecision):
            raise TypeError(f"store expects a RoutingDecision, got {type(decision).__name__}")
        if self.ttl_s <= 0:
            return
        norm = normalize_text(text)
        if not norm:
            return
        now = self._clock()
        with self._lock:
            self._data[norm] = (decision, now + self.ttl_s)
            self._data.move_to_end(norm)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self._evictions += 1

    def stats(self) -> dict[str, Any]:
        """Hit/miss/skip counters and occupancy. Same shape as the decision cache\'s."""
        with self._lock:
            return {
                "kind": "intent",
                "enabled": True,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / (self._hits + self._misses), 4) if (self._hits + self._misses) else 0.0,
                "exact_hits": self._exact_hits,
                "near_hits": self._near_hits,
                "semantic_hits": self._semantic_hits,
                "semantic_suspicious": self._semantic_suspicious,
                "semantic": self.semantic,
                "semantic_overlap_floor": self.semantic_overlap_floor,
                "gate_skips": self._gate_skips,
                "expired": self._expired,
                "evictions": self._evictions,
                "size": len(self._data),
                "max_entries": self.max_entries,
                "ttl_s": self.ttl_s,
                "jaccard_threshold": self.jaccard_threshold,
                "semantic_match": self.semantic_match is not None,
            }

    def clear(self) -> None:
        """Drop every entry. Counters are kept: they are the operator\'s history."""
        with self._lock:
            self._data.clear()


def build_intent_cache(policy_raw: Mapping[str, Any] | None) -> IntentCache | None:
    """Read the policy\'s ``intent_cache:`` section and build the cache it names.

    Config shape::

        intent_cache:
          enabled: true
          max_entries: 256
          ttl_s: 3600
          semantic: true   # v0.4: allow the model-based pass once a seam is wired

    ``semantic`` (default ``false``) is the policy half of the semantic pass:
    the pass runs only when it is true AND a seam is wired (``noul_fn`` /
    ``semantic_match``). The seam is code, not config, so the policy can
    declare the intent without naming a model; a deployment without a wired
    noul runs fully deterministic matching.

    Returns ``None`` when the section is absent or disabled -- the caller then
    skips the check entirely, so a policy without the key pays nothing.
    Invalid values raise ``ValueError`` at construction: a cache that silently
    ignores a typo\'d ``ttl_s`` is a config the operator believes is running.
    """
    if not isinstance(policy_raw, Mapping):
        return None
    cfg = policy_raw.get("intent_cache")
    if cfg is None:
        return None
    if not isinstance(cfg, Mapping):
        raise ValueError(f"intent_cache: must be a mapping, got {type(cfg).__name__}")
    if not cfg.get("enabled", True):
        return None
    return IntentCache(
        max_entries=_positive_int(cfg.get("max_entries", _DEFAULT_MAX_ENTRIES), "intent_cache.max_entries"),
        ttl_s=_non_negative_number(cfg.get("ttl_s", _DEFAULT_TTL_S), "intent_cache.ttl_s"),
        semantic=_bool(cfg.get("semantic", False), "intent_cache.semantic"),
    )


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{where} must be >= 1, got {value!r}")
    return value


def _non_negative_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{where} must be >= 0, got {value!r}")
    return number


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{where} must be a boolean, got {value!r}")
    return value


__all__ = [
    "DEFAULT_JACCARD_THRESHOLD",
    "DEFAULT_SEMANTIC_OVERLAP_FLOOR",
    "DEFAULT_SEMANTIC_THRESHOLD",
    "DEFAULT_SEMANTIC_TOP_K",
    "MAX_SEMANTIC_TOP_K",
    "CachedAnswer",
    "IntentCache",
    "NoulFn",
    "SemanticMatch",
    "build_intent_cache",
    "jaccard",
    "normalize_text",
    "semantic_match_from_noul",
    "token_set",
]
