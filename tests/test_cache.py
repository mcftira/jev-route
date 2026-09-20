"""Contract tests for :mod:`jev_route.cache`.

Why this file exists
--------------------
The cache sits *inline* on the request path of a proxy, so it has to be correct
in the boring directions: bounded memory, no deadlocks, no stale judgements. And
it has three promises that are easy to break accidentally and expensive to notice:

1. **It caches the judgement, not the decision.** The value stored is a
   :class:`~jev_route.backends.base.BackendResult` -- the four answers with their
   full probability distributions -- so a policy edit takes effect on the next
   request even for warm traffic. Every round-trip test below therefore compares
   whole distributions, never just the argmax: a cache that quietly dropped
   runner-up mass would still route correctly and would silently degrade the
   dataset the distillation pipeline trains on.
2. **A degraded answer is never cached.** That rule has its own named test, with
   the reasoning spelled out, because it is the difference between a one-second
   network blip and fifteen minutes of pinned fail-closed routing.
3. **It never connects to anything at construction time.** ``build_cache`` is
   called while a policy loads, and ``RedisCache`` builds a lazy client. A cache
   that dialled home on startup would make "runs offline with no key" false.

Everything here is offline. ``RedisCache`` is exercised against
:class:`FakeRedis`, an in-process double that records the key and the ``ex=`` TTL
that *would* have been sent. No test in this file talks to a Redis server, and
none needs one -- which is the point: the serialization contract is what could
break, not the network.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from jev_route.backends.base import BackendResult
from jev_route.cache import (
    DecisionCache,
    InMemoryTTLCache,
    NullCache,
    RedisCache,
    build_cache,
)
from jev_route.schema import (
    COMPLEXITY_LEVELS,
    DOMAINS,
    SENSITIVITY_LEVELS,
    ChoiceAnswer,
    DecisionAnswers,
    NoulAnswer,
)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def peaked(level: str, ladder: Sequence[str], confidence: float) -> ChoiceAnswer:
    """``level`` at ``confidence``, the rest of the ladder sharing the remainder."""
    others = [x for x in ladder if x != level]
    share = (1.0 - confidence) / len(others) if others else 0.0
    masses = dict.fromkeys(others, share)
    masses[level] = 1.0 - share * len(others)
    return ChoiceAnswer(
        choice=level,
        probabilities={name: masses[name] for name in ladder},
        confidence=confidence,
    )


def result(
    *,
    pii: float = 0.13,
    degraded: bool = False,
    degrade_reason: str | None = None,
    model_version: str = "fake-1.0.0",
    latency_ms: float = 7.5,
) -> BackendResult:
    """A judgement worth caching: confident, non-uniform, and unlike ``unknown()``.

    Deliberately *not* :meth:`DecisionAnswers.unknown`, so a test can tell a real
    cached answer apart from a degraded one by looking at the values alone.
    """
    return BackendResult(
        answers=DecisionAnswers(
            complexity=peaked("frontier", COMPLEXITY_LEVELS, 0.77),
            sensitivity=peaked("regulated", SENSITIVITY_LEVELS, 0.64),
            pii=NoulAnswer(value=pii),
            domain=peaked("code", DOMAINS, 0.91),
        ),
        model_version=model_version,
        questions_sent={"complexity": {"type": "choice"}, "pii_present": {"type": "noul"}},
        latency_ms=latency_ms,
        degraded=degraded,
        degrade_reason=degrade_reason,
    )


def degraded_result() -> BackendResult:
    """What a backend returns during an outage: maximum uncertainty, flagged."""
    return result(degraded=True, degrade_reason="timeout after 3000ms", model_version="fake-1.0.0")


class FakeRedis:
    """An in-process stand-in for ``redis.asyncio.Redis``.

    Implements the two commands :class:`RedisCache` uses and records every call,
    because what the tests need to assert is *what would have been sent*: the
    prefixed key, the JSON payload, and the ``ex=`` TTL. Values are stored as
    ``str`` to match ``decode_responses=True``, which is how the real client is
    constructed -- a double that returned bytes would hide a decoding bug.
    """

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int | None] = {}
        self.calls: list[tuple[Any, ...]] = []
        self.closed = 0

    async def get(self, key: str) -> str | None:
        self.calls.append(("get", key))
        return self.store.get(key)

    async def set(self, key: str, value: str, *, ex: int | None = None) -> bool:
        self.calls.append(("set", key, ex))
        self.store[key] = value
        self.ttls[key] = ex
        return True

    async def aclose(self) -> None:
        self.calls.append(("aclose",))
        self.closed += 1

    # -- conveniences for assertions ------------------------------------- #
    @property
    def keys(self) -> list[str]:
        return list(self.store)

    @property
    def sets(self) -> list[tuple[Any, ...]]:
        return [c for c in self.calls if c[0] == "set"]

    @property
    def gets(self) -> list[tuple[Any, ...]]:
        return [c for c in self.calls if c[0] == "get"]


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def redis_cache(fake_redis: FakeRedis) -> RedisCache:
    """A ``RedisCache`` wired to the fake, with a test-only prefix."""
    return RedisCache(client=fake_redis, ttl_seconds=900.0, key_prefix="jevroute:test:")


# --------------------------------------------------------------------------- #
# NullCache -- the "classify every request afresh" configuration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["", "abc", "a" * 64])
def test_null_cache_get_always_returns_none(key: str) -> None:
    assert NullCache().get(key) is None


def test_null_cache_put_is_a_no_op() -> None:
    cache = NullCache()
    assert cache.put("k", result()) is None
    assert cache.get("k") is None


def test_null_cache_stats_are_the_documented_disabled_shape() -> None:
    # Exact dict, not a subset: the CLI and the proxy's /stats endpoint print
    # this verbatim, and an extra or renamed key is a user-visible change.
    assert NullCache().stats() == {
        "backend": "null",
        "enabled": False,
        "hits": 0,
        "misses": 0,
        "size": 0,
    }


def test_null_cache_counters_never_move_under_traffic() -> None:
    # "enabled: False" has to stay true in the numbers too, or an operator reading
    # a hit rate cannot tell whether caching is on.
    cache = NullCache()
    for i in range(10):
        cache.put(f"k{i}", result())
        cache.get(f"k{i}")
    assert cache.stats()["hits"] == 0
    assert cache.stats()["misses"] == 0
    assert cache.stats()["size"] == 0


def test_null_cache_clear_returns_none_and_is_safe_to_repeat() -> None:
    cache = NullCache()
    assert cache.clear() is None
    assert cache.clear() is None
    assert cache.stats()["enabled"] is False


# --------------------------------------------------------------------------- #
# InMemoryTTLCache -- hit/miss accounting
# --------------------------------------------------------------------------- #
def test_put_then_get_returns_the_identical_object(memory_cache: InMemoryTTLCache) -> None:
    # Identity, not equality. The cache hands back the very BackendResult the
    # backend produced, so a warm request logs byte-identical soft targets and no
    # reserialization step can drop runner-up mass on the way through.
    cached = result()
    memory_cache.put("k", cached)
    assert memory_cache.get("k") is cached


def test_get_on_an_unknown_key_returns_none(memory_cache: InMemoryTTLCache) -> None:
    assert memory_cache.get("never-put") is None


def test_unknown_key_counts_as_a_miss_not_a_hit(memory_cache: InMemoryTTLCache) -> None:
    # The hit rate is the only signal an operator has that the cache is worth its
    # memory, so a lookup that found nothing must be counted as a miss.
    memory_cache.get("nope")
    stats = memory_cache.stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0
    assert stats["size"] == 0


def test_hit_and_miss_counters_and_hit_rate(memory_cache: InMemoryTTLCache) -> None:
    memory_cache.put("a", result())
    assert memory_cache.get("a") is not None  # hit
    assert memory_cache.get("a") is not None  # hit
    assert memory_cache.get("b") is None  # miss
    stats = memory_cache.stats()
    assert (stats["hits"], stats["misses"], stats["size"]) == (2, 1, 1)
    assert stats["hit_rate"] == pytest.approx(round(2 / 3, 4))


def test_hit_rate_is_zero_before_any_traffic(memory_cache: InMemoryTTLCache) -> None:
    # 0/0 must not raise or produce nan: stats() is called by health endpoints on
    # a process that may not have served a request yet.
    stats = memory_cache.stats()
    assert stats["hit_rate"] == 0.0
    assert stats["hits"] == 0 and stats["misses"] == 0


def test_stats_report_the_configured_shape(memory_cache: InMemoryTTLCache) -> None:
    stats = memory_cache.stats()
    assert stats["backend"] == "memory"
    assert stats["enabled"] is True
    assert stats["max_entries"] == 64
    assert stats["ttl_seconds"] == 60.0
    assert stats["evictions"] == 0


def test_construction_clamps_nonsense_configuration() -> None:
    # A policy typo (ttl: -1, max_entries: 0) must not produce a cache that
    # raises or one that grows without bound.
    assert InMemoryTTLCache(ttl_seconds=-5.0).ttl_seconds == 0.0
    assert InMemoryTTLCache(max_entries=0).max_entries == 1
    assert InMemoryTTLCache(max_entries=-3).max_entries == 1


@pytest.mark.parametrize("ttl", [0.0, -1.0])
def test_a_non_positive_ttl_stores_nothing(ttl: float) -> None:
    # ttl_seconds <= 0 means "caching off" rather than "cache forever": an entry
    # with no expiry is how a proxy ends up serving a judgement from last week.
    cache = InMemoryTTLCache(ttl_seconds=ttl, max_entries=8)
    cache.put("k", result())
    assert cache.get("k") is None
    assert cache.stats()["size"] == 0


def test_a_disabled_cache_still_counts_the_lookup_as_a_miss() -> None:
    cache = InMemoryTTLCache(ttl_seconds=0.0)
    cache.put("k", result())
    cache.get("k")
    assert cache.stats()["misses"] == 1
    assert cache.stats()["hit_rate"] == 0.0


# --------------------------------------------------------------------------- #
# InMemoryTTLCache -- TTL expiry
# --------------------------------------------------------------------------- #
def test_an_entry_is_readable_before_its_ttl_elapses() -> None:
    # The counterpart to the expiry test below: without it, "the entry vanished"
    # could equally mean put() never stored anything.
    cache = InMemoryTTLCache(ttl_seconds=5.0)
    cached = result()
    cache.put("k", cached)
    assert cache.get("k") is cached
    assert cache.stats()["size"] == 1


def test_an_entry_expires_after_its_ttl() -> None:
    # A short real sleep rather than a patched clock: expiry is measured against
    # time.monotonic(), and mocking that would test the mock. 80 ms against a
    # 50 ms TTL keeps the whole suite fast while leaving real margin.
    cache = InMemoryTTLCache(ttl_seconds=0.05)
    cache.put("k", result())
    assert cache.get("k") is not None
    time.sleep(0.08)
    assert cache.get("k") is None


def test_an_expired_get_deletes_the_entry_so_size_drops_to_zero() -> None:
    # Reading an expired key must reclaim it. Otherwise a cache that is only ever
    # read (never re-put) grows to max_entries of dead judgements and evicts live
    # ones to make room -- memory pressure caused entirely by garbage.
    cache = InMemoryTTLCache(ttl_seconds=0.05, max_entries=4)
    cache.put("k", result())
    assert cache.stats()["size"] == 1
    time.sleep(0.08)
    assert cache.get("k") is None
    assert cache.stats()["size"] == 0
    assert cache.get("k") is None  # still gone, and not resurrected
    assert cache.stats()["size"] == 0


def test_an_expired_read_counts_a_miss_not_a_hit() -> None:
    cache = InMemoryTTLCache(ttl_seconds=0.05)
    cache.put("k", result())
    cache.get("k")
    time.sleep(0.08)
    cache.get("k")
    stats = cache.stats()
    assert (stats["hits"], stats["misses"]) == (1, 1)
    assert stats["hit_rate"] == 0.5


def test_ttl_is_per_entry_not_global() -> None:
    # Each entry carries its own expiry, so a key written later outlives one
    # written earlier. A single global deadline would silently truncate the
    # cache's useful life under steady traffic. Timings leave ~20-40 ms of slack
    # on each side so a loaded CI machine cannot flip the result.
    cache = InMemoryTTLCache(ttl_seconds=0.10)
    cache.put("old", result())
    time.sleep(0.06)
    cache.put("new", result())
    time.sleep(0.06)  # t=0.12: "old" expired at 0.10, "new" not until 0.16
    assert cache.get("old") is None
    assert cache.get("new") is not None


# --------------------------------------------------------------------------- #
# InMemoryTTLCache -- bounded LRU
# --------------------------------------------------------------------------- #
def test_max_entries_is_enforced_and_evicts_the_oldest_first() -> None:
    cache = InMemoryTTLCache(ttl_seconds=60.0, max_entries=2)
    stored = {key: result() for key in "abcde"}
    for key, value in stored.items():
        cache.put(key, value)

    stats = cache.stats()
    assert stats["size"] == 2
    assert stats["max_entries"] == 2
    # Five keys through a two-slot cache is exactly three evictions: the counter
    # is how an operator notices the cache is too small for the traffic.
    assert stats["evictions"] == 3
    # The survivors are the two NEWEST keys, which is what makes this an LRU and
    # not a "first write wins" cache.
    assert cache.get("a") is None
    assert cache.get("b") is None
    assert cache.get("c") is None
    assert cache.get("d") is stored["d"]
    assert cache.get("e") is stored["e"]


def test_max_entries_of_one_is_a_valid_single_slot_cache() -> None:
    cache = InMemoryTTLCache(ttl_seconds=60.0, max_entries=1)
    first, second = result(model_version="one"), result(model_version="two")
    cache.put("a", first)
    cache.put("b", second)
    assert cache.get("a") is None
    assert cache.get("b") is second
    assert cache.stats()["evictions"] == 1


def test_a_read_refreshes_a_key_s_lru_position() -> None:
    # "Least recently USED", not "least recently written": a hot key that is read
    # constantly must not be evicted while idle keys survive. Verified ordering is
    # a -> get(a) -> [b, a] -> put(c) evicts b.
    cache = InMemoryTTLCache(ttl_seconds=60.0, max_entries=2)
    hot = result(model_version="hot")
    cache.put("a", hot)
    cache.put("b", result(model_version="cold"))
    assert cache.get("a") is hot  # this read is what saves "a"
    cache.put("c", result(model_version="newest"))

    assert cache.get("a") is hot
    assert cache.get("b") is None  # evicted instead, despite being written later
    assert cache.get("c") is not None


def test_rewriting_an_existing_key_refreshes_its_position_without_growing() -> None:
    cache = InMemoryTTLCache(ttl_seconds=60.0, max_entries=2)
    cache.put("a", result())
    cache.put("b", result())
    refreshed = result(model_version="refreshed")
    cache.put("a", refreshed)  # update in place, and move "a" to the hot end
    cache.put("c", result())

    assert cache.stats()["size"] == 2
    assert cache.get("a") is refreshed
    assert cache.get("b") is None
    assert cache.get("c") is not None


def test_a_hot_key_survives_a_long_stream_of_cold_ones() -> None:
    # The realistic proxy pattern: one prompt repeated by a retry loop or a
    # chatty client, plus a long tail of unique prompts. The hot key must stay.
    cache = InMemoryTTLCache(ttl_seconds=60.0, max_entries=4)
    hot = result(model_version="hot")
    cache.put("hot", hot)
    for i in range(50):
        cache.put(f"cold-{i}", result())
        assert cache.get("hot") is hot
    assert cache.stats()["size"] == 4
    # Exactly 47: the first three cold keys filled the slots left free by "hot",
    # and each of the remaining 47 writes forced one eviction.
    assert cache.stats()["evictions"] == 47


def test_evictions_do_not_count_rewrites_of_the_same_key() -> None:
    cache = InMemoryTTLCache(ttl_seconds=60.0, max_entries=4)
    for _ in range(5):
        cache.put("same", result())
    assert cache.stats()["size"] == 1
    assert cache.stats()["evictions"] == 0


# --------------------------------------------------------------------------- #
# The degraded-result promise
# --------------------------------------------------------------------------- #
def test_degraded_results_are_never_cached() -> None:
    """A backend outage must not be remembered.

    This one deserves its own test and its own explanation, because the failure
    mode is invisible from the routing side. When a backend times out it returns
    :meth:`DecisionAnswers.unknown` with ``degraded=True`` -- maximum uncertainty,
    which the policy engine fails closed on. If that answer were cached, a
    one-second network blip would pin every repeat of that prompt to fail-closed
    local routing for the whole TTL: fifteen minutes of degraded behaviour caused
    by a moment of it, with no error anywhere to explain why. Refusing to store
    it means the very next request asks the backend again.
    """
    cache = InMemoryTTLCache(ttl_seconds=60.0, max_entries=8)
    cache.put("blip", degraded_result())
    assert cache.get("blip") is None
    assert cache.stats()["size"] == 0


def test_refusing_a_degraded_result_has_no_side_effects() -> None:
    # Not stored, not evicting anything else, and not counted as an eviction:
    # stats() must describe real cache traffic only.
    cache = InMemoryTTLCache(ttl_seconds=60.0, max_entries=2)
    cache.put("a", result())
    cache.put("b", result())
    cache.put("c", degraded_result())
    assert cache.stats()["evictions"] == 0
    assert cache.stats()["size"] == 2
    assert cache.get("a") is not None and cache.get("b") is not None


def test_a_degraded_put_cannot_poison_an_already_cached_judgement() -> None:
    # The refusal happens before the write, so a good cached answer survives a
    # later outage for the same prompt. Verified behaviour, and the reason the
    # guard is an early return rather than a "store it but mark it" flag.
    cache = InMemoryTTLCache(ttl_seconds=60.0, max_entries=4)
    good = result(model_version="fresh")
    cache.put("k", good)
    cache.put("k", degraded_result())
    assert cache.get("k") is good


def test_the_cache_recovers_immediately_after_an_outage_ends() -> None:
    cache = InMemoryTTLCache(ttl_seconds=60.0, max_entries=4)
    cache.put("k", degraded_result())
    assert cache.get("k") is None
    recovered = result(model_version="back-up")
    cache.put("k", recovered)
    assert cache.get("k") is recovered


# --------------------------------------------------------------------------- #
# clear() and concurrency
# --------------------------------------------------------------------------- #
def test_clear_empties_the_data_and_resets_the_counters() -> None:
    cache = InMemoryTTLCache(ttl_seconds=60.0, max_entries=2)
    cache.put("a", result())
    cache.get("a")  # hit
    cache.get("z")  # miss
    cache.put("b", result())
    cache.put("c", result())  # evicts "a"
    assert cache.stats()["evictions"] == 1

    assert cache.clear() is None
    stats = cache.stats()
    assert stats["size"] == 0
    assert (stats["hits"], stats["misses"], stats["evictions"], stats["hit_rate"]) == (0, 0, 0, 0.0)
    assert stats["max_entries"] == 2  # configuration is not reset, only state
    assert cache.get("b") is None


def test_a_cleared_cache_is_immediately_reusable() -> None:
    cache = InMemoryTTLCache(ttl_seconds=60.0, max_entries=2)
    cached = result()
    cache.put("k", cached)
    cache.clear()
    cache.put("k", cached)
    assert cache.get("k") is cached
    assert cache.stats()["hits"] == 1


def test_concurrent_put_and_get_is_safe_and_stays_bounded() -> None:
    """Thread-safety smoke test: the proxy serves requests from many threads.

    8 threads x 200 operations, with a max_entries small enough that they are
    evicting each other's keys constantly -- that is the path where a missing
    lock shows up as a corrupted OrderedDict or a size that drifts past the
    bound. The assertions are the ones that matter: nothing raised, the cache
    never exceeded max_entries, and the counters add up to the number of reads.
    """
    cache = InMemoryTTLCache(ttl_seconds=60.0, max_entries=16)
    errors: list[BaseException] = []
    reads_per_thread = 200

    def worker(index: int) -> None:
        try:
            for op in range(reads_per_thread):
                cache.put(f"t{index}-{op % 40}", result())
                cache.get(f"t{index}-{(op + 1) % 40}")
                cache.get("never-present")
        except BaseException as exc:  # recorded, never swallowed: asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    stats = cache.stats()
    assert stats["size"] <= stats["max_entries"]
    assert stats["hits"] + stats["misses"] == 8 * reads_per_thread * 2


# --------------------------------------------------------------------------- #
# The DecisionCache protocol
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cache",
    [InMemoryTTLCache(), NullCache()],
    ids=["memory", "null"],
)
def test_shipped_caches_satisfy_the_protocol(cache: Any) -> None:
    # runtime_checkable, so the router can accept any object with the right four
    # methods -- which is what lets a test double or a third-party cache be
    # injected without inheriting from a base class.
    assert isinstance(cache, DecisionCache)


def test_redis_cache_satisfies_the_protocol(fake_redis: FakeRedis) -> None:
    # Constructed with an injected client, so this assertion needs no redis
    # package and opens no connection.
    assert isinstance(RedisCache(client=fake_redis), DecisionCache)


def test_an_object_missing_a_method_is_not_a_decision_cache() -> None:
    class Almost:
        def get(self, key: str) -> None: ...
        def put(self, key: str, result: BackendResult) -> None: ...
        def stats(self) -> dict[str, Any]: ...

    # Negative control: the protocol is structural, so it must actually be
    # checking for all four methods rather than passing everything.
    assert not isinstance(Almost(), DecisionCache)
    assert not isinstance(object(), DecisionCache)


def test_the_protocol_surface_is_exactly_four_methods() -> None:
    assert {name for name in dir(DecisionCache) if not name.startswith("_")} == {
        "get",
        "put",
        "stats",
        "clear",
    }


# --------------------------------------------------------------------------- #
# RedisCache -- always against an injected fake client, never a real server
# --------------------------------------------------------------------------- #
async def test_aput_then_aget_round_trips_the_whole_judgement(redis_cache: RedisCache) -> None:
    cached = result()
    await redis_cache.aput("abc123", cached)
    restored = await redis_cache.aget("abc123")

    assert restored is not None
    assert isinstance(restored, BackendResult)
    assert restored.model_version == cached.model_version
    assert restored.latency_ms == cached.latency_ms
    assert restored.degraded is False
    assert restored.degrade_reason is None
    assert dict(restored.questions_sent) == dict(cached.questions_sent)


async def test_the_round_trip_preserves_every_probability_of_every_ladder(
    redis_cache: RedisCache, fake_redis: FakeRedis
) -> None:
    """The shared-cache version of the schema's core promise.

    Redis stores JSON text, so unlike the in-memory cache this path really does
    serialize and re-parse. If the distributions did not survive, a multi-worker
    proxy would log *different* training data depending on which worker answered
    -- and nothing downstream could tell. Exact dict comparison, not approx.
    """
    cached = result()
    await redis_cache.aput("k", cached)
    restored = await redis_cache.aget("k")
    assert restored is not None

    for name, ladder in (
        ("complexity", COMPLEXITY_LEVELS),
        ("sensitivity", SENSITIVITY_LEVELS),
        ("domain", DOMAINS),
    ):
        before: ChoiceAnswer = getattr(cached.answers, name)
        after: ChoiceAnswer = getattr(restored.answers, name)
        assert dict(after.probabilities) == dict(before.probabilities)
        assert set(after.probabilities) == set(ladder)
        assert after.choice == before.choice
        assert after.confidence == before.confidence
        assert after.confidence_reported is before.confidence_reported
    assert restored.answers.pii.value == cached.answers.pii.value
    assert restored.answers == cached.answers


async def test_the_stored_payload_spans_the_full_ladder_in_json(redis_cache: RedisCache, fake_redis: FakeRedis) -> None:
    # Asserting on the wire format, not just the parsed object: another process
    # (or a Python version) may read these keys, so the text has to be complete.
    await redis_cache.aput("k", result())
    payload = json.loads(fake_redis.store["jevroute:test:k"])
    assert set(payload) == {
        "answers",
        "model_version",
        "questions_sent",
        "latency_ms",
        "degraded",
        "degrade_reason",
    }
    assert set(payload["answers"]["complexity"]["probabilities"]) == set(COMPLEXITY_LEVELS)
    assert payload["answers"]["pii"] == {"noul": 0.13}


async def test_the_payload_is_canonical_sorted_compact_json(redis_cache: RedisCache, fake_redis: FakeRedis) -> None:
    # sort_keys + compact separators: the same judgement written twice is
    # byte-identical, which makes a Redis value comparable and keeps the wire
    # size down on a path that runs per request.
    await redis_cache.aput("k", result())
    payload = fake_redis.store["jevroute:test:k"]
    assert '", "' not in payload
    assert '": "' not in payload
    assert payload == json.dumps(json.loads(payload), sort_keys=True, separators=(",", ":"))


async def test_keys_are_namespaced_with_the_prefix(redis_cache: RedisCache, fake_redis: FakeRedis) -> None:
    # The prefix is what makes "delete the key prefix explicitly" (the sanctioned
    # alternative to FLUSHDB) possible at all, and what keeps two environments
    # sharing one Redis from serving each other's judgements.
    await redis_cache.aput("abc123", result())
    assert await redis_cache.aget("abc123") is not None
    assert fake_redis.keys == ["jevroute:test:abc123"]
    assert fake_redis.sets == [("set", "jevroute:test:abc123", 900)]
    # Reads are prefixed too: a cache that namespaced writes but looked up bare
    # keys would report a 0% hit rate forever and never be suspected.
    assert fake_redis.gets == [("get", "jevroute:test:abc123")]


def test_the_default_key_prefix_is_jevroute(fake_redis: FakeRedis) -> None:
    assert RedisCache(client=fake_redis).key_prefix == "jevroute:"


async def test_a_custom_prefix_is_applied_to_reads_too(fake_redis: FakeRedis) -> None:
    cache = RedisCache(client=fake_redis, ttl_seconds=60.0, key_prefix="prod:router:")
    await cache.aput("k", result())
    assert fake_redis.keys == ["prod:router:k"]
    assert await cache.aget("k") is not None


async def test_a_degraded_result_is_not_stored_in_redis(redis_cache: RedisCache, fake_redis: FakeRedis) -> None:
    # Same promise as the in-memory cache, and more important here: a shared
    # cache would spread one worker's outage to every other worker for the TTL.
    await redis_cache.aput("blip", degraded_result())
    assert fake_redis.store == {}
    assert fake_redis.sets == []
    assert await redis_cache.aget("blip") is None


async def test_a_non_positive_ttl_stores_nothing_in_redis(
    fake_redis: FakeRedis,
) -> None:
    for ttl in (0.0, -1.0):
        cache = RedisCache(client=fake_redis, ttl_seconds=ttl)
        await cache.aput(f"k{ttl}", result())
    assert fake_redis.store == {}
    assert RedisCache(client=fake_redis, ttl_seconds=-1.0).ttl_seconds == 0.0


async def test_set_receives_the_ttl_as_an_integer_number_of_seconds(
    redis_cache: RedisCache, fake_redis: FakeRedis
) -> None:
    # Redis' EX argument takes whole seconds; the cache converts rather than
    # passing a float, which the server would reject.
    await redis_cache.aput("k", result())
    assert fake_redis.ttls["jevroute:test:k"] == 900
    assert isinstance(fake_redis.ttls["jevroute:test:k"], int)


async def test_a_fractional_ttl_is_truncated_to_whole_seconds(fake_redis: FakeRedis) -> None:
    # Documented reality: int(42.7) == 42. Truncation (not rounding) means the
    # entry expires slightly early, which is the safe direction for a cache.
    cache = RedisCache(client=fake_redis, ttl_seconds=42.7)
    await cache.aput("k", result())
    assert fake_redis.ttls["jevroute:k"] == 42


async def test_a_subsecond_ttl_still_sends_a_valid_expire_time(fake_redis: FakeRedis) -> None:
    cache = RedisCache(client=fake_redis, ttl_seconds=0.5)
    await cache.aput("k", result())
    assert fake_redis.ttls["jevroute:k"] >= 1


async def test_aget_on_a_missing_key_returns_none_and_counts_a_miss(
    redis_cache: RedisCache, fake_redis: FakeRedis
) -> None:
    assert await redis_cache.aget("never-written") is None
    stats = redis_cache.stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0
    assert stats["hit_rate"] == 0.0
    assert fake_redis.gets == [("get", "jevroute:test:never-written")]


async def test_stats_report_the_redis_backend_and_counters(redis_cache: RedisCache) -> None:
    await redis_cache.aput("k", result())
    await redis_cache.aget("k")  # hit
    await redis_cache.aget("other")  # miss
    stats = redis_cache.stats()
    assert stats["backend"] == "redis"
    assert stats["enabled"] is True
    assert (stats["hits"], stats["misses"]) == (1, 1)
    assert stats["hit_rate"] == 0.5
    assert stats["ttl_seconds"] == 900.0


async def test_redis_hit_rate_is_zero_before_any_traffic(redis_cache: RedisCache) -> None:
    assert redis_cache.stats()["hit_rate"] == 0.0


async def test_clear_refuses_to_flush_the_database(redis_cache: RedisCache) -> None:
    """Not an unimplemented stub: a deliberate refusal.

    A shared Redis usually holds more than jev-route's keys. FLUSHDB from a
    library would be an unreviewable, cross-application destructive action
    reachable from a config reload, so clear() raises and points at the safe
    alternative instead.
    """
    with pytest.raises(NotImplementedError) as excinfo:
        redis_cache.clear()
    message = str(excinfo.value)
    assert "FLUSHDB" in message
    assert "key prefix" in message


async def test_repeated_aput_overwrites_the_same_key(redis_cache: RedisCache, fake_redis: FakeRedis) -> None:
    await redis_cache.aput("k", result(model_version="first"))
    await redis_cache.aput("k", result(model_version="second"))
    assert fake_redis.keys == ["jevroute:test:k"]
    restored = await redis_cache.aget("k")
    assert restored is not None
    assert restored.model_version == "second"


def test_the_sync_shims_round_trip_from_plain_sync_code(fake_redis: FakeRedis) -> None:
    """get()/put() must work with no event loop running.

    This test is deliberately *not* ``async``: it is the LiteLLM-proxy and CLI
    call shape, where ``_run_sync`` has to create the loop itself via
    ``asyncio.run``. If it silently required a running loop, every synchronous
    integration would break at the first cache hit.
    """
    cache = RedisCache(client=fake_redis, ttl_seconds=30.0)
    cached = result(pii=0.9)
    cache.put("sync-key", cached)
    restored = cache.get("sync-key")
    assert restored is not None
    assert restored.answers.pii.value == pytest.approx(0.9)
    assert restored.answers == cached.answers
    assert cache.get("absent") is None
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


async def test_the_sync_shims_also_work_inside_a_running_loop(fake_redis: FakeRedis) -> None:
    # The other half of _run_sync: called from an async context it cannot block
    # the loop, so it hands the coroutine to a worker thread. A deadlock here
    # would hang the proxy rather than fail a test, which is why both call shapes
    # are covered.
    cache = RedisCache(client=fake_redis, ttl_seconds=30.0)
    await cache.aput("k", result(pii=0.42))
    restored = cache.get("k")  # sync call, running loop present
    assert restored is not None
    assert restored.answers.pii.value == pytest.approx(0.42)
    cache.put("j", result(pii=0.11))
    assert (await cache.aget("j")) is not None


def test_redis_cache_requires_a_url_or_an_injected_client() -> None:
    pytest.importorskip("redis", reason="RedisCache imports redis.asyncio when no client is given")
    with pytest.raises(ValueError, match=r"url=|client="):
        RedisCache()


def test_building_a_redis_cache_makes_no_connection() -> None:
    """Construction is lazy, so loading a policy never touches the network.

    The URL host is deliberately non-loopback and unresolvable. conftest's offline
    guard turns any connect attempt into ``NetworkBlockedError``, so this test passing
    is mechanical proof that ``redis.asyncio.from_url`` only builds a pool: if
    ``build_cache`` ever started pinging Redis, the test fails loudly instead of
    hanging on a timeout in CI.
    """
    pytest.importorskip("redis", reason="the redis extra is optional")
    cache = build_cache(
        {"kind": "redis", "url": "redis://cache.invalid:6379/0", "ttl_seconds": 30.0, "key_prefix": "p:"}
    )
    assert isinstance(cache, RedisCache)
    assert cache.ttl_seconds == 30.0
    assert cache.key_prefix == "p:"


# --------------------------------------------------------------------------- #
# build_cache -- policy config to implementation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("config", [None, {}], ids=["none", "empty"])
def test_the_default_cache_is_in_memory(config: Mapping[str, Any] | None) -> None:
    # No cache section in the policy must mean "cache in process", not "no
    # caching": the default has to be the one that works with zero infrastructure.
    cache = build_cache(config)
    assert isinstance(cache, InMemoryTTLCache)
    assert cache.ttl_seconds == 900.0
    assert cache.max_entries == 8192


@pytest.mark.parametrize(
    "config", [{"enabled": False}, {"enabled": False, "kind": "redis"}, {"enabled": False, "kind": "memory"}]
)
def test_enabled_false_selects_the_null_cache(config: dict[str, Any]) -> None:
    # `enabled` wins over `kind`, and it is checked first -- so turning the cache
    # off never constructs (or imports) a Redis client.
    cache = build_cache(config)
    assert isinstance(cache, NullCache)
    assert cache.stats()["enabled"] is False


@pytest.mark.parametrize("enabled", [True, "yes", 1], ids=["true", "string", "int"])
def test_a_truthy_enabled_value_selects_a_real_cache(enabled: Any) -> None:
    assert isinstance(build_cache({"enabled": enabled}), InMemoryTTLCache)


def test_an_explicit_null_enabled_value_disables_the_cache() -> None:
    # Verified behaviour, and worth pinning because it is surprising: `enabled:`
    # with no value in YAML parses as None, and `not None` is truthy, so an empty
    # `enabled:` turns caching OFF rather than on. That is the safe direction --
    # no caching means every request is classified afresh -- but a policy author
    # who writes a blank `enabled:` expecting the default gets the opposite.
    assert isinstance(build_cache({"enabled": None}), NullCache)


def test_memory_config_values_are_passed_through() -> None:
    cache = build_cache({"kind": "memory", "ttl_seconds": 5, "max_entries": 10})
    assert isinstance(cache, InMemoryTTLCache)
    assert cache.ttl_seconds == 5.0
    assert cache.max_entries == 10


@pytest.mark.parametrize("kind", ["memory", "MEMORY", "Memory", "  memory  ".strip()])
def test_the_kind_is_matched_case_insensitively(kind: str) -> None:
    # YAML config is hand-edited; "MEMORY" and "memory" must mean the same thing
    # rather than one of them raising at policy load.
    assert isinstance(build_cache({"kind": kind}), InMemoryTTLCache)


@pytest.mark.parametrize("kind", ["memcached", "disk", "", "in-memory", "Memory1"])
def test_an_unknown_kind_raises_and_names_the_alternatives(kind: str) -> None:
    # A typo in the policy must fail at load time with an actionable message, not
    # fall back to a different cache than the operator asked for.
    with pytest.raises(ValueError) as excinfo:
        build_cache({"kind": kind})
    message = str(excinfo.value)
    assert "memory" in message
    assert "redis" in message


def test_the_built_memory_cache_actually_caches() -> None:
    # End-to-end check of the factory's product: the object a policy yields has
    # to work, not merely be the right type.
    cache = build_cache({"kind": "memory", "ttl_seconds": 60.0, "max_entries": 4})
    cached = result()
    cache.put("k", cached)
    assert cache.get("k") is cached
    assert cache.stats()["backend"] == "memory"


def test_build_cache_redis_passes_the_config_through() -> None:
    pytest.importorskip("redis", reason="the redis extra is optional")
    cache = build_cache(
        {
            "kind": "redis",
            "url": "redis://cache.invalid:6379/2",
            "ttl_seconds": 45,
            "key_prefix": "tenant-a:",
        }
    )
    assert isinstance(cache, RedisCache)
    assert cache.ttl_seconds == 45.0
    assert cache.key_prefix == "tenant-a:"


def test_build_cache_never_mutates_the_policy_config() -> None:
    # The policy dict is shared by the whole process; a factory that popped keys
    # would leave a second caller with a different configuration.
    config: dict[str, Any] = {"kind": "memory", "ttl_seconds": 7, "max_entries": 3}
    build_cache(config)
    assert config == {"kind": "memory", "ttl_seconds": 7, "max_entries": 3}


def test_the_two_cache_kinds_share_the_same_observable_contract() -> None:
    """Whatever policy selects, the router sees the same four methods and shapes.

    That is the reason ``DecisionCache`` exists: swapping memory for Redis is a
    config change, and this asserts the swap cannot alter what the router reads
    back (an identical object identity for memory, an equal value for Redis).
    """
    cached = result()
    memory = build_cache({"kind": "memory", "ttl_seconds": 60.0})
    memory.put("k", cached)
    assert memory.get("k") is cached

    fake = FakeRedis()
    shared = RedisCache(client=fake, ttl_seconds=60.0)
    shared.put("k", cached)
    restored = shared.get("k")
    assert restored is not None
    assert restored.answers == cached.answers
    for cache in (memory, shared):
        assert set(cache.stats()) >= {"backend", "enabled", "hits", "misses", "hit_rate"}
