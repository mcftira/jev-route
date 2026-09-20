"""Decision cache.

Caches the backend's *judgement*, not the router's *decision*. That distinction is
the reason this module exists separately from the policy engine: if you cache the
final tier, a policy change silently does not apply to cached traffic for as long
as the TTL runs, and an operator who tightens a sensitivity rule will see it appear
to work while warm requests keep taking the old path. Caching the answers means a
policy edit takes effect on the next request, and the cache only ever saves the
network round-trip it was there to save.

Key is the hash of the *redacted* excerpt plus the backend identity, so two
backends never serve each other's cached answers and a change in redaction rules
produces a new key rather than reusing a stale judgement.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .backends.base import BackendResult


@runtime_checkable
class DecisionCache(Protocol):
    """A judgement cache. Implementations must be safe to share across requests."""

    def get(self, key: str) -> BackendResult | None: ...
    def put(self, key: str, result: BackendResult) -> None: ...
    def stats(self) -> dict[str, Any]: ...
    def clear(self) -> None: ...


class NullCache:
    """No caching. Use when every request must be classified afresh."""

    def get(self, key: str) -> BackendResult | None:
        return None

    def put(self, key: str, result: BackendResult) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {"backend": "null", "enabled": False, "hits": 0, "misses": 0, "size": 0}

    def clear(self) -> None:
        return None


class InMemoryTTLCache:
    """Bounded LRU with per-entry TTL.

    Single-process. In a multi-worker proxy each worker holds its own cache, which
    is fine -- the win rate drops but correctness does not. Use
    :class:`RedisCache` when workers must share.
    """

    def __init__(self, *, ttl_seconds: float = 900.0, max_entries: int = 8192) -> None:
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._data: OrderedDict[str, tuple[float, BackendResult]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> BackendResult | None:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            expires_at, result = entry
            if expires_at <= now:
                del self._data[key]
                self._misses += 1
                return None
            self._data.move_to_end(key)
            self._hits += 1
            return result

    def put(self, key: str, result: BackendResult) -> None:
        if self.ttl_seconds <= 0:
            return
        # Never cache a degraded answer: it is maximum-uncertainty noise, and
        # caching it would pin fail-closed behaviour for the whole TTL after a
        # one-second network blip.
        if result.degraded:
            return
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl_seconds, result)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self._evictions += 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "backend": "memory",
                "enabled": True,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "evictions": self._evictions,
                "size": len(self._data),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
            }

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = self._misses = self._evictions = 0


class RedisCache:
    """Shared cache across proxy workers. Requires ``redis>=5`` (extra: ``redis``).

    Stores the serialized :class:`BackendResult` as JSON. Full probability
    distributions survive the round trip, so a cached answer is indistinguishable
    from a fresh one downstream -- which matters, because the logged record must
    carry the same soft targets either way.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        client: Any = None,
        ttl_seconds: float = 900.0,
        key_prefix: str = "jevroute:",
    ) -> None:
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.key_prefix = key_prefix
        self._client = client
        if client is None:
            try:
                import redis.asyncio as aioredis
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("RedisCache requires the redis package: pip install 'jev-route[redis]'") from exc
            if not url:
                raise ValueError("RedisCache needs url= or an injected client=")
            self._client = aioredis.from_url(url, decode_responses=True)
        self._hits = 0
        self._misses = 0

    def _key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    async def aget(self, key: str) -> BackendResult | None:
        import json

        raw = await self._client.get(self._key(key))
        if raw is None:
            self._misses += 1
            return None
        self._hits += 1
        return _result_from_json(json.loads(raw))

    async def aput(self, key: str, result: BackendResult) -> None:

        if self.ttl_seconds <= 0 or result.degraded:
            return
        # Clamped up to at least one second. Redis expresses expiry in whole
        # seconds here, so `int(0.5)` is `EX 0`, and real Redis rejects that with
        # "ERR invalid expire time in set command". Because the router writes the
        # cache after a successful backend call, a sub-second TTL in policy config
        # would turn every request into an exception instead of a cache miss --
        # a config typo escalating into an outage. Rounding up keeps the operator's
        # intent (cache briefly) and never produces an invalid command.
        await self._client.set(self._key(key), _result_to_json(result), ex=max(1, int(self.ttl_seconds)))

    # Sync shims keep RedisCache usable where the router is called synchronously.
    def get(self, key: str) -> BackendResult | None:
        return _run_sync(self.aget(key))

    def put(self, key: str, result: BackendResult) -> None:
        _run_sync(self.aput(key, result))

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "backend": "redis",
            "enabled": True,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "ttl_seconds": self.ttl_seconds,
        }

    def clear(self) -> None:  # pragma: no cover - deliberately destructive
        raise NotImplementedError("refusing to FLUSHDB; delete the key prefix explicitly")


def _result_to_json(result: BackendResult) -> str:
    import json

    return json.dumps(
        {
            "answers": result.answers.to_dict(),
            "model_version": result.model_version,
            "questions_sent": dict(result.questions_sent or {}),
            "latency_ms": result.latency_ms,
            "degraded": result.degraded,
            "degrade_reason": result.degrade_reason,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _result_from_json(data: Mapping[str, Any]) -> BackendResult:
    from .schema import DecisionAnswers

    return BackendResult(
        answers=DecisionAnswers.from_dict(data["answers"]),
        model_version=str(data.get("model_version", "")),
        questions_sent=dict(data.get("questions_sent") or {}),
        latency_ms=float(data.get("latency_ms", 0.0)),
        degraded=bool(data.get("degraded", False)),
        degrade_reason=data.get("degrade_reason"),
    )


def _run_sync(coro: Any) -> Any:
    """Run ``coro`` from sync code, whether or not a loop is already running."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running here, so this thread can own one.
        return asyncio.run(coro)
    # Inside a running loop we cannot block on it. Submit to a worker thread with
    # its own loop rather than deadlocking the caller.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def build_cache(config: Mapping[str, Any] | None) -> DecisionCache:
    """Construct the cache named by policy config."""
    cfg = dict(config or {})
    if not cfg.get("enabled", True):
        return NullCache()
    kind = str(cfg.get("kind", "memory")).lower()
    if kind == "memory":
        return InMemoryTTLCache(
            ttl_seconds=float(cfg.get("ttl_seconds", 900.0)),
            max_entries=int(cfg.get("max_entries", 8192)),
        )
    if kind == "redis":
        return RedisCache(  # type: ignore[return-value]
            url=cfg.get("url"),
            ttl_seconds=float(cfg.get("ttl_seconds", 900.0)),
            key_prefix=str(cfg.get("key_prefix", "jevroute:")),
        )
    raise ValueError(f"unknown cache kind {kind!r}; expected 'memory' or 'redis'")


__all__ = [
    "DecisionCache",
    "InMemoryTTLCache",
    "NullCache",
    "RedisCache",
    "build_cache",
]
