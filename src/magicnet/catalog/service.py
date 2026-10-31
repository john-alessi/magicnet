"""The catalog HTTP service: lookups with single-flight refresh on miss.

Concurrency model: one asyncio event loop serves requests from a read-only
SQLite connection (WAL, so reads never block on the writer). A refresh runs in a
worker thread with its own connection, so present cards keep being served while
it downloads/embeds. Misses funnel into one shared "freshen" task (metadata
check, then refresh if stale); concurrent misses await that same task.

The same service hosts the key catalog (component-key descriptions, see
keys.py) and an embedding cache for arbitrary short strings. String misses are
embedded in a worker thread; a string already being embedded is awaited, not
embedded again.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from magicnet.catalog.embedder import Embedder
from magicnet.catalog.keys import Registry, registry_from_rows
from magicnet.catalog.scryfall import ScryfallSource
from magicnet.catalog.store import EMBED_CHUNK, CatalogStore, Resolved, parse_ts, sha256

log = logging.getLogger("magicnet.catalog")

MAX_STRINGS = 4096  # per /strings/embed or /keys/embed request
MAX_STRING_CHARS = 8192


@dataclass
class Outcome:
    ok: bool  # False: Scryfall/refresh failed, so misses are unconfirmed
    refreshed: bool = False
    error: str = ""


class CatalogService:
    def __init__(
        self,
        db_path: Path,
        source: ScryfallSource,
        embedder_factory: Callable[[], Embedder] | None,
        dump_dir: Path,
        *,
        check_interval: float = 300.0,
        negative_ttl: float = 12 * 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.db_path = Path(db_path)
        self.store = CatalogStore(self.db_path)
        self.source = source
        self.embedder_factory = embedder_factory
        self.dump_dir = Path(dump_dir)
        self.check_interval = check_interval
        self.negative_ttl = negative_ttl
        self.clock = clock
        self._task: asyncio.Task | None = None
        self._last_check: tuple[float, dict | None, str] | None = None  # (t, metadata, error)
        self._negative: dict[str, tuple[str | None, float]] = {}  # key -> (dump_updated_at, t)
        self.metadata_checks = 0
        self.refreshes = 0
        self._embedder: Embedder | None = None
        self._embedder_lock = threading.Lock()
        self._inflight: dict[str, asyncio.Task] = {}  # text_hash -> task embedding it
        self._registries: dict[tuple[int, int], Registry] = {}  # (as_of, latest) -> registry
        self._warned: set[str] = set()
        self.string_embed_calls = 0

    def check_as_of(self, as_of: int | None) -> int:
        latest = self.store.latest_version()
        if latest is None:
            raise HTTPException(503, "catalog is empty; run the import or a refresh first")
        if as_of is None:
            return latest
        if not 1 <= as_of <= latest:
            raise HTTPException(400, f"as_of must be in 1..{latest}")
        return as_of

    # -- lookups ------------------------------------------------------------
    def _negative_hit(self, key: str) -> bool:
        entry = self._negative.get(key)
        return (
            entry is not None
            and entry[0] == self.store.dump_updated_at()
            and self.clock() - entry[1] < self.negative_ttl
        )

    async def lookup(self, keys: list[str], as_of: int | None = None) -> tuple[int, dict[str, dict]]:
        as_of = self.check_as_of(as_of)
        results: dict[str, dict] = {}
        unresolved = []
        for key in dict.fromkeys(keys):
            r = self.store.resolve(key, as_of)
            if r.status != "missing":
                results[key] = _result(r)
            elif self._negative_hit(key):
                results[key] = {"status": "not_found", "reason": "not in current Scryfall dump (cached)"}
            else:
                unresolved.append(key)

        if unresolved:
            outcome = await self.freshen()
            for key in unresolved:
                r = self.store.resolve(key, as_of)
                if r.status != "missing":
                    results[key] = _result(r)
                elif not outcome.ok:
                    results[key] = {"status": "error", "reason": outcome.error}
                else:
                    self._negative[key] = (self.store.dump_updated_at(), self.clock())
                    results[key] = {"status": "not_found", "reason": "not in current Scryfall dump"}
        return as_of, {k: results[k] for k in keys}

    # -- strings ------------------------------------------------------------
    async def embed_strings(self, texts: list[str]) -> tuple[dict[str, np.ndarray], set[str]]:
        """Vectors for ``texts`` (text -> vector) and the subset that was
        already cached. Misses are embedded once even when requested
        concurrently: a hash being embedded maps to the task embedding it."""
        if len(texts) > MAX_STRINGS:
            raise HTTPException(413, f"at most {MAX_STRINGS} strings per request")
        if any(len(t) > MAX_STRING_CHARS for t in texts):
            raise HTTPException(413, f"strings are limited to {MAX_STRING_CHARS} characters")
        hashes = {t: sha256(t) for t in dict.fromkeys(texts)}
        have = self.store.string_vectors(hashes.values())
        cached = {t for t, h in hashes.items() if h in have}
        new = [t for t, h in hashes.items() if h not in have and h not in self._inflight]
        if new:
            task = asyncio.create_task(asyncio.to_thread(self._embed_sync, new))
            new_hashes = [hashes[t] for t in new]
            for h in new_hashes:
                self._inflight[h] = task

            def _done(_t, hs=new_hashes):
                for h in hs:
                    self._inflight.pop(h, None)

            task.add_done_callback(_done)
        # Captured before the first await: a task may finish (and leave
        # _inflight) while we wait on another.
        pending = {h: self._inflight[h] for t, h in hashes.items() if h not in have}
        out: dict[str, np.ndarray] = {}
        for t, h in hashes.items():
            if h in have:
                out[t] = have[h]
                continue
            try:
                out[t] = (await asyncio.shield(pending[h]))[h]
            except Exception as exc:
                log.warning("string embedding failed: %s", exc)
                raise HTTPException(503, f"embedding failed: {exc}") from exc
        return out, cached

    def _get_embedder(self) -> Embedder:
        with self._embedder_lock:
            if self._embedder is None:
                if self.embedder_factory is None:
                    raise RuntimeError("no embedder configured; cannot embed uncached strings")
                self._embedder = self.embedder_factory()
            return self._embedder

    def _embed_sync(self, texts: list[str]) -> dict[str, np.ndarray]:
        embedder = self._get_embedder()
        store = CatalogStore(self.db_path)  # own connection: this runs off the event loop
        try:
            store.check_embedder(embedder)
            out: dict[str, np.ndarray] = {}
            for i in range(0, len(texts), EMBED_CHUNK):
                chunk = texts[i : i + EMBED_CHUNK]
                self.string_embed_calls += 1
                vecs = np.asarray(embedder.embed(chunk), dtype="<f4")
                out.update(zip(store.put_strings(chunk, vecs), vecs))
            return out
        finally:
            store.close()

    # -- key catalog --------------------------------------------------------
    def registry(self, as_of: int) -> Registry:
        cache_key = (as_of, self.store.latest_version() or 0)
        reg = self._registries.get(cache_key)
        if reg is None:
            reg = self._registries[cache_key] = registry_from_rows(self.store.keys_at(as_of))
        return reg

    async def lookup_keys(self, keys: list[str], as_of: int | None = None, *,
                          vectors: bool = True) -> tuple[int, list[dict]]:
        """Describe (and embed) component keys as of a version. Unregistered
        keys get a description from a matching pattern, else one derived from
        the key's name; they are flagged, logged and recorded."""
        as_of = self.check_as_of(as_of)
        if len(keys) > MAX_STRINGS:
            raise HTTPException(413, f"at most {MAX_STRINGS} keys per request")
        reg = self.registry(as_of)
        results = []
        for key in dict.fromkeys(keys):
            source, entry = reg.resolve(key)
            if source != "registry":
                self.store.note_unregistered(key, source, entry.description)
                if key not in self._warned:
                    self._warned.add(key)
                    log.warning("unregistered key %r (%s): embedding %r", key, source, entry.description)
            results.append({"key": key, "registered": source == "registry", "source": source,
                            **entry.to_json()})
        if vectors:
            texts = [r["description"] for r in results if r["value_kind"] != "identity"]
            vecs, _ = await self.embed_strings(texts)
            for r in results:
                r["vector"] = vecs.get(r["description"]) if r["value_kind"] != "identity" else None
        return as_of, results

    async def list_keys(self, as_of: int | None = None, *, vectors: bool = False) -> tuple[int, list, list]:
        as_of = self.check_as_of(as_of)
        rows = self.store.keys_at(as_of)
        keys = [r for r in rows if not r["is_pattern"]]
        patterns = sorted((r for r in rows if r["is_pattern"]), key=lambda r: r.get("order", 0))
        if vectors:
            vecs, _ = await self.embed_strings([r["description"] for r in keys if r["value_kind"] != "identity"])
            for r in keys:
                r["vector"] = vecs.get(r["description"]) if r["value_kind"] != "identity" else None
        return as_of, keys, patterns

    # -- freshness ----------------------------------------------------------
    async def freshen(self, *, force: bool = False) -> Outcome:
        """Make sure the dump is current. Single-flight: if a check/refresh is
        already running, wait for it instead of starting another."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._check_and_refresh(force))
        return await asyncio.shield(self._task)

    async def _check_and_refresh(self, force: bool) -> Outcome:
        now = self.clock()
        if not force and self._last_check and now - self._last_check[0] < self.check_interval:
            _, meta, error = self._last_check
        else:
            try:
                self.metadata_checks += 1
                meta, error = await asyncio.to_thread(self.source.bulk_metadata), ""
            except Exception as exc:  # network, HTTP, bad JSON
                log.warning("Scryfall metadata check failed: %s", exc)
                meta, error = None, f"Scryfall metadata check failed: {exc}"
            self._last_check = (now, meta, error)
        if meta is None:
            return Outcome(ok=False, error=error)

        stored = self.store.dump_updated_at()
        if stored is not None and parse_ts(meta["updated_at"]) <= parse_ts(stored):
            return Outcome(ok=True)
        try:
            self.refreshes += 1
            await asyncio.to_thread(self._refresh_sync, meta)
        except Exception as exc:
            log.exception("refresh failed")
            error = f"refresh failed: {exc}"
            self._last_check = (now, None, error)  # back off until the next interval
            return Outcome(ok=False, error=error)
        return Outcome(ok=True, refreshed=True)

    def _refresh_sync(self, meta: dict) -> None:
        stamp = parse_ts(meta["updated_at"]).strftime("%Y%m%dT%H%M%S")
        dest = self.dump_dir / f"default-cards-{stamp}.jsonl.gz"
        if not dest.exists():
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            self.source.download(meta, dest)
        store = CatalogStore(self.db_path)  # own connection: this runs off the event loop
        try:
            report = store.refresh_from_dump(
                dest,
                dump_updated_at=meta["updated_at"],
                dump_source=meta.get("jsonl_download_uri") or meta.get("download_uri") or str(dest),
                embedder_factory=self.embedder_factory,
                note="refresh on miss",
            )
        finally:
            store.close()
        log.info("refresh: %s", report)


def _result(r: Resolved) -> dict:
    if r.status == "ambiguous":
        return {"status": "ambiguous", "reason": "name is claimed by several cards", "candidates": r.candidates}
    if r.status == "excluded":
        return {
            "status": "not_found",
            "reason": f"name maps to {r.oracle_id}, which is excluded from the catalog card set",
            "oracle_id": r.oracle_id,
        }
    return {"status": "found", "card": r.card, "vector": r.vector}


def _encode(result: dict, encoding: str) -> dict:
    vec = result.get("vector")
    if vec is None:
        return result
    out = dict(result)
    out["vector"] = _vec(vec, encoding)
    return out


def _vec(vec: np.ndarray, encoding: str) -> str | list[float]:
    return base64.b64encode(vec.astype("<f4").tobytes()).decode() if encoding == "b64" else vec.tolist()


class LookupRequest(BaseModel):
    keys: list[str]
    as_of: int | None = None
    encoding: str = "b64"  # "b64": base64 little-endian float32; "list": JSON floats


class KeysRequest(BaseModel):
    keys: list[str]
    as_of: int | None = None
    vectors: bool = True
    encoding: str = "b64"


class StringsRequest(BaseModel):
    strings: list[str]
    encoding: str = "b64"


def create_app(service: CatalogService) -> FastAPI:
    app = FastAPI(title="magicnet card catalog")
    app.state.service = service
    store = service.store

    def check_version(version: int | None) -> int:
        latest = store.latest_version()
        if latest is None:
            raise HTTPException(503, "catalog is empty")
        if version is None:
            return latest
        if not 1 <= version <= latest:
            raise HTTPException(404, f"no catalog version {version}")
        return version

    @app.get("/health")
    async def health() -> dict:
        model, dims = store.model
        return {
            "latest_version": store.latest_version(),
            "dump_updated_at": store.dump_updated_at(),
            "embed_model": model,
            "embed_dims": dims,
            "refreshing": service._task is not None and not service._task.done(),
        }

    @app.get("/versions")
    async def versions() -> list[dict]:
        return store.versions()

    @app.get("/versions/latest")
    async def latest_version() -> dict:
        return {"version": check_version(None)}

    @app.get("/versions/{version}/changes")
    async def changes(version: int) -> dict:
        return {"version": version, **store.changes(check_version(version))}

    @app.post("/lookup")
    async def lookup(req: LookupRequest) -> dict:
        version, results = await service.lookup(req.keys, req.as_of)
        return {
            "as_of": version,
            "embed_model": store.model[0],
            "results": [{"key": k, **_encode(v, req.encoding)} for k, v in results.items()],
        }

    @app.get("/cards/{key:path}")
    async def card(key: str, as_of: int | None = None, encoding: str = "list") -> dict:
        version, results = await service.lookup([key], as_of)
        res = results[key]
        if res["status"] == "error":
            raise HTTPException(503, res["reason"])
        if res["status"] != "found":
            raise HTTPException(404, res)
        return {"as_of": version, "key": key, **_encode(res, encoding)}

    @app.get("/names")
    async def names(version: int | None = None, format: str = Query("tsv", pattern="^(tsv|json)$")):
        v = check_version(version)
        table = store.name_table(v)
        if format == "json":
            return {"version": v, "names": dict(sorted(table.items()))}
        body = "".join(f"{name}\t{table[name]}\n" for name in sorted(table))
        return PlainTextResponse(body, headers={"X-Catalog-Version": str(v)})

    @app.get("/keys")
    async def keys(as_of: int | None = None, vectors: bool = False, encoding: str = "list") -> dict:
        version, rows, patterns = await service.list_keys(as_of, vectors=vectors)
        drop = {"text_hash"}
        return {
            "as_of": version,
            "embed_model": store.model[0],
            "keys": [_encode({k: v for k, v in r.items() if k not in drop}, encoding) for r in rows],
            "patterns": [{k: v for k, v in r.items() if k not in drop} for r in patterns],
        }

    @app.post("/keys/embed")
    async def keys_embed(req: KeysRequest) -> dict:
        version, results = await service.lookup_keys(req.keys, req.as_of, vectors=req.vectors)
        return {"as_of": version, "embed_model": store.model[0],
                "results": [_encode(r, req.encoding) for r in results]}

    @app.get("/keys/unregistered")
    async def keys_unregistered() -> list[dict]:
        return store.unregistered_keys()

    @app.post("/strings/embed")
    async def strings_embed(req: StringsRequest) -> dict:
        vecs, cached = await service.embed_strings(req.strings)
        return {
            "embed_model": store.model[0],
            "results": [{"text": t, "cached": t in cached, "vector": _vec(vecs[t], req.encoding)}
                        for t in dict.fromkeys(req.strings)],
        }

    @app.post("/admin/refresh")
    async def refresh() -> dict:
        outcome = await service.freshen(force=True)
        if not outcome.ok:
            raise HTTPException(503, outcome.error)
        return {"refreshed": outcome.refreshed, "latest_version": store.latest_version()}

    return app
