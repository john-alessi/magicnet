"""SQLite storage for the card catalog: versioned card rows, embeddings, names.

Everything is append-only. A *catalog version* is created per refresh; a card
row is (oracle_id, version) where ``version`` is the catalog version that
introduced that row, so older rows stay readable for ``as_of`` lookups.
Embeddings are keyed by (text_hash, model, dims), so a vector is reused exactly
when the embedded text and the embedding model are unchanged.

The key catalog (keys.py) is stored alongside: ``keys(key, version)`` rows,
written by ``sync_keys`` as a new catalog version when the registry changes, and
``strings`` recording the text of every non-card vector in ``embeddings``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from magicnet.cards import iter_unique_cards, name_claims
from magicnet.catalog.embedder import Embedder

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS versions (
    version         INTEGER PRIMARY KEY,
    created_at      TEXT NOT NULL,
    dump_updated_at TEXT NOT NULL,   -- Scryfall bulk-data updated_at of the source dump
    dump_source     TEXT NOT NULL,   -- download URI or local path
    note            TEXT
);
CREATE TABLE IF NOT EXISTS cards (
    oracle_id      TEXT NOT NULL,
    version        INTEGER NOT NULL REFERENCES versions,  -- version that introduced this row
    name           TEXT NOT NULL,
    mana_cost      TEXT NOT NULL,
    cmc            REAL NOT NULL,
    type_line      TEXT NOT NULL,
    colors         TEXT NOT NULL,    -- JSON list
    color_identity TEXT NOT NULL,    -- JSON list
    layout         TEXT NOT NULL,
    text           TEXT NOT NULL,    -- build_embedding_text(card)
    text_hash      TEXT NOT NULL,    -- sha256(text)
    record_hash    TEXT NOT NULL,    -- sha256 of all fields above
    PRIMARY KEY (oracle_id, version)
);
CREATE TABLE IF NOT EXISTS embeddings (
    text_hash  TEXT NOT NULL,
    model      TEXT NOT NULL,
    dims       INTEGER NOT NULL,
    vector     BLOB NOT NULL,        -- float32 little-endian
    created_at TEXT NOT NULL,
    PRIMARY KEY (text_hash, model, dims)
);
CREATE TABLE IF NOT EXISTS names (
    name       TEXT NOT NULL,
    version    INTEGER NOT NULL,     -- version from which this mapping holds
    oracle_id  TEXT,                 -- set: resolves; NULL: ambiguous or gone
    candidates TEXT,                 -- JSON list of oracle_ids when ambiguous
    PRIMARY KEY (name, version)
);
CREATE TABLE IF NOT EXISTS changes (  -- the errata log
    version   INTEGER NOT NULL,
    oracle_id TEXT NOT NULL,
    kind      TEXT NOT NULL CHECK (kind IN ('added', 'changed', 'updated', 'removed')),
    PRIMARY KEY (version, oracle_id)
);
-- Key catalog (additive): component-key registry rows, versioned like cards.
CREATE TABLE IF NOT EXISTS keys (
    key         TEXT NOT NULL,       -- concrete key, or a pattern such as ctr_{name}
    version     INTEGER NOT NULL REFERENCES versions,  -- version from which this row holds
    is_pattern  INTEGER NOT NULL,
    description TEXT,                -- NULL: removed from the registry at this version
    value_kind  TEXT,
    token_types TEXT,                -- JSON list
    meta        TEXT,                -- JSON: notes, alt_value_kinds, pattern, java, order
    text_hash   TEXT,                -- sha256(description)
    record_hash TEXT NOT NULL,
    PRIMARY KEY (key, version)
);
CREATE TABLE IF NOT EXISTS key_registry (  -- one row per version that changed the registry
    version       INTEGER PRIMARY KEY REFERENCES versions,
    registry_hash TEXT NOT NULL,
    n_keys        INTEGER NOT NULL,
    n_patterns    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS strings (  -- text of every non-card string with a vector
    text_hash  TEXT PRIMARY KEY,
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS unregistered_keys (  -- keys looked up but not in the registry
    key         TEXT PRIMARY KEY,
    source      TEXT NOT NULL,       -- pattern | derived
    description TEXT NOT NULL,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    hits        INTEGER NOT NULL
);
"""

CARD_FIELDS = ("name", "mana_cost", "cmc", "type_line", "colors", "color_identity", "layout", "text")
CHANGE_KINDS = ("added", "changed", "updated", "removed")
EMBED_CHUNK = 1000  # texts per embedder call + commit, so a crash mid-refresh doesn't re-pay


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_hash(rec: dict) -> str:
    return sha256(json.dumps({k: rec[k] for k in CARD_FIELDS}, sort_keys=True, ensure_ascii=False))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    return conn


@dataclass
class Resolved:
    status: str  # found | missing | ambiguous | excluded
    card: dict | None = None
    vector: np.ndarray | None = None
    oracle_id: str | None = None
    candidates: list[str] = field(default_factory=list)


@dataclass
class RefreshReport:
    version: int | None  # None: nothing changed, so no version was created
    n_cards: int
    added: int = 0
    changed: int = 0
    updated: int = 0
    removed: int = 0
    names_changed: int = 0
    embedded: int = 0
    missing_vectors: int = 0  # texts left without a vector (import without embedder)


@dataclass
class KeySyncReport:
    version: int | None  # None: registry unchanged (or dry run), so no version was created
    registry_hash: str
    n_keys: int
    n_patterns: int
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    to_embed: int = 0  # distinct key descriptions without a vector for the catalog's model
    est_tokens: int = 0
    embedded: int = 0
    dry_run: bool = False


class CatalogStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn = connect(self.db_path)

    def close(self) -> None:
        self.conn.close()

    # -- meta / versions ----------------------------------------------------
    def meta(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self.conn.execute("SELECT key, value FROM meta")}

    def set_meta(self, **values) -> None:
        for k, v in values.items():
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (k, str(v)))

    @property
    def model(self) -> tuple[str, int]:
        m = self.meta()
        return m["embed_model"], int(m["embed_dims"])

    def latest_version(self) -> int | None:
        return self.conn.execute("SELECT MAX(version) FROM versions").fetchone()[0]

    def dump_updated_at(self) -> str | None:
        row = self.conn.execute(
            "SELECT dump_updated_at FROM versions ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def versions(self) -> list[dict]:
        counts: dict[int, dict[str, int]] = {}
        for v, kind, n in self.conn.execute(
            "SELECT version, kind, COUNT(*) FROM changes GROUP BY version, kind"
        ):
            counts.setdefault(v, {})[kind] = n
        return [
            {**dict(r), **{k: counts.get(r["version"], {}).get(k, 0) for k in CHANGE_KINDS}}
            for r in self.conn.execute("SELECT * FROM versions ORDER BY version")
        ]

    def changes(self, version: int) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {k: [] for k in CHANGE_KINDS}
        for r in self.conn.execute(
            "SELECT oracle_id, kind FROM changes WHERE version = ? ORDER BY oracle_id", (version,)
        ):
            out[r["kind"]].append(r["oracle_id"])
        return out

    # -- reads --------------------------------------------------------------
    def _card_row(self, oracle_id: str, as_of: int) -> sqlite3.Row | None:
        # The row as of `as_of` if the card existed then, else its latest row.
        q = "SELECT * FROM cards WHERE oracle_id = ? {} ORDER BY version DESC LIMIT 1"
        return self.conn.execute(q.format("AND version <= ?"), (oracle_id, as_of)).fetchone() or (
            self.conn.execute(q.format(""), (oracle_id,)).fetchone()
        )

    def _name_row(self, name: str, as_of: int) -> sqlite3.Row | None:
        q = "SELECT * FROM names WHERE name = ? {} ORDER BY version DESC LIMIT 1"
        row = self.conn.execute(q.format("AND version <= ?"), (name, as_of)).fetchone()
        if row is None or (row["oracle_id"] is None and row["candidates"] is None):
            row = self.conn.execute(q.format(""), (name,)).fetchone()
        return row

    def vector(self, text_hash: str) -> np.ndarray | None:
        model, dims = self.model
        row = self.conn.execute(
            "SELECT vector FROM embeddings WHERE text_hash = ? AND model = ? AND dims = ?",
            (text_hash, model, dims),
        ).fetchone()
        return np.frombuffer(row[0], dtype="<f4") if row else None

    def resolve(self, key: str, as_of: int, *, with_vector: bool = True) -> Resolved:
        """Resolve an oracle_id or a card name (oracle_ids are tried first)."""
        row = self._card_row(key, as_of)
        if row is None:
            nrow = self._name_row(key, as_of)
            if nrow is None or (nrow["oracle_id"] is None and nrow["candidates"] is None):
                return Resolved("missing")
            if nrow["oracle_id"] is None:
                return Resolved("ambiguous", candidates=json.loads(nrow["candidates"]))
            row = self._card_row(nrow["oracle_id"], as_of)
            if row is None:
                # Named in the dump but filtered out of the card set (e.g. a
                # digital-only card while include_digital is off).
                return Resolved("excluded", oracle_id=nrow["oracle_id"])
        card = {
            "oracle_id": row["oracle_id"],
            "row_version": row["version"],
            **{k: row[k] for k in CARD_FIELDS},
            "colors": json.loads(row["colors"]),
            "color_identity": json.loads(row["color_identity"]),
            "text_hash": row["text_hash"],
        }
        vec = self.vector(row["text_hash"]) if with_vector else None
        return Resolved("found", card=card, vector=vec, oracle_id=row["oracle_id"])

    def name_table(self, version: int) -> dict[str, str]:
        rows = self.conn.execute(
            """SELECT n.name, n.oracle_id FROM names n
               JOIN (SELECT name, MAX(version) AS v FROM names WHERE version <= ? GROUP BY name) m
                 ON n.name = m.name AND n.version = m.v
               WHERE n.oracle_id IS NOT NULL""",
            (version,),
        )
        return {r[0]: r[1] for r in rows}

    # -- writes -------------------------------------------------------------
    def _latest_cards(self) -> tuple[dict[str, tuple[str, str]], set[str]]:
        """oracle_id -> (text_hash, record_hash) of each card's latest row, and
        the set of cards whose latest change is a removal."""
        rows = self.conn.execute(
            """SELECT c.oracle_id, c.text_hash, c.record_hash FROM cards c
               JOIN (SELECT oracle_id, MAX(version) AS v FROM cards GROUP BY oracle_id) m
                 ON c.oracle_id = m.oracle_id AND c.version = m.v"""
        )
        removed = {
            r[0]
            for r in self.conn.execute(
                """SELECT ch.oracle_id FROM changes ch
                   JOIN (SELECT oracle_id, MAX(version) AS v FROM changes GROUP BY oracle_id) m
                     ON ch.oracle_id = m.oracle_id AND ch.version = m.v
                   WHERE ch.kind = 'removed'"""
            )
        }
        return {r[0]: (r[1], r[2]) for r in rows}, removed

    def _latest_names(self) -> dict[str, tuple[str | None, str | None]]:
        rows = self.conn.execute(
            """SELECT n.name, n.oracle_id, n.candidates FROM names n
               JOIN (SELECT name, MAX(version) AS v FROM names GROUP BY name) m
                 ON n.name = m.name AND n.version = m.v"""
        )
        return {r[0]: (r[1], r[2]) for r in rows if r[1] is not None or r[2] is not None}

    def _missing_hashes(self, hashes: Iterable[str]) -> set[str]:
        model, dims = self.model
        have = {
            r[0]
            for r in self.conn.execute(
                "SELECT text_hash FROM embeddings WHERE model = ? AND dims = ?", (model, dims)
            )
        }
        return set(hashes) - have

    def put_embeddings(self, hashes: list[str], vectors: np.ndarray) -> None:
        model, dims = self.model
        vectors = np.asarray(vectors, dtype="<f4")
        if vectors.shape != (len(hashes), dims):
            raise ValueError(f"expected vectors of shape {(len(hashes), dims)}, got {vectors.shape}")
        ts = now_iso()
        self.conn.execute("BEGIN")
        self.conn.executemany(
            "INSERT OR IGNORE INTO embeddings VALUES (?, ?, ?, ?, ?)",
            ((h, model, dims, v.tobytes(), ts) for h, v in zip(hashes, vectors)),
        )
        self.conn.execute("COMMIT")

    # -- strings (key descriptions, keyword names, type lines, ...) ---------
    def string_vectors(self, hashes: Iterable[str]) -> dict[str, np.ndarray]:
        model, dims = self.model
        hashes = list(hashes)
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(hashes), 900):  # SQLite variable limit
            chunk = hashes[i : i + 900]
            rows = self.conn.execute(
                f"SELECT text_hash, vector FROM embeddings WHERE model = ? AND dims = ? "
                f"AND text_hash IN ({','.join('?' * len(chunk))})",
                (model, dims, *chunk),
            )
            out.update((r[0], np.frombuffer(r[1], dtype="<f4")) for r in rows)
        return out

    def put_strings(self, texts: list[str], vectors: np.ndarray) -> list[str]:
        """Store vectors for arbitrary strings (and the strings themselves, so
        they can be re-embedded under another model). Returns their hashes."""
        hashes = [sha256(t) for t in texts]
        ts = now_iso()
        self.conn.executemany("INSERT OR IGNORE INTO strings VALUES (?, ?, ?)",
                              ((h, t, ts) for h, t in zip(hashes, texts)))
        self.put_embeddings(hashes, vectors)
        return hashes

    def embed_strings(self, texts: Iterable[str], embedder: Embedder) -> int:
        """Embed and store the strings that have no vector yet; returns how many."""
        self.check_embedder(embedder)
        uniq = list(dict.fromkeys(texts))
        have = self.string_vectors(sha256(t) for t in uniq)
        todo = [t for t in uniq if sha256(t) not in have]
        for i in range(0, len(todo), EMBED_CHUNK):
            chunk = todo[i : i + EMBED_CHUNK]
            self.put_strings(chunk, embedder.embed(chunk))
        return len(todo)

    def check_embedder(self, embedder: Embedder) -> None:
        model, dims = self.model
        if (embedder.model, embedder.dims) != (model, dims):
            raise RuntimeError(
                f"embedder is {embedder.model}/{embedder.dims} but the catalog stores "
                f"{model}/{dims}; vectors from different models are not interchangeable"
            )

    # -- key catalog --------------------------------------------------------
    def _key_rows(self) -> dict[str, list[sqlite3.Row]]:
        rows: dict[str, list[sqlite3.Row]] = {}
        for r in self.conn.execute("SELECT * FROM keys ORDER BY key, version"):
            rows.setdefault(r["key"], []).append(r)
        return rows

    def keys_at(self, as_of: int) -> list[dict]:
        """The registry as of version ``as_of``: each key's row as it was then;
        a key added after ``as_of`` gets its first row, so a client pinned to
        any version (even one older than the key catalog) sees an immutable
        registry. Keys removed by ``as_of`` are omitted."""
        out = []
        for rows in self._key_rows().values():
            then = [r for r in rows if r["version"] <= as_of]
            row = then[-1] if then else rows[0]
            if row["description"] is None:
                continue
            out.append({
                "key": row["key"], "is_pattern": bool(row["is_pattern"]), "description": row["description"],
                "value_kind": row["value_kind"], "token_types": json.loads(row["token_types"]),
                "row_version": row["version"], "text_hash": row["text_hash"], **json.loads(row["meta"]),
            })
        return out

    def key_registry_versions(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM key_registry ORDER BY version")]

    def sync_keys(
        self,
        entries: list[dict],
        *,
        registry_hash: str,
        embedder_factory: Callable[[], Embedder] | None = None,
        dry_run: bool = False,
        note: str = "",
    ) -> KeySyncReport:
        """Record the key registry as a new catalog version if it changed, then
        embed key descriptions that have no vector. ``dry_run`` writes nothing
        and never calls the embedder; it only reports what would happen.

        A key-registry version reuses the latest version's dump metadata, so
        card lookups and Scryfall freshness are unaffected by it."""
        from magicnet.catalog.keys import estimate_tokens

        rows = []
        for e in entries:
            rec = {k: e[k] for k in ("key", "is_pattern", "description", "value_kind", "token_types", "meta")}
            rec["record_hash"] = sha256(json.dumps(rec, sort_keys=True, ensure_ascii=False))
            rows.append(rec)
        report = KeySyncReport(None, registry_hash, sum(not r["is_pattern"] for r in rows),
                               sum(r["is_pattern"] for r in rows), dry_run=dry_run)
        latest = {k: rs[-1] for k, rs in self._key_rows().items()}
        new_rows = []
        for r in rows:
            old = latest.get(r["key"])
            if old is None or old["description"] is None:
                report.added.append(r["key"])
            elif old["record_hash"] != r["record_hash"]:
                report.changed.append(r["key"])
            else:
                continue
            new_rows.append(r)
        report.removed = sorted(k for k, old in latest.items()
                                if old["description"] is not None and k not in {r["key"] for r in rows})

        texts = list(dict.fromkeys(r["description"] for r in rows
                                   if not r["is_pattern"] and r["value_kind"] != "identity"))
        have = self.string_vectors(sha256(t) for t in texts)
        todo = [t for t in texts if sha256(t) not in have]
        report.to_embed, report.est_tokens = len(todo), estimate_tokens(todo)
        if dry_run:
            return report

        if new_rows or report.removed:
            latest_v = self.latest_version()
            if latest_v is None:
                raise RuntimeError("catalog has no version yet; import or refresh cards first")
            prev = self.conn.execute("SELECT * FROM versions WHERE version = ?", (latest_v,)).fetchone()
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                version = (self.latest_version() or 0) + 1
                self.conn.execute(
                    "INSERT INTO versions VALUES (?, ?, ?, ?, ?)",
                    (version, now_iso(), prev["dump_updated_at"], prev["dump_source"],
                     note or f"key registry {registry_hash[:12]}"),
                )
                self.conn.executemany(
                    "INSERT INTO keys VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(r["key"], version, int(r["is_pattern"]), r["description"], r["value_kind"],
                      json.dumps(r["token_types"]), json.dumps(r["meta"], sort_keys=True),
                      sha256(r["description"]), r["record_hash"]) for r in new_rows]
                    + [(k, version, int(latest[k]["is_pattern"]), None, None, None, "{}", None,
                        "removed") for k in report.removed],
                )
                self.conn.execute("INSERT INTO key_registry VALUES (?, ?, ?, ?)",
                                  (version, registry_hash, report.n_keys, report.n_patterns))
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            report.version = version
        if todo and embedder_factory is not None:
            report.embedded = self.embed_strings(todo, embedder_factory())
        return report

    def note_unregistered(self, key: str, source: str, description: str) -> None:
        ts = now_iso()
        self.conn.execute(
            """INSERT INTO unregistered_keys VALUES (?, ?, ?, ?, ?, 1)
               ON CONFLICT(key) DO UPDATE SET last_seen = excluded.last_seen, hits = hits + 1""",
            (key, source, description, ts, ts),
        )

    def unregistered_keys(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM unregistered_keys ORDER BY hits DESC, key")]

    def apply_snapshot(
        self,
        records: list[dict],
        claims: dict[str, set[str]],
        *,
        dump_updated_at: str,
        dump_source: str,
        embedder_factory: Callable[[], Embedder] | None,
        note: str = "",
    ) -> RefreshReport:
        """Diff a full dump snapshot against the latest catalog state, embed
        only texts that have no vector for the catalog's model, and record a
        new version if anything (cards, names, or the dump itself) changed.

        ``embedder_factory=None`` skips embedding (used by the bootstrap import).
        """
        for rec in records:
            rec["text_hash"] = sha256(rec["text"])
            rec["record_hash"] = record_hash(rec)

        report = RefreshReport(version=None, n_cards=len(records))
        todo = self._missing_hashes(r["text_hash"] for r in records)
        if todo and embedder_factory is not None:
            embedder = embedder_factory()
            self.check_embedder(embedder)
            texts = {r["text_hash"]: r["text"] for r in records if r["text_hash"] in todo}
            pending = sorted(texts)
            for i in range(0, len(pending), EMBED_CHUNK):
                chunk = pending[i : i + EMBED_CHUNK]
                self.put_embeddings(chunk, embedder.embed([texts[h] for h in chunk]))
            report.embedded = len(pending)
        else:
            report.missing_vectors = len(todo)

        latest, removed = self._latest_cards()
        new_rows, kinds = [], {}
        for rec in records:
            oid, old = rec["oracle_id"], latest.get(rec["oracle_id"])
            if old is None or oid in removed:
                kinds[oid] = "added"
            elif old[0] != rec["text_hash"]:
                kinds[oid] = "changed"
            elif old[1] != rec["record_hash"]:
                kinds[oid] = "updated"
            if oid in kinds and (old is None or old[1] != rec["record_hash"]):
                new_rows.append(rec)
        for oid in latest.keys() - removed - {r["oracle_id"] for r in records}:
            kinds[oid] = "removed"

        old_names = self._latest_names()
        new_names = {
            name: (next(iter(ids)), None) if len(ids) == 1 else (None, json.dumps(sorted(ids)))
            for name, ids in claims.items()
        }
        name_rows = [(n, *v) for n, v in new_names.items() if old_names.get(n) != v]
        name_rows += [(n, None, None) for n in old_names.keys() - new_names.keys()]

        for kind in kinds.values():
            setattr(report, kind, getattr(report, kind) + 1)
        report.names_changed = len(name_rows)
        if not kinds and not name_rows and dump_updated_at == self.dump_updated_at():
            return report

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            version = (self.latest_version() or 0) + 1
            self.conn.execute(
                "INSERT INTO versions VALUES (?, ?, ?, ?, ?)",
                (version, now_iso(), dump_updated_at, dump_source, note),
            )
            self.conn.executemany(
                "INSERT INTO cards VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (r["oracle_id"], version, r["name"], r["mana_cost"], r["cmc"], r["type_line"],
                     json.dumps(r["colors"]), json.dumps(r["color_identity"]), r["layout"],
                     r["text"], r["text_hash"], r["record_hash"])
                    for r in new_rows
                ),
            )
            self.conn.executemany(
                "INSERT INTO changes VALUES (?, ?, ?)", ((version, o, k) for o, k in kinds.items())
            )
            self.conn.executemany(
                "INSERT INTO names VALUES (?, ?, ?, ?)", ((n, version, o, c) for n, o, c in name_rows)
            )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        report.version = version
        return report

    def refresh_from_dump(
        self,
        dump_path: Path,
        *,
        dump_updated_at: str,
        dump_source: str | None = None,
        embedder_factory: Callable[[], Embedder] | None,
        include_digital: bool = False,
        note: str = "",
    ) -> RefreshReport:
        """Refresh from a local Scryfall ``.jsonl.gz`` dump. ``include_digital``
        only applies when initialising an empty catalog; afterwards the stored
        setting wins so the card set stays consistent across versions."""
        if "embed_model" not in self.meta():
            if embedder_factory is None:
                raise RuntimeError("empty catalog: an embedder is needed to fix model/dims")
            emb = embedder_factory()
            self.set_meta(embed_model=emb.model, embed_dims=emb.dims, include_digital=include_digital)
        include_digital = self.meta().get("include_digital") == "True"
        records = [r.to_json() for r in iter_unique_cards(dump_path, include_digital=include_digital)]
        return self.apply_snapshot(
            records,
            name_claims(dump_path),
            dump_updated_at=dump_updated_at,
            dump_source=dump_source or str(dump_path),
            embedder_factory=embedder_factory,
            note=note,
        )
