"""Seed a catalog from the artefacts the offline scripts already produced, so
none of the existing embeddings has to be paid for again. Reads only; never
modifies the input files."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from magicnet.cards import iter_unique_cards, name_claims
from magicnet.catalog.embedder import NATIVE_DIMS
from magicnet.catalog.store import CatalogStore, RefreshReport, sha256
from magicnet.embedding import load_card_records

UNKNOWN = "unknown"


def embedding_provenance(matrix_dims: int) -> tuple[str, int, str]:
    """Model/dims the existing matrix was made with, from the same env vars
    scripts/embed_cards.py used (.env). Returns (model, dims, how-we-know)."""
    from dotenv import load_dotenv

    load_dotenv()
    model = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    env_dims = os.environ.get("AZURE_OPENAI_EMBEDDING_DIMENSIONS")
    if not model:
        return UNKNOWN, matrix_dims, "AZURE_OPENAI_EMBEDDING_DEPLOYMENT unset; model unknown, dims from matrix shape"
    if env_dims and int(env_dims) != matrix_dims:
        raise RuntimeError(f"env says {env_dims} dims but the matrix has {matrix_dims}")
    native = NATIVE_DIMS.get(model)
    if not env_dims and native and native != matrix_dims:
        raise RuntimeError(f"{model} is natively {native}-d but the matrix has {matrix_dims}")
    how = (
        "deployment from AZURE_OPENAI_EMBEDDING_DEPLOYMENT; dims "
        + (f"from AZURE_OPENAI_EMBEDDING_DIMENSIONS={env_dims}" if env_dims else "native (no truncation requested), matches matrix shape")
    )
    return model, matrix_dims, how


def import_existing(
    store: CatalogStore,
    *,
    dump_path: Path,
    cards_path: Path,
    embeddings_path: Path,
    ids_path: Path,
    model: str | None = None,
    dims: int | None = None,
    dump_updated_at: str | None = None,
) -> tuple[RefreshReport, dict]:
    if store.latest_version() is not None:
        raise RuntimeError(f"{store.db_path} already has versions; import only seeds an empty catalog")

    records = load_card_records(cards_path)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    matrix = np.load(embeddings_path, mmap_mode="r")
    if matrix.shape[0] != len(ids):
        raise RuntimeError(f"{embeddings_path} has {matrix.shape[0]} rows but {ids_path} has {len(ids)} ids")

    if model is None:
        model, env_dims, how = embedding_provenance(matrix.shape[1])
        dims = dims or env_dims
    else:
        how = "given on the command line"
        dims = dims or matrix.shape[1]
    if dims != matrix.shape[1]:
        raise RuntimeError(f"dims {dims} != matrix width {matrix.shape[1]}")

    # Which card set did extract_cards.py produce? The catalog must use the same
    # filter, or the first refresh would see "new" cards and embed them.
    have = {r["oracle_id"] for r in records}
    include_digital = None
    for flag in (False, True):
        if {r.oracle_id for r in iter_unique_cards(dump_path, include_digital=flag)} == have:
            include_digital = flag
            break
    if include_digital is None:
        include_digital = False
        how_set = "cards.jsonl matches neither filter of the dump; assuming include_digital=False"
    else:
        how_set = f"cards.jsonl matches the dump with include_digital={include_digital}"

    if dump_updated_at is None:
        mtime = datetime.fromtimestamp(dump_path.stat().st_mtime, tz=timezone.utc)
        dump_updated_at = mtime.isoformat(timespec="seconds")
        ts_how = "inferred from the dump file's mtime (the download time is an upper bound on Scryfall's updated_at)"
    else:
        ts_how = "given on the command line"

    store.set_meta(
        embed_model=model,
        embed_dims=dims,
        embed_provenance=how,
        include_digital=include_digital,
        card_set_provenance=how_set,
        bootstrap_dump_updated_at=ts_how,
    )

    text_by_id = {r["oracle_id"]: r["text"] for r in records}
    known = [(i, oid) for i, oid in enumerate(ids) if oid in text_by_id]
    for start in range(0, len(known), 2000):
        chunk = known[start : start + 2000]
        store.put_embeddings(
            [sha256(text_by_id[oid]) for _, oid in chunk],
            np.asarray(matrix[[i for i, _ in chunk]], dtype=np.float32),
        )

    report = store.apply_snapshot(
        records,
        name_claims(dump_path),
        dump_updated_at=dump_updated_at,
        dump_source=str(dump_path),
        embedder_factory=None,
        note=f"bootstrap import from {cards_path.name} + {embeddings_path.name}",
    )
    info = {
        "embed_model": model,
        "embed_dims": dims,
        "embed_provenance": how,
        "include_digital": include_digital,
        "card_set_provenance": how_set,
        "dump_updated_at": dump_updated_at,
        "dump_updated_at_provenance": ts_how,
        "vectors_imported": len(known),
        "vectors_without_card": len(ids) - len(known),
    }
    return report, info
