"""CLI for the card catalog.

    uv run python -m magicnet.catalog import            # seed from existing data/ files
    uv run python -m magicnet.catalog serve --port 8765
    uv run python -m magicnet.catalog refresh           # check Scryfall, refresh if stale
    uv run python -m magicnet.catalog refresh --dump PATH --updated-at TS   # from a local dump
    uv run python -m magicnet.catalog keys-sync --dry-run   # what a key-registry sync would embed
    uv run python -m magicnet.catalog keys-sync             # record keys.json, embed missing descriptions
    uv run python -m magicnet.catalog pca-fit [--components 256]   # fit + store a new PCA basis (deliberate)
    uv run python -m magicnet.catalog pca-list              # stored bases and which is current
    uv run python -m magicnet.catalog pca-use BASIS_ID      # make an older basis current again
    uv run python -m magicnet.catalog pca-drift             # drift log written by refreshes

`refresh` writes to the DB directly; while `serve` is running, use
`POST /admin/refresh` instead so the server's single-flight guard applies.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from magicnet.catalog.store import DRIFT_MARGIN

REPO = Path(__file__).resolve().parents[3]
DEFAULT_DB = REPO / "data/catalog/catalog.sqlite"
DEFAULT_DUMP_DIR = REPO / "data/catalog/dumps"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m magicnet.catalog", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP_DIR, help="where new Scryfall dumps are saved")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--check-interval", type=float, default=300.0, help="min seconds between Scryfall metadata checks")
    s.add_argument("--drift-margin", type=float, default=DRIFT_MARGIN,
                   help="warn when new cards' PCA unexplained variance exceeds the basis baseline by this much")

    i = sub.add_parser("import", help="seed an empty catalog from the existing offline artefacts")
    i.add_argument("--dump", type=Path, default=REPO / "data/raw/scryfall-default-cards.jsonl.gz")
    i.add_argument("--cards", type=Path, default=REPO / "data/processed/cards.jsonl")
    i.add_argument("--embeddings", type=Path, default=REPO / "data/processed/card_embeddings.npy")
    i.add_argument("--ids", type=Path, default=REPO / "data/processed/card_embeddings.ids.json")
    i.add_argument("--model", help="embedding deployment the matrix was made with (default: from .env)")
    i.add_argument("--dims", type=int)
    i.add_argument("--updated-at", help="Scryfall updated_at of the dump (default: file mtime)")

    r = sub.add_parser("refresh")
    r.add_argument("--dump", type=Path, help="refresh from this local .jsonl.gz instead of Scryfall")
    r.add_argument("--updated-at", help="updated_at to record for --dump (default: file mtime)")
    r.add_argument("--include-digital", action="store_true", help="only used when creating a new catalog")

    k = sub.add_parser("keys-sync", help="record the key registry (keys.json) and embed key descriptions")
    k.add_argument("--dry-run", action="store_true",
                   help="write nothing and call no API; report what would be recorded and embedded")
    k.add_argument("--no-embed", action="store_true", help="record the registry but embed nothing")
    k.add_argument("--registry", type=Path, help="keys.json to load (default: the packaged one)")

    p = sub.add_parser("pca-fit", help="fit a PCA basis on the current card pool and store it as a new version")
    p.add_argument("--components", type=int, default=256, help="K components to store (clients use any prefix)")
    p.add_argument("--note", default="")
    p.add_argument("--no-activate", action="store_true", help="store the basis without making it current")
    sub.add_parser("pca-list", help="list stored PCA bases")
    u = sub.add_parser("pca-use", help="make a stored basis current from a new catalog version on")
    u.add_argument("basis_id", type=int)
    sub.add_parser("pca-drift", help="print the drift log")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from magicnet.catalog.embedder import AzureEmbedder
    from magicnet.catalog.scryfall import HttpScryfall
    from magicnet.catalog.store import CatalogStore

    if args.cmd == "serve":
        import uvicorn

        from magicnet.catalog.service import CatalogService, create_app

        service = CatalogService(args.db, HttpScryfall(), AzureEmbedder.from_env, args.dump_dir,
                                 check_interval=args.check_interval, drift_margin=args.drift_margin)
        uvicorn.run(create_app(service), host=args.host, port=args.port)
        return 0

    store = CatalogStore(args.db)
    if args.cmd.startswith("pca-"):
        if args.cmd == "pca-fit":
            out = store.fit_pca(args.components, note=args.note, activate=not args.no_activate)
        elif args.cmd == "pca-list":
            out = store.pca_bases()
        elif args.cmd == "pca-use":
            out = {"version": store.use_pca(args.basis_id), "basis_id": args.basis_id}
        else:
            out = store.pca_drift()
        print(json.dumps(out, indent=2))
        return 0
    if args.cmd == "keys-sync":
        from magicnet.catalog.keys import KEYS_PATH, load_registry

        reg = load_registry(args.registry or KEYS_PATH)
        report = store.sync_keys(
            reg.entries(), registry_hash=reg.hash, dry_run=args.dry_run,
            embedder_factory=None if args.dry_run or args.no_embed else AzureEmbedder.from_env,
        )
        out = asdict(report)
        for kind in ("added", "changed", "removed"):  # long lists: counts plus a sample
            out[f"n_{kind}"] = len(out[kind])
            out[kind] = out[kind][:20]
        print(json.dumps(out, indent=2))
        return 0

    if args.cmd == "import":
        from magicnet.catalog.bootstrap import import_existing

        report, info = import_existing(
            store, dump_path=args.dump, cards_path=args.cards, embeddings_path=args.embeddings,
            ids_path=args.ids, model=args.model, dims=args.dims, dump_updated_at=args.updated_at,
        )
        print(json.dumps({"report": asdict(report), **info}, indent=2))
        return 0

    if args.dump:
        from datetime import datetime, timezone

        updated_at = args.updated_at or datetime.fromtimestamp(
            args.dump.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        report = store.refresh_from_dump(args.dump, dump_updated_at=updated_at,
                                         embedder_factory=AzureEmbedder.from_env,
                                         include_digital=args.include_digital, note="cli refresh")
    else:
        import asyncio

        from magicnet.catalog.service import CatalogService

        service = CatalogService(args.db, HttpScryfall(), AzureEmbedder.from_env, args.dump_dir)
        outcome = asyncio.run(service.freshen(force=True))
        print(json.dumps(asdict(outcome)))
        return 0 if outcome.ok else 1
    print(json.dumps(asdict(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
