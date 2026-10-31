#!/usr/bin/env python3
"""Build the name -> oracle_id table the Forge bridge loads at startup.

    uv run scripts/build_oracle_map.py

Rerun after refreshing the Scryfall bulk file. The bridge reads the output via
``-Dmagicnet.oracle=PATH`` (default ``../data/processed/oracle_ids.tsv``,
relative to ``forge-bridge/``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from magicnet.cards import oracle_name_index

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bulk", type=Path, default=REPO / "data/raw/scryfall-default-cards.jsonl.gz")
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/oracle_ids.tsv")
    args = ap.parse_args()

    index, ambiguous = oracle_name_index(args.bulk)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for name in sorted(index):
            f.write(f"{name}\t{index[name]}\n")
    print(f"wrote {len(index)} names to {args.out}")
    print(f"skipped {len(ambiguous)} ambiguous names: {sorted(ambiguous)[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
