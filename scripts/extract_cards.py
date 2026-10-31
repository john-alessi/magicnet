#!/usr/bin/env python3
"""CLI: extract deduplicated card records + embedding text from a Scryfall
bulk-data dump.

Usage:
    uv run scripts/extract_cards.py \
        --bulk-path data/raw/scryfall-default-cards.jsonl.gz \
        --out data/processed/cards.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from magicnet.cards import iter_unique_cards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bulk-path",
        type=Path,
        default=Path("data/raw/scryfall-default-cards.jsonl.gz"),
        help="Path to a Scryfall bulk-data .jsonl.gz file (e.g. 'default_cards').",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/cards.jsonl"),
        help="Output JSONL path: one deduplicated card record per line.",
    )
    parser.add_argument(
        "--include-digital",
        action="store_true",
        help="Include Arena-only digital cards (e.g. rebalanced 'A-' Alchemy cards).",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.out.open("w", encoding="utf-8") as f:
        for record in tqdm(
            iter_unique_cards(args.bulk_path, include_digital=args.include_digital),
            desc="Extracting unique cards",
            unit="card",
        ):
            f.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} unique cards to {args.out}")


if __name__ == "__main__":
    main()
