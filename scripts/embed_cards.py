#!/usr/bin/env python3
"""CLI: request embeddings for every card in cards.jsonl via an Azure OpenAI
embeddings deployment, then consolidate into one matrix.

Requires the following environment variables (put them in a gitignored .env,
see .env.example):
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_API_VERSION
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT   (your deployment name, not the base model name)
    AZURE_OPENAI_EMBEDDING_DIMENSIONS   (optional: request native truncation)

Usage:
    uv run scripts/embed_cards.py
    uv run scripts/embed_cards.py --batch-size 200   # tune for your rate limits

Safe to re-run: already-embedded cards (tracked via shard files) are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from magicnet.embedding import (
    EmbeddingConfig,
    consolidate_shards,
    embed_cards,
    load_card_records,
)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=Path("data/processed/cards.jsonl"))
    parser.add_argument("--shard-dir", type=Path, default=Path("data/processed/embedding_shards"))
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("data/processed/card_embeddings"),
        help="Writes <prefix>.npy and <prefix>.ids.json on consolidation.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--skip-consolidate",
        action="store_true",
        help="Only fetch embeddings into shards; don't merge into a final matrix yet.",
    )
    args = parser.parse_args()

    config = EmbeddingConfig.from_env()
    cards = load_card_records(args.cards)

    with tqdm(desc="Embedding cards", unit="card") as progress:
        embed_cards(
            cards,
            config,
            args.shard_dir,
            batch_size=args.batch_size,
            progress=progress,
        )

    if args.skip_consolidate:
        return

    embeddings_path, ids_path = consolidate_shards(args.shard_dir, args.out_prefix)
    print(f"Wrote embeddings matrix to {embeddings_path}")
    print(f"Wrote aligned card-id list to {ids_path}")


if __name__ == "__main__":
    main()
