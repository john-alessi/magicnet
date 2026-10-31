"""Request embeddings for card text from an Azure OpenAI embeddings deployment.

Design notes:
  * Requests are batched (many card texts per API call) and retried with
    exponential backoff on transient errors.
  * Progress is checkpointed as small shard files (``shard_000123.npz``) so an
    interrupted run can simply be re-launched and will pick up where it left
    off, without re-paying for cards already embedded.
  * ``consolidate_shards`` merges all shards into one embedding matrix + an
    aligned list of card ids, which is what downstream training code should
    actually load.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import AzureOpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential


@dataclass
class EmbeddingConfig:
    endpoint: str
    api_key: str
    api_version: str
    deployment: str
    dimensions: int | None = None  # native Matryoshka truncation, if supported

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        missing = [
            name
            for name in (
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_API_VERSION",
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            )
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError(
                "Missing required environment variable(s): "
                f"{', '.join(missing)}. Set them directly or via a .env file "
                "(see .env.example)."
            )
        dimensions_raw = os.environ.get("AZURE_OPENAI_EMBEDDING_DIMENSIONS")
        return cls(
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            dimensions=int(dimensions_raw) if dimensions_raw else None,
        )

    def build_client(self) -> AzureOpenAI:
        return AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )


def _is_retryable(exc: BaseException) -> bool:
    # Anything except a clear client-side/programming error is worth retrying:
    # rate limits, transient 5xx, connection resets, timeouts, etc. A 4xx other
    # than 429 (e.g. bad request, auth failure) won't be fixed by retrying.
    from openai import APIStatusError

    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return True


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(6),
    wait=wait_random_exponential(multiplier=1, max=60),
    reraise=True,
)
def _embed_batch_with_retry(
    client: AzureOpenAI, config: EmbeddingConfig, texts: Sequence[str]
) -> list[list[float]]:
    kwargs: dict = {"model": config.deployment, "input": list(texts)}
    if config.dimensions is not None:
        kwargs["dimensions"] = config.dimensions
    response = client.embeddings.create(**kwargs)
    return [item.embedding for item in response.data]


def embed_texts(
    config: EmbeddingConfig,
    texts: Sequence[str],
    *,
    batch_size: int = 100,
    client: AzureOpenAI | None = None,
) -> np.ndarray:
    """Embed ``texts`` in batches (with retries); returns a float32 (n, dim) matrix."""
    client = client or config.build_client()
    rows: list[list[float]] = []
    for batch in batched(texts, batch_size):
        rows.extend(_embed_batch_with_retry(client, config, batch))
    return np.asarray(rows, dtype=np.float32)


def batched(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_card_records(cards_path: Path) -> list[dict]:
    records = []
    with cards_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def already_embedded_ids(shard_dir: Path) -> set[str]:
    ids: set[str] = set()
    if not shard_dir.exists():
        return ids
    for shard_path in shard_dir.glob("shard_??????.npz"):
        with np.load(shard_path, allow_pickle=False) as data:
            ids.update(data["ids"].tolist())
    return ids


def write_shard(shard_dir: Path, shard_index: int, ids: Sequence[str], embeddings: np.ndarray) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    final_path = shard_dir / f"shard_{shard_index:06d}.npz"
    # np.savez_compressed appends ".npz" if the given name doesn't already end
    # with it, so the temp name must itself end in ".npz" to avoid a mismatch.
    tmp_path = shard_dir / f"shard_{shard_index:06d}.tmp.npz"
    np.savez_compressed(
        tmp_path,
        ids=np.array(ids),  # numpy infers a fixed-width unicode dtype (no pickle needed)
        embeddings=embeddings.astype(np.float32),
    )
    tmp_path.rename(final_path)  # atomic on the same filesystem


def embed_cards(
    cards: list[dict],
    config: EmbeddingConfig,
    shard_dir: Path,
    *,
    batch_size: int = 100,
    progress=None,
) -> None:
    client = config.build_client()
    done_ids = already_embedded_ids(shard_dir)
    remaining = [c for c in cards if c["oracle_id"] not in done_ids]

    existing_shards = list(shard_dir.glob("shard_??????.npz"))
    next_shard_index = len(existing_shards)

    if progress is not None:
        progress.total = len(cards)
        progress.update(len(done_ids))

    for batch in batched(remaining, batch_size):
        ids = [c["oracle_id"] for c in batch]
        texts = [c["text"] for c in batch]
        embeddings = _embed_batch_with_retry(client, config, texts)
        write_shard(shard_dir, next_shard_index, ids, np.array(embeddings))
        next_shard_index += 1
        if progress is not None:
            progress.update(len(batch))


def consolidate_shards(shard_dir: Path, out_prefix: Path) -> tuple[Path, Path]:
    """Merge all shards into one (N, dim) embedding matrix + an aligned id list.

    Returns the (embeddings_path, ids_path) that were written.
    """
    all_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []
    for shard_path in sorted(shard_dir.glob("shard_??????.npz")):
        with np.load(shard_path, allow_pickle=False) as data:
            all_ids.extend(data["ids"].tolist())
            all_embeddings.append(data["embeddings"])

    if not all_embeddings:
        raise RuntimeError(f"No shards found in {shard_dir}")

    matrix = np.concatenate(all_embeddings, axis=0)

    # De-duplicate defensively (e.g. if a run was interrupted mid-shard-write
    # and resumed, or the same card slipped into two batches).
    seen: dict[str, int] = {}
    for i, oracle_id in enumerate(all_ids):
        seen[oracle_id] = i  # last occurrence wins, they should be identical
    unique_ids = list(seen.keys())
    unique_indices = [seen[oid] for oid in unique_ids]
    matrix = matrix[unique_indices]

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    embeddings_path = out_prefix.with_suffix(".npy")
    ids_path = out_prefix.with_name(out_prefix.name + ".ids.json")
    np.save(embeddings_path, matrix)
    ids_path.write_text(json.dumps(unique_ids), encoding="utf-8")
    return embeddings_path, ids_path
