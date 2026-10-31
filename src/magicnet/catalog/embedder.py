"""Embedders the catalog can call. Injectable so tests never hit the real API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from magicnet.embedding import EmbeddingConfig, embed_texts

# Native output sizes, used when AZURE_OPENAI_EMBEDDING_DIMENSIONS is unset.
NATIVE_DIMS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


class Embedder(Protocol):
    model: str  # deployment/model name recorded with every vector
    dims: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 (len(texts), dims) matrix."""
        ...


class AzureEmbedder:
    def __init__(self, config: EmbeddingConfig, *, batch_size: int = 100):
        self.config = config
        self.batch_size = batch_size
        self.model = config.deployment
        dims = config.dimensions or NATIVE_DIMS.get(config.deployment)
        if dims is None:
            raise RuntimeError(
                f"cannot tell the output size of deployment {config.deployment!r}; "
                "set AZURE_OPENAI_EMBEDDING_DIMENSIONS"
            )
        self.dims = dims

    @classmethod
    def from_env(cls) -> "AzureEmbedder":
        from dotenv import load_dotenv

        load_dotenv()
        return cls(EmbeddingConfig.from_env())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        return embed_texts(self.config, texts, batch_size=self.batch_size)
