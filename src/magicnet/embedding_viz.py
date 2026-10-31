"""Helpers for exploring the card-embedding space: PCA dimensionality
reduction, a 2D UMAP projection for visualization, and assembly of a
dataframe (embedding coords + card metadata) suitable for an interactive
scatterplot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def load_embeddings(
    embeddings_path: Path, ids_path: Path
) -> tuple[np.ndarray, list[str]]:
    matrix = np.load(embeddings_path)
    ids = json.loads(ids_path.read_text())
    assert matrix.shape[0] == len(ids), "embedding rows and id list are misaligned"
    return matrix, ids


def load_card_metadata(cards_path: Path) -> pd.DataFrame:
    records = []
    with cards_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame.from_records(records)
    return df.set_index("oracle_id")


def fit_pca(embeddings: np.ndarray, n_components: int) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=0)
    reduced = pca.fit_transform(embeddings)
    return reduced, pca


def cumulative_explained_variance(pca: PCA) -> np.ndarray:
    return np.cumsum(pca.explained_variance_ratio_)


def n_components_for_variance(pca_full: PCA, target: float) -> int:
    """How many components are needed to explain >= `target` fraction of variance."""
    cum = cumulative_explained_variance(pca_full)
    idx = int(np.searchsorted(cum, target))
    return min(idx + 1, len(cum))


def fit_umap_2d(
    reduced_embeddings: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> np.ndarray:
    # Imported lazily: umap-learn has a heavier import (numba JIT) than the
    # rest of this module, only pay for it when actually projecting.
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        metric="cosine",
    )
    return reducer.fit_transform(reduced_embeddings)


def truncate_text(text: str, max_chars: int = 300) -> str:
    text = text.replace("\n//\n", " // ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\u2026"


def build_plot_frame(
    coords_2d: np.ndarray,
    ids: list[str],
    card_meta: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.DataFrame({"oracle_id": ids, "x": coords_2d[:, 0], "y": coords_2d[:, 1]})
    df = df.join(card_meta, on="oracle_id")
    df["hover_text"] = df["text"].fillna("").map(truncate_text)
    # A simple, visually useful color grouping: mono-color identity, else
    # "Multicolor"/"Colorless" -- good enough for a sanity-check plot.
    def color_group(colors: object) -> str:
        if not isinstance(colors, list) or len(colors) == 0:
            return "Colorless"
        if len(colors) > 1:
            return "Multicolor"
        return {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}.get(
            colors[0], "Colorless"
        )

    df["color_group"] = df["colors"].map(color_group)
    return df
