"""Frozen PCA basis for card embeddings.

The basis is the top-K eigenvectors of the card pool's covariance, fit once and
stored (see ``CatalogStore.fit_pca``); clients project onto a prefix ``d <= K``
and whiten by the per-component std. This module holds the pure computation
(fit, per-card unexplained variance) and the client-side ``PCABasis``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_COMPONENTS = 256


@dataclass
class PCAFit:
    mean: np.ndarray  # (dims,) float64
    components: np.ndarray  # (K, dims) float64, orthonormal rows, by decreasing variance
    eigenvalues: np.ndarray  # (K,) float64, variance along each component
    total_variance: float  # trace of the covariance (all dims)
    baseline_fuv: np.ndarray  # (K,) mean per-card fraction of unexplained variance at d = 1..K

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.eigenvalues)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.eigenvalues / self.total_variance


def fit(x: np.ndarray, k: int = DEFAULT_COMPONENTS) -> PCAFit:
    """PCA of the rows of ``x`` by eigendecomposition of the float64 covariance
    (population normalisation, 1/N, so whitened scores have unit variance on
    the pool). K is capped at min(dims, N - 1). Signs are fixed so the
    largest-|loading| entry of each component is positive, which makes the
    result deterministic and every prefix equal to a direct fit of that size."""
    x = np.asarray(x, dtype=np.float64)
    n, dims = x.shape
    k = min(k, dims, n - 1)
    if k < 1:
        raise ValueError(f"need at least 2 rows to fit a PCA, got {n}")
    mean = x.mean(axis=0)
    xc = x - mean
    cov = (xc.T @ xc) / n
    cov = (cov + cov.T) / 2
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:k]
    comps = evecs[:, order].T.copy()
    idx = np.argmax(np.abs(comps), axis=1)
    comps *= np.sign(comps[np.arange(k), idx])[:, None]
    evals = np.clip(evals[order], 0.0, None)
    total = float(np.trace(cov))
    return PCAFit(mean, comps, evals, total, fuv_curve(xc, comps).mean(axis=0))


def fuv_curve(xc: np.ndarray, components: np.ndarray) -> np.ndarray:
    """(N, K): fraction of each centred row's squared norm left unexplained by
    the first d components, for d = 1..K."""
    sq = np.einsum("nd,nd->n", xc, xc)
    explained = np.cumsum((xc @ components.T) ** 2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 1.0 - explained / sq[:, None]
    return np.nan_to_num(out, nan=0.0)


def _is_torch(x) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


@dataclass(eq=False)
class PCABasis:
    """A prefix of a stored basis: ``dims`` components of the ``embed_dims``-d
    embedding space. ``project`` maps (..., embed_dims) -> (..., dims) and
    ``inverse`` maps back (least-squares reconstruction). Torch inputs are
    handled on their own device/dtype; nothing is copied to the CPU."""

    basis_id: int
    dims: int
    mean: np.ndarray  # (embed_dims,) float32
    components: np.ndarray  # (dims, embed_dims) float32
    std: np.ndarray  # (dims,) float32
    explained_variance_ratio: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (dims,)
    meta: dict = field(default_factory=dict)  # model, catalog_version, n_cards, ...
    _torch: dict = field(default_factory=dict, repr=False)

    @property
    def embed_dims(self) -> int:
        return self.mean.shape[0]

    def _arrays(self, x):
        if _is_torch(x):
            key = (x.device, x.dtype)
            if key not in self._torch:
                import torch

                self._torch[key] = tuple(torch.as_tensor(np.array(a), device=x.device, dtype=x.dtype)
                                         for a in (self.mean, self.components, self.std))
            return self._torch[key]
        dt = np.result_type(x.dtype, np.float32)
        return self.mean.astype(dt, copy=False), self.components.astype(dt, copy=False), self.std.astype(dt, copy=False)

    def project(self, x, whiten: bool = True):
        if not _is_torch(x):
            x = np.asarray(x)
            if not np.issubdtype(x.dtype, np.floating):
                x = x.astype(np.float32)
        if x.shape[-1] != self.embed_dims:
            raise ValueError(f"expected (..., {self.embed_dims}) input, got {tuple(x.shape)}")
        mean, comps, std = self._arrays(x)
        z = (x - mean) @ comps.T
        return z / std if whiten else z

    def inverse(self, z, whitened: bool = True):
        if not _is_torch(z):
            z = np.asarray(z)
            if not np.issubdtype(z.dtype, np.floating):
                z = z.astype(np.float32)
        if z.shape[-1] != self.dims:
            raise ValueError(f"expected (..., {self.dims}) input, got {tuple(z.shape)}")
        mean, comps, std = self._arrays(z)
        return (z * std if whitened else z) @ comps + mean
