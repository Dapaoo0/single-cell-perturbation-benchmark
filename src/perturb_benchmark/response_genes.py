"""Response-gene recovery metrics with deterministic tie handling."""

import numpy as np


def _aligned_vectors(
    observed: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.ndim != 1 or predicted.shape != observed.shape or observed.size == 0:
        raise ValueError("response vectors must be nonempty aligned one-dimensional arrays")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("response vectors must be finite")
    return observed, predicted


def _top_indices(values: np.ndarray, genes: np.ndarray, k: int) -> np.ndarray:
    # np.lexsort uses the last key first: descending absolute response, then gene.
    return np.lexsort((genes.astype(str), -np.abs(values)))[:k]


def topk_recovery(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    k: int,
    genes: np.ndarray | None = None,
) -> float:
    """Return the fraction of observed top-k genes recovered by the prediction."""

    observed, predicted = _aligned_vectors(observed, predicted)
    if not 1 <= k <= observed.size:
        raise ValueError("k must be between 1 and the number of genes")
    if genes is None:
        genes = np.arange(observed.size).astype(str)
    else:
        genes = np.asarray(genes)
        if genes.shape != observed.shape or len(set(genes.astype(str))) != genes.size:
            raise ValueError("genes must be unique and aligned to the response vectors")
    observed_top = set(_top_indices(observed, genes, k).tolist())
    predicted_top = set(_top_indices(predicted, genes, k).tolist())
    return len(observed_top.intersection(predicted_top)) / k


def direction_accuracy(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Measure sign agreement among genes with a nonzero observed response."""

    observed, predicted = _aligned_vectors(observed, predicted)
    responsive = observed != 0
    if not responsive.any():
        return float("nan")
    return float(np.mean(np.sign(observed[responsive]) == np.sign(predicted[responsive])))
