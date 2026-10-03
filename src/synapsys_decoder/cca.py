"""Canonical correlation analysis (CCA) between an EEG window and a reference set.

Standard CCA as used here is **not trained**: it has no parameters learned from
data. For each window it simply measures how well *some* linear combination of
the EEG channels can match *some* linear combination of the reference
sinusoids, and reports that correlation.

Algorithm (Björck & Golub, 1973 — "angles between subspaces")
-----------------------------------------------------------
Given ``x`` of shape ``(p, n)`` and ``y`` of shape ``(q, n)`` (variables by
samples):

1. Validate: 2-D, same ``n``, all finite, ``n > p + q`` (see below).
2. Center every row (subtract its mean over samples).
3. Build an orthonormal basis of each centered column space in ``R^n``:
   thin SVD ``Xc^T = U S V^T`` and keep the columns of ``U`` whose singular
   value ``s_i > rank_tol * s_max``. This is rank revealing: a dead (all-zero
   after centering) or duplicated channel simply contributes no direction,
   instead of making a covariance matrix singular and producing NaN.
   ``Qx`` is ``n x r_x``; likewise ``Qy`` is ``n x r_y``.
4. The canonical correlations are the singular values of the small
   ``r_x x r_y`` matrix ``Qx^T Qy`` (cosines of the principal angles between
   the two subspaces), sorted descending and clipped to ``[0, 1]``.

This never forms or inverts ``Cxx``/``Cyy``, so its accuracy depends on the
conditioning of the data, not its square. The result is invariant to any
invertible linear transform of the rows of ``x`` (channel permutation,
per-channel scaling, re-referencing by an invertible matrix) because such a
transform does not change the column space.

Portability notes (ESP32-S3 C port)
-----------------------------------
* Sizes are tiny (``p <= 8`` channels, ``q = 2H <= 8``), so step 3 can use a
  one-sided Jacobi SVD or a column-pivoted Householder / modified Gram–Schmidt
  QR on the ``n x p`` matrix; any rank-revealing method spanning the same
  subspace gives the same correlations. The ``Qy`` basis for each target
  depends only on the reference table and can be precomputed offline.
* Step 4 needs singular values of an at most ``8 x 8`` matrix (Jacobi SVD, or
  ``sqrt`` of the eigenvalues of the symmetric ``M^T M`` via Jacobi
  eigen-iteration).
* ``rank_tol`` default ``1e-10`` is tuned for float64. For a float32 port use
  roughly ``1e-5`` (a few hundred float32 ulps); tolerances must track the
  arithmetic precision.

Degenerate input is explicit: :class:`DegenerateSignalError` is raised instead
of ever returning NaN.
"""

from __future__ import annotations

import numpy as np

#: Default relative singular-value tolerance for discarding rank-deficient
#: directions (float64). See module docstring for the float32 equivalent.
DEFAULT_RANK_TOL = 1e-10


class DegenerateSignalError(ValueError):
    """Raised when CCA is undefined for the given input.

    Cases: non-finite values, too few samples (``n <= p + q``), or a side with
    zero rank after centering (e.g. every EEG channel constant).
    """


def _orthonormal_basis(
    centered: np.ndarray, raw_scale: float, rank_tol: float, name: str
) -> np.ndarray:
    """Return an ``n x r`` orthonormal basis of the row space of ``centered`` (``(k, n)``).

    Directions with singular value ``<= rank_tol * s_max`` are dropped. The side
    is declared zero-rank when ``s_max <= rank_tol * raw_scale`` where
    ``raw_scale = sqrt(n) * max|raw|`` bounds the singular values of the
    uncentered data; this catches constant rows whose centering leaves only
    floating-point residue (``~1e-16`` relative), which a purely relative test
    would mistake for signal.
    """
    u, s, _ = np.linalg.svd(centered.T, full_matrices=False)
    s_max = float(s[0]) if s.size else 0.0
    if s_max == 0.0 or s_max <= rank_tol * raw_scale:
        raise DegenerateSignalError(f"{name} has zero rank after centering (constant rows?)")
    keep = s > rank_tol * s_max
    return u[:, keep]


def canonical_correlations(
    x: np.ndarray, y: np.ndarray, *, rank_tol: float = DEFAULT_RANK_TOL
) -> np.ndarray:
    """All canonical correlations between ``x`` and ``y``.

    Args:
        x: ``(p, n)`` array — variables by samples (e.g. EEG channels).
        y: ``(q, n)`` array — variables by samples (e.g. sin/cos references).
        rank_tol: Relative singular-value tolerance below which a direction is
            treated as linearly dependent and dropped (must be in ``(0, 1)``).

    Returns:
        1-D float64 array of length ``min(rank(x), rank(y))`` sorted descending,
        every value in ``[0, 1]``.

    Raises:
        ValueError: on wrong dimensionality, mismatched sample counts, or an
            invalid ``rank_tol``.
        DegenerateSignalError: on NaN/Inf, ``n <= p + q``, or a zero-rank side.

    Minimum samples rule ``n > p + q``: after centering, the data live in an
    ``(n-1)``-dimensional subspace of ``R^n``. If ``p + q > n - 1`` and both sides
    have full rank, the two column spaces must intersect, forcing a canonical
    correlation of exactly 1 regardless of the signal. Requiring ``n >= p + q + 1``
    rules out that guaranteed spurious result. (Real windows have
    ``n`` in the hundreds, far above this bound.)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"x and y must be 2-D (variables, samples); got {x.shape}, {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"sample counts differ: x has {x.shape[1]}, y has {y.shape[1]}")
    if not (0.0 < rank_tol < 1.0):
        raise ValueError(f"rank_tol must be in (0, 1); got {rank_tol}")
    p, n = x.shape
    q = y.shape[0]
    if p < 1 or q < 1:
        raise DegenerateSignalError("x and y must each have at least one variable")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise DegenerateSignalError("input contains NaN or Inf")
    if n <= p + q:
        raise DegenerateSignalError(
            f"need n_samples > p + q = {p + q} for meaningful CCA; got n_samples={n}"
        )

    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    sqrt_n = np.sqrt(n)
    qx = _orthonormal_basis(xc, sqrt_n * float(np.max(np.abs(x))), rank_tol, "x")
    qy = _orthonormal_basis(yc, sqrt_n * float(np.max(np.abs(y))), rank_tol, "y")

    rho = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return np.clip(np.sort(rho)[::-1], 0.0, 1.0)


def cca_score(
    eeg_window: np.ndarray, reference: np.ndarray, *, rank_tol: float = DEFAULT_RANK_TOL
) -> float:
    """Largest canonical correlation between an EEG window and a reference set.

    Args:
        eeg_window: ``(n_channels, n_samples)`` EEG (already preprocessed).
        reference: ``(2*H, n_samples)`` reference matrix, e.g. from
            :func:`synapsys_decoder.references.harmonic_references`.
        rank_tol: See :func:`canonical_correlations`.

    Returns:
        The first canonical correlation, a float in ``[0, 1]``.

    Raises:
        DegenerateSignalError / ValueError: see :func:`canonical_correlations`.
    """
    return float(canonical_correlations(eeg_window, reference, rank_tol=rank_tol)[0])
