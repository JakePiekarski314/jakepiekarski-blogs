"""NumPy media transforms shared by data generation and render-time reporting.

These mirror the pymc-marketing ``GeometricAdstock`` and ``TanhSaturation``
primitives used inside the fitted model, so contributions and response curves
can be recomputed from committed posterior draws without importing PyMC.
"""

from __future__ import annotations

import numpy as np


def geometric_adstock_1d(x: np.ndarray, alpha: float, l_max: int = 12) -> np.ndarray:
    """Geometric adstock for a single channel series (matches pymc-marketing)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    weights = alpha ** np.arange(l_max + 1)
    out = np.zeros(n)
    for t in range(n):
        lags = x[max(0, t - l_max) : t + 1][::-1]
        out[t] = np.sum(weights[: len(lags)] * lags)
    return out


def tanh_saturation_1d(x: np.ndarray, b: float, c: float) -> np.ndarray:
    """Tanh saturation ``b * tanh(x / (b * c))`` (matches pymc-marketing)."""
    return b * np.tanh(x / (b * c))


def geometric_adstock_draws(
    media: np.ndarray, alpha: np.ndarray, l_max: int = 12
) -> np.ndarray:
    """Vectorised geometric adstock over posterior draws.

    Parameters
    ----------
    media : (T, C) array
        Max-scaled spend per date and channel (shared across draws).
    alpha : (D, C) array
        Per-draw, per-channel decay rate.
    l_max : int
        Maximum carryover lag.

    Returns
    -------
    (D, T, C) array of adstocked media.
    """
    media = np.asarray(media, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    T, C = media.shape
    D = alpha.shape[0]
    out = np.zeros((D, T, C))
    for lag in range(l_max + 1):
        # shift media forward by `lag` weeks (earlier weeks contribute with decay)
        shifted = np.zeros_like(media)
        if lag == 0:
            shifted = media
        else:
            shifted[lag:] = media[:-lag]
        weight = alpha ** lag  # (D, C)
        out += weight[:, None, :] * shifted[None, :, :]
    return out


def tanh_saturation_draws(
    adstocked: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Vectorised tanh saturation over posterior draws.

    Parameters
    ----------
    adstocked : (D, T, C) array
    b, c : (D, C) arrays
    """
    b = np.asarray(b, dtype=float)[:, None, :]
    c = np.asarray(c, dtype=float)[:, None, :]
    return b * np.tanh(adstocked / (b * c))


def transformed_media_draws(
    media_scaled: np.ndarray,
    alpha: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    l_max: int = 12,
) -> np.ndarray:
    """Full adstock -> saturation transform, returning (D, T, C)."""
    adstocked = geometric_adstock_draws(media_scaled, alpha, l_max=l_max)
    return tanh_saturation_draws(adstocked, b, c)
