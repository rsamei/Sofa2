"""Evaluate threshold bands (see :mod:`sofa2.config`)."""

from __future__ import annotations

import operator

import numpy as np

from sofa2.config import Band, parse_bands

_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}


def as_bands(bands) -> list[Band]:
    """Accept parsed :class:`Band` objects or the raw YAML list."""
    if bands and isinstance(bands[0], Band):
        return list(bands)
    return parse_bands(bands)


def score(values, bands, support=None, max_without_support: int | None = None) -> np.ndarray:
    """Score values against bands.

    Each value scores the highest ``points`` whose condition holds, else 0. Missing values
    (NaN) give NaN. A band with ``requires_support`` only applies where ``support`` is true;
    where it is false the score is capped at ``max_without_support``.

    Args:
        values: scalar or array of values in canonical units.
        bands: band list (parsed or raw YAML).
        support: boolean scalar or array, same shape as ``values``.
        max_without_support: cap applied when a support-requiring band is not met because
            support is absent. Defaults to the highest points of the bands not requiring support.

    Returns:
        Float array of points (NaN where the value is missing).
    """
    bands = as_bands(bands)
    v = np.atleast_1d(np.asarray(values, dtype=float))
    has_support = (
        np.ones(v.shape, dtype=bool)
        if support is None
        else np.broadcast_to(np.asarray(support, dtype=bool), v.shape)
    )
    if max_without_support is None:
        free = [b.points for b in bands if not b.requires_support]
        max_without_support = max(free) if free else 0
    out = np.zeros(v.shape, dtype=float)
    with np.errstate(invalid="ignore"):
        for b in bands:
            hit = _OPS[b.op](v, b.value)
            if b.requires_support:
                out = np.where(hit & has_support, np.maximum(out, b.points), out)
                out = np.where(
                    hit & ~has_support, np.maximum(out, min(b.points, max_without_support)), out
                )
            else:
                out = np.where(hit, np.maximum(out, b.points), out)
    out[np.isnan(v)] = np.nan
    return out
