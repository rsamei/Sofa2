"""Hourly time grid relative to ICU admission.

Hour ``hr`` of a stay covers ``[intime + hr h, intime + (hr + 1) h)``. A stay of length L hours
has ``ceil(L)`` hours; an event at exactly ``outtime`` belongs to the last hour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)


def build_grid(stays: pd.DataFrame) -> pd.DataFrame:
    """One row per stay-hour: stay_id, hr, start, end."""
    n = np.ceil((stays["outtime"] - stays["intime"]) / HOUR).astype(int).clip(lower=1)
    grid = stays[["stay_id", "intime"]].loc[stays.index.repeat(n)].reset_index(drop=True)
    grid["hr"] = np.concatenate([np.arange(k) for k in n]) if len(n) else np.array([], int)
    grid["start"] = grid["intime"] + grid["hr"] * HOUR
    grid["end"] = grid["start"] + HOUR
    return grid[["stay_id", "hr", "start", "end"]]


def n_hours(stays: pd.DataFrame) -> pd.Series:
    """Number of grid hours per stay, indexed by stay_id."""
    n = np.ceil((stays["outtime"] - stays["intime"]) / HOUR).astype(int).clip(lower=1)
    return pd.Series(n.to_numpy(), index=stays["stay_id"].to_numpy())


def assign_hour(
    events: pd.DataFrame,
    stays: pd.DataFrame,
    time_col: str = "time",
    pre_hours: float = 0.0,
) -> pd.DataFrame:
    """Add an ``hr`` column and drop events outside the stay.

    Events from ``pre_hours`` before admission up to admission are put in hour 0.
    """
    if events.empty:
        return events.assign(hr=pd.Series(dtype=int))
    s = stays.set_index("stay_id")
    intime = events["stay_id"].map(s["intime"])
    outtime = events["stay_id"].map(s["outtime"])
    t = events[time_col]
    keep = (t >= intime - pd.Timedelta(hours=pre_hours)) & (t <= outtime)
    ev = events.loc[keep].copy()
    nh = events.loc[keep, "stay_id"].map(n_hours(stays))
    hr = np.floor((t[keep] - intime[keep]) / HOUR).astype(int)
    ev["hr"] = np.clip(hr, 0, nh - 1).astype(int)
    return ev


def overlap_hours(intervals: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    """Explode intervals (``start``, ``end``) into the grid hours they overlap.

    An interval overlaps hour ``hr`` when ``start < hour_end`` and ``end > hour_start``. A
    zero-length interval counts in the hour that contains it. Returns the interval columns plus
    ``hr``; parts outside the stay are dropped.
    """
    if intervals.empty:
        return intervals.assign(hr=pd.Series(dtype=int))
    s = stays.set_index("stay_id")
    intime = intervals["stay_id"].map(s["intime"])
    nh = intervals["stay_id"].map(n_hours(stays))
    first = np.floor((intervals["start"] - intime) / HOUR)
    last_excl = np.ceil((intervals["end"] - intime) / HOUR)
    last_excl = np.maximum(last_excl, first + 1)
    first = np.clip(first, 0, None)
    last_excl = np.minimum(last_excl, nh)
    count = (last_excl - first).clip(lower=0).astype(int)
    out = intervals.loc[intervals.index.repeat(count)].copy()
    offsets = np.concatenate([np.arange(c) for c in count]) if len(count) else np.array([], int)
    out["hr"] = (np.repeat(first.to_numpy(), count) + offsets).astype(int)
    return out.reset_index(drop=True)
