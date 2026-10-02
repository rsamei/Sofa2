"""Urine output over trailing windows, and body weight over time."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sofa2.derive.timegrid import HOUR, build_grid


def weight_at(times: pd.DataFrame, stays: pd.DataFrame, measurements: pd.DataFrame, pcfg: dict):
    """Body weight (kg) at each (stay_id, time) row.

    The last weight measured at or before the time; before the first measurement, the first
    measurement of the stay; without any measurement, ``stays.weight_kg``. Weights outside
    ``weight.min_kg``/``weight.max_kg`` (when set) are ignored.
    """
    w = measurements[measurements["variable"] == "weight"][["stay_id", "time", "value"]]
    lo, hi = pcfg["weight"].get("min_kg"), pcfg["weight"].get("max_kg")
    if lo is not None:
        w = w[w["value"] >= lo]
    if hi is not None:
        w = w[w["value"] <= hi]
    left = times[["stay_id", "time"]].reset_index(drop=True).reset_index(names="_row")
    out = pd.Series(np.nan, index=left["_row"])
    if not w.empty:
        m = pd.merge_asof(
            left.sort_values("time"), w.sort_values("time"), on="time", by="stay_id",
            direction="backward",
        ).set_index("_row")["value"]
        first = w.sort_values("time").groupby("stay_id")["value"].first()
        out = m.reindex(left["_row"]).fillna(left.set_index("_row")["stay_id"].map(first))
    fallback = left.set_index("_row")["stay_id"].map(stays.set_index("stay_id")["weight_kg"])
    if lo is not None:
        fallback = fallback.where(fallback >= lo)
    if hi is not None:
        fallback = fallback.where(fallback <= hi)
    return out.fillna(fallback).to_numpy()


def urine_windows(
    urine: pd.DataFrame,
    stays: pd.DataFrame,
    measurements: pd.DataFrame,
    pcfg: dict,
    windows=(6, 12, 24),
) -> pd.DataFrame:
    """Trailing-window urine output at the end of every grid hour.

    Each chart's collection interval is the time since the previous chart of the stay (for the
    first chart, since ICU admission). For a window of W hours ending at T, the charts with time
    in (T - W, T] give the volume and the summed collection intervals (coverage). With
    ``urine_output.require_full_coverage`` the window is evaluated only when the coverage is at
    least W hours.

    Returns:
        DataFrame indexed by (stay_id, hr) with, per window W: ``vol_W`` (mL), ``cov_W`` (h),
        ``n_W`` (charts), ``rate_W`` (mL/kg/h, NaN if not evaluable) and ``mlday_W``
        (mL per 24 h, NaN if not evaluable); and ``weight`` (kg).
    """
    full = pcfg["urine_output"]["require_full_coverage"]
    grid = build_grid(stays)
    grid["time"] = grid["end"]
    grid["weight"] = weight_at(grid, stays, measurements, pcfg)
    intime = stays.set_index("stay_id")["intime"]
    outtime = stays.set_index("stay_id")["outtime"]
    u = urine[["stay_id", "time", "volume_ml"]].copy()
    u = u[(u["time"] > u["stay_id"].map(intime)) & (u["time"] <= u["stay_id"].map(outtime))]
    u = u.sort_values(["stay_id", "time"])
    prev = u.groupby("stay_id")["time"].shift().fillna(u["stay_id"].map(intime))
    u["interval_h"] = (u["time"] - prev) / HOUR

    by_stay = {k: v for k, v in u.groupby("stay_id", sort=False)}
    empty = u.iloc[0:0]
    res = []
    for stay_id, g in grid.groupby("stay_id", sort=False):
        uu = by_stay.get(stay_id, empty)
        t = uu["time"].to_numpy()
        cv = np.concatenate([[0.0], np.cumsum(uu["volume_ml"].to_numpy())])
        ci = np.concatenate([[0.0], np.cumsum(uu["interval_h"].to_numpy())])
        ends = g["end"].to_numpy()
        cols = {"stay_id": g["stay_id"].to_numpy(), "hr": g["hr"].to_numpy(), "weight": g["weight"].to_numpy()}
        for w in windows:
            hi = np.searchsorted(t, ends, side="right")
            lo = np.searchsorted(t, ends - np.timedelta64(int(w * 3600), "s"), side="right")
            vol = cv[hi] - cv[lo]
            cov = ci[hi] - ci[lo]
            n = hi - lo
            ok = (n > 0) & ((cov >= w - 1e-9) if full else (cov > 0))
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = np.where(ok, vol / cov / cols["weight"], np.nan)
                mlday = np.where(ok, vol / cov * 24.0, np.nan)
            cols.update({f"vol_{w}": vol, f"cov_{w}": cov, f"n_{w}": n, f"rate_{w}": rate, f"mlday_{w}": mlday})
        res.append(pd.DataFrame(cols))
    if not res:
        return pd.DataFrame()
    return pd.concat(res, ignore_index=True).set_index(["stay_id", "hr"])


def anuria(uw: pd.DataFrame, window_hours: int, min_charts: int, full: bool = True) -> pd.Series:
    """True where the trailing window had zero volume with at least ``min_charts`` charts."""
    w = window_hours
    covered = uw[f"cov_{w}"] >= w - 1e-9 if full else uw[f"cov_{w}"] > 0
    return (uw[f"n_{w}"] >= min_charts) & (uw[f"vol_{w}"] == 0) & covered
