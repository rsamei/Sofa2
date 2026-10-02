"""Urine output over trailing windows, and body weight over time."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sofa2.derive.timegrid import HOUR, build_grid, n_hours


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
    """Trailing-window urine output per grid hour.

    Each chart's collection interval is the time since the previous chart of the stay (for the
    first chart, since ICU admission). A window of W hours ending at time T uses the charts with
    time in (T - W, T]: their volume and their summed collection intervals (coverage). With
    ``urine_output.require_full_coverage`` the window is evaluated only when the coverage is at
    least W hours. Windows are evaluated at every chart time T (a window ending between charts
    would include uncovered time after the last chart); the evaluation belongs to the hour
    (start, end] that contains T.

    Returns:
        DataFrame indexed by (stay_id, hr) with ``weight`` (kg, at the hour end) and, per window
        W: ``rate_W`` (lowest evaluable mean rate in the hour, mL/kg/h), ``mlday_W`` (mL per 24 h
        of the latest evaluable window in the hour) and ``anuria_W`` (an evaluable window with
        0 mL and at least ``urine_output.anuria_min_charts`` charts). NaN/False where nothing
        could be evaluated.
    """
    full = pcfg["urine_output"]["require_full_coverage"]
    min_charts = pcfg["urine_output"]["anuria_min_charts"]
    grid = build_grid(stays)
    grid["time"] = grid["end"]
    grid["weight"] = weight_at(grid, stays, measurements, pcfg)
    intime = stays.set_index("stay_id")["intime"]
    outtime = stays.set_index("stay_id")["outtime"]
    u = urine[["stay_id", "time", "volume_ml"]].copy()
    u = u[(u["time"] > u["stay_id"].map(intime)) & (u["time"] <= u["stay_id"].map(outtime))]
    u = u.sort_values(["stay_id", "time"], kind="stable")
    prev = u.groupby("stay_id")["time"].shift().fillna(u["stay_id"].map(intime))
    u["interval_h"] = (u["time"] - prev) / HOUR

    # evaluation times: chart times, with the weight at each
    ev = u[["stay_id", "time"]].drop_duplicates()
    ev = ev.sort_values(["stay_id", "time"], kind="stable").reset_index(drop=True)
    ev["weight"] = weight_at(ev, stays, measurements, pcfg)
    nh = ev["stay_id"].map(n_hours(stays))
    hr = np.ceil((ev["time"] - ev["stay_id"].map(intime)) / HOUR).astype(int) - 1
    ev["hr"] = np.clip(hr, 0, nh - 1)

    by_stay = {k: v for k, v in u.groupby("stay_id", sort=False)}
    empty_u = u.iloc[0:0]
    cols = {f"{k}_{w}": np.full(len(ev), np.nan) for w in windows for k in ("rate", "mlday")}
    cols.update({f"anuria_{w}": np.zeros(len(ev), dtype=bool) for w in windows})
    for stay_id, g in ev.groupby("stay_id", sort=False):
        uu = by_stay.get(stay_id, empty_u)
        if uu.empty:
            continue
        t = uu["time"].to_numpy()
        cv = np.concatenate([[0.0], np.cumsum(uu["volume_ml"].to_numpy())])
        ci = np.concatenate([[0.0], np.cumsum(uu["interval_h"].to_numpy())])
        ends = g["time"].to_numpy()
        pos = g.index.to_numpy()
        wt = g["weight"].to_numpy()
        for w in windows:
            hi = np.searchsorted(t, ends, side="right")
            lo = np.searchsorted(t, ends - np.timedelta64(int(w * 3600), "s"), side="right")
            vol, cov, n = cv[hi] - cv[lo], ci[hi] - ci[lo], hi - lo
            ok = (n > 0) & ((cov >= w - 1e-9) if full else (cov > 0))
            with np.errstate(divide="ignore", invalid="ignore"):
                # rounded so that float error cannot cross a band edge (eg 499.9999... mL/day)
                cols[f"rate_{w}"][pos] = np.round(np.where(ok & (wt > 0), vol / cov / wt, np.nan), 9)
                cols[f"mlday_{w}"][pos] = np.round(np.where(ok, vol / cov * 24.0, np.nan), 6)
            cols[f"anuria_{w}"][pos] = ok & (vol == 0) & (n >= min_charts)
    ev = ev.assign(**cols)
    grp = ev.groupby(["stay_id", "hr"])
    agg = {f"rate_{w}": "min" for w in windows}
    agg.update({f"anuria_{w}": "max" for w in windows})
    out = grp.agg(agg)
    for w in windows:
        out[f"mlday_{w}"] = grp[f"mlday_{w}"].last()  # last non-missing value in time order
    idx = pd.MultiIndex.from_frame(grid[["stay_id", "hr"]])
    out = out.reindex(idx)
    for w in windows:
        out[f"anuria_{w}"] = out[f"anuria_{w}"].fillna(False).astype(bool)
    out["weight"] = grid["weight"].to_numpy()
    return out


