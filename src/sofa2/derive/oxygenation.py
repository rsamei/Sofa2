"""PaO2:FiO2 and SpO2:FiO2 readings and their hourly respiratory points."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sofa2.derive.timegrid import assign_hour
from sofa2.scoring.components import ratio_points


def _pair_fio2(obs: pd.DataFrame, fio2: pd.DataFrame, minutes: float) -> pd.Series:
    """Last FiO2 of the same stay charted at or up to ``minutes`` before each observation."""
    if obs.empty or fio2.empty:
        return pd.Series(np.nan, index=obs.index)
    left = obs[["stay_id", "time"]].reset_index().sort_values("time")
    right = fio2[["stay_id", "time", "value"]].sort_values("time")
    merged = pd.merge_asof(
        left,
        right,
        on="time",
        by="stay_id",
        direction="backward",
        tolerance=pd.Timedelta(minutes=minutes),
    )
    return merged.set_index("index")["value"].reindex(obs.index)


def ratio_readings(measurements: pd.DataFrame, pipeline_cfg: dict) -> pd.DataFrame:
    """Build oxygenation ratio readings.

    PaO2 is divided by the FiO2 of the same blood gas (``specimen_id``) when present, otherwise by
    the last FiO2 within ``oxygenation.pao2_fio2_lookback_minutes`` before. SpO2 below the
    SOFA-2 limit is kept here with its paired FiO2 (last FiO2 within
    ``oxygenation.spo2_fio2_lookback_minutes`` before); the 98% rule is applied in
    :func:`respiratory_hourly`.

    Returns:
        DataFrame with stay_id, time, kind ('pf' or 'sf'), ratio, spo2.
    """
    ox = pipeline_cfg["oxygenation"]
    m = measurements
    fio2 = m[(m["variable"] == "fio2") & m["value"].between(ox["fio2_min"], ox["fio2_max"])]
    out = []

    pao2 = m[m["variable"] == "pao2"].copy()
    if not pao2.empty:
        f = pd.Series(np.nan, index=pao2.index)
        if "specimen_id" in m.columns:
            same = fio2.dropna(subset=["specimen_id"]).drop_duplicates(["stay_id", "specimen_id"])
            key = same.set_index(["stay_id", "specimen_id"])["value"]
            idx = pd.MultiIndex.from_frame(pao2[["stay_id", "specimen_id"]])
            f = pd.Series(key.reindex(idx).to_numpy(), index=pao2.index)
        lookup = _pair_fio2(pao2, fio2, ox["pao2_fio2_lookback_minutes"])
        f = f.fillna(lookup)
        pao2["ratio"] = pao2["value"] / f
        pao2["kind"] = "pf"
        pao2["spo2"] = np.nan
        out.append(pao2.dropna(subset=["ratio"]))

    spo2 = m[m["variable"] == "spo2"].copy()
    if not spo2.empty:
        f = _pair_fio2(spo2, fio2, ox["spo2_fio2_lookback_minutes"])
        spo2["ratio"] = spo2["value"] / f
        spo2["kind"] = "sf"
        spo2["spo2"] = spo2["value"]
        out.append(spo2.dropna(subset=["ratio"]))

    cols = ["stay_id", "time", "kind", "ratio", "spo2"]
    if not out:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    return pd.concat(out, ignore_index=True)[cols]


def support_at(events: pd.DataFrame, support: pd.DataFrame, types, time_col="time") -> np.ndarray:
    """True where an event time falls inside a support interval of one of ``types``."""
    flag = np.zeros(len(events), dtype=bool)
    sup = support[support["type"].isin(list(types))]
    if events.empty or sup.empty:
        return flag
    ev = events[["stay_id", time_col]].reset_index(drop=True).reset_index(names="_row")
    j = ev.merge(sup[["stay_id", "start", "end"]], on="stay_id")
    hit = j[(j[time_col] >= j["start"]) & (j[time_col] <= j["end"])]["_row"].unique()
    flag[hit] = True
    return flag


def drop_transient(readings: pd.DataFrame, minutes: float) -> pd.DataFrame:
    """Drop readings followed within ``minutes`` by a better-scoring reading of the same kind.

    Implements SOFA-2 footnote g ("changes ... within 1 hour, eg after suctioning, should not
    be considered"). Expects columns stay_id, kind, time, points.
    """
    if readings.empty or not minutes:
        return readings
    keep = np.ones(len(readings), dtype=bool)
    r = readings.reset_index(drop=True)
    win = np.timedelta64(int(minutes * 60), "s")
    for _, g in r.groupby(["stay_id", "kind"], sort=False):
        g = g.sort_values("time")
        t = g["time"].to_numpy()
        p = g["points"].to_numpy()
        idx = g.index.to_numpy()
        hi = np.searchsorted(t, t + win, side="right")
        for i in range(len(g)):
            later = p[i + 1 : hi[i]]
            if later.size and np.nanmin(later) < p[i]:
                keep[idx[i]] = False
    return r.loc[keep]


def respiratory_hourly(
    readings: pd.DataFrame,
    support: pd.DataFrame,
    stays: pd.DataFrame,
    score_cfg: dict,
) -> pd.DataFrame:
    """Hourly worst PaO2:FiO2 and SpO2:FiO2 points and ratios for one score.

    Returns:
        DataFrame indexed by (stay_id, hr) with pf_pts, sf_pts, pf_min, sf_min.
    """
    rc = score_cfg["respiratory"]
    r = readings
    if not rc.get("use_sf_ratio"):
        r = r[r["kind"] == "pf"]
    else:
        r = r[(r["kind"] == "pf") | (r["spo2"] < rc["sf_max_spo2_exclusive"])]
    r = r.copy()
    r["support"] = support_at(r, support, rc["support_types"])
    r["points"] = np.nan
    for kind, key in (("pf", "pf_bands"), ("sf", "sf_bands")):
        sel = r["kind"] == kind
        if sel.any():
            r.loc[sel, "points"] = ratio_points(
                r.loc[sel, "ratio"].to_numpy(), r.loc[sel, "support"].to_numpy(), rc[key], rc
            )
    r = assign_hour(r, stays)
    has_pf = r[r["kind"] == "pf"].groupby(["stay_id", "hr"]).size()
    r = drop_transient(r, rc.get("transient_minutes", 0))
    if r.empty:
        return pd.DataFrame(columns=["pf_pts", "sf_pts", "pf_min", "sf_min"])
    agg = r.pivot_table(
        index=["stay_id", "hr"], columns="kind", values=["points", "ratio"],
        aggfunc={"points": "max", "ratio": "min"},
    )
    out = pd.DataFrame(index=agg.index)
    for kind in ("pf", "sf"):
        out[f"{kind}_pts"] = agg[("points", kind)] if ("points", kind) in agg else np.nan
        out[f"{kind}_min"] = agg[("ratio", kind)] if ("ratio", kind) in agg else np.nan
    # SpO2:FiO2 only in hours without any PaO2:FiO2 (before the transient filter)
    pf_hours = out.index.isin(has_pf.index)
    out.loc[pf_hours, ["sf_pts", "sf_min"]] = np.nan
    return out
