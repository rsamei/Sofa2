"""Aggregate hourly organ points to scoring windows (worst value per window)."""

from __future__ import annotations

import pandas as pd

from sofa2.scoring import components as C
from sofa2.scoring.bands import score
from sofa2.scoring.engine import ORGANS

# How each auxiliary column is reduced over a window.
_MAX = ("respiratory", "cardiovascular", "brain", "liver", "kidney", "hemostasis",
        "ne_epi_max", "dopamine_max", "bilirubin_max", "creatinine_max", "potassium_max")
_MIN = ("pf_min", "sf_min", "map_min", "gcs_min", "platelets_min", "ph_min", "bicarbonate_min",
        "uo_rate_6h", "uo_rate_12h", "uo_rate_24h")
_ANY = ("oliguria", "rrt", "sedated", "delirium")
_LAST = ("uo_ml_day",)


def window_frame(hourly: pd.DataFrame, freq: str, pcfg: dict) -> pd.DataFrame:
    """Window bounds per stay: stay_id, window, hr_first, hr_last.

    ``daily``: consecutive blocks of ``windows.daily_hours`` from admission (window 0 = day 1).
    ``hourly``: one window per hour h covering the preceding ``windows.hourly_lookback_hours``
    hours including h (clipped at admission).
    """
    hrs = hourly.reset_index()[["stay_id", "hr"]]
    if freq == "daily":
        n = int(pcfg["windows"]["daily_hours"])
        w = hrs.assign(window=hrs["hr"] // n)
        f = w.groupby(["stay_id", "window"])["hr"].agg(hr_first="min", hr_last="max").reset_index()
        return f
    if freq == "hourly":
        n = int(pcfg["windows"]["hourly_lookback_hours"])
        return hrs.assign(window=hrs["hr"], hr_first=(hrs["hr"] - n + 1).clip(lower=0), hr_last=hrs["hr"])
    raise ValueError(f"freq must be 'daily' or 'hourly', got {freq!r}")


def aggregate(hourly: pd.DataFrame, freq: str, pcfg: dict) -> pd.DataFrame:
    """Reduce hourly columns to windows: organs and worst-high values by max, worst-low values
    by min, flags by any, and ``_LAST`` columns by their value in the last hour.

    Returns:
        DataFrame with stay_id, window, hr_first, hr_last and the reduced columns.
    """
    cols = list(hourly.columns)
    h = hourly.copy()
    for c in _ANY:
        if c in h:
            h[c] = h[c].astype(float)
    if freq == "daily":
        n = int(pcfg["windows"]["daily_hours"])
        key = [h.index.get_level_values("stay_id"), h.index.get_level_values("hr") // n]
        grp = h.groupby(key)
        parts = {}
        for c in cols:
            if c in _MIN:
                parts[c] = grp[c].min()
            elif c in _LAST:
                parts[c] = grp[c].agg(lambda x: x.iloc[-1])
            else:
                parts[c] = grp[c].max()
        agg = pd.DataFrame(parts)
        agg.index.names = ["stay_id", "window"]
        agg = agg.reset_index()
    else:
        n = int(pcfg["windows"]["hourly_lookback_hours"])
        grp = h.groupby(level="stay_id", sort=False)
        parts = {}
        for c in cols:
            if c in _LAST:
                parts[c] = h[c]
                continue
            roll = grp[c].rolling(n, min_periods=1)
            r = roll.min() if c in _MIN else roll.max()
            parts[c] = r.droplevel(0).reindex(h.index)
        agg = pd.DataFrame(parts).reset_index().rename(columns={"hr": "window"})
    for c in _ANY:
        if c in agg:
            agg[c] = agg[c].fillna(0).astype(bool)
    frame = window_frame(hourly, freq, pcfg)
    return frame.merge(agg, on=["stay_id", "window"], how="left")


def apply_window_rules(win: pd.DataFrame, score_name: str, score_cfg: dict) -> pd.DataFrame:
    """Kidney rules evaluated on the whole window.

    * SOFA-2 footnote p: not receiving RRT but fulfilling the criteria within the window -> 4.
    * Original SOFA: 24-hour urine volume at the end of the window (mL/day bands).
    """
    w = win.copy()
    kc = score_cfg["kidney"]
    if score_name == "sofa2" and kc.get("rrt_criteria", {}).get("enabled"):
        met = C.rrt_criteria_met(
            w["creatinine_max"], w["oliguria"], w["potassium_max"], w["ph_min"],
            w["bicarbonate_min"], kc,
        )
        w["rrt_criteria_met"] = met & ~w["rrt"]
        w.loc[w["rrt_criteria_met"], "kidney"] = float(kc["rrt_points"])
    if score_name == "sofa1":
        uo = score(w["uo_ml_day"], kc["urine_output_daily_bands"])
        w["kidney"] = C.fmax(w["kidney"].to_numpy(), uo)
    return w


def organ_columns() -> tuple[str, ...]:
    """The six organ columns, in output order."""
    return ORGANS


def last_observed(hourly: pd.DataFrame, organ: str) -> pd.DataFrame:
    """Per stay-hour: hour and value of the last hour (up to and including it) with data."""
    v = hourly[organ]
    hr = pd.Series(hourly.index.get_level_values("hr"), index=hourly.index, dtype=float)
    hr = hr.where(v.notna())
    g = hr.groupby(level="stay_id")
    return pd.DataFrame(
        {"last_hr": g.ffill(), "last_val": v.groupby(level="stay_id").ffill()}, index=hourly.index
    )

