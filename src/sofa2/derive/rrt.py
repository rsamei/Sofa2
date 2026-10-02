"""Hourly renal replacement therapy status."""

from __future__ import annotations

import pandas as pd

from sofa2.derive.timegrid import build_grid, overlap_hours


def rrt_hourly(support: pd.DataFrame, stays: pd.DataFrame, kidney_cfg: dict) -> pd.Series:
    """True for hours with RRT, after intermittent RRT until it is terminated, or chronic RRT.

    * Any RRT session (``kidney.rrt_types``) overlapping the hour.
    * SOFA-2 footnote q: after an intermittent session, the following
      ``kidney.intermittent_rrt_carry_hours`` hours also count (RRT is taken as terminated after
      that long without a session, or at ICU discharge).
    * ``stays.chronic_rrt``: every hour of the stay (Table 2: "includes chronic use").

    Returns:
        Boolean Series indexed by (stay_id, hr) over the whole grid.
    """
    grid = build_grid(stays).set_index(["stay_id", "hr"])
    flag = pd.Series(False, index=grid.index)
    sup = support[support["type"].isin(kidney_cfg["rrt_types"])].copy()
    carry = kidney_cfg.get("intermittent_rrt_carry_hours") or 0
    inter = sup["type"] == "rrt_intermittent"
    sup.loc[inter, "end"] = sup.loc[inter, "end"] + pd.Timedelta(hours=carry)
    if len(sup):
        h = overlap_hours(sup, stays)
        flag[flag.index.isin(pd.MultiIndex.from_frame(h[["stay_id", "hr"]]))] = True
    chronic = stays.loc[stays["chronic_rrt"], "stay_id"]
    flag[flag.index.get_level_values("stay_id").isin(chronic)] = True
    return flag
