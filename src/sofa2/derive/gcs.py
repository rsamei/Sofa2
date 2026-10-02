"""Hourly brain inputs: GCS points with the sedation and delirium rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sofa2.derive.oxygenation import support_at
from sofa2.derive.timegrid import assign_hour, build_grid, overlap_hours
from sofa2.scoring.components import brain, fmax, gcs_points


def sedation_intervals(infusions: pd.DataFrame, pcfg: dict) -> pd.DataFrame:
    """Intervals with a running sedative infusion (``sedation.drugs``, rate > 0)."""
    s = infusions[infusions["drug"].isin(pcfg["sedation"]["drugs"]) & (infusions["rate"] > 0)]
    return s[["stay_id", "start", "end"]].assign(type="sedation")


def brain_hourly(
    gcs: pd.DataFrame,
    infusions: pd.DataFrame,
    medications: pd.DataFrame,
    stays: pd.DataFrame,
    score_cfg: dict,
    pcfg: dict,
) -> pd.DataFrame:
    """Hourly brain points for one score.

    * GCS charted during a sedative infusion is ignored.
    * A complete GCS scores on its total; a GCS flagged ``unassessable`` (the three domains
      cannot be evaluated, eg intubated) scores on the motor response when the score allows it
      (SOFA-2 footnote d). Incompletely documented charts without the flag are not used. Within an hour the
      worst complete total and the best motor response of incomplete assessments are combined
      by taking the worse of the two.
    * In hours with sedation and no unsedated GCS, the last unsedated GCS points before are
      carried (footnote c, no time cap); with none, the hour has no GCS value, which scores 0.
    * Delirium treatment in the hour scores at least 1 (footnote e, SOFA-2 only).

    Returns:
        DataFrame indexed by (stay_id, hr) with gcs_min, gcs_pts, sedated, delirium, brain_pts.
    """
    bc = score_cfg["brain"]
    grid = build_grid(stays).set_index(["stay_id", "hr"])
    out = pd.DataFrame(index=grid.index)

    sed = sedation_intervals(infusions, pcfg)
    g = gcs.copy()
    comp = g["eye"] + g["verbal"] + g["motor"]
    g["total"] = g["total"].fillna(comp)
    g["assessable"] = ~g["unassessable"] & g["total"].notna()
    # charts that are neither complete nor flagged unassessable (incomplete documentation) are
    # not used
    g = g[g["assessable"] | g["unassessable"]]
    g = g[~support_at(g, sed, ["sedation"])]
    g["points"] = gcs_points(
        g["total"].to_numpy(), g["motor"].to_numpy(), g["assessable"].to_numpy(), bc
    ) if len(g) else np.array([])
    g = assign_hour(g, stays)

    a = g[g["assessable"]].groupby(["stay_id", "hr"]).agg(gcs_min=("total", "min"), pa=("points", "max"))
    u = g[~g["assessable"]].groupby(["stay_id", "hr"]).agg(pu=("points", "min"))
    out = out.join(a).join(u)
    out["gcs_pts"] = fmax(out["pa"].to_numpy(), out["pu"].to_numpy()) if len(out) else np.array([])
    out = out.drop(columns=["pa", "pu"])

    sed_hours = overlap_hours(sed, stays)
    out["sedated"] = out.index.isin(pd.MultiIndex.from_frame(sed_hours[["stay_id", "hr"]])) if len(sed_hours) else False
    if bc.get("carry_presedation_gcs"):
        last = out.groupby(level="stay_id")["gcs_pts"].ffill()
        fill = out["sedated"] & out["gcs_pts"].isna()
        out.loc[fill, "gcs_pts"] = last[fill]

    drugs = bc.get("delirium_drugs") or []
    med = assign_hour(medications[medications["drug"].isin(drugs)], stays)
    out["delirium"] = out.index.isin(pd.MultiIndex.from_frame(med[["stay_id", "hr"]])) if len(med) else False
    out["brain_pts"] = brain(out["gcs_pts"].to_numpy(), out["delirium"].to_numpy(), bc) if len(out) else np.array([])
    return out
