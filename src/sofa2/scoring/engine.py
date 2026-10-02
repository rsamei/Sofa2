"""Hourly organ points for one score, built from the internal schema.

Every grid hour gets the points of each organ from the worst values of that hour (NaN where the
organ had no data in the hour). Window aggregation and missing-data handling follow in
:mod:`sofa2.scoring.windows` and :mod:`sofa2.missing`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sofa2.derive.gcs import brain_hourly
from sofa2.derive.infusions import active_hours, hourly_rates, valid_infusions
from sofa2.derive.oxygenation import ratio_readings, respiratory_hourly
from sofa2.derive.rrt import rrt_hourly
from sofa2.derive.timegrid import assign_hour, build_grid
from sofa2.derive.urine import urine_windows
from sofa2.schema import VASOACTIVE_DRUGS, ICUData
from sofa2.scoring import components as C
from sofa2.scoring.bands import score

ORGANS = ("respiratory", "cardiovascular", "brain", "liver", "kidney", "hemostasis")
#: Hourly points from single measured variables (no treatment-derived points), used to carry
#: values forward. Columns are named ``<organ>_meas`` or ``<organ>_meas_<variable>``; each
#: variable is carried separately and the organ takes the worst carried variable.
MEAS_PREFIX = "_meas"


def meas_columns(columns) -> dict[str, list[str]]:
    """Map each organ to its measurement columns."""
    return {o: [c for c in columns if c == f"{o}_meas" or c.startswith(f"{o}_meas_")]
            for o in ORGANS}

_LAB_AGG = {
    "bilirubin": "max",
    "creatinine": "max",
    "platelets": "min",
    "potassium": "max",
    "ph": "min",
    "bicarbonate": "min",
    "map": "min",
}


@dataclass
class Shared:
    """Score-independent derivations, computed once per run."""

    grid: pd.DataFrame
    labs: pd.DataFrame
    readings: pd.DataFrame
    urine: pd.DataFrame


def _urine_windows_needed(cfg) -> list[int]:
    k = cfg.sofa2["kidney"]
    w = {int(c["window_hours"]) for c in k["urine_output"]}
    w.add(int(k["anuria"]["window_hours"]))
    w.add(int(k["rrt_criteria"]["oliguria"]["window_hours"]))
    w.add(int(cfg.sofa1["kidney"]["urine_output_window_hours"]))
    return sorted(w)


def shared_inputs(data: ICUData, cfg) -> Shared:
    """Hourly lab aggregates, oxygenation readings and urine windows."""
    p = cfg.pipeline
    grid = build_grid(data.stays).set_index(["stay_id", "hr"])
    m = data.measurements
    pre_vars = set(p.get("pre_icu_lab_variables") or [])
    pre = assign_hour(m[m["variable"].isin(pre_vars)], data.stays, pre_hours=p["pre_icu_lab_hours"])
    rest = assign_hour(m[~m["variable"].isin(pre_vars)], data.stays)
    mh = pd.concat([pre, rest], ignore_index=True)
    labs = pd.DataFrame(index=grid.index)
    for var, how in _LAB_AGG.items():
        s = mh[mh["variable"] == var].groupby(["stay_id", "hr"])["value"].agg(how)
        labs[f"{var}_{how}"] = s
    readings = ratio_readings(m, p)
    uw = urine_windows(data.urine_output, data.stays, m, p, windows=_urine_windows_needed(cfg))
    uw = uw.reindex(grid.index)
    return Shared(grid=grid, labs=labs, readings=readings, urine=uw)


def _flag(index: pd.Index, hours: pd.Index) -> np.ndarray:
    return index.isin(hours) if len(hours) else np.zeros(len(index), dtype=bool)


def _support_hours(data: ICUData, types) -> pd.Index:
    s = data.support[data.support["type"].isin(list(types))]
    return active_hours(s, data.stays)


def hourly_sofa2(data: ICUData, cfg, sh: Shared) -> pd.DataFrame:
    """Hourly SOFA-2 organ points plus the inputs needed at window level."""
    c, p = cfg.sofa2, cfg.pipeline
    idx = sh.grid.index
    out = pd.DataFrame(index=idx)

    # respiratory
    rh = respiratory_hourly(sh.readings, data.support, data.stays, c).reindex(idx)
    ecmo = _flag(idx, _support_hours(data, c["respiratory"]["ecmo_types"]))
    out["respiratory"] = C.respiratory(rh["pf_pts"], rh["sf_pts"], ecmo, c["respiratory"])
    out["respiratory_meas"] = C.respiratory(rh["pf_pts"], rh["sf_pts"], None, c["respiratory"])
    out["resp_support"] = _flag(idx, _support_hours(data, c["respiratory"]["support_types"]))
    out["pf_min"], out["sf_min"] = rh["pf_min"], rh["sf_min"]

    # cardiovascular
    cc = c["cardiovascular"]
    inf = valid_infusions(data.infusions, cc["min_infusion_minutes"], p["infusions"]["merge_gap_minutes"])
    rates = hourly_rates(
        inf, data.stays, VASOACTIVE_DRUGS, sums={"ne_epi": ["norepinephrine", "epinephrine"]}
    ).reindex(idx).fillna(0.0)
    other = (rates[[d for d in cc["other_agents"] if d in rates]] > 0).any(axis=1).to_numpy()
    mech = _flag(idx, _support_hours(data, cc["mechanical_support_types"]))
    out["cardiovascular"] = C.cardiovascular_sofa2(
        rates["ne_epi"], rates["dopamine"], other, mech, sh.labs["map_min"], cc
    )
    out["cardiovascular_meas"] = score(sh.labs["map_min"], cc["map_bands"])
    out["map_min"], out["ne_epi_max"], out["dopamine_max"] = (
        sh.labs["map_min"], rates["ne_epi"], rates["dopamine"]
    )

    # brain
    bh = brain_hourly(data.gcs, data.infusions, data.medications, data.stays, c, p).reindex(idx)
    out["brain"] = bh["brain_pts"]
    out["brain_meas"] = bh["gcs_pts"]
    out["gcs_min"], out["sedated"], out["delirium"] = bh["gcs_min"], bh["sedated"], bh["delirium"]
    out["presedation_gcs"] = bh["presedation_gcs"]

    # liver, hemostasis
    out["liver"] = out["liver_meas"] = C.liver(sh.labs["bilirubin_max"], c["liver"])
    out["hemostasis"] = out["hemostasis_meas"] = C.hemostasis(sh.labs["platelets_min"], c["hemostasis"])
    out["bilirubin_max"], out["platelets_min"] = sh.labs["bilirubin_max"], sh.labs["platelets_min"]

    # kidney
    kc = c["kidney"]
    uw = sh.urine
    rates_uo = {int(cr["window_hours"]): uw[f"rate_{int(cr['window_hours'])}"].to_numpy() for cr in kc["urine_output"]}
    an = uw[f"anuria_{int(kc['anuria']['window_hours'])}"].to_numpy()
    uo_pts = C.urine_points_sofa2(rates_uo, an, kc)
    rrt = rrt_hourly(data.support, data.stays, kc).reindex(idx).to_numpy()
    out["kidney"] = C.kidney_sofa2(sh.labs["creatinine_max"], uo_pts, rrt, kc)
    out["kidney_meas_creatinine"] = score(sh.labs["creatinine_max"], kc["creatinine_bands"])
    out["kidney_meas_urine"] = uo_pts
    out["creatinine_max"] = sh.labs["creatinine_max"]
    olig = kc["rrt_criteria"]["oliguria"]
    with np.errstate(invalid="ignore"):
        out["oliguria"] = C._test(uw[f"rate_{int(olig['window_hours'])}"], olig)
    out["potassium_max"] = sh.labs["potassium_max"]
    out["ph_min"] = sh.labs["ph_min"]
    out["bicarbonate_min"] = sh.labs["bicarbonate_min"]
    out["rrt"] = rrt
    for w in sorted(rates_uo):
        out[f"uo_rate_{w}h"] = rates_uo[w]
    return out


def hourly_sofa1(data: ICUData, cfg, sh: Shared) -> pd.DataFrame:
    """Hourly original-SOFA organ points plus the inputs needed at window level."""
    c, p = cfg.sofa1, cfg.pipeline
    idx = sh.grid.index
    out = pd.DataFrame(index=idx)

    rh = respiratory_hourly(sh.readings, data.support, data.stays, c).reindex(idx)
    out["respiratory"] = C.respiratory(rh["pf_pts"], None, None, c["respiratory"])
    out["respiratory_meas"] = out["respiratory"]
    out["resp_support"] = _flag(idx, _support_hours(data, c["respiratory"]["support_types"]))
    out["pf_min"] = rh["pf_min"]

    cc = c["cardiovascular"]
    inf = valid_infusions(data.infusions, cc["min_infusion_minutes"], p["infusions"]["merge_gap_minutes"])
    drugs = list(cc["drug_bands"])
    rates = hourly_rates(inf, data.stays, drugs).reindex(idx).fillna(0.0)
    out["cardiovascular"] = C.cardiovascular_sofa1(
        rates["norepinephrine"], rates["epinephrine"], rates["dopamine"], rates["dobutamine"],
        sh.labs["map_min"], cc,
    )
    out["cardiovascular_meas"] = score(sh.labs["map_min"], cc["map_bands"])
    out["map_min"] = sh.labs["map_min"]

    bh = brain_hourly(data.gcs, data.infusions, data.medications, data.stays, c, p).reindex(idx)
    out["brain"] = bh["brain_pts"]
    out["brain_meas"] = bh["gcs_pts"]
    out["gcs_min"] = bh["gcs_min"]
    out["presedation_gcs"] = bh["presedation_gcs"]

    out["liver"] = out["liver_meas"] = C.liver(sh.labs["bilirubin_max"], c["liver"])
    out["hemostasis"] = out["hemostasis_meas"] = C.hemostasis(sh.labs["platelets_min"], c["hemostasis"])
    out["bilirubin_max"], out["platelets_min"] = sh.labs["bilirubin_max"], sh.labs["platelets_min"]

    kc = c["kidney"]
    out["kidney"] = C.kidney_sofa1(sh.labs["creatinine_max"], np.nan, kc)
    out["creatinine_max"] = sh.labs["creatinine_max"]
    out["uo_ml_day"] = sh.urine[f"mlday_{int(kc['urine_output_window_hours'])}"]
    # carried values include the urine points (scored at window level in the score itself)
    out["kidney_meas_creatinine"] = score(sh.labs["creatinine_max"], kc["creatinine_bands"])
    out["kidney_meas_urine"] = score(out["uo_ml_day"], kc["urine_output_daily_bands"])
    return out
