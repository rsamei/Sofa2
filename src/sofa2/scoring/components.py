"""Per-organ point functions for SOFA-2 and the original SOFA.

All functions are vectorised over NumPy arrays (one element per scored time unit, usually an
hour) and take the relevant section of the YAML config. They return float arrays where NaN
means "no data to score this element"; missing-data handling happens later
(:mod:`sofa2.missing`). Inputs are already in canonical units and already reduced to the worst
value of the time unit (eg the lowest platelet count of the hour).
"""

from __future__ import annotations

import numpy as np

from sofa2.scoring.bands import score


def _arr(x, dtype=float) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=dtype))


def _bool(x, shape) -> np.ndarray:
    if x is None:
        return np.zeros(shape, dtype=bool)
    a = np.atleast_1d(np.asarray(x))
    if a.dtype.kind == "f":
        a = np.nan_to_num(a, nan=0.0)
    return np.broadcast_to(a.astype(bool), shape)


def _pos(x, shape) -> np.ndarray:
    """Positive dose indicator (NaN counts as no drug)."""
    if x is None:
        return np.zeros(shape, dtype=bool)
    a = np.broadcast_to(_arr(x), shape)
    return np.nan_to_num(a, nan=0.0) > 0


def fmax(*arrays) -> np.ndarray:
    """Element-wise maximum ignoring NaN (NaN only where all inputs are NaN)."""
    out = None
    for a in arrays:
        a = _arr(a)
        if out is None:
            out = a.copy()
        else:
            with np.errstate(invalid="ignore"):
                out = np.fmax(out, a)
    return out


# --------------------------------------------------------------------------------------------
# Brain
# --------------------------------------------------------------------------------------------


def gcs_points(total, motor=None, assessable=None, cfg: dict | None = None) -> np.ndarray:
    """Points for one GCS assessment.

    A complete GCS is scored on its total. If the three domains cannot be evaluated
    (``assessable`` false) and ``cfg['use_motor_when_unassessable']`` is true, the motor
    response is scored instead (SOFA-2 footnote d); otherwise the assessment gives NaN.
    """
    total = _arr(total)
    ok = np.ones(total.shape, dtype=bool) if assessable is None else _bool(assessable, total.shape)
    out = np.where(ok, score(total, cfg["gcs_total_bands"]), np.nan)
    if cfg.get("use_motor_when_unassessable") and motor is not None:
        motor_pts = score(np.broadcast_to(_arr(motor), total.shape), cfg["gcs_motor_bands"])
        out = np.where(ok, out, motor_pts)
    return out


def brain(gcs_pts, delirium=None, cfg: dict | None = None) -> np.ndarray:
    """Brain component from GCS points and the delirium-treatment flag (SOFA-2 footnote e).

    Delirium treatment scores at least ``cfg['delirium_min_points']`` even with GCS 15 or no
    GCS; without delirium treatment and without GCS the result is NaN.
    """
    gcs_pts = _arr(gcs_pts)
    minimum = cfg.get("delirium_min_points")
    if not minimum or delirium is None:
        return gcs_pts
    d = _bool(delirium, gcs_pts.shape)
    return np.where(d, fmax(gcs_pts, np.full(gcs_pts.shape, float(minimum))), gcs_pts)


# --------------------------------------------------------------------------------------------
# Respiratory
# --------------------------------------------------------------------------------------------


def ratio_points(ratio, support, bands, cfg: dict) -> np.ndarray:
    """Points of PaO2:FiO2 or SpO2:FiO2 readings with the advanced-support requirement.

    Without support the score is capped at ``cfg['max_points_without_support']`` unless
    ``cfg['allow_full_score_without_support']`` (SOFA-2 footnote h).
    """
    ratio = _arr(ratio)
    sup = _bool(support, ratio.shape)
    if cfg.get("allow_full_score_without_support"):
        sup = np.ones(ratio.shape, dtype=bool)
    return score(ratio, bands, support=sup, max_without_support=cfg["max_points_without_support"])


def respiratory(pf_pts, sf_pts=None, ecmo=None, cfg: dict | None = None) -> np.ndarray:
    """Respiratory component from the worst PaO2:FiO2 and SpO2:FiO2 points of the time unit.

    SpO2:FiO2 points are used only where no PaO2:FiO2 is available (SOFA-2 footnote f). ECMO
    scores ``cfg['ecmo_points']`` (footnote i).
    """
    pf_pts = _arr(pf_pts)
    out = pf_pts.copy()
    if sf_pts is not None and cfg.get("use_sf_ratio"):
        out = np.where(np.isnan(pf_pts), _arr(sf_pts), pf_pts)
    if ecmo is not None and cfg.get("ecmo_points"):
        e = _bool(ecmo, out.shape)
        out = np.where(e, fmax(out, np.full(out.shape, float(cfg["ecmo_points"]))), out)
    return out


# --------------------------------------------------------------------------------------------
# Cardiovascular
# --------------------------------------------------------------------------------------------


def cardiovascular_sofa2(
    ne_epi, dopamine, other_agent, mechanical, map_min, cfg: dict
) -> np.ndarray:
    """SOFA-2 cardiovascular component.

    Args:
        ne_epi: highest concurrent norepinephrine + epinephrine rate (ug/kg/min, base).
        dopamine: highest dopamine rate (ug/kg/min).
        other_agent: any other vasopressor or inotrope (``cfg['other_agents']``), excluding
            dopamine.
        mechanical: any mechanical circulatory support (footnote n, VA ECMO per footnote i).
        map_min: lowest MAP (mmHg).

    Rules (Table 2, footnotes i, l, m, n):
        * mechanical support -> 4
        * norepinephrine + epinephrine > 0: dose bands; any other agent (dopamine included)
          adds ``other_agent_adds_points`` (cap 4)
        * dopamine without norepinephrine/epinephrine: dopamine bands (footnote l); another
          agent adds ``other_agent_adds_points`` (cap 4)
        * another agent alone -> ``other_agent_alone_points``
        * no vasoactive drug: MAP bands (or footnote m bands if ``map_only_fallback``)
    """
    map_min = _arr(map_min)
    shape = np.broadcast_shapes(map_min.shape, _arr(ne_epi).shape, _arr(dopamine).shape)
    map_min = np.broadcast_to(map_min, shape)
    ne_epi = np.nan_to_num(np.broadcast_to(_arr(ne_epi), shape), nan=0.0)
    dopa = np.nan_to_num(np.broadcast_to(_arr(dopamine), shape), nan=0.0)
    other = _bool(other_agent, shape)
    mech = _bool(mechanical, shape)
    add = float(cfg["other_agent_adds_points"])

    map_bands = cfg["map_only_bands"] if cfg.get("map_only_fallback") else cfg["map_bands"]
    out = score(map_min, map_bands)

    ne_pts = score(ne_epi, cfg["ne_epi_bands"])
    ne_pts = np.where(other | (dopa > 0), np.minimum(4.0, ne_pts + add), ne_pts)
    dopa_pts = score(dopa, cfg["dopamine_alone_bands"])
    dopa_pts = np.where(other, np.minimum(4.0, dopa_pts + add), dopa_pts)

    out = np.where(other, float(cfg["other_agent_alone_points"]), out)
    out = np.where(dopa > 0, dopa_pts, out)
    out = np.where(ne_epi > 0, ne_pts, out)
    out = np.where(mech, float(cfg["mechanical_support_points"]), out)
    return out


def cardiovascular_sofa1(norepinephrine, epinephrine, dopamine, dobutamine, map_min, cfg: dict):
    """Original SOFA cardiovascular component (Vincent 1996, Table 3).

    The worst of the per-drug bands; with no qualifying drug, the MAP band.
    """
    map_min = _arr(map_min)
    drugs = {
        "norepinephrine": norepinephrine,
        "epinephrine": epinephrine,
        "dopamine": dopamine,
        "dobutamine": dobutamine,
    }
    shape = np.broadcast_shapes(map_min.shape, *(_arr(v).shape for v in drugs.values()))
    out = score(np.broadcast_to(map_min, shape), cfg["map_bands"])
    drug_pts = np.zeros(shape)
    any_drug = np.zeros(shape, dtype=bool)
    for name, bands in cfg["drug_bands"].items():
        rate = np.nan_to_num(np.broadcast_to(_arr(drugs[name]), shape), nan=0.0)
        drug_pts = np.maximum(drug_pts, score(rate, bands))
        any_drug |= rate > 0
    return np.where(any_drug, drug_pts, out)


# --------------------------------------------------------------------------------------------
# Liver, hemostasis
# --------------------------------------------------------------------------------------------


def liver(bilirubin_max, cfg: dict) -> np.ndarray:
    """Liver component from the highest total bilirubin (mg/dL)."""
    return score(bilirubin_max, cfg["bilirubin_bands"])


def hemostasis(platelets_min, cfg: dict) -> np.ndarray:
    """Hemostasis (coagulation) component from the lowest platelet count (x10^3/uL)."""
    return score(platelets_min, cfg["platelet_bands"])


# --------------------------------------------------------------------------------------------
# Kidney
# --------------------------------------------------------------------------------------------


def urine_points_sofa2(rates: dict[int, np.ndarray], anuria=None, cfg: dict | None = None):
    """SOFA-2 urine-output points.

    Args:
        rates: trailing-window mean urine output in mL/kg/h keyed by window length in hours
            (6, 12, 24); NaN where the window cannot be evaluated.
        anuria: True where the trailing ``cfg['anuria']['window_hours']`` window had 0 mL.
    """
    out = None
    for crit in cfg["urine_output"]:
        r = _arr(rates[int(crit["window_hours"])])
        pts = score(r, [{"points": crit["points"], "op": crit["op"], "value": crit["value"]}])
        out = pts if out is None else fmax(out, pts)
    if anuria is not None and cfg.get("anuria"):
        a = _bool(anuria, out.shape)
        out = np.where(a, fmax(out, np.full(out.shape, float(cfg["anuria"]["points"]))), out)
    return out


def kidney_sofa2(creatinine_max, urine_pts, rrt, cfg: dict) -> np.ndarray:
    """SOFA-2 kidney component: worst of creatinine, urine output and RRT (-> 4)."""
    cr = score(creatinine_max, cfg["creatinine_bands"])
    out = fmax(cr, urine_pts)
    r = _bool(rrt, out.shape)
    return np.where(r, float(cfg["rrt_points"]), out)


_CMP = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}


def _test(x, rule) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return _CMP[rule["op"]](_arr(x), rule["value"])


def rrt_criteria_met(creatinine_max, oliguria, potassium_max, ph_min, bicarbonate_min, cfg):
    """SOFA-2 footnote p: fulfils criteria for RRT (for patients not receiving RRT).

    (creatinine > 1.2 mg/dL or oliguria) and (potassium >= 6.0 mmol/L or (pH <= 7.20 and
    bicarbonate <= 12 mmol/L)). Arguments are the worst values of the scoring window;
    ``oliguria`` is true where the oliguria rule (< 0.3 mL/kg/h over 6 h) was met.
    """
    rule = cfg["rrt_criteria"]
    cr = _test(creatinine_max, rule["creatinine"])
    olig = _bool(oliguria, cr.shape)
    k = _test(potassium_max, rule["potassium"])
    acid = _test(ph_min, rule["ph"]) & _test(bicarbonate_min, rule["bicarbonate"])
    return (cr | olig) & (k | acid)


def kidney_sofa1(creatinine_max, urine_ml_per_day, cfg: dict) -> np.ndarray:
    """Original SOFA renal component: worst of creatinine and 24-hour urine volume."""
    cr = score(creatinine_max, cfg["creatinine_bands"])
    uo = score(urine_ml_per_day, cfg["urine_output_daily_bands"])
    return fmax(cr, uo)
