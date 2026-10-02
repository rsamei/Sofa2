"""Turn raw MIMIC-IV extracts into the internal schema (database-independent pandas code)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sofa2.schema import ICUData, empty
from sofa2.units import dose_to_ug_kg_min, norepinephrine_to_base

HOUR = pd.Timedelta(hours=1)


def _trim(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def _points(df: pd.DataFrame, typ: str) -> pd.DataFrame:
    """Charted observations as zero-length support intervals (they count in their hour)."""
    return pd.DataFrame({"stay_id": df["stay_id"], "start": df["charttime"],
                         "end": df["charttime"], "type": typ})


def _invert(groups: dict) -> dict:
    return {int(i): name for name, items in groups.items() for i in items}


# ------------------------------------------------------------------------------------ stays
def stays(raw_stays: pd.DataFrame, chronic: pd.DataFrame, iv: pd.DataFrame) -> pd.DataFrame:
    """Stays, with the first ``inputevents.patientweight`` as fallback weight (used only when no
    weight is charted)."""
    s = raw_stays[["stay_id", "patient_id", "intime", "outtime"]].copy()
    s["chronic_rrt"] = s["stay_id"].isin(chronic["stay_id"]) if len(chronic) else False
    w = iv[pd.to_numeric(iv["patientweight"], errors="coerce") > 0].sort_values("starttime")
    s["weight_kg"] = s["stay_id"].map(w.groupby("stay_id")["patientweight"].first()).astype(float)
    return s


# ------------------------------------------------------------------------------ measurements
def measurements(ce: pd.DataFrame, le: pd.DataFrame, mp: dict) -> pd.DataFrame:
    c, l = mp["chartevents"], mp["labevents"]
    parts = []
    v = ce["valuenum"]

    lo, hi = c["map_range"]
    sel = ce["itemid"].isin(c["map"]) & (v > lo) & (v < hi)
    parts.append(ce.loc[sel].assign(variable="map"))

    lo, hi = c["spo2_range"]
    sel = ce["itemid"].isin(c["spo2"]) & (v > lo) & (v <= hi)
    parts.append(ce.loc[sel].assign(variable="spo2"))

    f = ce[ce["itemid"].isin(c["fio2"])].copy()
    f["valuenum"] = _fio2_fraction(f["valuenum"])
    parts.append(f.dropna(subset=["valuenum"]).assign(variable="fio2"))

    sel = ce["itemid"].isin(c["weight"]) & (v > 0)
    parts.append(ce.loc[sel].assign(variable="weight"))
    sel = ce["itemid"].isin(c.get("weight_lbs", [])) & (v > 0)
    parts.append(ce.loc[sel].assign(variable="weight", valuenum=v[sel] * 0.45359237))

    chart = pd.concat(parts, ignore_index=True).drop(columns="value")
    chart = chart.rename(columns={"charttime": "time", "valuenum": "value"})
    chart["specimen_id"] = pd.NA

    lab_parts = []
    for var in ("creatinine", "bilirubin", "platelets", "potassium", "bicarbonate", "ph"):
        sel = le["itemid"].isin(l[var])
        if var in l.get("ranges", {}):
            lo, hi = l["ranges"][var]
            sel &= (le["valuenum"] > lo) & (le["valuenum"] <= hi)
        lab_parts.append(le.loc[sel].assign(variable=var))
    arterial = _trim(le["specimen_type"]).isin(l["arterial_specimen_values"]).fillna(False)
    lab_parts.append(le.loc[le["itemid"].isin(l["pao2"]) & arterial].assign(variable="pao2"))
    lf = le[le["itemid"].isin(l["fio2"])].copy()
    lf["valuenum"] = _fio2_fraction(lf["valuenum"])
    lab_parts.append(lf.dropna(subset=["valuenum"]).assign(variable="fio2"))
    lab = pd.concat(lab_parts, ignore_index=True)
    lab = lab.rename(columns={"charttime": "time", "valuenum": "value"})

    cols = ["stay_id", "time", "variable", "value", "specimen_id"]
    out = pd.concat([chart[cols], lab[cols]], ignore_index=True)
    out["specimen_id"] = out["specimen_id"].astype("object")
    return out.dropna(subset=["value"])


def _fio2_fraction(x: pd.Series) -> pd.Series:
    """FiO2 charted as a fraction (0.21-1) or a percentage (21-100); other values dropped."""
    x = pd.to_numeric(x, errors="coerce")
    return pd.Series(
        np.where(x.between(0.21, 1.0), x, np.where(x.between(21, 100), x / 100.0, np.nan)),
        index=x.index,
    )


# ------------------------------------------------------------------------------------- GCS
def gcs(ce: pd.DataFrame, mp: dict) -> pd.DataFrame:
    c = mp["chartevents"]
    g = ce[ce["itemid"].isin([c["gcs_eye"], c["gcs_verbal"], c["gcs_motor"]])].copy()
    if g.empty:
        return empty("gcs")
    g["unassessable"] = (g["itemid"] == c["gcs_verbal"]) & _trim(g["value"]).isin(
        c["gcs_verbal_unassessable_values"]
    ).fillna(False)
    g.loc[g["unassessable"], "valuenum"] = np.nan
    key = ["stay_id", "charttime"]
    wide = g.pivot_table(index=key, columns="itemid", values="valuenum", aggfunc="max")
    flag = g.groupby(key)["unassessable"].max()
    out = pd.DataFrame(index=flag.index)
    for name, item in (("eye", c["gcs_eye"]), ("verbal", c["gcs_verbal"]), ("motor", c["gcs_motor"])):
        out[name] = wide[item] if item in wide else np.nan
    out["unassessable"] = flag
    complete = out[["eye", "verbal", "motor"]].notna().all(axis=1) & ~out["unassessable"]
    out["total"] = (out["eye"] + out["verbal"] + out["motor"]).where(complete)
    out = out.reset_index().rename(columns={"charttime": "time"})
    return out[["stay_id", "time", "eye", "verbal", "motor", "total", "unassessable"]]


# ------------------------------------------------------------------------------- infusions
def infusions(iv: pd.DataFrame, mp: dict) -> pd.DataFrame:
    m = mp["inputevents"]
    drug_of = _invert(m["drugs"])
    x = iv[iv["itemid"].isin(list(drug_of)) & (iv["rate"] > 0) & iv["endtime"].notna()].copy()
    x["drug"] = x["itemid"].map(drug_of)
    keep = pd.Series(True, index=x.index)
    rate = x["rate"].astype(float).copy()
    for drug, units in m["dose_units"].items():
        sel = x["drug"] == drug
        if not sel.any():
            continue
        unit = _trim(x.loc[sel, "rateuom"]).fillna("")
        ok = unit.isin(units)
        n_bad = int((~ok).sum())
        if n_bad:
            warnings.warn(f"inputevents: dropped {n_bad} {drug} rows with rate units outside {units}",
                          stacklevel=2)
        keep[sel & ~ok.reindex(x.index, fill_value=False)] = False
        for u in units:
            su = sel & (_trim(x["rateuom"]) == u).fillna(False)
            if su.any():
                rate[su] = dose_to_ug_kg_min(x.loc[su, "rate"], u, x.loc[su, "patientweight"])
                n_nan = int(rate[su].isna().sum())
                if n_nan:
                    warnings.warn(f"inputevents: dropped {n_nan} {drug} rows in {u} without a "
                                  "patient weight", stacklevel=2)
    salt = m.get("norepinephrine_salt", "base")
    ne = x["drug"] == "norepinephrine"
    rate[ne] = norepinephrine_to_base(rate[ne], salt)
    x["rate"] = rate
    x = x[keep & x["rate"].notna() & (x["rate"] > 0)]
    return x.rename(columns={"starttime": "start", "endtime": "end"})[
        ["stay_id", "start", "end", "drug", "rate"]
    ]


# ----------------------------------------------------------------------------- medications
def medications(iv: pd.DataFrame, emar: pd.DataFrame, mp: dict) -> pd.DataFrame:
    bolus_of = _invert(mp["inputevents"]["bolus_drugs"])
    b = iv[iv["itemid"].isin(list(bolus_of))]
    parts = [pd.DataFrame({"stay_id": b["stay_id"], "time": b["starttime"],
                           "drug": b["itemid"].map(bolus_of)})]
    if len(emar):
        med = emar["medication"].astype("string").str.lower()
        excluded = pd.Series(False, index=emar.index)
        for p in mp["emar"].get("exclude", []):
            excluded |= med.str.contains(p.lower(), regex=False).fillna(False)
        for drug, patterns in mp["emar"]["medications"].items():
            sel = pd.Series(False, index=emar.index)
            for p in patterns:
                sel |= med.str.contains(p.lower(), regex=False).fillna(False)
            sel &= ~excluded
            e = emar[sel]
            parts.append(pd.DataFrame({"stay_id": e["stay_id"], "time": e["charttime"], "drug": drug}))
    out = pd.concat(parts, ignore_index=True)
    return out.dropna(subset=["time"]).drop_duplicates()


# --------------------------------------------------------------------------------- support
def ventilation_status(ce: pd.DataFrame, vm: dict, c: dict) -> pd.DataFrame:
    """Classify each charting time with a device or ventilator mode."""
    sel = ce["itemid"].isin([c["o2_device"], c["vent_mode"], c["vent_mode_hamilton"]])
    x = ce.loc[sel, ["stay_id", "charttime", "itemid", "value"]].copy()
    if x.empty:
        return pd.DataFrame(columns=["stay_id", "charttime", "status"])
    x["value"] = _trim(x["value"])
    dev = x["itemid"] == c["o2_device"]
    mode = x["itemid"] == c["vent_mode"]
    ham = x["itemid"] == c["vent_mode_hamilton"]
    flags = pd.DataFrame({
        "stay_id": x["stay_id"], "charttime": x["charttime"],
        "imv": (dev & x["value"].isin(vm["invasive_devices"]))
        | (mode & x["value"].isin(vm["invasive_modes"]))
        | (ham & x["value"].isin(vm["invasive_modes_hamilton"])),
        "bipap": dev & x["value"].isin(vm["bipap_devices"]),
        "cpap": (dev & x["value"].isin(vm["cpap_devices"]))
        | (mode & x["value"].isin(vm.get("cpap_modes", []))),
        "niv": ham & x["value"].isin(vm["niv_modes_hamilton"]),
        "hfnc": dev & x["value"].isin(vm["hfnc_devices"]),
        "oxygen": dev & x["value"].isin(vm["oxygen_devices"]),
        "none": dev & x["value"].isin(vm["none_devices"]),
    })
    order = ["imv", "bipap", "cpap", "niv", "hfnc", "oxygen", "none"]
    f = flags.fillna(False).groupby(["stay_id", "charttime"])[order].max()
    status = pd.Series(pd.NA, index=f.index, dtype="object")
    for name in reversed(order):  # highest priority last so it wins
        status[f[name].astype(bool)] = name
    return status.dropna().rename("status").reset_index()


def ventilation_episodes(status: pd.DataFrame, gap_hours: float) -> pd.DataFrame:
    """Episodes per mimic-code ventilation.sql (elapsed-time version of its 14-hour rule)."""
    if status.empty:
        return pd.DataFrame(columns=["stay_id", "start", "end", "status"])
    s = status.sort_values(["stay_id", "charttime"]).reset_index(drop=True)
    gap = pd.Timedelta(hours=gap_hours)
    g = s.groupby("stay_id")
    prev_t, prev_s, next_t = g["charttime"].shift(), g["status"].shift(), g["charttime"].shift(-1)
    new = prev_s.isna() | (prev_s != s["status"]) | ((s["charttime"] - prev_t) >= gap)
    s["episode"] = new.cumsum()
    s["end_c"] = np.where(next_t.isna() | ((next_t - s["charttime"]) >= gap), s["charttime"], next_t)
    s["end_c"] = pd.to_datetime(s["end_c"])
    e = s.groupby("episode").agg(stay_id=("stay_id", "first"), status=("status", "first"),
                                 start=("charttime", "min"), last=("charttime", "max"),
                                 end=("end_c", "max"))
    e = e[e["start"] != e["last"]]
    return e[["stay_id", "start", "end", "status"]].reset_index(drop=True)


def support(ce: pd.DataFrame, pe: pd.DataFrame, iv: pd.DataFrame, mp: dict) -> pd.DataFrame:
    c, vm = mp["chartevents"], mp["ventilation"]
    parts = []

    ep = ventilation_episodes(ventilation_status(ce, vm, c), vm["gap_hours"])
    ep = ep[ep["status"].isin(["imv", "bipap", "cpap", "niv", "hfnc"])]
    parts.append(ep.rename(columns={"status": "type"}))

    cfg = ce[ce["itemid"].isin(c["ecmo_configuration"])].copy()
    val = _trim(cfg["value"]).str.lower()
    parts.append(_points(cfg[(val == "vv").fillna(False)], "ecmo_vv"))
    parts.append(_points(cfg[val.isin(["va", "vav"]).fillna(False)], "ecmo_va"))
    flow = ce[ce["itemid"].isin(c["ecmo_flow"]) & (ce["valuenum"] > 0)]
    parts.append(_points(flow, "ecmo"))

    for typ, items in c["mcs"].items():
        parts.append(_points(ce[ce["itemid"].isin(items) & (ce["valuenum"] > 0)], typ))

    for typ, items in c["rrt_active"].items():
        parts.append(_points(ce[ce["itemid"].isin(items) & ce["value"].notna()], typ))
    pd_cat = ce[(ce["itemid"] == c["rrt_pd_catheter_status"])
                & _trim(ce["value"]).isin(c["rrt_pd_in_use_values"]).fillna(False)]
    parts.append(_points(pd_cat, "rrt_intermittent"))

    crrt = iv[iv["itemid"].isin(mp["inputevents"].get("rrt_continuous", []))
              & (pd.to_numeric(iv["amount"], errors="coerce") > 0)]
    end = crrt["endtime"].fillna(crrt["starttime"])
    parts.append(pd.DataFrame({"stay_id": crrt["stay_id"], "start": crrt["starttime"],
                               "end": end.where(end >= crrt["starttime"], crrt["starttime"]),
                               "type": "rrt_continuous"}))

    p = mp["procedureevents"]
    for typ in ("rrt_intermittent", "rrt_continuous", "imv", "niv"):
        x = pe[pe["itemid"].isin(p[typ])]
        end = x["endtime"].fillna(x["starttime"])
        parts.append(pd.DataFrame({"stay_id": x["stay_id"], "start": x["starttime"],
                                   "end": end.where(end >= x["starttime"], x["starttime"]),
                                   "type": typ}))
    out = pd.concat(parts, ignore_index=True)
    return out.dropna(subset=["start", "end"])[["stay_id", "start", "end", "type"]]


# ---------------------------------------------------------------------------- urine output
def urine_output(oe: pd.DataFrame, mp: dict) -> pd.DataFrame:
    o = mp["outputevents"]
    x = oe[oe["itemid"].isin(o["urine"] + o["urine_subtract"])].copy()
    x["value"] = pd.to_numeric(x["value"], errors="coerce")
    sub = x["itemid"].isin(o["urine_subtract"]) & (x["value"] > 0)
    x.loc[sub, "value"] = -x.loc[sub, "value"]
    s = x.groupby(["stay_id", "charttime"], as_index=False)["value"].sum()
    s["value"] = s["value"].clip(lower=0)
    return s.rename(columns={"charttime": "time", "value": "volume_ml"})


# ------------------------------------------------------------------------------------- all
def to_icudata(raw: dict[str, pd.DataFrame], mp: dict) -> ICUData:
    """Build :class:`ICUData` from the raw extracts (keys as the SQL file names)."""
    raw = {k: v.copy() for k, v in raw.items()}
    for key in ("chartevents", "labevents", "inputevents", "outputevents", "procedureevents", "emar"):
        df = raw[key]
        for col in ("charttime", "starttime", "endtime"):
            if col in df:
                df[col] = pd.to_datetime(df[col])
    st = raw["stays"].copy()
    st["intime"], st["outtime"] = pd.to_datetime(st["intime"]), pd.to_datetime(st["outtime"])
    ce, le, iv = raw["chartevents"], raw["labevents"], raw["inputevents"]
    return ICUData(
        stays=stays(st, raw["chronic_rrt"], iv),
        measurements=measurements(ce, le, mp),
        gcs=gcs(ce, mp),
        infusions=infusions(iv, mp),
        medications=medications(iv, raw["emar"], mp),
        support=support(ce, raw["procedureevents"], iv, mp),
        urine_output=urine_output(raw["outputevents"], mp),
    )
