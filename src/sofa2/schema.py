"""Internal data schema shared by all adapters and the scoring engine.

An adapter only has to produce an :class:`ICUData` with these tables; the scoring code never
sees database-specific names. All times are timezone-naive ``datetime64`` and all values are in
the canonical units of :mod:`sofa2.units`.

Tables
------
``stays``         one row per ICU stay: stay_id, patient_id, intime, outtime,
                  weight_kg (optional), chronic_rrt (optional bool)
``measurements``  stay_id, time, variable, value, specimen_id (optional; pairs PaO2 with the
                  FiO2 of the same blood gas)
``gcs``           stay_id, time, eye, verbal, motor, total, verbal_unassessable
``infusions``     stay_id, start, end, drug, rate (vasoactive drugs in ug/kg/min; vasopressin in
                  units/min; sedatives in any unit, only presence is used)
``medications``   stay_id, time, drug (discrete administrations, eg delirium drugs)
``support``       stay_id, start, end, type (respiratory, mechanical circulatory and renal support)
``urine_output``  stay_id, time, volume_ml
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import pandas as pd

MEASUREMENT_VARIABLES = (
    "pao2",
    "fio2",
    "spo2",
    "map",
    "bilirubin",
    "creatinine",
    "platelets",
    "potassium",
    "ph",
    "bicarbonate",
    "weight",
)

VASOACTIVE_DRUGS = (
    "norepinephrine",
    "epinephrine",
    "dopamine",
    "dobutamine",
    "vasopressin",
    "phenylephrine",
    "milrinone",
    "levosimendan",
)
SEDATIVE_DRUGS = ("propofol", "midazolam", "lorazepam", "dexmedetomidine")
INFUSION_DRUGS = VASOACTIVE_DRUGS + SEDATIVE_DRUGS

SUPPORT_TYPES = (
    # respiratory
    "imv",
    "niv",
    "cpap",
    "bipap",
    "hfnc",
    "home_vent",
    # extracorporeal and mechanical circulatory
    "ecmo_vv",
    "ecmo_va",
    "iabp",
    "impella",
    "lvad",
    # renal
    "rrt_continuous",
    "rrt_intermittent",
)

COLUMNS: dict[str, dict[str, str]] = {
    "stays": {"stay_id": "id", "patient_id": "id", "intime": "time", "outtime": "time"},
    "measurements": {"stay_id": "id", "time": "time", "variable": "str", "value": "float"},
    "gcs": {
        "stay_id": "id",
        "time": "time",
        "eye": "float",
        "verbal": "float",
        "motor": "float",
        "total": "float",
        "verbal_unassessable": "bool",
    },
    "infusions": {"stay_id": "id", "start": "time", "end": "time", "drug": "str", "rate": "float"},
    "medications": {"stay_id": "id", "time": "time", "drug": "str"},
    "support": {"stay_id": "id", "start": "time", "end": "time", "type": "str"},
    "urine_output": {"stay_id": "id", "time": "time", "volume_ml": "float"},
}
OPTIONAL_COLUMNS: dict[str, dict[str, str]] = {
    "stays": {"weight_kg": "float", "chronic_rrt": "bool"},
    "measurements": {"specimen_id": "id"},
}

# Plausibility ranges in canonical units; values outside are rejected by validate().
VALUE_RANGES = {
    "pao2": (0, 800),
    "fio2": (0.21, 1.0),
    "spo2": (0, 100),
    "map": (0, 300),
    "bilirubin": (0, 100),
    "creatinine": (0, 40),
    "platelets": (0, 3000),
    "potassium": (0, 15),
    "ph": (6.0, 8.0),
    "bicarbonate": (0, 80),
    "weight": (0, 700),
}


class SchemaError(ValueError):
    """Raised when input tables do not follow the internal schema."""


def empty(table: str) -> pd.DataFrame:
    """An empty table with the required columns of ``table``."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS[table]})


def _coerce(df: pd.DataFrame, table: str) -> pd.DataFrame:
    spec = {**COLUMNS[table], **{k: v for k, v in OPTIONAL_COLUMNS.get(table, {}).items() if k in df}}
    missing = [c for c in COLUMNS[table] if c not in df.columns]
    if missing:
        raise SchemaError(f"{table}: missing columns {missing}")
    out = df.copy()
    for col, kind in spec.items():
        if kind == "time":
            out[col] = pd.to_datetime(out[col]).astype("datetime64[ns]")
        elif kind == "float":
            out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
        elif kind == "bool":
            out[col] = out[col].fillna(False).astype(bool)
        elif kind == "str":
            out[col] = out[col].astype(str).str.strip().str.lower()
    return out


@dataclass
class ICUData:
    """All inputs for scoring, in the internal schema."""

    stays: pd.DataFrame
    measurements: pd.DataFrame = field(default_factory=lambda: empty("measurements"))
    gcs: pd.DataFrame = field(default_factory=lambda: empty("gcs"))
    infusions: pd.DataFrame = field(default_factory=lambda: empty("infusions"))
    medications: pd.DataFrame = field(default_factory=lambda: empty("medications"))
    support: pd.DataFrame = field(default_factory=lambda: empty("support"))
    urine_output: pd.DataFrame = field(default_factory=lambda: empty("urine_output"))

    def validate(self) -> "ICUData":
        """Coerce types, check columns, allowed categories and value ranges; return a new copy.

        Raises:
            SchemaError: on any violation.
        """
        tables = {f.name: _coerce(getattr(self, f.name), f.name) for f in fields(self)}
        stays = tables["stays"]
        if stays["stay_id"].duplicated().any():
            raise SchemaError("stays: duplicated stay_id")
        if (stays["outtime"] <= stays["intime"]).any():
            raise SchemaError("stays: outtime must be after intime")
        if "weight_kg" not in stays:
            stays["weight_kg"] = float("nan")
        if "chronic_rrt" not in stays:
            stays["chronic_rrt"] = False

        m = tables["measurements"]
        bad = set(m["variable"]) - set(MEASUREMENT_VARIABLES)
        if bad:
            raise SchemaError(f"measurements: unknown variables {sorted(bad)}")
        for var, (lo, hi) in VALUE_RANGES.items():
            vals = m.loc[m["variable"] == var, "value"]
            if ((vals < lo) | (vals > hi)).any():
                raise SchemaError(
                    f"measurements: {var} outside [{lo}, {hi}] (check units; see sofa2.units)"
                )
        if m["value"].isna().any():
            raise SchemaError("measurements: value has missing entries")

        for name, col, allowed in (
            ("infusions", "drug", INFUSION_DRUGS),
            ("support", "type", SUPPORT_TYPES),
        ):
            bad = set(tables[name][col]) - set(allowed)
            if bad:
                raise SchemaError(f"{name}: unknown {col} {sorted(bad)}")
            t = tables[name]
            if (t["end"] < t["start"]).any():
                raise SchemaError(f"{name}: end before start")
        if (tables["infusions"]["rate"] < 0).any():
            raise SchemaError("infusions: negative rate")
        if (tables["urine_output"]["volume_ml"] < 0).any():
            raise SchemaError("urine_output: negative volume")

        known = set(stays["stay_id"])
        for name, t in tables.items():
            if name != "stays" and not set(t["stay_id"]).issubset(known):
                raise SchemaError(f"{name}: stay_id not in stays")
        return ICUData(**tables)
