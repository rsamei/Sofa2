"""Explicit unit conversions to the canonical units used by the scoring code.

Canonical units:

========================  ==================
variable                  unit
========================  ==================
bilirubin, creatinine     mg/dL
platelets                 x10^3/uL (= x10^9/L)
pao2, map                 mmHg
fio2                      fraction (0.21-1.0)
spo2                      %
potassium, bicarbonate    mmol/L
ph                        pH units
weight                    kg
catecholamine infusions   ug/kg/min (norepinephrine as base)
urine output              mL
========================  ==================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: 1 mg/dL creatinine in umol/L (molar mass 113.12 g/mol).
CREATININE_UMOL_PER_MGDL = 88.4
#: 1 mg/dL bilirubin in umol/L (molar mass 584.66 g/mol).
BILIRUBIN_UMOL_PER_MGDL = 17.1
#: 1 kPa in mmHg. SOFA-2 Table 2 converts with 7.5 (300 mmHg = 40 kPa, 75 mmHg = 10 kPa);
#: the exact factor (7.50062) would move ratios charted in kPa off the published cut-offs.
MMHG_PER_KPA = 7.5

#: mg of salt equivalent to 1 mg of norepinephrine base (SOFA-2 Table 2, footnote k).
NOREPINEPHRINE_SALT_FACTORS = {
    "base": 1.0,
    "bitartrate_monohydrate": 2.0,
    "bitartrate_anhydrous": 1.89,
    "tartrate": 1.89,
    "hydrochloride": 1.22,
}


def creatinine_umol_to_mgdl(x):
    """Creatinine umol/L -> mg/dL."""
    return np.asarray(x, dtype=float) / CREATININE_UMOL_PER_MGDL


def bilirubin_umol_to_mgdl(x):
    """Total bilirubin umol/L -> mg/dL."""
    return np.asarray(x, dtype=float) / BILIRUBIN_UMOL_PER_MGDL


def kpa_to_mmhg(x):
    """Pressure kPa -> mmHg."""
    return np.asarray(x, dtype=float) * MMHG_PER_KPA


def fio2_to_fraction(x):
    """FiO2 given as a fraction or a percentage -> fraction.

    Values above 1 are taken as percentages (21-100) and divided by 100.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x > 1.0, x / 100.0, x)


def norepinephrine_to_base(dose, salt: str = "base"):
    """Norepinephrine dose expressed as a salt -> dose of base (footnote k)."""
    try:
        factor = NOREPINEPHRINE_SALT_FACTORS[salt]
    except KeyError as exc:
        raise ValueError(f"unknown norepinephrine salt {salt!r}") from exc
    return np.asarray(dose, dtype=float) / factor


# Multipliers that turn (rate in unit) into ug/kg/min; "per_kg" False means divide by weight.
_DOSE_UNITS: dict[str, tuple[float, bool]] = {
    "ug/kg/min": (1.0, True),
    "mcg/kg/min": (1.0, True),
    "mg/kg/min": (1000.0, True),
    "ug/kg/h": (1.0 / 60.0, True),
    "mcg/kg/h": (1.0 / 60.0, True),
    "mg/kg/h": (1000.0 / 60.0, True),
    "ug/min": (1.0, False),
    "mcg/min": (1.0, False),
    "mg/min": (1000.0, False),
    "ug/h": (1.0 / 60.0, False),
    "mcg/h": (1.0 / 60.0, False),
    "mg/h": (1000.0 / 60.0, False),
}


def dose_to_ug_kg_min(rate, unit: str, weight_kg=None):
    """Infusion rate in ``unit`` -> ug/kg/min.

    Args:
        rate: rate value(s).
        unit: one of the keys of ``_DOSE_UNITS`` (eg ``"mcg/kg/min"``, ``"mg/h"``).
        weight_kg: body weight, required for units that are not per kg.

    Raises:
        ValueError: unknown unit, or a weight is needed but missing.
    """
    key = _norm_unit(unit).replace("hour", "h").replace("/hr", "/h")
    if key not in _DOSE_UNITS:
        raise ValueError(f"unknown dose unit {unit!r}")
    factor, per_kg = _DOSE_UNITS[key]
    rate = np.asarray(rate, dtype=float) * factor
    if per_kg:
        return rate
    if weight_kg is None:
        raise ValueError(f"unit {unit!r} needs a body weight")
    weight = np.asarray(weight_kg, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(weight > 0, rate / weight, np.nan)


# (variable, unit) -> function to canonical. Canonical units map to identity.
_MEASUREMENT_UNITS = {
    ("creatinine", "mg/dl"): lambda x: np.asarray(x, dtype=float),
    ("creatinine", "umol/l"): creatinine_umol_to_mgdl,
    ("bilirubin", "mg/dl"): lambda x: np.asarray(x, dtype=float),
    ("bilirubin", "umol/l"): bilirubin_umol_to_mgdl,
    ("platelets", "k/ul"): lambda x: np.asarray(x, dtype=float),
    ("platelets", "10^3/ul"): lambda x: np.asarray(x, dtype=float),
    ("platelets", "10^9/l"): lambda x: np.asarray(x, dtype=float),
    ("pao2", "mmhg"): lambda x: np.asarray(x, dtype=float),
    ("pao2", "kpa"): kpa_to_mmhg,
    ("map", "mmhg"): lambda x: np.asarray(x, dtype=float),
    ("map", "kpa"): kpa_to_mmhg,
    ("fio2", "fraction"): lambda x: np.asarray(x, dtype=float),
    ("fio2", "%"): lambda x: np.asarray(x, dtype=float) / 100.0,
    ("spo2", "%"): lambda x: np.asarray(x, dtype=float),
    ("potassium", "mmol/l"): lambda x: np.asarray(x, dtype=float),
    ("potassium", "meq/l"): lambda x: np.asarray(x, dtype=float),
    ("bicarbonate", "mmol/l"): lambda x: np.asarray(x, dtype=float),
    ("bicarbonate", "meq/l"): lambda x: np.asarray(x, dtype=float),
    ("ph", "units"): lambda x: np.asarray(x, dtype=float),
    ("weight", "kg"): lambda x: np.asarray(x, dtype=float),
    ("weight", "lb"): lambda x: np.asarray(x, dtype=float) * 0.45359237,
}


def _norm_unit(unit: str) -> str:
    return str(unit).strip().lower().replace("µ", "u").replace("μ", "u").replace(" ", "")


def to_canonical(measurements: pd.DataFrame) -> pd.DataFrame:
    """Convert a measurements table with a ``unit`` column to canonical units.

    Rows whose ``unit`` is missing are assumed to be canonical already. The ``unit`` column is
    dropped from the result.

    Raises:
        ValueError: a (variable, unit) pair has no known conversion.
    """
    if "unit" not in measurements.columns:
        return measurements
    out = measurements.copy()
    units = out["unit"].map(lambda u: None if pd.isna(u) else _norm_unit(u))
    for (var, unit), idx in out.groupby([out["variable"], units], dropna=True).groups.items():
        fn = _MEASUREMENT_UNITS.get((var, unit))
        if fn is None:
            raise ValueError(f"no conversion for variable {var!r} in unit {unit!r}")
        out.loc[idx, "value"] = fn(out.loc[idx, "value"].to_numpy())
    return out.drop(columns="unit")
