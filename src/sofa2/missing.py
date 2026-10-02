"""Missing-data strategies for windowed organ scores.

Strategies (``pipeline.missing.strategy``):

``locf``
    A window without data for an organ takes the points of the last hour with data, if that hour
    is at most ``locf_max_hours`` before the window start (status ``carried_forward``);
    otherwise 0 (status ``imputed_normal``). On the first day there is nothing to carry, so a
    missing organ scores 0, as recommended in SOFA-2 Table 2 footnote b.
``normal``
    Missing organs score 0 (``imputed_normal``).
``none``
    Missing organs stay empty (``missing``) and so does the total.

Organs with data in the window have status ``observed``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sofa2.scoring.windows import last_observed

OBSERVED = "observed"
CARRIED = "carried_forward"
IMPUTED = "imputed_normal"
MISSING = "missing"


def apply_missing(win: pd.DataFrame, hourly: pd.DataFrame, organs, pcfg: dict) -> pd.DataFrame:
    """Fill missing organ scores in ``win`` and add ``<organ>_status`` columns.

    Args:
        win: windowed scores with stay_id, hr_first and one column per organ.
        hourly: hourly organ points (indexed by stay_id, hr) the windows were built from.
        organs: organ column names.
        pcfg: pipeline config.
    """
    strategy = pcfg["missing"]["strategy"]
    max_h = float(pcfg["missing"]["locf_max_hours"])
    w = win.copy()
    prev = pd.MultiIndex.from_arrays([w["stay_id"], w["hr_first"] - 1])
    for organ in organs:
        obs = w[organ].notna()
        status = np.where(obs, OBSERVED, MISSING).astype(object)
        vals = w[organ].to_numpy(dtype=float).copy()
        if strategy == "locf":
            lo = last_observed(hourly, organ).reindex(prev)
            last_hr = lo["last_hr"].to_numpy()
            last_val = lo["last_val"].to_numpy()
            with np.errstate(invalid="ignore"):
                carry = ~obs.to_numpy() & ~np.isnan(last_hr) & (
                    w["hr_first"].to_numpy() - last_hr <= max_h
                )
            vals[carry] = last_val[carry]
            status[carry] = CARRIED
        if strategy in ("locf", "normal"):
            rest = status == MISSING
            vals[rest] = 0.0
            status[rest] = IMPUTED
        w[organ] = vals
        w[f"{organ}_status"] = status
    return w
