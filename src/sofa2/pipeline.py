"""End-to-end scoring: internal schema in, one row per stay per window out."""

from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np
import pandas as pd

from sofa2.config import Config, load_config
from sofa2.derive.timegrid import HOUR
from sofa2.missing import apply_missing
from sofa2.schema import ICUData
from sofa2.scoring.engine import ORGANS, hourly_sofa1, hourly_sofa2, shared_inputs
from sofa2.scoring.windows import aggregate, apply_window_rules

SCORES = {"sofa2": hourly_sofa2, "sofa1": hourly_sofa1}
_KEYS = ["stay_id", "window", "hr_first", "hr_last"]


class Loadable(Protocol):
    """Anything with a ``load()`` returning :class:`ICUData` (eg an adapter)."""

    def load(self) -> ICUData: ...


def compute_scores(
    data: ICUData | Loadable,
    freq: str = "daily",
    config: Config | None = None,
    scores: Iterable[str] = ("sofa2", "sofa1"),
    include_values: bool = False,
) -> pd.DataFrame:
    """Compute SOFA-2 and/or original SOFA scores.

    Args:
        data: an :class:`ICUData` or an adapter with ``load()``.
        freq: ``"daily"`` (24-hour blocks from ICU admission) or ``"hourly"`` (each hour scored
            over the preceding 24 hours).
        config: rules and settings; defaults to :func:`sofa2.config.load_config`.
        scores: which scores to compute (``"sofa2"``, ``"sofa1"``).
        include_values: add the worst raw values behind the scores (eg ``sofa2_pf_min``).

    Returns:
        One row per stay per window with columns ``stay_id``, ``window_index`` (0-based),
        ``window_start``, ``window_end``, and per score ``<score>_total``,
        ``<score>_<organ>`` and ``<score>_<organ>_status`` for the six organs
        (respiratory, cardiovascular, brain, liver, kidney, hemostasis). Status is one of
        observed, carried_forward, imputed_normal, missing.
    """
    if not isinstance(data, ICUData):
        data = data.load()
    cfg = config or load_config()
    data = data.validate()
    shared = shared_inputs(data, cfg)
    result = None
    for name in scores:
        if name not in SCORES:
            raise ValueError(f"unknown score {name!r}; choose from {sorted(SCORES)}")
        score_cfg = getattr(cfg, name)
        hourly = SCORES[name](data, cfg, shared)
        win = aggregate(hourly, freq, cfg.pipeline)
        win = apply_window_rules(win, name, score_cfg)
        win = apply_missing(win, hourly[list(ORGANS)], ORGANS, cfg.pipeline)
        organ_vals = win[list(ORGANS)].to_numpy(dtype=float)
        total = organ_vals.sum(axis=1)  # NaN if any organ is missing (strategy "none")
        cols = {k: win[k] for k in _KEYS}
        cols[f"{name}_total"] = total
        for o in ORGANS:
            cols[f"{name}_{o}"] = win[o]
        for o in ORGANS:
            cols[f"{name}_{o}_status"] = win[f"{o}_status"]
        if include_values:
            extra = [c for c in hourly.columns if c not in ORGANS]
            if name == "sofa2":
                extra.append("rrt_criteria_met")
            for c in extra:
                if c in win:
                    cols[f"{name}_{c}"] = win[c]
        part = pd.DataFrame(cols)
        result = part if result is None else result.merge(part, on=_KEYS, how="outer")
    if result is None:
        raise ValueError("no scores requested")

    stays = data.stays.set_index("stay_id")
    intime = result["stay_id"].map(stays["intime"])
    outtime = result["stay_id"].map(stays["outtime"])
    result.insert(2, "window_start", intime + result["hr_first"] * HOUR)
    end = intime + (result["hr_last"] + 1) * HOUR
    result.insert(3, "window_end", end.where(end <= outtime, outtime))
    result = result.rename(columns={"window": "window_index"}).drop(columns=["hr_first", "hr_last"])
    for c in result.columns:
        if c.endswith("_total") or any(c.endswith(f"_{o}") for o in ORGANS):
            if result[c].notna().all() and np.allclose(result[c], result[c].round()):
                result[c] = result[c].astype(int)
    return result.sort_values(["stay_id", "window_index"]).reset_index(drop=True)
