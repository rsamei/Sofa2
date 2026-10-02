"""Continuous infusions: episode duration rule and hourly rates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sofa2.derive.timegrid import overlap_hours


def valid_infusions(
    infusions: pd.DataFrame, min_minutes: float, merge_gap_minutes: float
) -> pd.DataFrame:
    """Keep infusion rows that belong to a continuous infusion lasting ``min_minutes`` or more.

    Rows with rate > 0 of the same stay and drug form one continuous infusion when each row
    starts within ``merge_gap_minutes`` of the latest end so far (SOFA-2 footnote j; original
    SOFA Table 3 footnote).
    """
    inf = infusions[infusions["rate"] > 0].sort_values(["stay_id", "drug", "start", "end"])
    if inf.empty:
        return inf
    gap = pd.Timedelta(minutes=merge_gap_minutes)
    prev_end = inf.groupby(["stay_id", "drug"])["end"].transform(
        lambda s: s.cummax().shift()
    )
    new = prev_end.isna() | (inf["start"] > prev_end + gap)
    episode = new.cumsum()
    dur = inf.groupby(episode)["end"].transform("max") - inf.groupby(episode)["start"].transform("min")
    return inf.loc[dur >= pd.Timedelta(minutes=min_minutes)]


def _segments(inf: pd.DataFrame, drugs) -> pd.DataFrame:
    """Split infusion rows into elementary segments with each drug's rate.

    Segment boundaries are all row start and end times of the stay. Where rows of the same drug
    overlap, the highest rate is used.

    Returns:
        DataFrame with stay_id, start, end and one rate column per drug.
    """
    codes, uniq = pd.factorize(inf["stay_id"])
    origin = inf["start"].min()
    sec_s = ((inf["start"] - origin).dt.total_seconds()).to_numpy().astype(np.int64)
    sec_e = ((inf["end"] - origin).dt.total_seconds()).to_numpy().astype(np.int64)
    big = np.int64(max(sec_e.max(), 0) + 1)
    keys = np.unique(np.concatenate([codes * big + sec_s, codes * big + sec_e]))
    k_code, k_sec = keys // big, keys % big
    same = k_code[:-1] == k_code[1:]
    seg_lo, seg_hi = keys[:-1][same], keys[1:][same]
    seg_code, seg_s, seg_e = k_code[:-1][same], k_sec[:-1][same], k_sec[1:][same]
    # segments covered by each row: those with seg_lo >= row start and seg_hi <= row end
    a = np.searchsorted(seg_lo, codes * big + sec_s, side="left")
    b = np.searchsorted(seg_hi, codes * big + sec_e, side="right")
    count = np.clip(b - a, 0, None)
    row_idx = np.repeat(np.arange(len(inf)), count)
    seg_idx = np.repeat(a, count) + (np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count))
    cov = pd.DataFrame({"seg": seg_idx, "drug": inf["drug"].to_numpy()[row_idx],
                        "rate": inf["rate"].to_numpy()[row_idx]})
    wide = cov.pivot_table(index="seg", columns="drug", values="rate", aggfunc="max")
    seg = pd.DataFrame({
        "stay_id": uniq[seg_code],
        "start": origin + pd.to_timedelta(seg_s, unit="s"),
        "end": origin + pd.to_timedelta(seg_e, unit="s"),
    })
    for drug in drugs:
        col = wide[drug] if drug in wide else pd.Series(dtype=float)
        seg[drug] = col.reindex(np.arange(len(seg))).fillna(0.0).to_numpy()
    return seg


def hourly_rates(
    infusions: pd.DataFrame, stays: pd.DataFrame, drugs, sums: dict[str, list[str]] | None = None
) -> pd.DataFrame:
    """Hourly maximum rate of each drug and of concurrent sums of drugs.

    Args:
        infusions: valid infusion rows (see :func:`valid_infusions`).
        stays: stays table.
        drugs: drugs to report.
        sums: name -> drugs whose concurrent rates are added before taking the hourly maximum
            (eg ``{"ne_epi": ["norepinephrine", "epinephrine"]}``).

    Returns:
        DataFrame indexed by (stay_id, hr) with one column per drug and per sum; hours without
        any infusion are absent.
    """
    drugs = list(drugs)
    sums = sums or {}
    inf = infusions[infusions["drug"].isin(drugs) & (infusions["end"] > infusions["start"])]
    cols = drugs + list(sums)
    if inf.empty:
        return pd.DataFrame(columns=cols, index=pd.MultiIndex.from_arrays([[], []], names=["stay_id", "hr"]))
    seg = _segments(inf, drugs)
    for name, members in sums.items():
        seg[name] = seg[members].sum(axis=1)
    seg = seg[seg[cols].gt(0).any(axis=1)]
    hours = overlap_hours(seg, stays)
    return hours.groupby(["stay_id", "hr"])[cols].max()


def active_hours(intervals: pd.DataFrame, stays: pd.DataFrame) -> pd.Index:
    """(stay_id, hr) pairs overlapped by any interval."""
    h = overlap_hours(intervals, stays)
    return pd.MultiIndex.from_frame(h[["stay_id", "hr"]].drop_duplicates())
