"""End-to-end scores on the synthetic dataset, checked by hand against Table 2 / Vincent 1996."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sofa2.adapters.internal import InternalAdapter
from sofa2.config import load_config
from sofa2.pipeline import compute_scores

DATA = Path(__file__).parent / "data" / "synthetic"
ORG = ["respiratory", "cardiovascular", "brain", "liver", "kidney", "hemostasis"]

# (stay, window): SOFA-2 organs, SOFA-1 organs (in ORG order)
EXPECTED = {
    # day 1: PF 140 on IMV -> 3/3; NE 0.15 + vasopressin -> 3 (low dose + other) / NE >0.1 -> 4;
    # GCS 15 -> 0/0; bilirubin 2.5 (pre-ICU) -> 1/2; UO 0.375 mL/kg/h over 12 h -> 2,
    # creatinine 1.5 -> 1 (SOFA-1: 720 mL/day -> 0); platelets 90 -> 2/2
    (1, 0): ([3, 3, 0, 1, 2, 2], [3, 4, 0, 2, 1, 2]),
    # day 2: resp carried; MAP 75 -> 0; motor 5 with verbal not assessable -> 1 (SOFA-1: sedated
    # hours carry GCS 15 -> 0); liver and kidney carried; platelets 45 -> 4/3
    (1, 1): ([3, 0, 1, 1, 2, 4], [3, 0, 0, 2, 1, 3]),
    (2, 0): ([0] * 6, [0] * 6),
    (2, 1): ([0] * 6, [0] * 6),
    # SpO2:FiO2 230 without support -> 2 (SOFA-1: no PaO2 -> 0); dobutamine -> 2/2 (dopamine
    # 30 min not scored); creatinine 2.0 + K 6.2 meet RRT criteria -> 4 (SOFA-1: 2)
    (3, 0): ([2, 2, 0, 0, 4, 0], [0, 2, 0, 0, 2, 0]),
    # VA ECMO -> resp 4 and CV 4 (SOFA-1: none); GCS 7 -> 3/3; chronic RRT -> 4 (SOFA-1: none)
    (4, 0): ([4, 4, 3, 0, 4, 0], [0, 0, 3, 0, 0, 0]),
}


@pytest.fixture(scope="module")
def daily():
    return compute_scores(InternalAdapter(DATA))


def test_rows(daily):
    assert sorted(zip(daily.stay_id, daily.window_index)) == sorted(EXPECTED)


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_expected_scores(daily, key):
    row = daily.set_index(["stay_id", "window_index"]).loc[key]
    s2, s1 = EXPECTED[key]
    assert [row[f"sofa2_{o}"] for o in ORG] == s2
    assert [row[f"sofa1_{o}"] for o in ORG] == s1
    assert row["sofa2_total"] == sum(s2)
    assert row["sofa1_total"] == sum(s1)


def test_statuses(daily):
    r = daily.set_index(["stay_id", "window_index"])
    assert r.loc[(1, 1), "sofa2_respiratory_status"] == "carried_forward"
    assert r.loc[(1, 1), "sofa2_cardiovascular_status"] == "observed"
    assert r.loc[(2, 0), "sofa2_brain_status"] == "imputed_normal"
    assert r.loc[(3, 0), "sofa2_kidney_status"] == "observed"


def test_window_bounds(daily):
    r = daily.set_index(["stay_id", "window_index"])
    assert r.loc[(2, 1), "window_start"] == pd.Timestamp("2100-01-02 00:00")
    assert r.loc[(2, 1), "window_end"] == pd.Timestamp("2100-01-02 06:00")


def test_strategy_normal(daily):
    cfg = load_config(overrides={"pipeline": {"missing": {"strategy": "normal"}}})
    r = compute_scores(InternalAdapter(DATA), config=cfg).set_index(["stay_id", "window_index"])
    assert r.loc[(1, 1), "sofa2_respiratory"] == 0
    assert r.loc[(1, 1), "sofa2_respiratory_status"] == "imputed_normal"


def test_strategy_none():
    cfg = load_config(overrides={"pipeline": {"missing": {"strategy": "none"}}})
    r = compute_scores(InternalAdapter(DATA), config=cfg).set_index(["stay_id", "window_index"])
    assert np.isnan(r.loc[(1, 1), "sofa2_respiratory"])
    assert np.isnan(r.loc[(1, 1), "sofa2_total"])
    assert r.loc[(1, 1), "sofa2_respiratory_status"] == "missing"
    assert r.loc[(1, 0), "sofa2_total"] == 11


def test_locf_limit():
    cfg = load_config(overrides={"pipeline": {"missing": {"locf_max_hours": 12}}})
    r = compute_scores(InternalAdapter(DATA), config=cfg).set_index(["stay_id", "window_index"])
    # last PaO2 at hour 2, day 2 starts at hour 24: 22 h > 12 h -> not carried
    assert r.loc[(1, 1), "sofa2_respiratory"] == 0
    assert r.loc[(1, 1), "sofa2_respiratory_status"] == "imputed_normal"
    # kidney: last hourly value at hour 23 -> carried
    assert r.loc[(1, 1), "sofa2_kidney_status"] == "carried_forward"


def test_hourly_matches_daily_at_end_of_day(daily):
    hourly = compute_scores(InternalAdapter(DATA), freq="hourly")
    assert len(hourly) == 48 + 30 + 24 + 24
    h = hourly.set_index(["stay_id", "window_index"])
    d = daily.set_index(["stay_id", "window_index"])
    # the hourly window ending at hour 23 covers the same 24 hours as day 1
    for stay in (1, 3, 4):
        for o in ORG:
            assert h.loc[(stay, 23), f"sofa2_{o}"] == d.loc[(stay, 0), f"sofa2_{o}"]
            assert h.loc[(stay, 23), f"sofa1_{o}"] == d.loc[(stay, 0), f"sofa1_{o}"]


def test_hourly_early_hours():
    h = compute_scores(InternalAdapter(DATA), freq="hourly").set_index(["stay_id", "window_index"])
    # hour 0 of stay 1: only bilirubin (pre-ICU) and weight -> liver 1, rest imputed 0
    assert h.loc[(1, 0), "sofa2_liver"] == 1
    assert h.loc[(1, 0), "sofa2_respiratory_status"] == "imputed_normal"
    # hour 2: PaO2 140 on IMV -> 3
    assert h.loc[(1, 2), "sofa2_respiratory"] == 3


def test_include_values():
    r = compute_scores(InternalAdapter(DATA), include_values=True).set_index(["stay_id", "window_index"])
    assert r.loc[(1, 0), "sofa2_pf_min"] == pytest.approx(140)
    assert r.loc[(3, 0), "sofa2_sf_min"] == pytest.approx(230)
    assert bool(r.loc[(3, 0), "sofa2_rrt_criteria_met"])
    assert r.loc[(1, 0), "sofa1_uo_ml_day"] == pytest.approx(720)


def test_single_score():
    r = compute_scores(InternalAdapter(DATA), scores=["sofa2"])
    assert not any(c.startswith("sofa1") for c in r.columns)
