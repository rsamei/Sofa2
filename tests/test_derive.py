"""Derivation rules: pairing, transient readings, infusion duration, urine windows, sedation, RRT."""

import numpy as np
import pandas as pd
import pytest

from sofa2.config import load_config
from sofa2.derive.infusions import hourly_rates, valid_infusions
from sofa2.derive.oxygenation import drop_transient, ratio_readings
from sofa2.derive.rrt import rrt_hourly
from sofa2.derive.timegrid import assign_hour, build_grid, overlap_hours
from sofa2.derive.urine import anuria, urine_windows, weight_at
from sofa2.pipeline import compute_scores
from sofa2.schema import ICUData

T0 = pd.Timestamp("2100-01-01")
CFG = load_config()


def t(h):
    return T0 + pd.Timedelta(hours=h)


def stays(hours=24, **kw):
    d = dict(stay_id=[1], patient_id=[1], intime=[T0], outtime=[t(hours)])
    d.update({k: [v] for k, v in kw.items()})
    return pd.DataFrame(d)


def meas(rows):
    return pd.DataFrame(rows, columns=["stay_id", "time", "variable", "value"])


def score_day1(**tables):
    st = tables.pop("stays", stays())
    d = ICUData(stays=st, **tables)
    return compute_scores(d, include_values=True).iloc[0]


# ------------------------------------------------------------------ time grid
def test_grid_and_assign():
    s = ICUData(stays=stays(2.5)).validate().stays
    g = build_grid(s)
    assert g["hr"].tolist() == [0, 1, 2]
    ev = pd.DataFrame({"stay_id": [1, 1, 1, 1], "time": [t(-1), t(0), t(2.5), t(3)]})
    a = assign_hour(ev, s)
    assert a["hr"].tolist() == [0, 2]  # before admission and after discharge dropped
    a = assign_hour(ev, s, pre_hours=2)
    assert a["hr"].tolist() == [0, 0, 2]


def test_overlap_hours():
    s = ICUData(stays=stays(5)).validate().stays
    iv = pd.DataFrame({"stay_id": [1, 1], "start": [t(0.5), t(3)], "end": [t(2), t(3)]})
    assert overlap_hours(iv, s)["hr"].tolist() == [0, 1, 3]


# ------------------------------------------------------------------ oxygenation
def test_pf_pairs_specimen_then_lookback():
    m = pd.DataFrame(
        [
            (1, t(1), "fio2", 0.8, None),
            (1, t(2), "pao2", 80, "a"), (1, t(2), "fio2", 0.4, "a"),  # same specimen -> 200
            (1, t(5), "pao2", 80, None),                               # last FiO2 at 2 h -> 200
            (1, t(10), "pao2", 80, None),                              # FiO2 > 4 h old -> none
        ],
        columns=["stay_id", "time", "variable", "value", "specimen_id"],
    )
    r = ratio_readings(m, CFG.pipeline)
    assert r[r.kind == "pf"]["ratio"].tolist() == [200, 200]


def test_sf_lookback_and_98_rule():
    m = meas([(1, t(1), "fio2", 0.5), (1, t(1.5), "spo2", 95), (1, t(3), "spo2", 90),
              (1, t(5), "fio2", 0.5), (1, t(5.5), "spo2", 98)])
    r = score_day1(measurements=m)
    # SpO2 95 / 0.5 = 190 without support -> 2; SpO2 90 at 3 h has no FiO2 within 60 min;
    # SpO2 98 is not used
    assert r["sofa2_respiratory"] == 2
    assert r["sofa2_sf_min"] == pytest.approx(190)


def test_sf_not_used_in_hour_with_pf():
    m = meas([(1, t(1), "fio2", 0.5), (1, t(1.2), "pao2", 200), (1, t(1.5), "spo2", 80)])
    r = score_day1(measurements=m)
    assert r["sofa2_respiratory"] == 0  # PF 400; SF 160 ignored in the same hour


def test_transient_reading_dropped_footnote_g():
    r = pd.DataFrame({"stay_id": 1, "kind": "pf", "time": [t(1), t(1.5), t(3), t(4.5)],
                      "points": [3, 1, 3, 3]})
    kept = drop_transient(r, 60)
    assert kept["time"].tolist() == [t(1.5), t(3), t(4.5)]


def test_transient_rule_in_pipeline():
    sup = pd.DataFrame({"stay_id": [1], "start": [t(0)], "end": [t(24)], "type": ["imv"]})
    m = meas([(1, t(1), "fio2", 0.5), (1, t(2), "pao2", 60), (1, t(2.5), "pao2", 160)])
    # PF 120 (3 points) improves to 320 (0 points) within 60 min -> 120 not considered
    assert score_day1(measurements=m, support=sup)["sofa2_respiratory"] == 0
    m = meas([(1, t(1), "fio2", 0.5), (1, t(2), "pao2", 60), (1, t(3.5), "pao2", 160)])
    assert score_day1(measurements=m, support=sup)["sofa2_respiratory"] == 3


def test_hfnc_counts_as_advanced_support_only_for_sofa2():
    sup = pd.DataFrame({"stay_id": [1], "start": [t(0)], "end": [t(24)], "type": ["hfnc"]})
    m = meas([(1, t(1), "fio2", 0.5), (1, t(2), "pao2", 70)])  # PF 140
    r = score_day1(measurements=m, support=sup)
    assert r["sofa2_respiratory"] == 3
    assert r["sofa1_respiratory"] == 2  # <200 but HFNC is not respiratory support in SOFA-1 config


# ------------------------------------------------------------------ infusions
def inf(rows):
    return pd.DataFrame(rows, columns=["stay_id", "start", "end", "drug", "rate"])


def test_duration_rule_with_merging():
    rows = inf([
        (1, t(1), t(1.5), "norepinephrine", 0.1),
        (1, t(1.5), t(2.1), "norepinephrine", 0.2),   # contiguous -> one 66-min infusion
        (1, t(5), t(5.5), "epinephrine", 0.1),        # 30 min -> dropped
        (1, t(8), t(8.4), "dopamine", 5),
        (1, t(8.5), t(9.0), "dopamine", 5),           # 6-min gap -> two short infusions, dropped
    ])
    v = valid_infusions(rows, 60, 1)
    assert v["drug"].tolist() == ["norepinephrine", "norepinephrine"]


def test_concurrent_sum_not_sum_of_maxima():
    s = ICUData(stays=stays()).validate().stays
    rows = inf([
        (1, t(1), t(1.5), "norepinephrine", 0.3),
        (1, t(1.5), t(2), "epinephrine", 0.3),
        (1, t(3), t(4), "norepinephrine", 0.15),
        (1, t(3), t(4), "epinephrine", 0.1),
    ])
    h = hourly_rates(rows, s, ["norepinephrine", "epinephrine"],
                     {"ne_epi": ["norepinephrine", "epinephrine"]})
    assert h.loc[(1, 1), "ne_epi"] == pytest.approx(0.3)   # never concurrent
    assert h.loc[(1, 3), "ne_epi"] == pytest.approx(0.25)


def test_cv_in_pipeline():
    rows = inf([(1, t(1), t(3), "norepinephrine", 0.15), (1, t(1), t(3), "epinephrine", 0.1)])
    r = score_day1(infusions=rows)
    assert r["sofa2_cardiovascular"] == 3      # sum 0.25 -> medium dose
    assert r["sofa1_cardiovascular"] == 4      # NE 0.15 > 0.1


# ------------------------------------------------------------------ urine output
def urine(times, vol):
    return pd.DataFrame({"stay_id": 1, "time": [t(h) for h in times], "volume_ml": vol})


def test_weight_fallbacks():
    s = ICUData(stays=stays(weight_kg=90)).validate().stays
    m = meas([(1, t(5), "weight", 80), (1, t(10), "weight", 82)])
    times = pd.DataFrame({"stay_id": 1, "time": [t(1), t(6), t(12)]})
    np.testing.assert_allclose(weight_at(times, s, m, CFG.pipeline), [80, 80, 82])
    np.testing.assert_allclose(weight_at(times, s, m.iloc[0:0], CFG.pipeline), [90, 90, 90])


def test_urine_coverage_rule():
    s = ICUData(stays=stays(weight_kg=100)).validate().stays
    u = urine([2, 4, 6], [40, 40, 40]).astype({"volume_ml": float})
    uw = urine_windows(u, s, meas([]), CFG.pipeline)
    # at the end of hour 5 (6:00): 120 mL over 6 h / 100 kg = 0.2 mL/kg/h
    assert uw.loc[(1, 5), "rate_6"] == pytest.approx(0.2)
    # end of hour 6 (7:00): charts at 2, 4, 6 h are in (1, 7]; their intervals sum to 6 h -> ok
    assert uw.loc[(1, 6), "rate_6"] == pytest.approx(0.2)
    # end of hour 7 (8:00): only charts at 4 and 6 h in (2, 8]: 4 h of coverage -> not scored
    assert uw.loc[(1, 7), "cov_6"] == pytest.approx(4.0)
    assert np.isnan(uw.loc[(1, 7), "rate_6"])
    assert np.isnan(uw.loc[(1, 5), "rate_12"])


def test_anuria():
    s = ICUData(stays=stays(weight_kg=100)).validate().stays
    u = urine([4, 8, 12], [0.0, 0.0, 0.0])
    uw = urine_windows(u, s, meas([]), CFG.pipeline, windows=(12,))
    an = anuria(uw, 12, 2)
    assert bool(an.loc[(1, 11)])
    assert not bool(an.loc[(1, 10)])


def test_urine_in_pipeline_sofa1_daily_volume():
    u = urine(range(1, 25), [8.0] * 24)  # 192 mL/day
    r = score_day1(urine_output=u, stays=stays(weight_kg=100))
    assert r["sofa1_kidney"] == 4
    assert r["sofa2_kidney"] == 3  # 0.08 mL/kg/h over 24 h < 0.3


# ------------------------------------------------------------------ brain
def gcs(rows):
    return pd.DataFrame(rows, columns=["stay_id", "time", "eye", "verbal", "motor", "total",
                                       "unassessable"])


def test_presedation_gcs_carried():
    g = gcs([(1, t(1), 3, 4, 5, 12, False), (1, t(5), 1, 1, 1, 3, False)])
    i = inf([(1, t(3), t(20), "propofol", 20)])
    r = score_day1(gcs=g, infusions=i)
    assert r["sofa2_brain"] == 2  # GCS 12 before sedation; GCS 3 during propofol ignored


def test_sedation_without_previous_gcs_scores_0():
    g = gcs([(1, t(5), 1, 1, 1, 3, False)])
    i = inf([(1, t(3), t(20), "propofol", 20)])
    r = score_day1(gcs=g, infusions=i)
    assert r["sofa2_brain"] == 0
    assert r["sofa2_brain_status"] == "imputed_normal"


def test_worst_complete_vs_best_motor_in_hour():
    g = gcs([(1, t(1), 4, 5, 6, 15, False), (1, t(1.2), 1, None, 4, None, True),
             (1, t(1.5), 1, None, 5, None, True)])
    r = score_day1(gcs=g)
    assert r["sofa2_brain"] == 1  # best motor 5 -> 1, complete 15 -> 0, worse of the two
    assert r["sofa1_brain"] == 0


def test_delirium_with_gcs_15():
    g = gcs([(1, t(1), 4, 5, 6, 15, False)])
    med = pd.DataFrame({"stay_id": [1], "time": [t(2)], "drug": ["haloperidol"]})
    r = score_day1(gcs=g, medications=med)
    assert r["sofa2_brain"] == 1
    assert r["sofa1_brain"] == 0


# ------------------------------------------------------------------ RRT
def test_intermittent_rrt_carry_footnote_q():
    s = ICUData(stays=stays(120)).validate().stays
    sup = pd.DataFrame({"stay_id": [1], "start": [t(2)], "end": [t(6)], "type": ["rrt_intermittent"]})
    f = rrt_hourly(sup, s, CFG.sofa2["kidney"])
    assert f.loc[(1, 1)] == False  # noqa: E712
    assert f.loc[(1, 2)] and f.loc[(1, 77)]
    assert f.loc[(1, 78)] == False  # noqa: E712  (6 h + 72 h)


def test_rrt_criteria_need_same_window():
    m = meas([(1, t(3), "creatinine", 2.5), (1, t(30), "potassium", 6.5)])
    d = ICUData(stays=stays(48), measurements=m)
    r = compute_scores(d).set_index("window_index")
    assert r.loc[0, "sofa2_kidney"] == 2
    assert r.loc[1, "sofa2_kidney"] == 2  # creatinine carried; K alone does not meet criteria


def test_pre_icu_lab_window():
    m = meas([(1, t(-5), "creatinine", 4.0), (1, t(-7), "bilirubin", 13.0)])
    r = score_day1(measurements=m)
    assert r["sofa2_kidney"] == 3   # 5 h before admission -> day 1
    assert r["sofa2_liver"] == 0    # 7 h before -> outside the 6-h lookback


def test_incomplete_gcs_without_flag_ignored():
    g = gcs([(1, t(1), None, None, 3, None, False)])
    r = score_day1(gcs=g)
    assert r["sofa2_brain"] == 0
    assert r["sofa2_brain_status"] == "imputed_normal"
