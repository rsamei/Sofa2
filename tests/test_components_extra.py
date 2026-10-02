"""Edge cases: missing flags, broadcasting, NaN doses, gap values, combined inputs."""

import numpy as np
import pandas as pd
import pytest

from sofa2.scoring.components import (
    _bool,
    brain,
    cardiovascular_sofa1,
    cardiovascular_sofa2,
    fmax,
    gcs_points,
    kidney_sofa1,
    kidney_sofa2,
    liver,
    ratio_points,
    respiratory,
    rrt_criteria_met,
    urine_points_sofa2,
)


def eq(a, b):
    np.testing.assert_array_equal(np.asarray(a, float), np.asarray(b, float))


# ---------------------------------------------------------------- helpers
def test_bool_missing_is_false():
    eq(_bool(pd.Series([True, False, np.nan]).to_numpy(), (3,)), [1, 0, 0])
    eq(_bool(pd.array([True, pd.NA, False], dtype="boolean"), (3,)), [1, 0, 0])
    eq(_bool(np.array([1.0, np.nan, 0.0]), (3,)), [1, 0, 0])


def test_fmax():
    eq(fmax([1, np.nan, np.nan], [np.nan, 2, np.nan])[:2], [1, 2])
    assert np.isnan(fmax([np.nan], [np.nan])[0])
    eq(fmax(1.0, [0, 2]), [1, 2])


# ---------------------------------------------------------------- missing flags never score
def test_missing_flags_do_not_score(s2):
    flags = pd.Series([np.nan], dtype=object).to_numpy()
    assert respiratory([0.0], ecmo=flags, cfg=s2["respiratory"])[0] == 0
    assert kidney_sofa2([1.0], [0.0], flags, s2["kidney"])[0] == 0
    assert cardiovascular_sofa2(0, 0, flags, flags, 80, s2["cardiovascular"])[0] == 0
    assert brain([0.0], delirium=flags, cfg=s2["brain"])[0] == 0
    r = s2["respiratory"]
    assert ratio_points(100, flags, r["pf_bands"], r)[0] == 2


# ---------------------------------------------------------------- broadcasting
def test_scalar_main_input_with_array_flags(s2):
    c = s2["cardiovascular"]
    eq(cardiovascular_sofa2(0, 0, [True, False, True], False, 80, c), [2, 0, 2])
    eq(gcs_points(15, motor=[6, 3], assessable=[True, False], cfg=s2["brain"]), [0, 3])
    eq(brain(0.0, delirium=[True, False], cfg=s2["brain"]), [1, 0])
    r = s2["respiratory"]
    eq(ratio_points(100, [True, False], r["pf_bands"], r), [3, 2])
    eq(respiratory(1.0, ecmo=[True, False], cfg=r), [4, 1])
    eq(urine_points_sofa2({6: 0.6, 12: 0.6, 24: 0.6}, anuria=[True, False], cfg=s2["kidney"]), [3, 0])
    eq(rrt_criteria_met(2.0, [True, False], 6.5, 7.4, 24, s2["kidney"]), [1, 1])


# ---------------------------------------------------------------- cardiovascular SOFA-2
CV2 = [
    (0.1, 0, False, False, np.nan, 2),     # NaN MAP with a drug -> drug points
    (np.nan, np.nan, False, False, 65, 1),  # NaN doses mean no drug
    (0, 0, True, False, 55, 2),            # other agent with MAP < 70 -> 2
    (0, 0.01, False, False, 80, 2),        # tiny dopamine dose -> 2
    (0.3, 2, False, False, 80, 4),         # medium NE + dopamine (other) -> 4
    (0, 20, True, False, 80, 3),           # dopamine + other: +1 at the band edges, cap 4
    (0, 20.1, True, False, 80, 4),
    (0, 40, True, False, 80, 4),
    (0, 40.1, True, False, 80, 4),
    (0.1, 0, True, True, 80, 4),           # mechanical support with drugs -> 4
]


@pytest.mark.parametrize("ne,dopa,other,mech,mapv,expected", CV2)
def test_cv2_extra(s2, ne, dopa, other, mech, mapv, expected):
    assert cardiovascular_sofa2(ne, dopa, other, mech, mapv, s2["cardiovascular"])[0] == expected


def test_cv2_fallback_with_drug_uses_drug_points(s2):
    # footnote m applies only when vasoactive drugs are unavailable or precluded
    c = dict(s2["cardiovascular"], map_only_fallback=True)
    assert cardiovascular_sofa2(0.1, 0, False, False, 35, c)[0] == 2


# ---------------------------------------------------------------- cardiovascular original SOFA
CV1 = [
    (0.001, 0, 0, 0, 80, 3),
    (0, 0.001, 0, 0, 80, 3),
    (0, 0, 0.001, 0, 80, 2),
    (0, 0, 6, 3, 80, 3),          # dobutamine + dopamine >5 -> highest single drug
    (np.nan, np.nan, np.nan, np.nan, 65, 1),
    (0.2, 0, 0, 0, np.nan, 4),
]


@pytest.mark.parametrize("ne,epi,dopa,dobu,mapv,expected", CV1)
def test_cv1_extra(s1, ne, epi, dopa, dobu, mapv, expected):
    assert cardiovascular_sofa1(ne, epi, dopa, dobu, mapv, s1["cardiovascular"])[0] == expected


# ---------------------------------------------------------------- respiratory
def test_resp_nan_ratio(s2):
    r = s2["respiratory"]
    assert np.isnan(ratio_points(np.nan, True, r["pf_bands"], r)[0])


@pytest.mark.parametrize("ratio,points", [(301, 0), (300, 1), (251, 1), (250, 2)])
def test_sf_without_support(s2, ratio, points):
    r = s2["respiratory"]
    assert ratio_points(ratio, False, r["sf_bands"], r)[0] == points


def test_pf_without_support_301(s2):
    r = s2["respiratory"]
    assert ratio_points(301, False, r["pf_bands"], r)[0] == 0


def test_ecmo_combinations(s2, s1):
    eq(respiratory([2.0, np.nan, 1.0], ecmo=[True, True, False], cfg=s2["respiratory"])[:3], [4, 4, 1])
    # original SOFA has no ECMO rule
    assert respiratory([1.0], ecmo=[True], cfg=s1["respiratory"])[0] == 1


def test_support_array_with_nan(s2):
    r = s2["respiratory"]
    eq(ratio_points([100, 100], [np.nan, 1.0], r["pf_bands"], r), [2, 3])


# ---------------------------------------------------------------- brain
def test_brain_extra(s2):
    b = s2["brain"]
    assert np.isnan(gcs_points(np.nan, motor=np.nan, assessable=False, cfg=b)[0])
    assert gcs_points(15, motor=1, assessable=True, cfg=b)[0] == 0
    eq(gcs_points([15, 3, 15], motor=[6, 6, 4], assessable=[True, True, False], cfg=b), [0, 4, 2])
    eq(brain([0, 2, 4, 1], delirium=[True, True, False, False], cfg=b), [1, 2, 4, 1])


# ---------------------------------------------------------------- kidney
def test_urine_partial_windows(s2):
    k = s2["kidney"]
    assert urine_points_sofa2({6: np.nan, 12: np.nan, 24: 0.2}, cfg=k)[0] == 3
    assert urine_points_sofa2({6: np.nan, 12: 0.4, 24: np.nan}, cfg=k)[0] == 2
    eq(urine_points_sofa2({6: [np.nan, 0.6], 12: [np.nan, 0.6], 24: [np.nan, 0.6]},
                          anuria=[False, True], cfg=k)[1:], [3])
    assert np.isnan(urine_points_sofa2({6: [np.nan, 0.6], 12: np.nan, 24: np.nan},
                                       anuria=[False, True], cfg=k)[0])


def test_rrt_flag_float_nan(s2):
    eq(kidney_sofa2([1.0, 1.0], [0, 0], np.array([np.nan, 1.0]), s2["kidney"]), [0, 4])


def test_rrt_criteria_oliguria_nan(s2):
    eq(rrt_criteria_met([1.0, 1.0], np.array([np.nan, 1.0]), [6.5, 6.5], 7.4, 24, s2["kidney"]), [0, 1])


def test_sofa1_creatinine_above_5(s1):
    assert kidney_sofa1(5.01, np.nan, s1["kidney"])[0] == 4


# ---------------------------------------------------------------- liver gap values (half-open)
@pytest.mark.parametrize("value,points", [(1.95, 1), (5.95, 2), (11.95, 3)])
def test_sofa1_liver_gap_values(s1, value, points):
    assert liver(value, s1["liver"])[0] == points
