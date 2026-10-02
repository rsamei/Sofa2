import numpy as np
import pytest

from sofa2.scoring.components import (
    kidney_sofa1,
    kidney_sofa2,
    rrt_criteria_met,
    urine_points_sofa2,
)

# SOFA-2 creatinine: <=1.20 -> 0; <=2.0 -> 1; <=3.50 -> 2; >3.50 -> 3
SOFA2_CR = [(1.2, 0), (1.21, 1), (2.0, 1), (2.01, 2), (3.5, 2), (3.51, 3), (8.0, 3)]


@pytest.mark.parametrize("cr,points", SOFA2_CR)
def test_sofa2_creatinine(s2, cr, points):
    assert kidney_sofa2(cr, np.nan, False, s2["kidney"])[0] == points


def test_sofa2_rrt_scores_4(s2):
    assert kidney_sofa2(np.nan, np.nan, True, s2["kidney"])[0] == 4
    assert kidney_sofa2(0.8, 0, True, s2["kidney"])[0] == 4


def test_sofa2_missing_is_nan(s2):
    assert np.isnan(kidney_sofa2(np.nan, np.nan, False, s2["kidney"])[0])


# (rate6, rate12, rate24, anuria12, expected) -- Table 2:
# <0.5 for 6-12 h -> 1; <0.5 for >=12 h -> 2; <0.3 for >=24 h -> 3; anuria >=12 h -> 3
UO = [
    (0.5, 0.5, 0.5, False, 0),
    (0.49, 0.6, 0.6, False, 1),
    (0.49, 0.5, 0.6, False, 1),
    (0.49, 0.49, 0.6, False, 2),
    (0.2, 0.4, 0.3, False, 2),
    (0.2, 0.2, 0.29, False, 3),
    (0.0, 0.0, np.nan, True, 3),
    (np.nan, np.nan, np.nan, False, np.nan),
]


@pytest.mark.parametrize("r6,r12,r24,an,expected", UO)
def test_sofa2_urine(s2, r6, r12, r24, an, expected):
    out = urine_points_sofa2({6: r6, 12: r12, 24: r24}, anuria=an, cfg=s2["kidney"])[0]
    if np.isnan(expected):
        assert np.isnan(out)
    else:
        assert out == expected


def test_sofa2_worst_of_creatinine_and_urine(s2):
    assert kidney_sofa2(1.5, 2.0, False, s2["kidney"])[0] == 2
    assert kidney_sofa2(4.0, 1.0, False, s2["kidney"])[0] == 3


# footnote p: (cr >1.2 or oliguria) and (K >=6.0 or (pH <=7.20 and HCO3 <=12))
RRT = [
    (1.21, False, 6.0, 7.4, 24, True),
    (1.2, False, 6.0, 7.4, 24, False),
    (1.0, True, 6.0, 7.4, 24, True),
    (2.0, False, 5.9, 7.4, 24, False),
    (2.0, False, 5.0, 7.20, 12, True),
    (2.0, False, 5.0, 7.21, 12, False),
    (2.0, False, 5.0, 7.20, 12.1, False),
    (2.0, False, np.nan, np.nan, np.nan, False),
    (np.nan, False, 7.0, 7.0, 10, False),
]


@pytest.mark.parametrize("cr,olig,k,ph,hco3,expected", RRT)
def test_rrt_criteria(s2, cr, olig, k, ph, hco3, expected):
    assert bool(rrt_criteria_met(cr, olig, k, ph, hco3, s2["kidney"])[0]) is expected


# Vincent 1996: 1.2-1.9 -> 1; 2.0-3.4 -> 2; 3.5-4.9 -> 3 or <500 mL/day; >5.0 -> 4 or <200 mL/day
SOFA1 = [
    (1.19, np.nan, 0), (1.2, np.nan, 1), (1.99, np.nan, 1), (2.0, np.nan, 2), (3.49, np.nan, 2),
    (3.5, np.nan, 3), (4.99, np.nan, 3), (5.0, np.nan, 4), (np.nan, 500, 0), (np.nan, 499, 3),
    (np.nan, 200, 3), (np.nan, 199, 4), (1.0, 300, 3), (np.nan, np.nan, np.nan),
]


@pytest.mark.parametrize("cr,uo,expected", SOFA1)
def test_sofa1(s1, cr, uo, expected):
    out = kidney_sofa1(cr, uo, s1["kidney"])[0]
    if np.isnan(expected):
        assert np.isnan(out)
    else:
        assert out == expected
