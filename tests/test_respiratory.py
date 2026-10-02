import numpy as np
import pytest

from sofa2.scoring.components import ratio_points, respiratory

# SOFA-2 PaO2:FiO2 (Table 2): >300 -> 0; <=300 -> 1; <=225 -> 2;
# <=150 + advanced support -> 3; <=75 + advanced support -> 4; max 2 without support (fn h)
SOFA2_PF = [
    (301, True, 0), (300.5, True, 0), (300, True, 1), (226, True, 1), (225, True, 2),
    (151, True, 2), (150, True, 3), (76, True, 3), (75, True, 4), (40, True, 4),
    (300, False, 1), (225, False, 2), (150, False, 2), (75, False, 2), (40, False, 2),
]
# SOFA-2 SpO2:FiO2 (footnote f): >300 -> 0; <=300 -> 1; <=250 -> 2; <=200 + support -> 3;
# <=120 + support -> 4
SOFA2_SF = [
    (301, True, 0), (300, True, 1), (251, True, 1), (250, True, 2), (201, True, 2),
    (200, True, 3), (121, True, 3), (120, True, 4), (200, False, 2), (120, False, 2),
]
# Vincent 1996: <400 -> 1; <300 -> 2; <200 with respiratory support -> 3; <100 with support -> 4
SOFA1_PF = [
    (400, True, 0), (399.9, True, 1), (300, True, 1), (299, True, 2), (200, True, 2),
    (199, True, 3), (100, True, 3), (99, True, 4), (199, False, 2), (99, False, 2),
]


@pytest.mark.parametrize("ratio,support,points", SOFA2_PF)
def test_sofa2_pf(s2, ratio, support, points):
    r = s2["respiratory"]
    assert ratio_points(ratio, support, r["pf_bands"], r)[0] == points


@pytest.mark.parametrize("ratio,support,points", SOFA2_SF)
def test_sofa2_sf(s2, ratio, support, points):
    r = s2["respiratory"]
    assert ratio_points(ratio, support, r["sf_bands"], r)[0] == points


@pytest.mark.parametrize("ratio,support,points", SOFA1_PF)
def test_sofa1_pf(s1, ratio, support, points):
    r = s1["respiratory"]
    assert ratio_points(ratio, support, r["pf_bands"], r)[0] == points


def test_sf_used_only_without_pf(s2):
    out = respiratory([1.0, np.nan, np.nan], sf_pts=[3.0, 2.0, np.nan], cfg=s2["respiratory"])
    np.testing.assert_array_equal(out[:2], [1, 2])
    assert np.isnan(out[2])


def test_ecmo_scores_4_footnote_i(s2):
    out = respiratory([0.0, np.nan], ecmo=[True, True], cfg=s2["respiratory"])
    np.testing.assert_array_equal(out, [4, 4])


def test_sofa1_ignores_sf(s1):
    out = respiratory([np.nan], sf_pts=[3.0], cfg=s1["respiratory"])
    assert np.isnan(out[0])


def test_ceiling_of_treatment_switch(s2):
    r = dict(s2["respiratory"], allow_full_score_without_support=True)
    assert ratio_points(70, False, r["pf_bands"], r)[0] == 4
