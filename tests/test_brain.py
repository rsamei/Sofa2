import numpy as np
import pytest

from sofa2.scoring.components import brain, gcs_points

# SOFA-2 Table 2: 15 -> 0; 13-14 -> 1; 9-12 -> 2; 6-8 -> 3; 3-5 -> 4
SOFA2_TOTAL = [(15, 0), (14, 1), (13, 1), (12, 2), (9, 2), (8, 3), (6, 3), (5, 4), (3, 4)]
# Table 2 parentheticals / footnote d: M6 -> 0, M5 -> 1, M4 -> 2, M3 -> 3, M2 -> 4, M1 -> 4
SOFA2_MOTOR = [(6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 4)]
# Vincent 1996: 15 -> 0; 13-14 -> 1; 10-12 -> 2; 6-9 -> 3; <6 -> 4
SOFA1_TOTAL = [(15, 0), (14, 1), (13, 1), (12, 2), (10, 2), (9, 3), (6, 3), (5, 4), (3, 4)]


@pytest.mark.parametrize("total,points", SOFA2_TOTAL)
def test_sofa2_total(s2, total, points):
    assert gcs_points(total, cfg=s2["brain"])[0] == points


@pytest.mark.parametrize("motor,points", SOFA2_MOTOR)
def test_sofa2_motor_when_unassessable(s2, motor, points):
    # total is ignored when the three domains cannot be assessed
    assert gcs_points(15, motor=motor, assessable=False, cfg=s2["brain"])[0] == points


@pytest.mark.parametrize("total,points", SOFA1_TOTAL)
def test_sofa1_total(s1, total, points):
    assert gcs_points(total, cfg=s1["brain"])[0] == points


def test_sofa1_unassessable_is_nan(s1):
    assert np.isnan(gcs_points(10, motor=4, assessable=False, cfg=s1["brain"])[0])


def test_delirium_footnote_e(s2):
    # GCS 15 with delirium drug -> 1; GCS 13 -> 1; GCS 8 -> 3; no GCS + delirium -> 1
    out = brain([0, 1, 3, np.nan, np.nan], delirium=[True, True, True, True, False], cfg=s2["brain"])
    np.testing.assert_array_equal(out[:4], [1, 1, 3, 1])
    assert np.isnan(out[4])


def test_sofa1_has_no_delirium_rule(s1):
    assert brain([0], delirium=[True], cfg=s1["brain"])[0] == 0
