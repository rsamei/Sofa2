import numpy as np
import pytest

from sofa2.scoring.components import liver

# SOFA-2 Table 2: <=1.20 -> 0; <=3.0 -> 1; <=6.0 -> 2; <=12.0 -> 3; >12 -> 4
SOFA2 = [
    (0.5, 0), (1.2, 0), (1.21, 1), (3.0, 1), (3.01, 2), (6.0, 2), (6.01, 3),
    (12.0, 3), (12.01, 4), (30.0, 4),
]
# Vincent 1996 Table 3: <1.2 -> 0; 1.2-1.9 -> 1; 2.0-5.9 -> 2; 6.0-11.9 -> 3; >12.0 -> 4
SOFA1 = [
    (1.19, 0), (1.2, 1), (1.9, 1), (1.99, 1), (2.0, 2), (5.9, 2), (6.0, 3), (11.9, 3),
    (12.0, 4), (12.1, 4),
]


@pytest.mark.parametrize("value,points", SOFA2)
def test_sofa2(s2, value, points):
    assert liver(value, s2["liver"])[0] == points


@pytest.mark.parametrize("value,points", SOFA1)
def test_sofa1(s1, value, points):
    assert liver(value, s1["liver"])[0] == points


def test_missing_is_nan(s2):
    assert np.isnan(liver(np.nan, s2["liver"])[0])
