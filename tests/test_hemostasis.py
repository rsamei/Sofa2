import numpy as np
import pytest

from sofa2.scoring.components import hemostasis

# SOFA-2 Table 2: >150 -> 0; <=150 -> 1; <=100 -> 2; <=80 -> 3; <=50 -> 4
SOFA2 = [
    (151, 0), (150.5, 0), (150, 1), (101, 1), (100, 2), (81, 2), (80, 3), (51, 3),
    (50, 4), (5, 4),
]
# Vincent 1996 Table 3: <150 -> 1; <100 -> 2; <50 -> 3; <20 -> 4
SOFA1 = [
    (150, 0), (149.9, 1), (100, 1), (99, 2), (50, 2), (49, 3), (20, 3), (19.9, 4),
]


@pytest.mark.parametrize("value,points", SOFA2)
def test_sofa2(s2, value, points):
    assert hemostasis(value, s2["hemostasis"])[0] == points


@pytest.mark.parametrize("value,points", SOFA1)
def test_sofa1(s1, value, points):
    assert hemostasis(value, s1["hemostasis"])[0] == points


def test_missing_is_nan(s2):
    assert np.isnan(hemostasis(np.nan, s2["hemostasis"])[0])
