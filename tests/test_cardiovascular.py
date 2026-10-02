import numpy as np
import pytest

from sofa2.scoring.components import cardiovascular_sofa1, cardiovascular_sofa2

# (ne_epi, dopamine, other_agent, mechanical, map, expected) -- SOFA-2 Table 2 + footnotes l, n
SOFA2 = [
    (0, 0, False, False, 70, 0),
    (0, 0, False, False, 69.9, 1),
    (0, 0, False, False, np.nan, np.nan),
    (0.01, 0, False, False, 50, 2),
    (0.2, 0, False, False, 50, 2),
    (0.21, 0, False, False, 50, 3),
    (0.4, 0, False, False, 50, 3),
    (0.41, 0, False, False, 50, 4),
    (0.2, 0, True, False, 50, 3),     # low dose + other -> 3
    (0.21, 0, True, False, 50, 4),    # medium dose + other -> 4
    (0.4, 0, True, False, 50, 4),
    (0.5, 0, True, False, 50, 4),
    (0.1, 5, False, False, 50, 3),    # dopamine with NE/epi counts as other agent
    (0, 0, True, False, 80, 2),       # any dose of other agent alone -> 2
    (0, 20, False, False, 80, 2),     # dopamine alone <=20 -> 2 (fn l)
    (0, 20.1, False, False, 80, 3),   # >20-40 -> 3
    (0, 40, False, False, 80, 3),
    (0, 40.1, False, False, 80, 4),   # >40 -> 4
    (0, 10, True, False, 80, 3),      # dopamine + other non-NE agent: +1 (agreed rule)
    (0, 30, True, False, 80, 4),
    (0, 0, False, True, 80, 4),       # mechanical support -> 4 (fn n)
    (0, 0, False, True, np.nan, 4),
]


@pytest.mark.parametrize("ne,dopa,other,mech,mapv,expected", SOFA2)
def test_sofa2(s2, ne, dopa, other, mech, mapv, expected):
    out = cardiovascular_sofa2(ne, dopa, other, mech, mapv, s2["cardiovascular"])[0]
    if np.isnan(expected):
        assert np.isnan(out)
    else:
        assert out == expected


# footnote m (off by default): >=70 -> 0; 60-69 -> 1; 50-59 -> 2; 40-49 -> 3; <40 -> 4
@pytest.mark.parametrize("mapv,expected", [(70, 0), (69, 1), (60, 1), (59, 2), (50, 2), (49, 3), (40, 3), (39, 4)])
def test_sofa2_map_only_fallback(s2, mapv, expected):
    c = dict(s2["cardiovascular"], map_only_fallback=True)
    assert cardiovascular_sofa2(0, 0, False, False, mapv, c)[0] == expected


# (ne, epi, dopa, dobu, map, expected) -- Vincent 1996 Table 3
SOFA1 = [
    (0, 0, 0, 0, 70, 0),
    (0, 0, 0, 0, 69, 1),
    (0, 0, 5, 0, 80, 2),     # dopamine <=5 -> 2
    (0, 0, 0, 2, 80, 2),     # dobutamine any dose -> 2
    (0, 0, 5.1, 0, 80, 3),   # dopamine >5 -> 3
    (0, 0, 15, 0, 80, 3),
    (0, 0, 15.1, 0, 80, 4),  # dopamine >15 -> 4
    (0.1, 0, 0, 0, 80, 3),   # NE <=0.1 -> 3
    (0.11, 0, 0, 0, 80, 4),  # NE >0.1 -> 4
    (0, 0.1, 0, 0, 80, 3),   # epi <=0.1 -> 3
    (0, 0.11, 0, 0, 80, 4),
    (0.05, 0.05, 0, 0, 80, 3),  # doses are not summed in the original SOFA
    (0, 0, 0, 0, np.nan, np.nan),
]


@pytest.mark.parametrize("ne,epi,dopa,dobu,mapv,expected", SOFA1)
def test_sofa1(s1, ne, epi, dopa, dobu, mapv, expected):
    out = cardiovascular_sofa1(ne, epi, dopa, dobu, mapv, s1["cardiovascular"])[0]
    if np.isnan(expected):
        assert np.isnan(out)
    else:
        assert out == expected
