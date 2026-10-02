import numpy as np

from sofa2.scoring.bands import score

BANDS = [
    {"points": 1, "op": "<=", "value": 300},
    {"points": 2, "op": "<=", "value": 225},
    {"points": 3, "op": "<=", "value": 150, "requires_support": True},
    {"points": 4, "op": "<=", "value": 75, "requires_support": True},
]


def test_basic_and_nan():
    out = score([301, 300, 225, np.nan], BANDS)
    np.testing.assert_array_equal(out[:3], [0, 1, 2])
    assert np.isnan(out[3])


def test_support_cap():
    vals = [150, 75, 150, 75]
    sup = [True, True, False, False]
    np.testing.assert_array_equal(score(vals, BANDS, support=sup), [3, 4, 2, 2])
