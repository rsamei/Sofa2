import pandas as pd
import pytest

from sofa2.schema import ICUData, SchemaError


def _stays():
    return pd.DataFrame(
        {"stay_id": [1], "patient_id": [10], "intime": ["2100-01-01 08:00"], "outtime": ["2100-01-03 08:00"]}
    )


def test_minimal_validates_and_fills_optional():
    d = ICUData(stays=_stays()).validate()
    assert d.stays["chronic_rrt"].tolist() == [False]
    assert d.stays["intime"].dtype.kind == "M"


def test_unknown_variable_rejected():
    m = pd.DataFrame({"stay_id": [1], "time": ["2100-01-01 09:00"], "variable": ["lactate"], "value": [2.0]})
    with pytest.raises(SchemaError):
        ICUData(stays=_stays(), measurements=m).validate()


def test_fio2_percent_rejected():
    m = pd.DataFrame({"stay_id": [1], "time": ["2100-01-01 09:00"], "variable": ["fio2"], "value": [40.0]})
    with pytest.raises(SchemaError):
        ICUData(stays=_stays(), measurements=m).validate()


def test_unknown_stay_rejected():
    u = pd.DataFrame({"stay_id": [2], "time": ["2100-01-01 09:00"], "volume_ml": [100.0]})
    with pytest.raises(SchemaError):
        ICUData(stays=_stays(), urine_output=u).validate()


def test_unknown_support_type_rejected():
    s = pd.DataFrame({"stay_id": [1], "start": ["2100-01-01 09:00"], "end": ["2100-01-01 10:00"], "type": ["oxygen"]})
    with pytest.raises(SchemaError):
        ICUData(stays=_stays(), support=s).validate()
