import numpy as np
import pandas as pd
import pytest

from sofa2 import units


def test_creatinine_and_bilirubin():
    assert units.creatinine_umol_to_mgdl(88.4) == pytest.approx(1.0)
    assert units.bilirubin_umol_to_mgdl(17.1) == pytest.approx(1.0)


def test_kpa():
    # Table 2 equivalences: 40 kPa = 300 mmHg, 10 kPa = 75 mmHg (exact, so cut-offs coincide)
    assert units.kpa_to_mmhg(40) == 300.0
    assert units.kpa_to_mmhg(10) == 75.0


def test_fio2_fraction_and_percent():
    np.testing.assert_allclose(units.fio2_to_fraction([0.4, 40, 100, 1.0]), [0.4, 0.4, 1.0, 1.0])


def test_norepinephrine_salt_factors_footnote_k():
    # 1 mg base = 2 mg bitartrate monohydrate = 1.89 mg anhydrous bitartrate = 1.22 mg HCl
    assert units.norepinephrine_to_base(2.0, "bitartrate_monohydrate") == pytest.approx(1.0)
    assert units.norepinephrine_to_base(1.89, "bitartrate_anhydrous") == pytest.approx(1.0)
    assert units.norepinephrine_to_base(1.22, "hydrochloride") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        units.norepinephrine_to_base(1.0, "sulfate")


def test_dose_units():
    assert units.dose_to_ug_kg_min(0.1, "mcg/kg/min") == pytest.approx(0.1)
    assert units.dose_to_ug_kg_min(0.0001, "mg/kg/min") == pytest.approx(0.1)
    assert units.dose_to_ug_kg_min(8, "mcg/min", 80) == pytest.approx(0.1)
    assert units.dose_to_ug_kg_min(0.48, "mg/h", 80) == pytest.approx(0.1)
    assert units.dose_to_ug_kg_min(0.48, "mg/hour", 80) == pytest.approx(0.1)
    assert units.dose_to_ug_kg_min(6, "μg/kg/hr") == pytest.approx(0.1)
    with pytest.raises(ValueError):
        units.dose_to_ug_kg_min(8, "mcg/min")
    with pytest.raises(ValueError):
        units.dose_to_ug_kg_min(8, "units/min", 80)


def test_to_canonical():
    df = pd.DataFrame(
        {
            "variable": ["creatinine", "creatinine", "bilirubin", "pao2", "fio2"],
            "value": [176.8, 2.0, 34.2, 10.0, 50.0],
            "unit": ["µmol/L", "mg/dL", "umol/L", "kPa", "%"],
        }
    )
    out = units.to_canonical(df)
    np.testing.assert_allclose(out["value"], [2.0, 2.0, 2.0, 75.0, 0.5])
    assert "unit" not in out


def test_to_canonical_unknown_unit():
    df = pd.DataFrame({"variable": ["creatinine"], "value": [1.0], "unit": ["g/L"]})
    with pytest.raises(ValueError):
        units.to_canonical(df)
