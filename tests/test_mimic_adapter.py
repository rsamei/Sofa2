"""MIMIC-IV adapter on made-up MIMIC-shaped files (DuckDB) and its transform rules."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from sofa2.adapters.mimic_iv import MimicIVAdapter, load_mapping  # noqa: E402
from sofa2.adapters.mimic_iv import transform as T  # noqa: E402
from sofa2.pipeline import compute_scores  # noqa: E402

DATA = Path(__file__).parent / "data" / "mimic_synthetic"
MP = load_mapping()
T0 = pd.Timestamp("2150-01-01")
ORG = ["respiratory", "cardiovascular", "brain", "liver", "kidney", "hemostasis"]


def t(h):
    return T0 + pd.Timedelta(hours=h)


@pytest.fixture(scope="module")
def data():
    with pytest.warns(UserWarning, match="norepinephrine"):
        return MimicIVAdapter.local(DATA).load()


def test_tables(data):
    sup = data.support.set_index("type")
    assert sup.loc["imv", "start"] == t(0) and sup.loc["imv", "end"] == t(24)
    assert sup.loc["bipap", "end"] == t(7)          # trailing space in "Bipap mask " handled
    assert {"ecmo_va", "iabp", "rrt_continuous", "rrt_intermittent"} <= set(sup.index)
    assert data.stays.set_index("stay_id")["chronic_rrt"].to_dict() == {1000: False, 2000: True}
    m = data.measurements
    assert m[m.variable == "pao2"]["value"].tolist() == [70]          # venous PO2 dropped
    assert sorted(m[m.variable == "fio2"]["value"].tolist()) == [0.5, 0.5, 0.6]
    assert m[m.variable == "creatinine"]["time"].min() == t(-4)
    assert not (m.variable == "bilirubin").any()                       # 10 h before ICU
    g = data.gcs.set_index("stay_id")
    assert bool(g.loc[1000, "unassessable"]) and pd.isna(g.loc[1000, "total"])
    assert g.loc[2000, "total"] == 15
    inf = data.infusions
    ne = inf[inf.drug == "norepinephrine"].sort_values("start")
    assert ne["rate"].tolist() == pytest.approx([0.25, 0.1])          # 8 mcg/min / 80 kg
    assert set(data.medications["drug"]) == {"haloperidol", "quetiapine"}  # "Not Given" dropped
    u = data.urine_output.set_index("time")["volume_ml"]
    assert u.loc[t(5)] == 0                                            # 20 - 30 irrigant -> 0


EXPECTED = {
    (1000, 0): ([3, 3, 2, 0, 3, 4], [3, 4, 0, 0, 3, 3]),
    (1000, 1): ([3, 2, 1, 0, 3, 4], [3, 3, 0, 0, 2, 3]),
    (2000, 0): ([4, 4, 0, 0, 4, 0], [0, 0, 0, 0, 1, 0]),
}


def test_scores(data):
    r = compute_scores(data).set_index(["stay_id", "window_index"])
    for key, (s2, s1) in EXPECTED.items():
        assert [r.loc[key, f"sofa2_{o}"] for o in ORG] == s2, key
        assert [r.loc[key, f"sofa1_{o}"] for o in ORG] == s1, key


def test_stay_subset_and_ids():
    a = MimicIVAdapter.local(DATA)
    assert a.stay_ids() == [1000, 2000]
    d = a.load(stay_ids=[2000])
    assert d.stays["stay_id"].tolist() == [2000]
    assert set(d.support["stay_id"]) == {2000}


def test_bigquery_sql_rendering():
    seen = []

    def runner(sql):
        seen.append(sql)
        return pd.DataFrame()

    a = MimicIVAdapter(runner, "`physionet-data.mimiciv_3_1_icu`", "`physionet-data.mimiciv_3_1_hosp`",
                       "bigquery")
    a.extract(stay_ids=[1, 2])
    labs = next(s for s in seen if "labevents le" in s)
    assert "DATETIME_SUB(ie.intime, INTERVAL 6 HOUR)" in labs
    assert all("ie.stay_id IN (1, 2)" in s for s in seen)
    assert "`physionet-data.mimiciv_3_1_icu`.chartevents" in "".join(seen)
    assert "{" not in "".join(seen)


# --------------------------------------------------------------------------- ventilation
def status_rows(rows):
    return pd.DataFrame(rows, columns=["stay_id", "charttime", "status"])


def test_episode_gap_and_status_change():
    s = status_rows([(1, t(0), "imv"), (1, t(4), "imv"), (1, t(20), "imv"), (1, t(22), "imv"),
                     (1, t(23), "hfnc"), (1, t(24), "hfnc")])
    ep = T.ventilation_episodes(s, 14)
    # 16-h gap splits the imv charts; the first episode ends at its last chart (next >= 14 h)
    assert ep.values.tolist() == [
        [1, t(0), t(4), "imv"], [1, t(20), t(23), "imv"], [1, t(23), t(24), "hfnc"]]


def test_single_chart_episode_dropped():
    s = status_rows([(1, t(0), "hfnc"), (1, t(2), "oxygen"), (1, t(3), "oxygen")])
    ep = T.ventilation_episodes(s, 14)
    assert ep["status"].tolist() == ["oxygen"]


def test_status_priority_and_trach():
    ce = pd.DataFrame([
        (1, t(0), 226732, "Nasal cannula"), (1, t(0), 226732, "Endotracheal tube"),
        (1, t(1), 226732, "Tracheostomy tube"),
        (1, t(2), 226732, "Tracheostomy tube"), (1, t(2), 223849, "CPAP/PSV"),
        (1, t(3), 229314, "NIV"), (1, t(4), 226732, "High flow nasal cannula"),
        (1, t(5), 223849, "Standby"),
    ], columns=["stay_id", "charttime", "itemid", "value"])
    st = T.ventilation_status(ce, MP["ventilation"], MP["chartevents"])
    assert st["status"].tolist() == ["imv", "oxygen", "imv", "niv", "hfnc"]


def test_fio2_fraction():
    out = T._fio2_fraction(pd.Series([0.5, 50, 21, 0.21, 4, 0, 101, 0.2]))
    assert out.tolist()[:4] == [0.5, 0.5, 0.21, 0.21]
    assert out.iloc[4:].isna().all()


def test_batched_equals_single(data):
    import warnings

    from sofa2.pipeline import compute_scores_batched

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = compute_scores_batched(MimicIVAdapter.local(DATA), batch_size=1)
    pd.testing.assert_frame_equal(b, compute_scores(data))
