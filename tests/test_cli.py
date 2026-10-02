from pathlib import Path

import pandas as pd

from sofa2.cli import main

DATA = Path(__file__).parent / "data" / "synthetic"


def test_cli_internal(tmp_path):
    out = tmp_path / "scores.csv"
    assert main(["score", "--source", "internal", "--path", str(DATA), "--out", str(out)]) == 0
    r = pd.read_csv(out)
    assert len(r) == 6 and "sofa2_total" in r and "sofa1_total" in r


def test_cli_options(tmp_path):
    out = tmp_path / "scores.parquet"
    assert main(["score", "--source", "internal", "--path", str(DATA), "--freq", "hourly",
                 "--scores", "sofa2", "--missing", "normal", "--out", str(out)]) == 0
    r = pd.read_parquet(out)
    assert len(r) == 126 and not any(c.startswith("sofa1") for c in r)
