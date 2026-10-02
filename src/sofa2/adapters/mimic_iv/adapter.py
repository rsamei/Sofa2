"""MIMIC-IV adapter: BigQuery or local CSV/Parquet files (through DuckDB).

Both back ends run the same SQL (``sql/*.sql``) and the same pandas transform
(:mod:`sofa2.adapters.mimic_iv.transform`); only the table paths and the date arithmetic differ.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd
import yaml

from sofa2.adapters.mimic_iv.transform import to_icudata
from sofa2.schema import ICUData

Runner = Callable[[str], pd.DataFrame]

_TABLES = {
    "icu": ["icustays", "chartevents", "inputevents", "outputevents", "procedureevents"],
    "hosp": ["labevents", "emar", "diagnoses_icd"],
}


def load_mapping(path: str | Path | None = None) -> dict:
    """The MIMIC-IV item mapping (packaged ``mapping.yaml`` unless a path is given)."""
    if path is None:
        ref = resources.files("sofa2.adapters.mimic_iv").joinpath("mapping.yaml")
        return yaml.safe_load(ref.read_text(encoding="utf-8"))
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _sql(name: str) -> str:
    return resources.files("sofa2.adapters.mimic_iv").joinpath("sql", f"{name}.sql").read_text(
        encoding="utf-8"
    )


def _ints(values: Iterable) -> str:
    return ", ".join(str(int(v)) for v in values)


def _strs(values: Iterable[str]) -> str:
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def _chart_items(mp: dict) -> list[int]:
    c = mp["chartevents"]
    items = c["map"] + c["spo2"] + c["fio2"] + c["weight"]
    items += [c["gcs_eye"], c["gcs_verbal"], c["gcs_motor"], c["o2_device"], c["vent_mode"],
              c["vent_mode_hamilton"], c["rrt_pd_catheter_status"]]
    items += c["ecmo_configuration"] + c["ecmo_flow"]
    for group in (c["mcs"], c["rrt_active"]):
        for v in group.values():
            items += v
    return sorted(set(int(i) for i in items))


def _lab_items(mp: dict) -> list[int]:
    lab = mp["labevents"]
    keys = ("creatinine", "bilirubin", "platelets", "potassium", "bicarbonate", "ph", "pao2", "fio2")
    return sorted({int(i) for k in keys for i in lab[k]})


class MimicIVAdapter:
    """Extract MIMIC-IV into the internal schema.

    Use :meth:`bigquery` or :meth:`local` to build one.

    Args:
        runner: function that runs a SQL string and returns a DataFrame.
        icu, hosp: SQL prefixes of the icu and hosp modules (eg ``physionet-data.mimiciv_3_1_icu``
            in backticks for BigQuery, ``icu`` for DuckDB).
        dialect: ``"bigquery"`` or ``"duckdb"``.
        mapping: item mapping; defaults to the packaged ``mapping.yaml``.
        pre_icu_lab_hours: how far before ICU admission laboratory values are extracted (should be
            at least ``pipeline.pre_icu_lab_hours``).
    """

    def __init__(
        self,
        runner: Runner,
        icu: str,
        hosp: str,
        dialect: str,
        mapping: dict | None = None,
        pre_icu_lab_hours: float = 6,
    ):
        if dialect not in ("bigquery", "duckdb"):
            raise ValueError("dialect must be 'bigquery' or 'duckdb'")
        self.runner, self.icu, self.hosp, self.dialect = runner, icu, hosp, dialect
        self.mapping = mapping or load_mapping()
        self.pre_icu_lab_hours = pre_icu_lab_hours
        self._stay_ids: Sequence[int] | None = None

    # ------------------------------------------------------------------ constructors
    @classmethod
    def bigquery(
        cls,
        project: str,
        icu_dataset: str = "physionet-data.mimiciv_3_1_icu",
        hosp_dataset: str = "physionet-data.mimiciv_3_1_hosp",
        **kwargs,
    ) -> "MimicIVAdapter":
        """Adapter on BigQuery. ``project`` is the billing project running the queries.

        Requires ``pip install sofa2[bigquery]`` and Google Cloud credentials with PhysioNet
        access to MIMIC-IV.
        """
        from google.cloud import bigquery  # optional dependency

        client = bigquery.Client(project=project)

        def run(sql: str) -> pd.DataFrame:
            return client.query(sql).to_dataframe()

        return cls(run, f"`{icu_dataset}`", f"`{hosp_dataset}`", "bigquery", **kwargs)

    @classmethod
    def local(cls, directory: str | Path, **kwargs) -> "MimicIVAdapter":
        """Adapter on local files laid out as on PhysioNet: ``<directory>/icu/<table>.csv.gz``
        and ``<directory>/hosp/<table>.csv.gz`` (``.csv`` and ``.parquet`` also work).

        Requires ``pip install sofa2[local]`` (DuckDB).
        """
        import duckdb  # optional dependency

        con = duckdb.connect()
        root = Path(directory)
        for module, tables in _TABLES.items():
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {module}")
            for table in tables:
                path = _find(root / module, table)
                reader = (
                    f"read_parquet('{path}')" if path.suffix == ".parquet"
                    else f"read_csv_auto('{path}', all_varchar=false, sample_size=-1)"
                )
                con.execute(f"CREATE VIEW {module}.{table} AS SELECT * FROM {reader}")

        def run(sql: str) -> pd.DataFrame:
            return con.execute(sql).df()

        return cls(run, "icu", "hosp", "duckdb", **kwargs)

    # ------------------------------------------------------------------ extraction
    def _render(self, name: str, **params) -> str:
        if self._stay_ids is None:
            stay_filter = ""
        elif len(self._stay_ids) == 0:
            stay_filter = "AND 1 = 0"
        else:
            stay_filter = f"AND ie.stay_id IN ({_ints(self._stay_ids)})"
        hours = int(self.pre_icu_lab_hours)
        lab_start = (
            f"DATETIME_SUB(ie.intime, INTERVAL {hours} HOUR)" if self.dialect == "bigquery"
            else f"ie.intime - INTERVAL {hours} HOUR"
        )
        return _sql(name).format(icu=self.icu, hosp=self.hosp, stay_filter=stay_filter,
                                 lab_start=lab_start, **params)

    def stay_ids(self) -> list[int]:
        """All ICU stay ids (eg to process MIMIC-IV in batches)."""
        sql = f"SELECT stay_id FROM {self.icu}.icustays ORDER BY stay_id"
        return self.runner(sql)["stay_id"].astype(int).tolist()

    def extract(self, stay_ids: Sequence[int] | None = None) -> dict[str, pd.DataFrame]:
        """Run all extraction queries; returns raw DataFrames keyed by query name."""
        self._stay_ids = None if stay_ids is None else list(stay_ids)
        mp = self.mapping
        inp = mp["inputevents"]
        input_items = [i for v in list(inp["drugs"].values()) + list(inp["bolus_drugs"].values()) for i in v]
        out_items = mp["outputevents"]["urine"] + mp["outputevents"]["urine_subtract"]
        proc_items = mp["procedureevents"]["rrt_intermittent"] + mp["procedureevents"]["rrt_continuous"]
        icd = mp["diagnoses_icd"]["chronic_rrt"]
        like = " OR ".join(
            f"LOWER(e.medication) LIKE '%{p.lower()}%'"
            for pats in mp["emar"]["medications"].values() for p in pats
        )
        try:
            return {
                "stays": self.runner(self._render("stays")),
                "chronic_rrt": self.runner(self._render(
                    "chronic_rrt", icd9=_strs(icd["9"]), icd10=_strs(icd["10"]))),
                "chartevents": self.runner(self._render("chartevents", items=_ints(_chart_items(mp)))),
                "labevents": self.runner(self._render(
                    "labevents", items=_ints(_lab_items(mp)),
                    specimen_item=int(mp["labevents"]["specimen_type"]))),
                "inputevents": self.runner(self._render("inputevents", items=_ints(input_items))),
                "outputevents": self.runner(self._render("outputevents", items=_ints(out_items))),
                "procedureevents": self.runner(self._render("procedureevents", items=_ints(proc_items))),
                "emar": self.runner(self._render(
                    "emar", medication_like=like, events=_strs(mp["emar"]["given_events"]))),
            }
        finally:
            self._stay_ids = None

    def load(self, stay_ids: Sequence[int] | None = None) -> ICUData:
        """Extract and transform into :class:`ICUData` (all stays, or only ``stay_ids``)."""
        return to_icudata(self.extract(stay_ids), self.mapping)


def _find(folder: Path, table: str) -> Path:
    for ext in (".parquet", ".csv.gz", ".csv"):
        p = folder / f"{table}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{folder}/{table}(.parquet|.csv.gz|.csv) not found")
