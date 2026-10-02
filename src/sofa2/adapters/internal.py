"""Read tables that are already in the internal schema from CSV or Parquet files."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pandas as pd

from sofa2.schema import ICUData, empty
from sofa2.units import to_canonical


class InternalAdapter:
    """Load ``<directory>/<table>.csv`` (or ``.csv.gz`` / ``.parquet``) for each schema table.

    Only ``stays`` is required; missing tables are empty. A ``measurements`` table may carry a
    ``unit`` column, in which case values are converted with :func:`sofa2.units.to_canonical`.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def _read(self, table: str) -> pd.DataFrame | None:
        for ext in (".parquet", ".csv", ".csv.gz"):
            path = self.directory / f"{table}{ext}"
            if path.exists():
                return pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
        return None

    def load(self) -> ICUData:
        """Read all tables into an :class:`ICUData` (not yet validated)."""
        tables = {}
        for f in fields(ICUData):
            df = self._read(f.name)
            if df is None:
                if f.name == "stays":
                    raise FileNotFoundError(f"no stays table in {self.directory}")
                df = empty(f.name)
            tables[f.name] = df
        tables["measurements"] = to_canonical(tables["measurements"])
        return ICUData(**tables)
