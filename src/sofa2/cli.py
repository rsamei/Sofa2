"""Command-line interface: ``sofa2 score ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sofa2.config import load_config
from sofa2.pipeline import compute_scores, compute_scores_batched


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sofa2", description="SOFA-2 and original SOFA scores.")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="compute scores and write them to CSV or Parquet")
    s.add_argument("--source", required=True, choices=["internal", "mimic-local", "mimic-bigquery"],
                   help="internal: CSV/Parquet in the sofa2 schema; mimic-local: MIMIC-IV files; "
                        "mimic-bigquery: MIMIC-IV on BigQuery")
    s.add_argument("--path", help="data directory (internal, mimic-local)")
    s.add_argument("--project", help="billing project (mimic-bigquery)")
    s.add_argument("--icu-dataset", default="physionet-data.mimiciv_3_1_icu")
    s.add_argument("--hosp-dataset", default="physionet-data.mimiciv_3_1_hosp")
    s.add_argument("--freq", default="daily", choices=["daily", "hourly"])
    s.add_argument("--scores", default="sofa2,sofa1", help="comma-separated: sofa2,sofa1")
    s.add_argument("--missing", choices=["locf", "normal", "none"], help="missing-data strategy")
    s.add_argument("--sofa2-config", help="YAML replacing the packaged sofa2.yaml")
    s.add_argument("--sofa1-config", help="YAML replacing the packaged sofa1.yaml")
    s.add_argument("--pipeline-config", help="YAML replacing the packaged pipeline.yaml")
    s.add_argument("--batch-size", type=int, default=0, help="stays per batch (MIMIC sources)")
    s.add_argument("--values", action="store_true", help="include the worst raw values")
    s.add_argument("--out", required=True, help="output file (.csv or .parquet)")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; returns the process exit code."""
    args = _parser().parse_args(argv)
    overrides = {"pipeline": {"missing": {"strategy": args.missing}}} if args.missing else None
    cfg = load_config(args.sofa2_config, args.sofa1_config, args.pipeline_config, overrides)
    hours = cfg.pipeline["pre_icu_lab_hours"]

    if args.source == "internal":
        from sofa2.adapters.internal import InternalAdapter

        adapter = InternalAdapter(args.path)
    elif args.source == "mimic-local":
        from sofa2.adapters.mimic_iv import MimicIVAdapter

        adapter = MimicIVAdapter.local(args.path, pre_icu_lab_hours=hours)
    else:
        from sofa2.adapters.mimic_iv import MimicIVAdapter

        if not args.project:
            print("--project is required for mimic-bigquery", file=sys.stderr)
            return 2
        adapter = MimicIVAdapter.bigquery(args.project, args.icu_dataset, args.hosp_dataset,
                                          pre_icu_lab_hours=hours)

    kwargs = dict(freq=args.freq, config=cfg, scores=args.scores.split(","),
                  include_values=args.values)
    if args.batch_size and args.source != "internal":
        result = compute_scores_batched(adapter, batch_size=args.batch_size, **kwargs)
    else:
        result = compute_scores(adapter, **kwargs)

    out = Path(args.out)
    if out.suffix == ".parquet":
        result.to_parquet(out, index=False)
    else:
        result.to_csv(out, index=False)
    print(f"wrote {len(result)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
