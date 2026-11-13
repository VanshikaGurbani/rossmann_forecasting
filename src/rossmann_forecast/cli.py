"""Command line entry point: `rossmann check-data` and `rossmann run-all`."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import DEFAULT_ORIGINS, HORIZON, PROPHET_SAMPLE_STORES
from .data import check_data, load_panel


def _synthetic_origins(panel, horizon: int) -> list[str]:
    end = panel.dates[-1]
    first = end - pd.Timedelta(days=horizon - 1)
    return [str((first - pd.Timedelta(days=(horizon + 14) * k)).date()) for k in (2, 1, 0)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rossmann", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    chk = sub.add_parser("check-data", help="verify the Kaggle files are present")
    chk.add_argument("--data-dir", default="data")

    run = sub.add_parser("run-all", help="run CV, intervals, promo analysis, SHAP, write reports")
    run.add_argument("--data-dir", default="data")
    run.add_argument("--out-dir", default="reports")
    run.add_argument("--docs-dir", default="docs")
    run.add_argument(
        "--readme", default="README.md", help="file with RESULTS markers; 'none' skips"
    )
    run.add_argument("--horizon", type=int, default=HORIZON)
    run.add_argument("--origins", nargs="+", default=list(DEFAULT_ORIGINS))
    run.add_argument("--n-prophet-stores", type=int, default=PROPHET_SAMPLE_STORES)
    run.add_argument("--no-rf", action="store_true")
    run.add_argument("--no-prophet", action="store_true")
    run.add_argument("--synthetic", action="store_true", help="tiny synthetic data (CI smoke run)")

    args = parser.parse_args(argv)
    if args.command == "check-data":
        ok, msg = check_data(args.data_dir)
        print(msg)
        return 0 if ok else 1

    from .pipeline import run_all

    if args.synthetic:
        from .synthetic import synthetic_panel

        panel = synthetic_panel(n_stores=15, n_days=640)
        horizon = 14
        origins = _synthetic_origins(panel, horizon)
        n_perm, cache = 5, "data/processed/synthetic"
    else:
        try:
            panel = load_panel(args.data_dir)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        horizon, origins = args.horizon, args.origins
        n_perm, cache = 100, "data/processed"

    run_all(
        panel,
        out_dir=args.out_dir,
        docs_dir=args.docs_dir,
        readme=None if args.readme.lower() == "none" else args.readme,
        cache_dir=cache,
        origins=origins,
        horizon=horizon,
        n_prophet_stores=args.n_prophet_stores if not args.synthetic else 3,
        include_rf=not args.no_rf,
        include_prophet=not args.no_prophet,
        n_perm=n_perm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
