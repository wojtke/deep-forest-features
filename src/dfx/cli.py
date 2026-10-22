"""Command line: `dfx data | run | status | report`. See README.md and `dfx <command> -h`."""
import os

# One thread per process, set before numpy is imported anywhere (workers inherit it).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402


def _list(text: str | None, cast=str):
    return None if text is None else [cast(x) for x in text.split(",") if x]


def cmd_data(args) -> None:
    from .data import outer_folds, prepare, save, write_audit
    from .datasets import ALL, DATASETS
    from .run import experiment_settings

    main = experiment_settings("main")
    for name in args.names or ALL:
        print(f"{name} ...", flush=True)
        data, facts = prepare(DATASETS[name])
        save(data, facts)
        for seed in main["seeds"]:
            outer_folds(data, seed, main["outer_folds"], create=True)
        print(f"  {len(data.y):,} rows x {len(data.features)} features, sha256 {data.sha256[:12]}")
    print(f"wrote {write_audit()}")


def cmd_run(args) -> None:
    from .run import run_experiment

    ok = run_experiment(args.experiment, workers=args.workers, results=Path(args.results),
                        datasets=_list(args.datasets), methods=_list(args.methods),
                        seeds=_list(args.seeds, int), folds=_list(args.folds, int),
                        checks=_list(args.checks), dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


def cmd_status(args) -> None:
    from .run import status

    status(args.experiment, Path(args.results))


def cmd_report(args) -> None:
    from .report import report

    report(args.experiment, Path(args.results), Path(args.out), allow_incomplete=args.allow_incomplete)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="dfx", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="download, prepare and split datasets; write data/AUDIT.md")
    p.add_argument("names", nargs="*", help="dataset names (default: all 32)")
    p.set_defaults(func=cmd_data)

    for name, func, text in (("run", cmd_run, "run an experiment (resumes; skips finished tasks)"),
                             ("status", cmd_status, "count finished, failed and missing tasks"),
                             ("report", cmd_report, "collect results, run statistics, write tables")):
        p = sub.add_parser(name, help=text)
        p.add_argument("experiment", help="name of experiments/<name>.toml")
        p.add_argument("--results", default="results", help="results root (default: results)")
        if name == "run":
            p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                           help="parallel processes (default: cores - 2)")
            p.add_argument("--datasets", help="comma-separated subset")
            p.add_argument("--methods", help="comma-separated subset")
            p.add_argument("--seeds", help="comma-separated subset, e.g. 0")
            p.add_argument("--folds", help="comma-separated outer folds, e.g. 0,1")
            p.add_argument("--checks", help="comma-separated sensitivity checks")
            p.add_argument("--dry-run", action="store_true", help="list what would run")
        if name == "report":
            p.add_argument("--out", default="reports", help="output root (default: reports)")
            p.add_argument("--allow-incomplete", action="store_true",
                           help="interim report from a partial run (marked INCOMPLETE)")
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
