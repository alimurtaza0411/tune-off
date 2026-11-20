"""Command-line interface: ``python -m mvt <command>`` (README §Command-line interface)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from mvt.core.config import load_base, load_experiment, load_search_space
from mvt.core.env import env_report
from mvt.core.results import export_cell, iter_results, results_to_frame, validate_frame
from mvt.core.types import make_cell_id

log = logging.getLogger("mvt")


def _owner() -> str:
    return os.environ.get("MVT_OWNER", "")


def _results_root(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "root", None) or load_base().runtime.results_root)


# ---------------------------------------------------------------- commands
def cmd_info(args: argparse.Namespace) -> int:
    for key, value in env_report().items():
        print(f"{key:20s} {value}")
    return 0


def cmd_sobol(args: argparse.Namespace) -> int:
    from mvt.search.sobol import check_table, write_table

    cfg = load_search_space()
    if args.write:
        print(f"wrote {write_table(cfg)}")
        return 0
    ok = check_table(cfg)
    print("sobol table OK" if ok else "sobol table DIFFERS from regeneration")
    return 0 if ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    from mvt.search.runner import CellRunner

    archs = ["transformer", "mamba"] if args.arch == "both" else [args.arch]
    runner = CellRunner(args.task, args.fraction, args.scale, archs, _results_root(args), _owner())
    n = runner.run(args.phase, limit=args.limit, retry_failed=args.retry_failed)
    print(f"{runner.cell_id}: ran {n} trial(s)")
    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    from mvt.search.pilot import rule_outcomes, run_pilot

    fractions = [args.fraction] if args.fraction else load_experiment().fractions
    summary = run_pilot(
        args.task, fractions, ["transformer", "mamba"], _results_root(args), _owner() or "pilot"
    )
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False))
    print("\n".join(rule_outcomes(summary)))
    return 0


def _frame(root: Path) -> pd.DataFrame:
    return results_to_frame(iter_results(root))


def cmd_status(args: argparse.Namespace) -> int:
    df = _frame(_results_root(args))
    if df.empty:
        print("no results yet")
        return 0
    if args.task:
        df = df[df["task"] == args.task]
    if args.scale:
        df = df[df["scale"] == args.scale]
    table = df.pivot_table(
        index=["cell_id", "arch"], columns="phase", values="run_id", aggfunc="count", fill_value=0
    )
    failed = df[df["status"] == "failed"].groupby(["cell_id", "arch"]).size()
    table["failed"] = failed.reindex(table.index, fill_value=0)
    print(table.to_string())
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    root = _results_root(args)
    if args.cell:
        cells = [args.cell]
    else:
        present = {p.name for p in root.iterdir() if p.is_dir()} if root.exists() else set()
        cells = sorted(c for c in present if args.all or c.startswith(f"{args.task}-"))
    for cell in cells:
        print(f"wrote {export_cell(root, cell, args.out)}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    files = sorted(Path(args.path).glob("*.csv"))
    if not files:
        print(f"no CSV files in {args.path}")
        return 1
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    problems = validate_frame(df, load_experiment().phases.expected())
    print("\n".join(problems) if problems else f"OK: {len(files)} cell file(s) valid")
    return 1 if problems else 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """CPU end-to-end check on the debug tasks with reduced phases."""
    import torch

    from mvt.search.runner import CellRunner

    torch.set_num_threads(1)  # tiny models: thread overhead dominates, one thread is fastest
    start = time.perf_counter()
    smoke = load_experiment().smoke
    out = Path(args.out or tempfile.mkdtemp(prefix="mvt_smoke_"))
    runs, csvs = out / "runs", out / "results"
    for task in smoke.tasks:
        runner = CellRunner(
            task,
            smoke.fraction,
            smoke.scale,
            ["transformer", "mamba"],
            runs,
            owner="smoke",
            counts=smoke.phases,
            enforce_rules=False,
        )
        runner.run("all")
        export_cell(runs, make_cell_id(task, smoke.fraction, smoke.scale), csvs)
    df = pd.concat([pd.read_csv(f) for f in sorted(csvs.glob("*.csv"))], ignore_index=True)
    problems = validate_frame(df, smoke.phases.expected())
    elapsed = time.perf_counter() - start
    if problems:
        print("SMOKE FAILED:\n" + "\n".join(problems))
        return 1
    print(f"SMOKE OK: {len(df)} trials, validated, in {elapsed:.0f}s (output: {out})")
    return 0


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mvt", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="environment report").set_defaults(fn=cmd_info)

    p = sub.add_parser("smoke", help="CPU end-to-end check on the debug tasks")
    p.add_argument("--out", help="output directory (default: a temp dir)")
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("sobol", help="generate or verify configs/sobol_32.csv")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    p.set_defaults(fn=cmd_sobol)

    p = sub.add_parser("pilot", help="learning-rate sweep + scale check for one task")
    p.add_argument("--task", required=True)
    p.add_argument("--fraction", type=float)
    p.add_argument("--root", help="results root (default: MVT_RESULTS_ROOT)")
    p.set_defaults(fn=cmd_pilot)

    p = sub.add_parser("run", help="run or resume one cell")
    p.add_argument("--task", required=True)
    p.add_argument("--fraction", type=float, required=True)
    p.add_argument("--scale", required=True)
    p.add_argument("--arch", default="both", choices=["both", "transformer", "mamba"])
    p.add_argument("--phase", default="all", choices=["all", "sobol", "tpe", "replicates"])
    p.add_argument("--limit", type=int, help="stop after N new trials")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--root", help="results root (default: MVT_RESULTS_ROOT)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("status", help="progress per cell x arch x phase")
    p.add_argument("--task")
    p.add_argument("--scale")
    p.add_argument("--root")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("export", help="write results/{cell_id}.csv")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--task")
    g.add_argument("--cell")
    g.add_argument("--all", action="store_true")
    p.add_argument("--out", default="results")
    p.add_argument("--root")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("validate", help="check exported CSVs against the results contract")
    p.add_argument("--path", default="results")
    p.set_defaults(fn=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
