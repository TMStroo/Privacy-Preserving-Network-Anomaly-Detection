"""Command-line interface.

    python -m driftguard data inspect
    python -m driftguard data fetch --dataset ugr16
    python -m driftguard train       --config configs/benchmark.yaml
    python -m driftguard evaluate    --config configs/benchmark.yaml
    python -m driftguard detect-drift
    python -m driftguard adapt
    python -m driftguard benchmark   --config configs/benchmark.yaml
    python -m driftguard report
    python -m driftguard demo
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from driftguard.data.registry import get_adapter
from driftguard.utils import resolve_raw_dir

DEFAULT_CONFIG = "configs/benchmark.yaml"


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _print_dataset(name: str, raw_dir: str, options: Optional[Dict] = None) -> int:
    adapter = get_adapter(name)
    missing = adapter.missing_files(raw_dir)
    if missing:
        print(f"dataset '{name}' is not present in {raw_dir}")
        print("missing files:")
        for path in missing:
            print(f"    {path}")
        print(f"\nFetch it with: python -m driftguard data fetch --dataset {name}")
        return 1
    frame = adapter.load(raw_dir, **(options or {}))
    print(json.dumps(frame.describe_time(), indent=2))
    print("\ncolumns in the common schema:")
    for column in frame.feature_names():
        print(f"    {column}")
    return 0


def cmd_data_inspect(args) -> int:
    config = load_config(args.config) if os.path.exists(args.config) else {}
    raw_dir = args.raw_dir or resolve_raw_dir(config)
    names = [args.dataset] if args.dataset else ["unsw_nb15", "ugr16", "synthetic"]
    status = 0
    for name in names:
        print("=" * 70)
        print(f"dataset: {name}")
        if name == "synthetic":
            from driftguard.data.synthetic import generate_flows

            frame = generate_flows(rows=2000, days=5, seed=1)
            print(json.dumps({
                "dataset": "synthetic", "rows": len(frame),
                "note": "generated on demand; not a research dataset",
            }, indent=2))
            continue
        status |= _print_dataset(name, raw_dir, config.get("dataset", {}).get("load_options"))
    return status


def cmd_data_fetch(args) -> int:
    """Download a research dataset. Never called by tests or the demo."""
    from driftguard.data.fetch import FETCHERS

    if args.dataset not in FETCHERS:
        print(f"unknown dataset '{args.dataset}'. Available: {sorted(FETCHERS)}")
        return 1
    if not args.yes:
        print(f"This will download data into {args.raw_dir}.")
        print("Dataset files are large and are never committed. Re-run with --yes to proceed.")
        return 0
    return FETCHERS[args.dataset](args.raw_dir, weeks=args.weeks)


def cmd_benchmark(args) -> int:
    from driftguard.pipeline import prune_private, run_benchmark

    config = load_config(args.config)
    if args.set:
        for assignment in args.set:
            key, _, value = assignment.partition("=")
            node = config
            parts = key.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            try:
                node[parts[-1]] = yaml.safe_load(value)
            except Exception:
                node[parts[-1]] = value

    result = run_benchmark(config, experiment_root=config.get("output", {}).get("experiments_dir", "results/experiments"))
    print(f"\nexperiment written to: {result['experiment']['path']}")

    metrics = prune_private(result["metrics"])
    for entry in metrics["models"]:
        result_row = entry["result"]
        backtest = result_row["backtest"]
        print(f"\n  {result_row['model']}: threshold={result_row['threshold']:.4f}")
        print(f"    backtest  F1={backtest['f1']:.4f} recall={backtest['recall']:.4f} FPR={backtest['false_positive_rate']:.4f}")
        if "forward" in result_row:
            forward = result_row["forward"]
            print(f"    forward   F1={forward['f1']:.4f} recall={forward['recall']:.4f} FPR={forward['false_positive_rate']:.4f}")
            print(f"    F1 degradation = {result_row['f1_degradation']:+.4f}")
        else:
            print("    forward   (no forward test in this configuration)")
    return 0


def cmd_demo(args) -> int:
    from driftguard.pipeline import prune_private, run_benchmark

    config = load_config(args.config)
    result = run_benchmark(config, experiment_root=config.get("output", {}).get("experiments_dir", "results/demo"))
    metrics = prune_private(result["metrics"])
    split = metrics["split"]

    print("\n" + "=" * 70)
    print("DRIFTGUARD DEMO  -  generated data, pipeline verification only")
    print("This is NOT a research benchmark. The numbers below come from a")
    print("deterministic synthetic fixture with a planted distribution shift.")
    print("=" * 70)
    print(f"\ntrain      {split['train_period']}")
    print(f"backtest   {split['backtest_period']}")
    print(f"forward    {split['forward_period']}")

    print("\n  model                 backtest F1   forward F1   drift?   adapted F1")
    print("  " + "-" * 66)
    for entry in metrics["models"]:
        row = entry["result"]
        forward = row.get("forward", {})
        adapted = None
        for strategy in entry["adaptation"]["strategies"]:
            if strategy.get("strategy") == "recent_window_retrain" and "metrics" in strategy:
                adapted = strategy["metrics"]["f1"]
        drift_alerts = metrics.get("drift_summary", {}).get("alerts", 0)
        print(
            f"  {row['model']:<22} {row['backtest']['f1']:>11.4f} {forward.get('f1', float('nan')):>11.4f} "
            f"{('yes' if drift_alerts else 'no'):>8} {adapted if adapted is not None else float('nan'):>11.4f}"
        )
    print(f"\nexperiment: {result['experiment']['path']}")
    return 0


def cmd_report(args) -> int:
    from driftguard.reporting.generate import generate_report

    path = generate_report(experiments_dir=args.experiments_dir, output=args.output)
    print(f"report written to: {path}")
    return 0


def cmd_shift(args) -> int:
    from driftguard.data.registry import get_adapter
    from driftguard.experiments.tracking import (
        ExperimentRun,
        experiment_id,
        file_checksums,

    )
    from driftguard.shift.experiment import run_controlled_shift_experiments

    config = load_config(args.config)
    for assignment in args.set:
        key, _, value = assignment.partition("=")
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        try:
            node[parts[-1]] = yaml.safe_load(value)
        except Exception:
            node[parts[-1]] = value

    dataset = config["dataset"]
    raw_dir = resolve_raw_dir(config)
    frame = get_adapter(dataset["name"]).load(
        raw_dir, **dataset.get("load_options", {})
    )
    models = args.models or config.get("shift", {}).get("models", ["logistic_regression", "random_forest"])
    params = model_params(config)
    root = config.get("output", {}).get("experiments_dir", "results/experiments")

    run = ExperimentRun(
        experiment_id("shift", frame.name, config.get("tag")), root=root
    )
    meta = run.metadata(
        dataset=frame.name,
        dataset_checksums=file_checksums(
            [f"{raw_dir}/{f}" for f in frame.source_files]
        ),
        features=list(frame.frame.columns),
        seed=int(config.get("seed", 42)),
        config=config,
        split_description=period_summary(frame, config.get("temporal", {})),
    )
    # The model parameters, adaptation strategies and drift methods are already
    # inside `config`; recording them separately as well means a reader does not
    # have to know where in the config to look.
    meta["models"] = {m: params.get(m, {}) for m in models}
    meta["adaptation"] = config.get("adaptation", {})
    meta["drift"] = config.get("drift", {})
    run.write_json("metadata.json", meta)
    run.write_text("config.yaml", yaml.safe_dump(config, sort_keys=True))

    print(f"running controlled shifts on {len(models)} model(s): {', '.join(models)}")
    summary = run_controlled_shift_experiments(
        frame, config, models, run.path, run.experiment_id,
    )
    run.write_json("controlled_shift_summary.json", summary)
    print(f"\n  rows evaluated : {summary['rows']}")
    print(f"  degraded      : {summary['degraded']}")
    print(f"  recovered     : {summary['recovered']}")
    print(f"  skipped       : {summary['skipped']}")
    print(f"\nexperiment written to: {run.path}")
    return 0


def model_params(config: Dict) -> Dict[str, Dict]:
    """Per-model hyperparameters, whichever way the config spells them.

    ``models`` is either a list of names, meaning "use the registry defaults",
    or a mapping of name to parameter overrides.
    """
    block = config.get("models")
    if isinstance(block, dict):
        return {k: (v if isinstance(v, dict) else {}) for k, v in block.items()}
    if isinstance(block, list):
        return {name: {} for name in block}
    return {}


def period_summary(frame, temporal_config: Dict) -> Dict:
    """Record the period boundaries so a later reader can check them."""
    from driftguard.temporal import build_temporal_split

    split = build_temporal_split(frame, **(temporal_config or {}))
    periods = ["train", "validation", "backtest"]
    if split.has_forward:
        periods.append("forward")
    return {
        name: [str(lo), str(hi)]
        for name, (lo, hi) in ((p, split.time_range(p)) for p in periods)
    }


def cmd_experiments(args) -> int:
    from driftguard.experiments.tracking import list_experiments

    entries = list_experiments(args.experiments_dir)
    if not entries:
        print(f"no experiments found in {args.experiments_dir}")
        return 0
    for entry in entries:
        print(f"{entry['experiment_id']}  dataset={entry.get('dataset')}  commit={str(entry.get('git_commit'))[:8]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="driftguard", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="dataset commands").add_subparsers(dest="subcommand", required=True)
    inspect = p.add_parser("inspect", help="describe the datasets found in the raw directory")
    inspect.add_argument("--config", default=DEFAULT_CONFIG)
    inspect.add_argument("--raw-dir", default=None)
    inspect.add_argument("--dataset", default=None, choices=["unsw_nb15", "ugr16", "synthetic"])
    inspect.set_defaults(func=cmd_data_inspect)

    fetch = p.add_parser("fetch", help="download a research dataset (never run by tests)")
    fetch.add_argument("--dataset", required=True, choices=["unsw_nb15", "ugr16"])
    fetch.add_argument("--raw-dir", default="data/raw")
    fetch.add_argument("--weeks", default="all", help="'all' or a comma-separated list of UGR'16 weeks")
    fetch.add_argument("--yes", action="store_true", help="actually download")
    fetch.set_defaults(func=cmd_data_fetch)

    train = sub.add_parser("train", help="fit the models and score the backtest")
    train.add_argument("--config", default=DEFAULT_CONFIG)
    train.add_argument("--set", action="append", default=[], help="override a config key, e.g. --set models.enabled='[random_forest]'")
    train.set_defaults(func=cmd_benchmark)

    evaluate = sub.add_parser("evaluate", help="score the forward test (never picks a threshold)")
    evaluate.add_argument("--config", default=DEFAULT_CONFIG)
    evaluate.add_argument("--set", action="append", default=[])
    evaluate.set_defaults(func=cmd_benchmark)

    drift = sub.add_parser("detect-drift", help="drift analysis only")
    drift.add_argument("--config", default=DEFAULT_CONFIG)
    drift.add_argument("--set", action="append", default=[])
    drift.set_defaults(func=cmd_benchmark)

    adapt = sub.add_parser("adapt", help="adaptation study only")
    adapt.add_argument("--config", default=DEFAULT_CONFIG)
    adapt.add_argument("--set", action="append", default=[])
    adapt.set_defaults(func=cmd_benchmark)

    bench = sub.add_parser("benchmark", help="run the full research benchmark")
    bench.add_argument("--config", default=DEFAULT_CONFIG)
    bench.add_argument("--set", action="append", default=[])
    bench.set_defaults(func=cmd_benchmark)

    shift = sub.add_parser("shift", help="run controlled distribution-shift experiments")
    shift.add_argument("--config", default=DEFAULT_CONFIG)
    shift.add_argument("--set", action="append", default=[])
    shift.add_argument("--models", nargs="*", default=None)
    shift.set_defaults(func=cmd_shift)

    ablate = sub.add_parser("ablate", help="run the targeted ablation suite")
    ablate.add_argument("--config", default="configs/ablations.yaml")
    ablate.add_argument("--set", action="append", default=[])
    ablate.set_defaults(func=cmd_ablate)

    report = sub.add_parser("report", help="render the technical report from experiment output")
    report.add_argument("--experiments-dir", default="results/experiments")
    report.add_argument("--output", default="docs/technical_report.pdf")
    report.set_defaults(func=cmd_report)

    experiments = sub.add_parser("experiments", help="list recorded experiments")
    experiments.add_argument("--experiments-dir", default="results/experiments")
    experiments.set_defaults(func=cmd_experiments)

    demo = sub.add_parser("demo", help="end-to-end demo on generated data (no download)")
    demo.add_argument("--config", default="configs/demo.yaml")
    demo.set_defaults(func=cmd_demo)

    return parser



def cmd_ablate(args) -> int:
    """Run the targeted ablation suite and record it as an experiment."""
    from driftguard.experiments.ablations import run_ablations
    from driftguard.experiments.tracking import (
        ExperimentRun,
        experiment_id,
        file_checksums,
    )
    import yaml as _yaml

    config = load_config(args.config)
    for override in getattr(args, "set", []) or []:
        key, _, value = override.partition("=")
        config[key] = _yaml.safe_load(value)

    dataset = config["dataset"]
    raw_dir = resolve_raw_dir(config)
    run = ExperimentRun(
        experiment_id("ablation", dataset["name"], config.get("tag")),
        root=config.get("output", {}).get("experiments_dir", "results/experiments"),
    )
    meta = run.metadata(
        dataset=dataset["name"],
        dataset_checksums={},
        features=list(config.get("features", [])),
        seed=int(config.get("seed", 42)),
        config=config,
        split_description={"note": "the ablation suite re-splits inside each function; "
                                   "see ablation_results.csv for the rows it produced"},
    )
    meta["models"] = config.get("ablation", {}).get("models", [])
    meta["raw_dir"] = raw_dir
    run.write_json("metadata.json", meta)
    run.write_text("config.yaml", _yaml.safe_dump(config, sort_keys=True))

    print(f"running ablations: {', '.join(config.get('ablation', {}).get('names', []))}")
    summary = run_ablations(
        config, Path(run.path), run.experiment_id,
        ablations=config.get("ablation", {}).get("names"),
        models=config.get("ablation", {}).get("models"),
    )
    run.write_json("ablation_summary.json", summary)
    print(f"  rows      : {summary['rows']}")
    print(f"  failed    : {summary['failed']}")
    print(f"experiment written to: {run.path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
