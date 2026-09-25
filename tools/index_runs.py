"""Build results/index.csv: one row per recorded experiment, no bulk data.

The experiment directories hold 100+ MB of per-row prediction CSVs, which do not
belong in git. But the summary that says which run produced which number is
exactly what a reviewer needs and is a few kilobytes, so that is tracked
instead. The index is generated, never typed, so it cannot claim a run exists
that is not on disk.

Regenerate with:  python tools/index_runs.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "results" / "experiments"
SUPERSEDED = ROOT / "results" / "superseded"
OUT = ROOT / "results" / "index.csv"

FIELDS = [
    "experiment_id",
    "kind",
    "dataset",
    "status",
    "git_commit",
    "seed",
    "models",
    "adaptation_strategies",
    "drift_comparisons",
    "train_rows",
    "forward_rows",
    "forward_start",
    "forward_end",
    "best_forward_f1",
    "worst_f1_degradation",
    "package_python",
    "notes",
]


def _kind(experiment_id: str) -> str:
    for candidate in ("temporal", "shift", "ablation"):
        if f"_{candidate}_" in experiment_id:
            return candidate
    return "unknown"


def _count_strategies(metrics: dict) -> str:
    found = set()
    for entry in metrics.get("models", []):
        for key in (entry.get("adaptation") or {}):
            found.add(key)
    return ";".join(sorted(found))


def _describe(directory: Path, status: str) -> dict:
    row = {field: "" for field in FIELDS}
    row["experiment_id"] = directory.name
    row["kind"] = _kind(directory.name)
    # The caller's verdict wins. A directory under superseded/ stays superseded
    # even if it also looks finished, because a finished-but-wrong run is exactly
    # what the caller is telling us.
    row["status"] = status or "unknown"

    meta_path = directory / "metadata.json"
    if not meta_path.is_file():
        row["notes"] = "no metadata.json: this run never wrote its header"
        return row
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        row["notes"] = f"metadata unreadable: {exc}"
        return row

    row["dataset"] = str(meta.get("dataset", ""))
    row["git_commit"] = str(meta.get("git_commit", ""))[:12]
    row["seed"] = str(meta.get("random_seed", ""))
    row["package_python"] = str((meta.get("package_versions") or {}).get("python", ""))

    if meta.get("superseded"):
        row["notes"] = "SUPERSEDED: " + str(meta["superseded"].get("reason", ""))[:200]

    split = meta.get("temporal_split") or {}
    row["train_rows"] = str((split.get("train_count") or {}).get("rows", ""))
    forward = split.get("forward_period")
    if isinstance(forward, dict):
        row["forward_start"], row["forward_end"] = str(forward.get("start", "")), str(forward.get("end", ""))
    elif isinstance(forward, (list, tuple)) and len(forward) == 2:
        row["forward_start"], row["forward_end"] = str(forward[0]), str(forward[1])
    row["forward_rows"] = str((split.get("forward_count") or {}).get("rows", ""))

    metrics_path = directory / "metrics.json"
    if not metrics_path.is_file():
        if row["status"] == "unknown":
            row["status"] = "incomplete"
        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + "no metrics.json: run did not finish"
        return row

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if row["status"] == "unknown":
            row["status"] = "corrupt"
        row["notes"] = f"metrics.json unreadable: {exc}"
        return row

    models = metrics.get("models") or []
    row["models"] = ";".join(str(m.get("result", {}).get("model", "")) for m in models)
    row["adaptation_strategies"] = _count_strategies(metrics)
    if metrics.get("drift_summary"):
        row["drift_comparisons"] = str(sum(
            int(v.get("comparisons", 0) or 0) for v in metrics["drift_summary"].values()
            if isinstance(v, dict)
        ) or "")
    if models:
        fwd = [(m.get("result", {}).get("forward", {}) or {}).get("f1") for m in models]
        fwd = [f for f in fwd if isinstance(f, (int, float))]
        if fwd:
            row["best_forward_f1"] = f"{max(fwd):.4f}"
        degs = [m.get("result", {}).get("f1_degradation") for m in models]
        degs = [d for d in degs if isinstance(d, (int, float))]
        if degs:
            row["worst_f1_degradation"] = f"{max(degs):+.4f}"
    if row["status"] == "unknown":
        row["status"] = "complete"
    return row


def build() -> list:
    rows = []
    if EXPERIMENTS.is_dir():
        for directory in sorted(EXPERIMENTS.iterdir()):
            if directory.is_dir():
                rows.append(_describe(directory, ""))
    if SUPERSEDED.is_dir():
        for directory in sorted(SUPERSEDED.iterdir()):
            if directory.is_dir():
                rows.append(_describe(directory, "superseded"))
    return rows


def main() -> int:
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    complete = sum(1 for r in rows if r["status"] == "complete")
    print(f"{OUT.relative_to(ROOT)}: {len(rows)} runs, {complete} complete")
    for row in rows:
        if row["status"] not in ("complete", "superseded"):
            print(f"  {row['status']:11s} {row['experiment_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
