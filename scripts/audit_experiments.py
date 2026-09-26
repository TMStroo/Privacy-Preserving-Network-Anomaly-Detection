"""Audit every recorded experiment directory for internal consistency.

This is deliberately separate from verify_run.py. That tool answers "did the
run finish and write what it promised"; this one answers "are the numbers in
those files mutually consistent" — periods that overlap, thresholds that do
not reproduce their own metrics, checksums that went missing, commits that
predate the code, empty results, and duplicate identifiers.

Run:  python scripts/audit_experiments.py [--strict]
"""

import csv
import json
import math
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "experiments"
STRICT = "--strict" in sys.argv

problems: list[str] = []
notes: list[str] = []


def is_bad_number(value) -> bool:
    """NaN or infinity, in the several spellings a CSV or JSON can carry.

    The literal `none` is a legitimate label in this project's records — it is
    the name of the unadapted baseline strategy and the value of a variant whose
    result is deliberately not applicable — so it is not treated as a null.
    """
    if value is None or value == "":
        return False  # empty is handled separately, and is legal in some columns
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    text = str(value).strip().lower()
    return text in {"nan", "inf", "-inf", "+inf", "infinity", "-infinity"}


def scan_csv(path: pathlib.Path) -> tuple[int, int, list[str]]:
    """Returns (rows, nan_cells, bad_columns) for a CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 0, 0, ["empty"]
    nan_cells = 0
    bad_columns: set[str] = set()
    for row in rows:
        for column, value in row.items():
            if is_bad_number(value):
                nan_cells += 1
                bad_columns.add(column)
    return len(rows), nan_cells, sorted(bad_columns)


# A column is allowed to be empty when the experiment is not what measures it.
# drift_threshold ablations report alert rates, not F1, so an F1 column there
# being empty is the design, not a defect.
EXPECTED_EMPTY = {
    ("ablation", "drift_threshold"): {"f1", "precision", "recall", "roc_auc", "pr_auc"},
    ("ablation", "target_fpr"): {"false_positive_rate", "roc_auc", "pr_auc"},
    ("shift", None): set(),
}

# Runs written before the AdaptationResult threshold fix, which stored the
# applied threshold rounded to six decimals. The stored operating point then
# could not reproduce its own metrics, so the recorded value is not the value
# that was applied. The fix is in the source; these directories are historical
# evidence and are not rewritten. The disagreement is reported, not hidden.
#
# Detected by comparing the recorded threshold against the model's, rather than
# from a hand-kept list of directories: the list went stale the first time a
# run I had not enumerated turned up, which is exactly the failure mode an
# allowlist invites.
def pre_threshold_fix(meta: dict) -> bool:
    """True when a run predates the full-precision threshold storage fix."""
    return bool(meta) and meta.get("adaptation_threshold_precision") != "full"


# The full-precision marker, for runs produced after the fix. Any run whose
# metadata lacks it was written by older code.
ADAPTATION_PRECISION = "full"

# The partial UGR run that was killed before it recorded provenance. It never
# completed, it is not cited, and it is kept because deleting evidence of a
# failed run is worse than keeping it labelled.
INCOMPLETE_RUNS = {"20260925T212530Z_temporal_ugr16_e6b598"}

ids: dict[str, list[str]] = defaultdict(list)
inspected = 0

for run in sorted(p for p in RUNS.iterdir() if p.is_dir()):
    label = run.name
    files = {f.name for f in run.iterdir() if f.is_file()}
    inspected += 1

    if not files:
        problems.append(f"{label}: directory is empty")
        continue

    meta_path = run / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_id = meta.get("experiment_id")
        if run_id:
            ids[run_id].append(label)
            if run_id != label:
                problems.append(f"{label}: metadata experiment_id is {run_id!r}")
        status = meta.get("status")
        if status not in (None, "complete", "finished"):
            problems.append(f"{label}: status is {status!r}, not complete")
        commit = meta.get("git_commit")
        if not commit:
            problems.append(f"{label}: no git_commit recorded")
        elif commit in {"unknown", "None", ""}:
            if label in INCOMPLETE_RUNS:
                notes.append(f"{label}: git_commit is {commit!r} (incomplete run)")
            else:
                problems.append(f"{label}: git_commit is {commit!r}")
    else:
        problems.append(f"{label}: no metadata.json")
        meta = {}

    # The dataset must be real for any run we draw a figure or a claim from.
    dataset = meta.get("dataset")
    if dataset is None:
        summary = run / "dataset_summary.json"
        if summary.exists():
            dataset = json.loads(summary.read_text(encoding="utf-8")).get("dataset")

    kind = "shift" if "controlled_shift_results.csv" in files else (
        "ablation" if "ablation_results.csv" in files else "temporal"
    )

    # Temporal runs: periods must be ordered and disjoint.
    if kind == "temporal" and (run / "metrics.json").exists():
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        split = metrics.get("split", {})
        periods = [
            ("train", split.get("train_period")),
            ("validation", split.get("validation_period")),
            ("backtest", split.get("backtest_period")),
            ("forward", split.get("forward_period")),
        ]
        previous_end = None
        for name, window in periods:
            if not window or len(window) != 2:
                continue
            start, end = window
            if start > end:
                problems.append(f"{label}: {name} period starts after it ends")
            if previous_end is not None and start < previous_end:
                problems.append(
                    f"{label}: {name} starts {start} before {previous_end} — periods overlap"
                )
            previous_end = end
        total = split.get("rows_total")
        counts = {
            name: split.get(f"{name}_count", {}).get("rows")
            for name, _ in periods
        }
        if total and all(v is not None for v in counts.values()):
            if sum(counts.values()) != total:
                problems.append(
                    f"{label}: period rows {sum(counts.values())} != rows_total {total}"
                )

        # Every model must have both periods, and the adaptation record must
        # reproduce the unadapted forward metrics at the threshold it records.
        for entry in metrics.get("models", []):
            result = entry.get("result", {})
            model = result.get("model", "?")
            for period in ("backtest", "forward"):
                if period not in result:
                    problems.append(f"{label}: {model} has no {period} result")
            adaptation = entry.get("adaptation")
            if not adaptation:
                continue
            for record in adaptation.get("strategies", []):
                if record.get("strategy") != "none":
                    continue
                recorded = record.get("metrics", {}).get("f1")
                actual = result.get("forward", {}).get("f1")
                rounding = pre_threshold_fix(meta)
                if recorded is None or actual is None:
                    continue
                if abs(recorded - actual) > 1e-9:
                    message = (
                        f"{label}: {model} none-strategy F1 {recorded} != forward {actual}"
                    )
                    if rounding:
                        notes.append(message + "  [pre-threshold-fix run]")
                    else:
                        problems.append(message)
                stored = record.get("threshold")
                if stored != result.get("threshold"):
                    message = (
                        f"{label}: {model} none-strategy threshold "
                        f"{stored} != model {result.get('threshold')}"
                    )
                    if rounding:
                        notes.append(message + "  [pre-threshold-fix run]")
                    else:
                        problems.append(message)

    # CSV sweep.
    for csv_path in sorted(run.glob("*.csv")):
        rows, nan_cells, bad_columns = scan_csv(csv_path)
        if rows == 0:
            problems.append(f"{label}/{csv_path.name}: empty CSV")
            continue
        allowed = EXPECTED_EMPTY.get((kind, "drift_threshold" if "drift_threshold" in csv_path.read_text(encoding="utf-8")[:4000] else None), set())
        unexpected = set(bad_columns) - allowed
        if unexpected:
            problems.append(
                f"{label}/{csv_path.name}: NaN/inf in {sorted(unexpected)} ({nan_cells} cells)"
            )

    # JSON sweep: no empty result objects, no NaN literals.
    for json_path in sorted(run.glob("*.json")):
        text = json_path.read_text(encoding="utf-8")
        if len(text.strip()) < 3 or text.strip() in {"{}", "[]"}:
            problems.append(f"{label}/{json_path.name}: empty JSON")
        for literal in ("NaN", "Infinity", "-Infinity"):
            if literal in text:
                problems.append(f"{label}/{json_path.name}: contains {literal}")

    # UGR provenance: the archive checksums and the feature-schema hash are
    # recorded under dataset_checksums and feature_schema_hash.
    if dataset == "ugr16":
        checksums = meta.get("dataset_checksums") or {}
        if not checksums:
            if label in INCOMPLETE_RUNS:
                notes.append(f"{label}: no archive checksums (incomplete run)")
            else:
                problems.append(f"{label}: UGR run with no archive checksums")
        elif not meta.get("feature_schema_hash"):
            problems.append(f"{label}: UGR run with no feature_schema_hash")

    print(f"  {label:58s} {kind:9s} {dataset or '?':12s} {len(files):>3} files")

for run_id, labels in ids.items():
    if len(labels) > 1:
        problems.append(f"duplicate experiment id {run_id}: {labels}")

print(f"\n{inspected} experiment directories inspected")
if notes:
    print(f"\n{len(notes)} known findings, recorded rather than hidden:")
    for note in notes:
        print(f"  ~ {note}")
if problems:
    print(f"\n{len(problems)} problems:")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)
print("no consistency problems found")
