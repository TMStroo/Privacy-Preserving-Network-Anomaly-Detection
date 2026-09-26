"""Check the README's numbers against the experiment artifacts.

The README is prose, so the only way to know a figure in it is still true after
an experiment is re-run is to compare it with the file it claims to come from.
This reads the artifacts and reports every headline value, so a mismatch is a
failure rather than something a reader has to notice.
"""

import json
import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "experiments"
UNSW = RUNS / "20260925T195935Z_temporal_unsw_nb15_full_6167ea"
UGR = RUNS / "20260925T220604Z_temporal_ugr16_78d5c1"
SHIFT = RUNS / "20260926T003707Z_shift_unsw_nb15_eae6e4"
ABLATION = RUNS / "20260925T205657Z_ablation_unsw_nb15_d6d364"

README = (ROOT / "scripts" / "readme_source.md").read_text(encoding="utf-8")

problems = []
checked = 0


def expect(label: str, value: float, places: int = 4) -> None:
    """The README must contain this value, formatted the way it is written."""
    global checked
    checked += 1
    if f"{value:.{places}f}" not in README:
        problems.append(f"{label}: {value:.{places}f} is not in the README")


def model_metrics(run: pathlib.Path) -> dict:
    payload = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    return {e["result"]["model"]: e for e in payload["models"]}


for tag, run in (("UNSW", UNSW), ("UGR", UGR)):
    entries = model_metrics(run)
    for name, entry in entries.items():
        result = entry["result"]
        expect(f"{tag} {name} backtest f1", result["backtest"]["f1"])
        expect(f"{tag} {name} forward f1", result["forward"]["f1"])
        expect(f"{tag} {name} forward fpr", result["forward"]["false_positive_rate"])
        if name == "majority":
            continue
        for record in entry["adaptation"]["strategies"]:
            metrics = record["metrics"]
            expect(f"{tag} {name} {record['strategy']} f1", metrics["f1"])
            expect(f"{tag} {name} {record['strategy']} fpr", metrics["false_positive_rate"], 3)

    # Calibration
    for path in sorted(run.glob("calibration_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = path.stem.replace("calibration_", "")
        expect(f"{tag} {name} backtest brier", payload["backtest"]["brier"], 5)
        expect(f"{tag} {name} forward brier", payload["forward"]["brier"], 5)
        expect(f"{tag} {name} forward ece", payload["forward"]["ece"], 5)

    # Drift alert rates
    events = pd.read_csv(run / "drift_events.csv")
    for method, group in events.groupby("method"):
        rate = group["alert"].sum() / len(group) * 100
        expect(f"{tag} {method} alert rate", rate, 1)

# Controlled shifts
shifts = pd.read_csv(SHIFT / "controlled_shift_results.csv")
verified = int(shifts["realized_verified"].astype(str).str.lower().eq("true").sum())
if f"**All {verified} of {len(shifts)} rows were verified**" not in README:
    problems.append(f"shift verification: README does not state {verified}/{len(shifts)}")
for family, group in shifts.groupby("shift"):
    if family not in README:
        problems.append(f"shift family missing from README: {family}")
tvd = shifts.loc[shifts["shift"] == "protocol_mixture", "realized_max_categorical_tvd"].max()
expect("protocol_mixture tvd", tvd, 4)
byte_rate = shifts.loc[shifts["shift"] == "byte_rate", "realized_max_abs_cohens_d"]
expect("byte_rate max d", byte_rate.max(), 3)
expect("byte_rate min d", byte_rate.min(), 3)

# Ablations
ablations = pd.read_csv(ABLATION / "ablation_results.csv")
summary = json.loads((ABLATION / "ablation_summary.json").read_text(encoding="utf-8"))
if f"{summary['rows']} rows, {summary['failed']} failures" not in README:
    problems.append(f"ablation: README does not state {summary['rows']} rows, {summary['failed']} failures")
for family in ablations["ablation"].unique():
    if family not in README:
        problems.append(f"ablation family missing from README: {family}")

# The target-FPR sweep, quoted in full, including the best F1 in the project.
target_fpr = ablations[ablations["ablation"] == "target_fpr"]
if target_fpr.empty:
    problems.append("no target_fpr rows in the ablation artifact")
else:
    for _, row in target_fpr.iterrows():
        for column in ("f1", "precision", "recall", "achieved_fpr"):
            value = row[column]
            if pd.isna(value):
                continue
            if f"{value:.6f}" not in README and f"{value:.4f}" not in README:
                problems.append(
                    f"target_fpr {row['model']} {row['variant']} {column}={value} not in README"
                )
        # The budget is a label. The table may write it as 0.1 or 0.10, so
        # either form is accepted; what matters is that the number appears.
        budget = row["target_fpr"]
        forms = {f"{budget:g}", f"{budget:.2f}"}
        if not any(form in README for form in forms):
            problems.append(
                f"target_fpr {row['model']} {row['variant']} budget={budget} not in README"
            )
    best = target_fpr.loc[target_fpr["f1"].idxmax()]
    # The best row must be quoted to six places and named as an ablation result.
    if f"{best['f1']:.6f}" not in README:
        problems.append(f"best target_fpr F1 {best['f1']:.6f} not quoted in full")
    for token in ("random_forest", "ablation"):
        if token not in README:
            problems.append(f"target_fpr context missing: {token}")
    # The budget must sit in the same sentence region as the result, so the
    # number cannot be read without it.
    for phrase in ("target budget", "achieved false-positive rate", "default 0.05 budget"):
        if phrase not in README:
            problems.append(f"target_fpr budget context missing: {phrase!r}")
    # It must not be promoted into a "best model" claim.
    for claim in ("RF is best", "random forest is the best model", "best model overall"):
        if claim in README:
            problems.append(f"target_fpr result was over-generalised: {claim!r}")

# Nothing stale
for pattern, description in (
    (r"\bTODO\b", "TODO"),
    (r"\bFIXME\b", "FIXME"),
    (r"\bplaceholder\b", "placeholder"),
    (r"not run", "not run"),
    (r"\bpending\b", "pending"),
    (r"[A-Z]:[\\/]Users", "absolute Windows path"),
    (r"[A-Z]:[\\/]projects", "absolute Windows path"),
    (r"/home/[a-z]", "absolute home path"),
):
    for match in re.finditer(pattern, README, re.IGNORECASE):
        line = README[: match.start()].count("\n") + 1
        problems.append(f"{description} on line {line}")

print(f"checked {checked} values from the artifacts")
if problems:
    print(f"\n{len(problems)} problems:")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)
print("README agrees with every artifact it cites")
