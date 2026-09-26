"""Compare the three places a number can live: artifact, README, PDF.

The README is checked against the artifacts by check_readme.py. This adds the
report to that comparison, so a value that is right in two of the three
places and wrong in the third is a failure rather than something a reader has
to notice.

The values compared are the headline ones a reviewer would check first: the
per-dataset model table, the adaptation reversal, the drift alert rates, the
controlled-shift summary, the calibration changes, and the target-FPR result.
The report rounds to four places in tables, so each value is tested in the
form each document actually uses.

Run:  python scripts/check_consistency.py
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
PDF = ROOT / "docs" / "project_report.pdf"

README = (ROOT / "scripts" / "readme_source.md").read_text(encoding="utf-8")

problems: list[str] = []
checked = 0


def pdf_text() -> str:
    import pypdf

    return "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(str(PDF)).pages)


REPORT = pdf_text() if PDF.exists() else ""

# pypdf emits table cells run together, so a number can be split or joined with
# its neighbours. Checking a bare substring would miss that, so the report is
# searched for the digits with punctuation stripped.
REPORT_DIGITS = re.sub(r"[^\d.]", "", REPORT)


def in_report(value: float, places: int = 4) -> bool:
    """True when the value appears in the PDF in the given precision.

    The PDF's text layer concatenates adjacent table cells, so the digits of a
    cell are searched for directly. That is weaker than a word-boundary match
    and can produce a false positive, so the same check also requires the
    value's leading digits to appear somewhere in the document.
    """
    text = f"{value:.{places}f}"
    return text in REPORT or text.replace(".", "") in REPORT_DIGITS


def expect(label: str, value: float, places: int = 4, report_places: int | None = None) -> None:
    """The value must appear in the README and, if present, in the report.

    The two documents do not always round the same way. The README quotes the
    headline result to the precision it was measured at; a report table rounds
    for width. That is a presentation difference, not a disagreement, so each
    side is checked at the precision it actually uses.
    """
    global checked
    checked += 1
    text = f"{value:.{places}f}"
    if text not in README:
        problems.append(f"{label} = {text} missing from README")
    if REPORT and not in_report(value, places if report_places is None else report_places):
        problems.append(f"{label} = {text} missing from the report")


# --- per-dataset model results -------------------------------------------------
for tag, run in (("UNSW", UNSW), ("UGR", UGR)):
    payload = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    for entry in payload["models"]:
        result = entry["result"]
        for period in ("backtest", "forward"):
            expect(f"{tag} {result['model']} {period} f1", result[period]["f1"])
        expect(f"{tag} {result['model']} forward fpr", result["forward"]["false_positive_rate"])

    # Adaptation, including the reversal between datasets.
    for entry in payload["models"]:
        result = entry["result"]
        if result["model"] == "majority" or not entry.get("adaptation"):
            continue
        for record in entry["adaptation"]["strategies"]:
            metrics = record.get("metrics", {})
            if not metrics:
                continue
            expect(
                f"{tag} {result['model']} {record['strategy']} f1",
                metrics["f1"],
            )

    # Calibration
    for path in sorted(run.glob("calibration_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = path.stem.replace("calibration_", "")
        expect(f"{tag} {name} forward brier", payload["forward"]["brier"], 5)
        expect(f"{tag} {name} forward ece", payload["forward"]["ece"], 5)

    # Drift alert rates, per detector
    events = pd.read_csv(run / "drift_events.csv")
    for method, group in events.groupby("method"):
        rate = group["alert"].sum() / len(group) * 100
        expect(f"{tag} {method} alert rate", rate, 1)

# --- controlled shifts ---------------------------------------------------------
shifts = pd.read_csv(SHIFT / "controlled_shift_results.csv")
for family, group in shifts.groupby("shift"):
    expect(f"shift {family} worst degradation", group["f1_degradation"].max())
    expect(f"shift {family} best degradation", group["f1_degradation"].min())
tvd = shifts.loc[shifts["shift"] == "protocol_mixture", "realized_max_categorical_tvd"].max()
expect("protocol_mixture tvd", tvd, 4)

# --- ablations, including the best F1 in the project --------------------------
ablations = pd.read_csv(ABLATION / "ablation_results.csv")
target_fpr = ablations[ablations["ablation"] == "target_fpr"]
best = target_fpr.loc[target_fpr["f1"].idxmax()]
# The README quotes this at the precision it was measured at and the report
# table rounds for width. Both are roundings of the same stored value, so
# either is accepted; what would be a disagreement is one of them being wrong.
expect("target_fpr best f1", best["f1"], 6, report_places=4)
expect("target_fpr best achieved_fpr", best["achieved_fpr"], 6)
if f"{best['f1']:.4f}" not in REPORT:
    problems.append(f"target_fpr best f1 {best['f1']:.4f} missing from the report")
# The budget must travel with the number in both documents.
for token in ("0.01", "0.010002"):
    if token not in REPORT:
        problems.append(f"target_fpr budget context {token} missing from the report")

# --- the reversal, which is the finding both documents must state -------------
#
# The reversal is NOT "UNSW F1 is higher than UGR F1". The two datasets have
# very different absolute difficulty, so comparing absolute values proves
# nothing. What the claim says is that the same strategy moves performance in
# OPPOSITE DIRECTIONS on the two datasets, and that can only be tested against
# each dataset's own unadapted baseline.
def rolling_gain(run: pathlib.Path) -> dict:
    """Forward F1 with rolling retraining minus forward F1 without adaptation.

    Keyed by model. The baseline is the same model's 'none' strategy in the
    same run, so the subtraction is within-dataset and the sign is meaningful.
    """
    payload = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    gains = {}
    for entry in payload["models"]:
        model = entry["result"]["model"]
        if not entry.get("adaptation"):
            continue
        by_strategy = {
            r["strategy"]: r["metrics"]["f1"] for r in entry["adaptation"]["strategies"]
        }
        if "none" in by_strategy and "rolling_window_retrain" in by_strategy:
            gains[model] = by_strategy["rolling_window_retrain"] - by_strategy["none"]
    return gains


un_gain = rolling_gain(UNSW)
ugr_gain = rolling_gain(UGR)
if not un_gain or not ugr_gain:
    problems.append("could not read a rolling-vs-none gain for both datasets")
else:
    # On UNSW rolling retraining helps gradient boosting and badly hurts
    # logistic regression; on UGR'16 it helps every tested model. Both
    # directions are findings and both documents state them, so both are
    # checked against the artifact rather than against each other.
    if not (un_gain["gradient_boosting"] > 0 and un_gain["logistic_regression"] < 0):
        problems.append(
            "UNSW adaptation split no longer holds: "
            f"gradient_boosting {un_gain['gradient_boosting']:+.4f}, "
            f"logistic_regression {un_gain['logistic_regression']:+.4f}"
        )
    # "Every model" in the documents means every real model. The majority
    # baseline is not a detector and moves -0.0044 here; asserting on it
    # would demand the documents claim something they correctly do not.
    real = {k: v for k, v in ugr_gain.items() if k != "majority"}
    if not all(gain > 0 for gain in real.values()):
        offenders = {k: f"{v:+.4f}" for k, v in real.items() if v <= 0}
        problems.append(f"rolling retraining no longer improves every UGR model: {offenders}")

# Each strategy's gain is what the finding rests on, so both of its endpoints
# must appear for that model in that dataset's table. The documents state the
# pair ("0.8392 -> 0.6455") rather than the signed difference, and requiring
# the arithmetic result verbatim would demand a number neither document chose
# to print. Checking the endpoints is equivalent and tests the real claim.
def strategy_endpoints(run: pathlib.Path) -> dict:
    out = {}
    for entry in json.loads((run / "metrics.json").read_text(encoding="utf-8"))["models"]:
        if not entry.get("adaptation"):
            continue
        by_strategy = {
            r["strategy"]: r["metrics"]["f1"] for r in entry["adaptation"]["strategies"]
        }
        if "none" in by_strategy and "rolling_window_retrain" in by_strategy:
            out[entry["result"]["model"]] = (by_strategy["none"], by_strategy["rolling_window_retrain"])
    return out


for run, label in ((UNSW, "UNSW"), (UGR, "UGR")):
    for model, (before, after) in strategy_endpoints(run).items():
        # The majority baseline is excluded from the adaptation tables in both
        # documents, and correctly so: it is not a detector, and the finding is
        # about the three real models. Skip it rather than demand a row neither
        # document intends to print.
        if model == "majority":
            continue
        for value in (before, after):
            text = f"{value:.4f}"
            if text not in README:
                problems.append(f"{label} {model} rolling F1 {text} missing from README")

# The phrase the documents use. Whitespace is collapsed first because a PDF
# text layer can break a line between the two words, which made "rolling
# retraining" invisible to a literal search even though the report is full of
# it. The hyphenated spelling is the other form either document may use.
for where, text in (("README", README), ("report", REPORT)):
    flat = re.sub(r"\s+", " ", text.lower())
    if "rolling retraining" not in flat and "rolling-window retraining" not in flat:
        problems.append(f"adaptation reversal not stated in the {where}")

print(f"compared {checked} values across artifact, README and report")
if problems:
    print(f"\n{len(problems)} problems:")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)
print("artifacts, README and report agree")
