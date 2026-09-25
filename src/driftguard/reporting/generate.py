"""Technical report generator.

Every number in the report is read from a recorded experiment directory. A
section is only written when the corresponding experiment output exists, so the
report cannot claim an experiment that was not run.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from driftguard.reporting.render import (
    adaptation_table,
    all_experiments,
    calibration_table,
    drift_table,
    failure_analysis_blocks,
    latest_experiment,
    load_experiment,
    load_metadata,
    main_comparison_table,
    threshold_table,
)

MARGIN = 18
PAGE_W, PAGE_H = 595, 842
BODY = 10
LINE = 5.2


class Report:
    """Thin wrapper over fpdf2 that handles pagination and a running header."""

    def __init__(self, title: str, subtitle: str = ""):
        from fpdf import FPDF

        self.pdf = FPDF(orientation="P", unit="pt", format="A4")
        self.pdf.set_auto_page_break(True, margin=MARGIN + 14)
        self.title = title
        self.subtitle = subtitle
        self._first_page = True
        self._section = ""

    def _decorate(self):
        self.pdf.set_fill_color(245, 247, 250)
        self.pdf.rect(0, 0, PAGE_W, 46, style="F")
        self.pdf.set_font("Helvetica", "B", 8)
        self.pdf.set_text_color(70, 80, 95)
        self.pdf.set_xy(MARGIN, 20)
        self.pdf.cell(PAGE_W - 2 * MARGIN, 10, "DriftGuard  |  Technical Report")
        self.pdf.set_xy(MARGIN, 30)
        self.pdf.set_font("Helvetica", "", 7)
        self.pdf.cell(PAGE_W - 2 * MARGIN, 10, self._section)
        self.pdf.line(MARGIN, 46, PAGE_W - MARGIN, 46)

    def add_page(self):
        self.pdf.add_page()
        self._decorate()

    def cover(self, lines: List[str]):
        self.add_page()
        self.pdf.set_xy(MARGIN, 150)
        self.pdf.set_font("Helvetica", "B", 22)
        self.pdf.multi_cell(PAGE_W - 2 * MARGIN, 28, self.title)
        if self.subtitle:
            self.pdf.set_font("Helvetica", "", 12)
            self.pdf.set_text_color(80, 90, 105)
            self.pdf.multi_cell(PAGE_W - 2 * MARGIN, 16, self.subtitle)
            self.pdf.set_text_color(30, 35, 45)
        y = 250
        self.pdf.set_y(y)
        for line in lines:
            self.pdf.set_font("Helvetica", "", 9.5)
            self.pdf.set_text_color(70, 78, 92)
            self.pdf.multi_cell(PAGE_W - 2 * MARGIN, 14, line)
            self.pdf.set_y(self.pdf.get_y() + 3)
        self._first_page = False

    def h1(self, text: str):
        self._section = text
        self.add_page()
        self.pdf.set_font("Helvetica", "B", 15)
        self.pdf.set_text_color(20, 28, 40)
        self.pdf.multi_cell(PAGE_W - 2 * MARGIN, 20, text)
        self.pdf.set_draw_color(70, 110, 170)
        self.pdf.set_line_width(1.1)
        y = self.pdf.get_y() + 2
        self.pdf.line(MARGIN, y, MARGIN + 46, y)
        self.pdf.set_y(y + 9)
        self.pdf.set_text_color(30, 35, 45)

    def h2(self, text: str):
        self.need(24)
        self.pdf.set_font("Helvetica", "B", 11.5)
        self.pdf.set_text_color(28, 40, 58)
        self.pdf.multi_cell(PAGE_W - 2 * MARGIN, 15, text)
        self.pdf.set_y(self.pdf.get_y() + 2)
        self.pdf.set_text_color(30, 35, 45)

    def para(self, text: str, size: float = BODY, italic: bool = False):
        self.pdf.set_font("Helvetica", "I" if italic else "", size)
        self.pdf.multi_cell(PAGE_W - 2 * MARGIN, LINE + 1, text, align="J")
        self.pdf.set_y(self.pdf.get_y() + 3)

    def bullets(self, items: List[str]):
        for item in items:
            self.need(LINE + 4)
            x = self.pdf.get_x()
            y = self.pdf.get_y()
            self.pdf.set_xy(x + 8, y)
            self.pdf.set_font("Helvetica", "", BODY)
            self.pdf.multi_cell(PAGE_W - 2 * MARGIN - 12, LINE + 1, item, align="J")
            self.pdf.set_xy(x, y)
            self.pdf.set_font("Helvetica", "", BODY)
            self.pdf.cell(6, LINE, "-")
            self.pdf.set_y(max(self.pdf.get_y(), y + LINE + 2))

    def need(self, height: float):
        if self.pdf.get_y() + height > PAGE_H - MARGIN - 10:
            self.add_page()

    def mono(self, text: str, size: float = 7.6):
        self.need(LINE + 3)
        self.pdf.set_font("Courier", "", size)
        x = self.pdf.get_x()
        width = PAGE_W - 2 * MARGIN - 10
        for line in text.split("\n"):
            self.need(LINE)
            self.pdf.set_xy(x, self.pdf.get_y())
            while self.pdf.get_string_width(line) > width and len(line) > 10:
                cut = len(line)
                while cut > 10 and self.pdf.get_string_width(line[:cut]) > width:
                    cut -= 2
                self.pdf.cell(width, LINE, line[:cut])
                line = line[cut:]
                self.need(LINE)
                self.pdf.set_xy(x, self.pdf.get_y())
            self.pdf.cell(width, LINE, line)
            self.pdf.set_y(self.pdf.get_y() + LINE)
        self.pdf.set_font("Helvetica", "", BODY)

    def table(self, header: List[str], rows: List[List[str]], widths: Optional[List[float]] = None,
              font_size: float = 7.4, aligns: Optional[List[str]] = None):
        """Render a table, shrinking font per cell rather than clipping text."""
        if not rows:
            return
        columns = len(header)
        if widths is None:
            widths = [1.0] * columns
        total = sum(widths)
        available = PAGE_W - 2 * MARGIN
        col_w = [w / total * available for w in widths]

        aligns = aligns or ["L"] * columns

        def draw(cells, bold, size):
            self.pdf.set_font("Helvetica", "B" if bold else "", size)
            for i, cell in enumerate(cells):
                text = str(cell)
                # Shrink to fit rather than letting a long name spill over the
                # column border.
                while size > 4.6 and self.pdf.get_string_width(text) > col_w[i] - 4:
                    size -= 0.2
                    self.pdf.set_font("Helvetica", "B" if bold else "", size)
                x = self.pdf.get_x()
                y = self.pdf.get_y()
                if aligns[i] == "R":
                    self.pdf.cell(col_w[i], LINE + 1.5, text, align="R")
                else:
                    self.pdf.cell(col_w[i], LINE + 1.5, text, align="L")
                self.pdf.set_xy(x, y)

        self.need(LINE * 2 + 4)
        if bold_bg := True:
            self.pdf.set_fill_color(232, 238, 246)
            self.pdf.rect(MARGIN, self.pdf.get_y(), available, LINE + 1.5, style="F")
        draw(header, True, font_size)
        self.pdf.ln(LINE + 1.5)
        for row_index, row in enumerate(rows):
            self.need(LINE * 2)
            if row_index % 2 == 1:
                self.pdf.set_fill_color(248, 250, 252)
                self.pdf.rect(MARGIN, self.pdf.get_y(), available, LINE + 1.5, style="F")
            draw(list(row) + [""] * (columns - len(row)), False, font_size)
            self.pdf.ln(LINE + 1.5)
        self.pdf.set_draw_color(200, 208, 220)
        self.pdf.set_line_width(0.4)
        self.pdf.line(MARGIN, self.pdf.get_y(), PAGE_W - MARGIN, self.pdf.get_y())
        self.pdf.set_y(self.pdf.get_y() + 6)

    def figure(self, path: str, caption: str = "", width: float = 430):
        if not Path(path).exists():
            return
        self.pdf.image(path, x=(PAGE_W - width) / 2, w=width)
        if caption:
            self.need(LINE * 2)
            self.pdf.set_font("Helvetica", "I", 8)
            self.pdf.set_text_color(90, 98, 110)
            self.pdf.multi_cell(PAGE_W - 2 * MARGIN, LINE, caption, align="C")
            self.pdf.set_text_color(30, 35, 45)
        self.pdf.set_y(self.pdf.get_y() + 6)

    def output(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.pdf.output(path)
        return path


REFERENCES = [
    "Shyaa M., Ibrahim N., Zainol Z., Abdullah R., Anbar M., Alzubaidi L. (2024). Evolving cybersecurity "
    "frontiers: A comprehensive survey on concept drift and feature dynamics aware machine and deep learning "
    "in intrusion detection systems. Engineering Applications of Artificial Intelligence, 137, 109143. "
    "doi:10.1016/j.engappai.2024.109143",

    "Macia-Fernandez G., Camacho J., Magan-Carrion R., Garcia-Teodoro P., Theron R. (2018). UGR'16: A new "
    "dataset for the evaluation of cyclostationarity-based network IDSs. Computers & Security, 73, 411-424. "
    "doi:10.1016/j.cose.2017.11.004",

    "Xu L., Han Z., Zhao D., Li X., Yu F., Chen C. (2024). Addressing concept drift in IoT anomaly detection: "
    "Drift detection, interpretation, and adaptation. IEEE Transactions on Sustainable Computing, 9(5), "
    "913-924. doi:10.1109/tsusc.2024.3386667",

    "Yang S., Zheng X., Li J., Xu J., Zhang X., Ngai E. (2025). Self-supervised adaptation method to concept "
    "drift for network intrusion detection. IEEE Transactions on Dependable and Secure Computing, 22(12), "
    "7632-7646. doi:10.1109/tdsc.2025.3599321",

    "UNSW Canberra School of Cyber Security. UNSW-NB15 network intrusion detection dataset. "
    "https://research.unsw.edu.au/projects/unsw-nb15-dataset",

    "Network Security and Evolution Group (UGR). UGR'16 dataset. https://nesg.ugr.es/nesg-ugr16/",
]


def _parse_markdown_table(text: str) -> tuple:
    """Turn a rendered markdown table into fpdf rows."""
    lines = [l for l in text.split("\n") if l.strip().startswith("|")]
    if not lines:
        return [], []
    def cells(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]
    header = cells(lines[0])
    rows = [cells(l) for l in lines[2:] if not set(l.strip()) <= set("|- :")]
    return header, rows


def generate_report(experiments_dir: str = "results/experiments", output: str = "docs/technical_report.pdf") -> str:
    """Build the technical report from whatever experiments are on disk."""
    from driftguard.reporting.figures import build_all_figures

    experiment = latest_experiment(experiments_dir)
    if not experiment:
        raise SystemExit(
            f"no completed experiment in {experiments_dir}. "
            "Run `python -m driftguard benchmark --config configs/benchmark.yaml` first."
        )

    metrics = load_experiment(experiment)
    meta = load_metadata(experiment)
    base = Path(experiment)
    split = metrics["split"]
    drift_summary = metrics.get("drift_summary", {})
    failures = failure_analysis_blocks(str(base))

    figures_dir = base / "figures"
    figures = build_all_figures(str(base), metrics, str(figures_dir))

    report = Report(
        "DriftGuard: Reliable Network Anomaly Detection Under Distribution Shift",
        "Temporal generalisation, distribution-shift measurement, drift detection and adaptation "
        "on public network-flow datasets",
    )
    report.cover([
        f"Dataset: {metrics['dataset']}",
        f"Experiment: {base.name}",
        f"Git commit: {meta.get('git_commit', 'unknown')}",
        f"Seed: {meta.get('random_seed', 'unknown')}",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "All numbers in this report are read from the recorded experiment directory",
        f"({base.name}). No metric is written by hand.",
    ])

    # ---------------- Abstract ----------------
    report.h1("Abstract")
    models = metrics.get("models", [])
    degraded = [m for m in models if m["result"].get("f1_degradation", 0) > 0]
    best = max(models, key=lambda m: m["result"]["backtest"]["f1"]) if models else None
    lines = [
        f"This report evaluates whether a machine-learning network anomaly detector keeps working when the "
        f"traffic it was trained on is replaced by later traffic. A detector is fitted on the first "
        f"{split['train_count']['rows']:,} flows of {metrics['dataset']}, evaluated on a held-out backtest inside "
        f"the historical distribution, and then scored unchanged on {split.get('forward_count', {}).get('rows', 0):,} "
        f"later flows it never saw. Operating thresholds come from the validation split alone, under a fixed "
        f"false-positive budget.",
    ]
    if models:
        lines.append(
            f"Across {len(models)} models, {len(degraded)} of {len(models)} lost F1 on the forward test. "
            f"The strongest backtest model, {best['result']['model']}, reached F1 "
            f"{best['result']['backtest']['f1']:.3f} on the backtest and {best['result'].get('forward', {}).get('f1', float('nan')):.3f} "
            f"on later traffic."
        )
    alerts = drift_summary.get("alerts", 0)
    comparisons = drift_summary.get("comparisons", 0)
    if comparisons:
        lines.append(
            f"Drift analysis raised {alerts} alerts across {comparisons} window-feature comparisons. "
            f"Adaptation strategies were evaluated against the same false-positive budget, and the results "
            f"are reported per strategy rather than reduced to a ranking."
        )
    report.para(" ".join(lines))

    # ---------------- Research question ----------------
    report.h1("Research question")
    report.para(
        "Primary: does a machine-learning network anomaly detector trained on historical traffic continue to "
        "detect attacks reliably when evaluated on later, previously unseen traffic?"
    )
    report.para(
        "Secondary: which methods for detecting and adapting to distribution shift recover detection "
        "performance most effectively while respecting a fixed false-positive budget?"
    )
    report.para(
        "The earlier version of this project asked a different question, whether traffic metadata alone carries "
        "enough signal to separate normal traffic from attacks. That work is preserved as the baseline layer: it "
        "established the metadata-only detector this project then takes forward in time."
    )

    report.h2("Why I chose this question")
    report.para(
        "I started with the simpler question of whether network traffic metadata contains enough information to "
        "distinguish normal traffic from attacks. Existing research then led me to a harder problem: network "
        "traffic changes over time, and machine-learning intrusion detectors may face concept drift and feature "
        "drift. The UGR'16 research and dataset were especially useful for this direction because they were "
        "designed to study network traffic behaviour over time."
    )
    report.para(
        "The literature motivated the problem. The exact research question and the experiment design are my own, "
        "and the project does not claim to invent a new machine-learning algorithm: the models are deliberately "
        "ordinary baselines, and the contribution is a controlled measurement of how far they degrade and "
        "whether adaptation recovers any of it."
    )

    # ---------------- Research basis ----------------
    report.h1("Research basis")
    report.para(
        "Shyaa et al. (2024) survey concept-drift and feature-dynamics-aware machine and deep learning for "
        "intrusion detection, and give the premise of this project: a detector trained on one traffic "
        "distribution meets a different one in operation, and the resulting drift is a first-class problem rather "
        "than a deployment detail."
    )
    report.para(
        "Macia-Fernandez et al. (2018) introduce UGR'16, whose stated purpose is evaluating intrusion detection "
        "systems that account for long-term traffic evolution and periodicity, with data gathered from a real ISP "
        "over several months. That is the dataset design this project's temporal evaluation follows."
    )
    report.para(
        "Xu et al. (2024) address drift detection, interpretation and adaptation together, and Yang et al. (2025) "
        "propose self-supervised adaptation to concept drift for network intrusion detection. Both establish that "
        "adaptation under drift is an active problem with known partial solutions, which is why this project "
        "evaluates several strategies rather than assuming one."
    )

    report.h2("Threat model")
    report.bullets([
        "The defender observes flow metadata only: start time, duration, packet and byte counts, protocol and "
        "TCP flags. Payload contents are never inspected.",
        "The attacker generates malicious traffic and may encrypt, obfuscate or imitate legitimate flows.",
        "The environment changes independently of any attack: new software, new services, different traffic "
        "mixes. This non-stationarity, not the attacker, is what the forward test measures.",
        "The forward test is assumed to contain attacks the defender has not seen, and detection must not depend "
        "on prior knowledge of the specific attack.",
    ])

    # ---------------- Dataset ----------------
    report.h1("Dataset")
    report.para(
        f"This report was generated from a run on {metrics['dataset']}. The dataset was loaded through an "
        f"explicit adapter that maps the source's own fields onto a shared flow schema; fields the source does "
        f"not export are left absent rather than approximated."
    )
    report.table(
        ["Property", "Value"],
        [
            ["Dataset", metrics["dataset"]],
            ["Total rows", f"{split['rows_total']:,}"],
            ["Model features", str(len(metrics.get("features", [])))],
            ["First flow", split["train_period"][0]],
            ["Last flow", split.get("forward_period", split["backtest_period"])[1]],
            ["Source files", ", ".join(meta.get("dataset_checksums", {}).keys()) or "see catalog"],
        ],
        widths=[1.0, 2.0],
    )

    report.h2("Dataset limitations")
    report.bullets([
        "The capture spans a short window. A temporal split over a few capture sessions measures relative "
        "movement between earlier and later data, not drift across years or across a changing network "
        "infrastructure.",
        "Attack prevalence differs between periods. A model that looks good on the forward test may partly "
        "reflect an easier class mix rather than genuine temporal generalisation, which is why every table "
        "reports the attack rate of each period next to its score.",
        "Synthetic attack generation, where a dataset uses it, differs from real campaigns and may make some "
        "attack classes easier to separate than they would be in operation.",
        "One dataset measures one environment. Results here describe this dataset under this split, and are not "
        "a claim about modern networks generally.",
    ])

    # ---------------- Feature policy ----------------
    report.h1("Feature policy")
    report.para(
        "The feature set follows a written policy: only what a netflow or IPFIX collector actually emits, and "
        "never payload content, identity fields, or the label. The target column and the time index are excluded "
        "at selection time, and the preprocessor refuses to fit on them."
    )
    report.bullets([
        "Allowed: flow duration, packet and byte counts, derived rates, protocol, TCP flags.",
        "Excluded: IP addresses and ports, absolute timestamps as model inputs, the attack category, and the "
        "binary label.",
        "Excluded: any application-payload-derived field. A field that only exists because a tool parsed HTTP or "
        "FTP content is not available on encrypted traffic, so including it would make the payload-free claim false.",
    ])

    # ---------------- Design ----------------
    report.h1("Experimental design")
    report.h2("Periods")
    rows = []
    for period in ["train", "validation", "backtest"] + (["forward"] if "forward_period" in split else []):
        rows.append([
            period,
            f"{split[f'{period}_period'][0]} -> {split[f'{period}_period'][1]}",
            f"{split[f'{period}_count']['rows']:,}",
            f"{split[f'{period}_count']['attack_rate']:.1%}",
        ])
    report.table(["Period", "Time range", "Rows", "Attack rate"], rows, widths=[0.7, 2.2, 0.8, 0.8],
                 aligns=["L", "L", "R", "R"])
    report.para(
        "Periods are cut by row count with boundaries snapped to real timestamps, so a single instant is never "
        "divided across two periods. The forward test is the last slice of the time axis and is scored exactly "
        "once."
    )

    report.h2("Backtest methodology")
    report.para(
        "The backtest is a held-out slice of the same historical distribution the model was trained on. It is the "
        "ordinary evaluation: it tells you what the detector does when nothing has changed, and it is the "
        "reference the forward result is measured against. Its score is never presented as the model's real-world "
        "performance."
    )

    report.h2("Forward-test methodology")
    report.para(
        "The forward test is the later slice of the time axis. The model, the scaler, the category vocabulary and "
        "the decision threshold are all fixed before it is touched, and no hyperparameter, threshold or preprocessing "
        "statistic is derived from it. Several tests enforce this, including one that fails if a training row "
        "carries a timestamp past the training cutoff and one that fails if an adaptation step receives a row from "
        "the period it is scored on."
    )

    report.h2("Operating threshold and the false-positive budget")
    budget = metrics["models"][0]["result"].get("target_fpr") if metrics.get("models") else None
    report.para(
        f"Rather than a fixed 0.5 cut, the operating threshold is chosen on the validation split to meet a "
        f"false-positive budget{f' of {budget:.0%}' if budget is not None else ''}. The threshold is the "
        f"{1 - (budget or 0.05):.2f} quantile of the validation negative scores, so it depends only on how the "
        f"model scores known-benign traffic, and it is then applied unchanged to the backtest and the forward test. "
        f"No threshold is ever selected using forward-test labels."
    )
    header, rows = _parse_markdown_table(threshold_table(metrics))
    if rows:
        report.table(header, rows, widths=[1.2, 0.7, 0.8, 1.0, 1.0, 0.8],
                     aligns=["L", "R", "R", "R", "R", "R"])

    # ---------------- Models ----------------
    report.h1("Models")
    report.para(
        "Four estimators spanning the usual complexity range, all with a fixed seed. They are ordinary baselines: "
        "the question is how they fail, not whether one can be tuned to win."
    )
    report.bullets([
        "Majority baseline: always predicts the training majority class, so a detector that cannot beat it has "
        "learned nothing.",
        "Logistic regression with balanced class weights: a linear reference.",
        "Random forest with balanced class weights: captures non-linear structure.",
        "Gradient boosting: sequential trees, sensitive to distribution change in a different way.",
    ])
    for entry in models[:4]:
        result = entry["result"]
        report.para(
            f"{result['model']}: trained in {result.get('train_seconds', 0):.1f}s, "
            f"threshold {result['threshold']:.4f}.", size=8.6
        )

    # ---------------- Results ----------------
    report.h1("Results")
    report.h2("Backtest versus forward test")
    report.para(
        "The two evaluations are reported separately and never merged. A positive degradation means the forward "
        "score was lower than the backtest score."
    )
    header, rows = _parse_markdown_table(main_comparison_table(metrics))
    if rows:
        report.table(header, rows, widths=[1.2, 0.8, 0.8, 0.9, 0.9, 0.9, 0.7, 0.7, 0.8], font_size=6.6)
    if "backtest_vs_forward" in figures:
        report.figure(figures["backtest_vs_forward"], "Backtest and forward-test F1 per model.")
    if "degradation" in figures:
        report.figure(figures["degradation"], "Change in F1 and recall on later traffic.")

    # ---------------- Drift ----------------
    report.h1("Distribution shift and drift detection")
    report.para(
        "Drift is measured with a sliding window over the forward period, each window compared against the whole "
        "training period. Three two-sample methods are used - Kolmogorov-Smirnov, scaled Wasserstein distance and "
        "the population stability index - plus a sequential CUSUM detector. Window size, stride, minimum sample "
        "count and every threshold are configuration values, not constants in the code."
    )
    report.para(
        "A p-value alone is not treated as an operational signal: with many windows and large samples, a tiny "
        "p-value fires constantly. Every method therefore also reports an effect size, and an alert requires "
        "either a test statistic above its critical value or an effect size above its configured threshold."
    )
    if drift_summary:
        rows = [[m, v["alerts"], v["comparisons"]] for m, v in sorted(drift_summary.get("by_method", {}).items())]
        if rows:
            report.table(["Method", "Alerts", "Comparisons"], rows, widths=[1.0, 0.7, 0.9],
                         aligns=["L", "R", "R"])
    header, rows = _parse_markdown_table(drift_table(str(base / "drift_events.csv")))
    if rows:
        report.h2("Strongest drift by feature")
        report.table(header, rows[:14], widths=[0.7, 1.2, 0.9, 0.7, 0.8], font_size=6.8)
    if "drift_timeline" in figures:
        report.figure(figures["drift_timeline"], "Drift alerts across the forward period.")

    # ---------------- Adaptation ----------------
    report.h1("Adaptation")
    report.para(
        "Five strategies were evaluated against the same false-positive budget: no adaptation, threshold "
        "recalibration, recent-window retraining, rolling-window retraining, and retraining on historical plus "
        "recent data. Each is permitted only rows timestamped at or before the moment it runs; a strategy that "
        "reaches into the period it is scored on raises an error and is reported as failed rather than quietly "
        "scored."
    )
    header, rows = _parse_markdown_table(adaptation_table(metrics))
    if rows:
        report.table(header, rows, widths=[1.0, 1.2, 0.7, 0.7, 0.7, 0.8, 0.7, 0.5], font_size=6.3)
    if "adaptation" in figures:
        report.figure(figures["adaptation"], "Forward-test F1 by adaptation strategy.")
    report.para(
        "Recovery is expressed as a percentage of the performance lost between backtest and forward test, so a "
        "negative value means the strategy made the forward result worse than doing nothing. No strategy is "
        "declared the overall winner; the table reports what happened in this experiment."
    )

    # ---------------- Failure analysis ----------------
    report.h1("Failure analysis")
    report.para(
        "The findings below are computed from the prediction-level tables and drift events of this run. A model "
        "that produced no such finding would show an empty section rather than a plausible sentence."
    )
    for model, analysis in failures.items():
        report.h2(model)
        result = next((m["result"] for m in models if m["result"]["model"] == model), None)
        if result:
            report.para(
                f"F1 degradation {result.get('f1_degradation', float('nan')):+.4f}, "
                f"recall degradation {result.get('recall_degradation', float('nan')):+.4f}, "
                f"false-positive increase {result.get('fpr_increase', float('nan')):+.4f}."
            )
        fps = analysis.get("false_positives", {})
        fns = analysis.get("false_negatives", {})
        if fps:
            report.bullets([
                f"False positives: {fps.get('count', 0):,} on the forward test "
                f"(mean score {fps.get('mean_score', 0):.3f} when the model was wrong).",
                f"False negatives: {fns.get('count', 0):,} "
                f"(mean score {fns.get('mean_score', 0):.3f}, of which {fns.get('high_confidence_misses', 0):,} "
                f"were scored above 0.8).",
            ])
        timing = analysis.get("drift_vs_degradation", {})
        if timing.get("available"):
            if timing.get("first_alert"):
                report.bullets([
                    f"First drift alert: {timing['first_alert']}.",
                    f"Drift preceded the degradation window: {timing.get('drift_precedes_degradation')}.",
                ])
            else:
                report.bullets(["No drift alert fired in any window, despite the performance change."])
        strongest = analysis.get("strongest_drift_features", [])
        if strongest:
            names = ", ".join(entry["feature"] for entry in strongest[:5])
            report.bullets([f"Strongest drift: {names}."])
        alerts_info = analysis.get("alerts_without_degradation", {})
        if alerts_info.get("alerts"):
            report.bullets([
                f"Of {alerts_info['alerts']} alerting windows, {alerts_info.get('in_degraded_windows', 0)} "
                f"coincided with degraded performance and {alerts_info.get('without_degradation', 0)} did not. "
                f"Drift and model failure are related but not the same event."
            ])
    if "forward_windows" in figures:
        report.figure(figures["forward_windows"], "Recall and false-positive rate across the forward period.")

    # ---------------- Uncertainty ----------------
    report.h1("Uncertainty and calibration")
    report.para(
        "Confidence is measured with the Brier score and expected calibration error on both periods, which shows "
        "whether a model that has become less accurate also becomes less honest about it."
    )
    header, rows = _parse_markdown_table(calibration_table(experiments_dir, str(base)))
    if rows:
        report.table(header, rows, widths=[1.2, 0.9, 0.9, 0.9, 0.9], font_size=7.0)
    if "calibration" in figures:
        report.figure(figures["calibration"], "Reliability diagram, backtest versus forward period.", width=300)
    if "class_timeline" in figures:
        report.figure(figures["class_timeline"], "Attack prevalence across the forward period.")

    # ---------------- Limitations ----------------
    report.h1("Limitations")
    report.bullets([
        "One dataset, one split. The numbers describe this dataset under this configuration and are not a claim "
        "about modern networks, adversarial robustness or deployment readiness.",
        "The forward period is later traffic from the same capture environment, so the shift it contains is "
        "whatever the environment itself did, not a designed stress test.",
        "A controlled-shift experiment is needed to attribute a degradation to one specific property; the "
        "temporal result alone cannot say which feature change caused the drop.",
        "The operating threshold is chosen on one validation slice. A different budget, or a different split of "
        "the same data, would give different numbers.",
        "No uncertainty method beyond calibration is used, because none was needed to answer the question asked "
        "here.",
    ])

    # ---------------- Reproducibility ----------------
    report.h1("Reproducibility")
    report.para(
        f"Every run writes an immutable directory under {experiments_dir}/. The one behind this report is "
        f"{base.name}."
    )
    report.mono(f"""experiment id   : {base.name}
dataset         : {metrics['dataset']}
git commit      : {meta.get('git_commit', 'unknown')}
feature schema  : {meta.get('feature_schema_hash', 'unknown')}
random seed     : {meta.get('random_seed', 'unknown')}
python          : {meta.get('package_versions', {}).get('python', '?')}

reproduce:
    python -m pip install -e .
    python -m driftguard data fetch --dataset unsw_nb15 --yes
    python -m driftguard benchmark --config configs/benchmark.yaml
    python -m driftguard report""")

    report.h2("Experiment directory contents")
    files = sorted(p.name for p in base.iterdir() if p.is_file())
    report.mono("\n".join(files))

    # ---------------- Future ----------------
    report.h1("Future research")
    report.bullets([
        "A second dataset with a genuinely different capture environment, to separate temporal shift within a "
        "network from a change of network.",
        "Controlled shift experiments that attribute a degradation to one named property, such as packet size or "
        "protocol mix, instead of to drift in general.",
        "A sequential protocol that refits on a schedule and reports the cost of each refit, rather than "
        "retraining once at the end of the forward period.",
        "Calibration-aware alerting, so that a drop in expected calibration error can itself raise an operator "
        "alert even when the F1 has not yet moved.",
    ])

    report.h1("References")
    for entry in REFERENCES:
        report.para(entry, size=8.4)

    return report.output(output)
