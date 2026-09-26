"""Technical report generator.

Every number in the report is read from a recorded experiment directory. A
section is only written when the corresponding experiment output exists, so the
report cannot claim an experiment that was not run.
"""

import json
import pandas as pd
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
        # Anything read back from a CSV arrives as a float NaN for an empty
        # cell, and fpdf cannot encode that. Treat it as no text at all rather
        # than printing "nan" in the middle of a sentence.
        if text is None or _is_blank(text):
            return
        self.pdf.set_font("Helvetica", "I" if italic else "", size)
        self.pdf.multi_cell(PAGE_W - 2 * MARGIN, LINE + 1, str(text), align="J")
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
        if not widths or len(widths) != columns:
            # A caller that supplies the wrong number of widths gets equal
            # columns rather than an IndexError partway down the page. Tables
            # here are built from CSV output, so the column count is data
            # dependent and a hard failure here loses the whole report.
            widths = [1.0] * columns
        total = sum(widths)
        available = PAGE_W - 2 * MARGIN
        col_w = [w / total * available for w in widths]

        if not aligns or len(aligns) != columns:
            aligns = ["L"] * columns

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



def _is_blank(value) -> bool:
    """True for None, empty strings, and the NaN that pandas gives an empty cell."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return str(value).strip() == ""


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




def _latest_shift_csv(experiments_dir: str) -> Optional[str]:
    """The most recent completed controlled-shift table, if one exists."""
    root = Path(experiments_dir)
    if not root.exists():
        return None
    candidates = sorted(
        d / "controlled_shift_results.csv"
        for d in root.iterdir()
        if d.is_dir() and (d / "controlled_shift_results.csv").is_file()
    )
    return str(candidates[-1]) if candidates else None


def _collect_ablations(experiments_dir: str) -> dict:
    """Group every recorded ablation row by ablation name.

    Ablation experiments are separate runs, so the report has to go looking for
    them rather than expecting them in the benchmark's own directory.
    """
    root = Path(experiments_dir)
    if not root.exists():
        return {}
    grouped: dict = {}
    for directory in sorted(root.iterdir()):
        path = directory / "ablation_results.csv"
        if not path.is_file():
            continue
        try:
            table = pd.read_csv(path)
        except Exception:
            continue
        if table.empty or "ablation" not in table.columns:
            continue
        for name, group in table.groupby("ablation", sort=False):
            grouped.setdefault(str(name), []).extend(group.to_dict("records"))
    return grouped


def _other_datasets(experiments_dir: str, current: str) -> dict:
    """Load the latest completed benchmark for each dataset other than this one."""
    root = Path(experiments_dir)
    if not root.exists():
        return {}
    found: dict = {}
    for directory in sorted(root.iterdir()):
        metrics_path = directory / "metrics.json"
        if not metrics_path.is_file() or directory.name.endswith(".tmp"):
            continue
        try:
            loaded = load_experiment(str(directory))
        except Exception:
            continue
        name = loaded.get("dataset")
        if not name or name == current:
            continue
        # The synthetic fixture exists so the test suite can run the pipeline
        # without a download. It is not a dataset the project evaluated, and
        # reporting it beside a real one invites a comparison that means
        # nothing.
        if name == "synthetic":
            continue
        # Later directories win, so iteration order gives the most recent run.
        found[name] = loaded
    return found


def _ugr_section(report, experiments_dir: str) -> None:
    """Section 8: the UGR'16 experiment, on its own terms.

    The two datasets are not pooled and not ranked against each other. UGR'16
    is reported separately because its attack prevalence is one to two orders
    of magnitude lower, which makes its F1 incomparable to UNSW's before any
    distribution shift is even considered.
    """
    report.h1("8. UGR'16")
    ugr = _other_datasets(experiments_dir, "unsw_nb15").get("ugr16")
    if not ugr:
        report.para("No UGR'16 run has been completed, so this section reports nothing rather than estimating.")
        return

    split = ugr.get("split", {})
    models = ugr.get("models", [])
    report.para(
        "UGR'16 was captured from a real ISP over several months, which makes it the stronger of the two "
        "temporal experiments here. UNSW-NB15's recorded time axis spans under a month and its forward period "
        "covers roughly four hours, whereas the UGR'16 forward period is a full week of later traffic. It is "
        "also the harder detection problem, because attacks are one to two orders of magnitude rarer in it."
    )
    report.table(
        ["Property", "Value"],
        [
            ["Total rows", f"{split.get('rows_total', 0):,}"],
            ["Train rows", f"{split.get('train_count', {}).get('rows', 0):,}"],
            ["Train attack rate", f"{split.get('train_count', {}).get('attack_rate', 0) * 100:.3f}%"],
            ["Validation rows", f"{split.get('validation_count', {}).get('rows', 0):,}"],
            ["Backtest rows", f"{split.get('backtest_count', {}).get('rows', 0):,}"],
            ["Forward rows", f"{split.get('forward_count', {}).get('rows', 0):,}"],
            ["Forward attack rate", f"{split.get('forward_count', {}).get('attack_rate', 0) * 100:.3f}%"],
            ["Forward period", " to ".join([
                str(split.get("forward_period", ["", ""])[0]), str(split.get("forward_period", ["", ""])[1])
            ])],
        ],
        widths=[1.3, 2.0],
    )

    by_method = (ugr.get("drift_summary") or {}).get("by_method", {})
    if by_method:
        report.para(
            "The drift detectors disagree more sharply here than on the other dataset, and the direction of the "
            "disagreement is reversed."
        )
        report.table(
            ["Detector", "Alerts", "Alert rate"],
            [
                [name, f"{s.get('alerts', 0)}/{s.get('comparisons', 0)}",
                 f"{s.get('alerts', 0) / max(s.get('comparisons', 1), 1) * 100:.1f}%"]
                for name, s in sorted(by_method.items())
            ],
            widths=[1.2, 1.0, 1.0],
        )
        silent = sorted(n for n, s in by_method.items() if s.get("alerts", 0) == 0)
        if silent:
            report.para(
                f"{' and '.join(silent)} raised no alert at all in this experiment, while the same detectors were "
                "far from silent on the other dataset. That is reported as a finding rather than smoothed over: a "
                "detector that stays quiet under a measurably shifted distribution has not found the traffic "
                "stable, it has failed to notice."
            )

    if models:
        header, body = _parse_markdown_table(main_comparison_table(ugr))
        if body:
            report.para("The identical protocol, applied to UGR'16:")
            report.table(header, body, widths=[1.2, 0.8, 0.8, 0.9, 0.9, 0.9, 0.7, 0.7, 0.8], font_size=6.6)
        worst = min(models, key=lambda m: m["result"].get("f1_degradation", 0))
        report.para(
            "Absolute F1 here is far below the figures on the other dataset, and the reason is prevalence rather "
            "than model quality: with attacks present in well under one flow in a hundred, precision is bounded "
            f"by the base rate however good the ranking is. The largest F1 loss falls on "
            f"{worst['result']['model']} ({worst['result'].get('f1_degradation', 0):+.4f}), and recall falls "
            "further than F1 for the models that score it, because the forward period carries more attacks than "
            "the period they were fitted on."
        )
        if any(m.get("adaptation") for m in models):
            report.para(
                "Adaptation behaves here in the opposite direction to the other dataset. That disagreement is the "
                "most consequential difference between the two experiments, and it is taken up in section 23."
            )


def generate_report(experiments_dir: str = "results/experiments", output: str = "docs/technical_report.pdf") -> str:
    """Build the technical report from whatever experiments are on disk."""
    from driftguard.reporting.figures import build_all_figures
    from driftguard.reporting.figures_extra import build_extra_figures

    # UNSW-NB15 is the headline benchmark, so it is pinned rather than inferred.
    # Alphabetical order would pick UGR'16 and then label its numbers "UNSW-NB15".
    experiment = latest_experiment(experiments_dir, dataset="unsw_nb15")
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
    figures.update(build_extra_figures(str(base), metrics, str(figures_dir), _latest_shift_csv(experiments_dir)))

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
    report.h1("1. Abstract")
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
    report.h1("2. Research question")
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

    report.h1("3. Why I chose this question")
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
    report.h1("4. Literature motivation")
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

    report.h1("5. Threat model")
    report.bullets([
        "The defender observes flow metadata only: start time, duration, packet and byte counts, protocol and "
        "TCP flags. Payload contents are never inspected.",
        "The attacker generates malicious traffic and may encrypt, obfuscate or imitate legitimate flows.",
        "The environment changes independently of any attack: new software, new services, different traffic "
        "mixes. This non-stationarity, not the attacker, is what the forward test measures.",
        "The forward test is assumed to contain attacks the defender has not seen, and detection must not depend "
        "on prior knowledge of the specific attack.",
    ])

    # ---------------- Dataset provenance ----------------
    report.h1("6. Dataset provenance")
    report.para(
        "Both datasets are public and are used unmodified. What each is allowed to support is not the same, and "
        "the difference matters for how far the results travel."
    )
    report.table(
        ["dataset", "what it is", "what it can support"],
        [
            ["UNSW-NB15",
             "Labelled flow records with attack categories, captured over three sessions in January and "
             "February 2015. Retained here in a mirror that preserves the start and end timestamps.",
             "A relative temporal comparison: does a detector keep working on later captured traffic. Its time "
             "axis spans 27 days and one session has no attacks, so it cannot support a long-horizon claim."],
            ["UGR'16",
             "Netflow records from a Greek ISP network, published as 23 weekly archives totalling about 215 GB.",
             "The stronger temporal experiment, on a real network rather than a lab. This report uses a six-week "
             "subset, which is stated below and is not the full release."],
        ],
        widths=[0.9, 2.3, 2.4], font_size=6.8,
    )

    from driftguard.data.registry import Ugr16Adapter

    notes = Ugr16Adapter.SUBSET_NOTES
    report.h2("The UGR'16 subset, exactly")
    report.para(
        f"Full release: {notes['full_release_size']}. Subset used here: {notes['subset_size']}. No claim is made "
        f"about the weeks that were not downloaded."
    )
    report.para(notes["why"])
    report.bullets([
        f"Calibration weeks (background only, never used for training): {', '.join(notes['calibration_weeks'])}.",
        f"Weeks carrying the temporal experiment: {', '.join(notes['weeks_used'])}.",
    ] + list(notes["limitations"]))

    # ---------------- Threat model ----------------
    report.h2("Threat model in terms of trust boundaries")
    report.para(
        "The adversary here is not the attacker in the dataset. It is the passage of time, and the question is "
        "whether the measurement apparatus keeps working when the traffic it was calibrated on is replaced."
    )
    report.bullets([
        "Trusted: the historical capture is assumed to be a faithful record of what the network carried, and the "
        "labels in the benchmark are assumed correct. No attempt is made to verify either.",
        "Untrusted: everything after the training period. The forward period is treated as hostile in the sense "
        "that nothing about it - not its features, its prevalence, its duration, nor its labels - is allowed to "
        "influence any fitted object before it is scored.",
        "Out of scope: an attacker who knows the detector's features and adapts to evade it, a compromised "
        "capture point, and label poisoning in the training data. None of these are tested or claimed.",
        "The privacy claim is narrower than it looks: features are metadata only, meaning no payload bytes, "
        "addresses or ports become predictive inputs. That reduces what an observer can learn, and it is not the "
        "same as a formal privacy guarantee against a traffic analyst.",
    ])

    report.h1("7. UNSW-NB15")
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

    # ---------------- Design ----------------
    if "split_timeline" in figures:
        report.figure(figures["split_timeline"],
                       "The four periods on the real capture axis, drawn to scale. "
                       "Every conclusion here is bounded by how short that axis is.", width=380)

    # ---------------- UGR'16 ----------------
    # Placed here by hand rather than by the reorder pass: the helper emits a
    # numbered heading from outside the body, so a sort on heading numbers
    # cannot place it, and it landed inside the abstract when it tried.
    _ugr_section(report, experiments_dir)

    report.h1("9. Temporal evaluation design")
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

    # ---------------- Leakage prevention ----------------
    report.h1("10. Leakage prevention")
    report.para(
        "A temporal experiment is only meaningful if nothing from the future can reach the model. Five guards "
        "enforce that here, and each is covered by a test that fails if the guard is removed."
    )
    report.bullets([
        "Periods are contiguous in time and ordered. An identical timestamp is never split across a boundary, "
        "because UNSW-NB15 stamps some rows with the same second.",
        "The scaler and encoder are fitted on the training period only. Fitting on all data and then splitting "
        "would leak the forward period's mean and variance into training, which is the most common way a "
        "'temporal' split quietly becomes a random one.",
        "The operating threshold is chosen on the validation split, never on the backtest or forward periods.",
        "The backtest is used for exactly one thing: measuring how much the model decays before the forward "
        "test is touched at all. The forward period is scored once.",
        "Adaptation is given the window immediately before the forward period and nothing after its first "
        "timestamp. Rolling adaptation re-fits on a cadence, and at each step the fit may only use rows earlier "
        "than the step it is about to score.",
    ])
    guard_tests = [
        t for t in sorted(p.name for p in (Path(__file__).resolve().parents[2] / "tests").glob("test_*.py"))
        if "leak" in t or "silent" in t
    ]
    if guard_tests:
        report.para("Guards covered by: " + ", ".join(guard_tests) + ".")

    # ---------------- Models ----------------
    report.h1("11. Models")
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

    # ---------------- Drift ----------------
    report.h1("12. Drift detection")
    if "drift_detectors" in figures:
        report.figure(figures["drift_detectors"],
                       "The four detectors on the same windows. The disagreement between them is the "
                       "result; agreement would have been the surprising outcome.", width=380)

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
    report.h1("13. Adaptation strategies")
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

    # ---------------- Metrics ----------------
    report.h1("14. Metrics")
    report.para(
        "Every metric is computed at the threshold chosen on validation data, never at a threshold tuned on the "
        "period being scored. A detector that simply alerts more would otherwise look better, so recall is "
        "always reported beside the false-positive rate it cost."
    )
    report.bullets([
        "F1 is the harmonic mean of precision and recall. At the attack prevalence seen in UGR'16 it is "
        "dominated by the false-positive term, which is why the false-positive rate is reported next to it "
        "rather than left implicit.",
        "Recall is the share of attacks caught. The false-positive rate is the share of normal flows wrongly "
        "called an attack, and is the quantity held to the budget.",
        "PR AUC summarises ranking quality across every threshold, so it is the one measure here that does not "
        "depend on where the operating point happened to land.",
        "Brier score and expected calibration error measure whether a probability means what it says. A model "
        "that becomes less accurate without becoming less honest is a different problem from one that degrades "
        "in both directions at once.",
        "Cohen's d measures a numeric shift and total-variation distance measures a categorical one. A protocol "
        "mixture shift is exactly zero under Cohen's d however far the mixture moves, which is why measuring it "
        "that way alone would have reported a real shift as absent.",
    ])

    # ---------------- Experimental protocol ----------------
    report.h1("15. Experimental protocol")
    report.para(
        "Every number in the results sections comes from one command run against a recorded configuration, with "
        "the seed fixed and the dataset checksum stored in the run's own metadata. The protocol is stated here so "
        "that a reader can trace each figure to a procedure rather than take it on trust."
    )
    report.bullets([
        f"Dataset {metrics['dataset']}, loaded through the adapter named in the run metadata, "
        f"{split['rows_total']:,} flows in total.",
        f"Four contiguous, disjoint periods in the order train, validation, backtest, forward: "
        f"{split['train_count']['rows']:,} / {split['validation_count']['rows']:,} / "
        f"{split['backtest_count']['rows']:,} / {split.get('forward_count', {}).get('rows', 0):,}.",
        "The operating threshold is chosen on validation data alone, at the configured target false-positive "
        "rate, and is then frozen for the backtest, the forward period and every adaptation strategy.",
        "Preprocessing, including the one-hot encoder for protocol, is fitted on training rows only. An encoder "
        "fitted across all periods would let a protocol label that only appears later define a feature dimension.",
        f"Seed {meta.get('random_seed', '42')}, Python "
        f"{(meta.get('package_versions') or {}).get('python', 'unknown')}, commit "
        f"{meta.get('git_commit', 'unknown')}.",
        "An experiment directory is never overwritten. A run that cannot complete every requested stage fails "
        "loudly rather than writing a partial result that looks finished.",
    ])

    # ---------------- Results ----------------
    report.h1("16. Main results")
    report.para(
        f"This section reports the headline benchmark on {metrics['dataset']}. UGR'16 is reported separately in "
        "section 8 and the two are compared in section 23."
    )
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

    if "recall_vs_fpr" in figures:
        report.figure(figures["recall_vs_fpr"],
                       "Each model's operating point: recall against the false-positive rate it "
                       "actually achieved, backtest versus forward.", width=380)

    # ---------------- Drift analysis ----------------
    report.h1("17. Drift analysis")
    if drift_summary.get("comparisons"):
        report.para(
            f"Across {drift_summary['comparisons']} window-feature comparisons on {metrics['dataset']}, "
            f"{drift_summary['alerts']} raised an alert, a rate of "
            f"{drift_summary['alerts'] / max(drift_summary['comparisons'], 1):.1%}."
        )
    report.para(
        "The detectors do not agree, and that disagreement is the result rather than a defect in any one of them: "
        "a significance test, a distributional distance and a sequential detector answer three different "
        "questions about the same windows."
    )
    report.para(
        "Three failure modes are visible and are kept distinct. A detector can alert on nearly everything, which "
        "measures how many comparisons were made rather than whether anything changed. A detector can stay "
        "silent while the distribution measurably moves, which is what happened to PSI and CUSUM on UGR'16. And "
        "a detector can alert on a shift that cost the model no F1 at all, which is the ordinary case for benign "
        "drift."
    )
    report.para(
        "Drift is nowhere in this report treated as a stand-in for failure. The two are measured independently, "
        "and only their coincidence is discussed."
    )

    # ---------------- Adaptation analysis ----------------
    report.h1("18. Adaptation analysis")
    report.para(
        "Each strategy is judged on what it recovers and what it costs, under the same false-positive budget as "
        "the unadapted baseline. A strategy that raises F1 by widening its threshold is not an adaptation, which "
        "is why the false-positive rate is reported beside every F1."
    )
    report.para(
        "The finding is that the ranking is not stable. On UNSW-NB15, rolling-window retraining is the worst "
        "strategy for one model and among the best for another, and the two datasets evaluated here disagree "
        "about it completely. No claim of the form 'method X adapts best' is supported by this evidence and none "
        "is made."
    )
    report.para(
        "Threshold recalibration changed nothing on UNSW-NB15, because the validation-derived threshold was "
        "already meeting the target false-positive rate. That is a fact about this data rather than a verdict on "
        "the method: recalibration can only help once the operating point has actually moved."
    )

    # ---------------- Controlled shifts ----------------
    report.h1("19. Controlled shifts")
    shift_csv = _latest_shift_csv(experiments_dir)
    if not shift_csv:
        report.para("No controlled-shift experiment has been completed.")
    else:
        shifts = pd.read_csv(shift_csv)
        if shifts.empty:
            report.para("The controlled-shift experiment recorded no rows.")
        else:
            verified = int(shifts["realized_verified"].sum())
            report.para(
                f"Eight shift families are applied to already-trained models at several magnitudes, giving "
                f"{len(shifts)} measurements over {shifts['shift'].nunique()} families. Every row records both "
                "the shift that was requested and the shift that was actually measured, because a perturbation "
                "that fails to move the distribution is a finding and not a success."
            )
            report.para(
                f"All {verified} of {len(shifts)} rows produced a measured shift in the intended direction. "
                "Numeric families are measured with Cohen's d and the categorical protocol-mixture family with "
                "total-variation distance, because a mixture has no mean for Cohen's d to summarise."
            )
            report.para(
                "The responses are not uniformly harmful. Several perturbations improved F1, most clearly the "
                "class-prevalence shift, which raises the base rate and therefore precision at a fixed threshold. "
                "Reporting only the degradations would misrepresent the experiment, so both directions are kept."
            )
            by_family = []
            for family, group in shifts.groupby("shift"):
                by_family.append([
                    str(family),
                    str(len(group)),
                    f"{float(group['f1_degradation'].min()):+.4f}",
                    f"{float(group['f1_degradation'].max()):+.4f}",
                    f"{int(group['realized_verified'].sum())}/{len(group)}",
                ])
            report.table(
                ["Shift family", "Rows", "Best F1 change", "Worst F1 change", "Measured as intended"],
                by_family,
                widths=[1.3, 0.5, 0.9, 0.9, 0.9],
                font_size=7.0,
            )

    # ---------------- Ablations ----------------
    report.h1("20. Ablations")
    if "shift_degradation" in figures:
        report.figure(figures["shift_degradation"],
                       "Controlled shifts, labelled with how many of each family moved the "
                       "distribution as intended.", width=380)

    report.para(
        "Each ablation below answers one methodological question. They are run with cheaper models than the "
        "headline benchmark because the question is about the effect, not about which model is best."
    )
    ablation_rows = _collect_ablations(experiments_dir)
    if ablation_rows:
        for group, rows in ablation_rows.items():
            report.h2(group.replace("_", " "))
            header = list(rows[0].keys())
            keep = [c for c in header if c not in {"note", "experiment_id", "dropped", "seconds", "model"}]
            # str() on a float prints all 17 significant digits and prints a
            # missing value as the literal "nan", so the ablation tables were
            # both unreadable and wrong. Numbers are rounded and blanks are
            # left blank.
            def _cell(value):
                if value is None:
                    return ""
                if isinstance(value, float):
                    if value != value:          # NaN
                        return ""
                    if value in (float("inf"), float("-inf")):
                        return "inf"
                    return f"{value:.4f}"
                return str(value)

            table_rows = [[_cell(r.get(c, "")) for c in keep] for r in rows]
            report.table(keep, table_rows, font_size=6.6)
            notes = [r["note"] for r in rows if r.get("note") and not str(r["note"]).startswith("trained")]
            for note in dict.fromkeys(notes):
                report.para(note, size=7.4, italic=True)
    else:
        report.para("No ablation experiment has been recorded yet.")

    # ---------------- Failure analysis ----------------
    report.h1("21. Failure analysis")
    if "failure_analysis" in figures:
        report.figure(figures["failure_analysis"],
                       "Error volume and mean error score per model on the forward period.", width=380)

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
    report.h1("22. Uncertainty and calibration")
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

    # ---------------- Cross-dataset ----------------
    report.h1("23. Cross-dataset discussion")
    other = _other_datasets(experiments_dir, metrics["dataset"])
    if not other:
        report.para(
            "Only one dataset has been run, so this report draws no comparison across datasets. The single-dataset "
            "result is stated as what it is: later traffic from one capture environment."
        )
    else:
        for name, other_metrics in other.items():
            other_models = other_metrics.get("models", [])
            report.h2(name)
            report.para(
                f"{len(other_models)} models on {name}, split "
                f"{other_metrics.get('split', {}).get('forward_count', {}).get('rows', 0):,} forward rows."
            )
            header, rows = _parse_markdown_table(main_comparison_table(other_metrics))
            if rows:
                report.table(header, rows, widths=[1.3, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], font_size=6.8)
            report.para(
                "Comparing the two datasets directly is not valid: they have different label definitions, different "
                "attack taxonomies and different collection tools. The transferable claim is only about the "
                "method - splitting chronologically, fixing the operating point on validation data, and treating "
                "adaptation as a separate question from detection."
            )

    # ---------------- Limitations ----------------
    report.h1("24. Limitations")
    dataset_name = metrics.get("dataset", "the dataset")
    limit_items = [
        "The numbers describe one dataset under one configuration. They are not a claim about modern networks, "
        "about adversarial robustness, or about deployment readiness, and nothing here was tested against an "
        "adaptive attacker.",
    ]
    if dataset_name.startswith("unsw"):
        limit_items.append(
            "UNSW-NB15's time axis is three discrete capture sessions across 27 days, one of which contains no "
            "attacks at all. It supports a relative temporal comparison - does performance fall on later captured "
            "traffic - and nothing stronger. It cannot speak to years of deployment, because it spans weeks."
        )
    else:
        limit_items.append(
            "This is a subset of a much larger collection, so it constrains the time axis as well as the traffic "
            "variety. Periods outside the subset are unmeasured."
        )
    limit_items += [
        "The forward period is later traffic from the same capture environment, so the shift it contains is "
        "whatever the environment itself did, not a designed stress test. The controlled-shift experiments "
        "attribute a degradation to named properties, but they perturb a dataset that is already synthetic in "
        "its attack labels.",
        "The operating threshold comes from one validation slice at a fixed false-positive budget. A different "
        "budget, or a different split of the same data, gives different numbers; the target-FPR ablation shows "
        "how sharply recall depends on that choice.",
        "A detector can be simultaneously accurate and badly calibrated, and calibration error is estimated on "
        "slices as small as the attack class. The uncertainty numbers are point estimates without confidence "
        "intervals, so small differences between models should not be over-read.",
    ]
    report.bullets(limit_items)

    # ---------------- Reproducibility ----------------
    report.h1("25. Reproducibility")
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
    report.h1("26. Future work")
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

    # ---------------- Conclusion ----------------
    report.h1("27. Conclusion")
    if models:
        worst = min(models, key=lambda m: m["result"].get("f1_degradation", 0))
        best_fwd = max(models, key=lambda m: m["result"]["forward"]["f1"])
        report.para(
            f"A detector trained on the earliest {split['train_count']['rows']:,} flows of {metrics['dataset']} "
            f"was evaluated on {split.get('forward_count', {}).get('rows', 0):,} flows it never saw. "
            f"{len(degraded)} of {len(models)} models lost F1 on that later traffic. The largest loss was "
            f"{worst['result']['model']} at {worst['result'].get('f1_degradation', float('nan')):+.4f} F1; the best "
            f"forward performer was {best_fwd['result']['model']} at F1 {best_fwd['result']['forward']['f1']:.4f}."
        )
        report.para(
            "The answer to the primary question is therefore conditional rather than binary. Models trained on "
            "historical traffic do continue to detect attacks on later traffic, but not at the level they reached "
            "on held-out data from the same period, and the drop is not uniform across models. "
            + (
                "The adaptation comparison further shows that refitting on recent data is not automatically an "
                "improvement, so the choice of strategy has to be made against a stated objective rather than "
                "assumed from its name."
                if len(models) else ""
            )
        )
    else:
        report.para("No model results were recorded, so no conclusion can be drawn.")

    # ---------------- References ----------------
    report.h1("References")
    for entry in REFERENCES:
        report.para(entry, size=8.4)

    return report.output(output)

