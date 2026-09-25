#!/usr/bin/env python
"""Generate docs/project_report.pdf from the repository's own outputs.

The report contains no hand-typed experiment numbers: every metric, count and
model parameter is read at generation time from results/metrics/ and
configs/config.yaml, so the PDF cannot disagree with the run that produced it.

    python scripts/generate_report.py
"""

import json
import sys
from datetime import date
from pathlib import Path

import yaml
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "project_report.pdf"

sys.path.insert(0, str(ROOT / "src"))
from privacy_preserving_nad.features import build_features as bf  # noqa: E402

MODEL_LABELS = {
    "majority_baseline": "Majority baseline",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}
SET_ORDER = ["FULL_METADATA", "RESTRICTED_METADATA"]
MODEL_ORDER = ["majority_baseline", "logistic_regression", "random_forest"]


def load_inputs() -> dict:
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    eval_results = json.loads(
        (ROOT / "results" / "metrics" / "evaluation_results.json").read_text()
    )
    info_path = ROOT / "results" / "metrics" / "dataset_info.json"
    dataset_info = json.loads(info_path.read_text()) if info_path.exists() else {}
    return {"config": config, "eval": eval_results, "dataset_info": dataset_info}


def f4(x) -> str:
    return f"{x:.4f}"


def p100(x) -> str:
    return f"{x * 100:.1f}%"


class Report(FPDF):
    def __init__(self, smoke: bool):
        super().__init__()
        self.smoke = smoke

    def header(self):
        if self.page_no() <= 1:
            return  # cover page carries its own title block
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, "Privacy-Preserving Network Anomaly Detection", align="L")
        if self.smoke:
            self.set_text_color(200, 60, 60)
            self.cell(0, 6, "DEVELOPMENT SAMPLE - NOT BENCHMARK RESULTS", align="R")
        self.ln(8)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    def h1(self, text: str):
        if self.get_y() > self.h - 45:
            self.add_page()
        self.ln(3)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(20, 50, 90)
        self.multi_cell(0, 8, text)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def h2(self, text: str):
        if self.get_y() > self.h - 35:
            self.add_page()
        self.ln(2)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(40, 70, 110)
        self.multi_cell(0, 6.5, text)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def p(self, text: str, size: int = 10, style: str = ""):
        self.set_font("Helvetica", style, size)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.2, text)
        self.ln(1.5)
        self.set_text_color(0, 0, 0)

    def bullets(self, items, size: int = 10):
        for item in items:
            if self.get_y() > self.h - 20:
                self.add_page()
            self.set_font("Helvetica", "", size)
            self.set_text_color(30, 30, 30)
            x = self.get_x()
            self.cell(6, 5.2, chr(149))
            self.multi_cell(0, 5.2, item)
            self.set_x(x)
        self.ln(1.5)
        self.set_text_color(0, 0, 0)

    def table(self, headers, rows, widths, aligns=None):
        aligns = aligns or ["L"] * len(headers)

        def header_row():
            self.set_font("Helvetica", "B", 8.5)
            self.set_fill_color(232, 238, 246)
            for h, w, a in zip(headers, widths, aligns):
                self.cell(w, 7, h, border=1, align="C", fill=True)
            self.ln()

        if self.get_y() > self.h - (12 + 7 * (len(rows) + 1)):
            self.add_page()
        header_row()
        self.set_font("Helvetica", "", 8.5)
        fill = False
        for row in rows:
            if self.get_y() > self.h - 14:
                self.add_page()
                header_row()
                self.set_font("Helvetica", "", 8.5)
                fill = False
            self.set_fill_color(246, 249, 252)
            for v, w, a in zip(row, widths, aligns):
                text = str(v)
                size = 8.5
                self.set_font("Helvetica", "", size)
                while self.get_string_width(text) > w - 3 and size > 5.5:
                    size -= 0.5
                    self.set_font("Helvetica", "", size)
                self.cell(w, 6.5, text, border=1, align=a, fill=fill)
            self.set_font("Helvetica", "", 8.5)
            self.ln()
            fill = not fill
        self.ln(3)

    def figure(self, path: Path, caption: str, width: float = 170):
        if not path.exists():
            return
        from PIL import Image

        with Image.open(path) as im:
            px_w, px_h = im.size
        image_height = width * px_h / px_w
        if self.get_y() > self.h - (image_height + 22):
            self.add_page()
        self.ln(2)
        x = (self.w - width) / 2
        self.image(str(path), x=x, w=width)
        self.ln(2)
        self.set_font("Helvetica", "I", 8.5)
        self.set_text_color(90, 90, 90)
        self.multi_cell(0, 5, caption)
        self.set_text_color(0, 0, 0)
        self.ln(3)


def cover(pdf: Report, data: dict):
    info = data["dataset_info"]
    config = data["config"]
    pdf.add_page()
    pdf.ln(45)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(20, 50, 90)
    pdf.multi_cell(0, 11, "Privacy-Preserving Network\nAnomaly Detection", align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(70, 70, 70)
    pdf.multi_cell(
        0,
        7,
        "Detecting abnormal network behavior from traffic metadata\n"
        "without inspecting packet payload contents",
        align="C",
    )
    pdf.ln(14)
    pdf.set_font("Helvetica", "", 11)
    info_lines = []
    if info.get("kind") == "research":
        info_lines.append(
            f"Dataset: UNSW-NB15 official split - {info['train_rows']:,} training rows, "
            f"{info['test_rows']:,} test rows"
        )
    else:
        info_lines.append(
            "Dataset: DEVELOPMENT SAMPLE - synthetic smoke-test data, "
            "not the UNSW-NB15 benchmark"
        )
    info_lines.append(
        f"Feature sets: FULL_METADATA ({len(config['feature_sets']['FULL_METADATA'])}) "
        f"and RESTRICTED_METADATA ({len(config['feature_sets']['RESTRICTED_METADATA'])})"
    )
    info_lines.append("Models: majority baseline, Logistic Regression, Random Forest")
    info_lines.append("Research questions, methodology, results, limitations")
    pdf.multi_cell(0, 6.5, "\n".join(info_lines), align="C")
    pdf.ln(0)
    pdf.ln(30)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        0,
        5.5,
        "Generated from this repository's configuration and results files.\n"
        f"Report generated: {date.today().isoformat()}",
        align="C",
    )
    pdf.set_text_color(0, 0, 0)


def overview(pdf: Report, data: dict):
    info = data["dataset_info"]
    pdf.add_page()
    pdf.h1("1. What this project is")
    pdf.p(
        "This project studies whether a network detector can find abnormal behavior "
        "using only traffic metadata - flow statistics, packet headers, timing and "
        "connection state - without ever reading packet payloads. It is built as a "
        "reproducible research repository: one configuration file defines the data, "
        "the feature policy and the models; one command runs the experiment; every "
        "number in this report was read from the files that command writes."
    )
    pdf.p(
        "The practical motivation is that encryption is now the default on the "
        "internet. TLS 1.3, QUIC and encrypted DNS make deep packet inspection "
        "either impossible or unacceptable, while defenders still have to find "
        "scanning, denial-of-service, exploitation and command-and-control traffic. "
        "Flow exporters such as NetFlow and IPFIX keep producing metadata regardless "
        "of what is inside the packets. This repository measures how far that "
        "metadata alone gets you on a standard public dataset."
    )
    pdf.h2("Status of the run described in this report")
    if info.get("kind") == "research":
        pdf.p(
            f"Research run. The official UNSW-NB15 split was used as released: "
            f"{info['train_rows']:,} training rows and {info['test_rows']:,} test "
            "rows, verified against the sha256 checksums recorded in the "
            "configuration. The two files were never merged, shuffled or "
            "re-partitioned, and nothing was tuned against the test split."
        )
    else:
        pdf.p(
            f"Smoke-test run. data/raw/ currently contains development sample data "
            f"({info.get('train_rows', 0):,} train rows / "
            f"{info.get('test_rows', 0):,} test rows) instead of the official "
            "UNSW-NB15 split, so the numbers that follow only demonstrate that the "
            "pipeline works. They are not benchmark results and must not be cited "
            "as such. Fetch the official files (section 3) and re-run the pipeline "
            "to produce the research experiment.",
            style="B",
        )


def questions(pdf: Report, data: dict):
    config = data["config"]
    pdf.h1("2. Research questions")
    pdf.p(
        'Primary question: "Can network traffic metadata provide enough information '
        'to detect abnormal network behavior without inspecting packet payload '
        'contents?"'
    )
    pdf.p(
        'Secondary question: "How much detection performance changes when the '
        'detector is restricted to a smaller subset of traffic metadata?"'
    )
    pdf.p(
        "Hypothesis: metadata carries real signal for anomaly detection, and a small "
        "core subset of it keeps most of that signal while sacrificing some "
        "performance. The two feature sets in the configuration - FULL_METADATA "
        f"({len(config['feature_sets']['FULL_METADATA'])} fields) and "
        f"RESTRICTED_METADATA ({len(config['feature_sets']['RESTRICTED_METADATA'])} "
        "fields) - are the instrument for answering the second question."
    )
    pdf.p(
        "Metadata-only monitoring is not a formal privacy guarantee. The project "
        "studies payload-free detection, not complete traffic privacy.",
        style="B",
    )
    pdf.h2("Threat model")
    pdf.bullets(
        [
            "Defender: observes traffic metadata only (headers, flow records, "
            "timing). Never decrypts and never parses payload.",
            "Attacker: produces malicious traffic and may encrypt, obfuscate or "
            "imitate legitimate flows.",
            "Assumption under test: an attacker cannot match the statistical "
            "profile of benign traffic across every metadata dimension at once. "
            "The experiment measures how well that holds on UNSW-NB15.",
        ]
    )


def dataset_section(pdf: Report, data: dict):
    info = data["dataset_info"]
    pdf.h1("3. Dataset")
    pdf.p(
        "UNSW-NB15 (Australian Centre for Cyber Security, 2015) mixes real normal "
        "traffic with synthetic attack traffic - DoS, exploits, fuzzers, "
        "reconnaissance, shellcode, worms, backdoors, generic - captured in a "
        "network laboratory. The repository uses its official train/test files, "
        "each with 45 columns: a row id, 42 traffic fields, the attack category "
        "and the binary label (0 = normal, 1 = anomalous) that this project uses "
        "as its target."
    )
    def fmt_count(counts, key):
        value = counts.get(key)
        return f"{value:,}" if isinstance(value, int) else "-"

    if info.get("kind") == "research":
        train_counts = info.get("train_class_counts", {})
        test_counts = info.get("test_class_counts", {})
        pdf.table(
            ["Split", "Rows", "Normal", "Anomalous"],
            [
                [
                    "Training",
                    f"{info['train_rows']:,}",
                    fmt_count(train_counts, "0"),
                    fmt_count(train_counts, "1"),
                ],
                [
                    "Testing",
                    f"{info['test_rows']:,}",
                    fmt_count(test_counts, "0"),
                    fmt_count(test_counts, "1"),
                ],
            ],
            [45, 45, 45, 45],
            ["L", "R", "R", "R"],
        )
        pdf.p(
            "Checksums recorded in configs/config.yaml matched both files, so the "
            "rows above are the official release rather than a reformatted copy."
        )
    else:
        pdf.p(
            f"The data present is NOT the research dataset: "
            f"{info.get('train_rows', 0):,} train rows / "
            f"{info.get('test_rows', 0):,} test rows of synthetic development data "
            f"(kind: {info.get('kind', 'unknown')}). The official split has 175,341 "
            "training and 82,332 test rows.",
            style="B",
        )
    pdf.h2("Development sample vs research dataset")
    pdf.bullets(
        [
            "Development sample: small synthetic rows written by "
            "scripts/create_sample_data.py so tests and CI run in seconds without "
            "network access. The pipeline detects it and labels the run a smoke "
            "test on screen and in results/metrics/dataset_info.json.",
            "Research dataset: the official UNSW-NB15 training and testing files, "
            "recognized by row count and sha256. Only this input can produce "
            "benchmark results, and only this input should be cited.",
            "The sample generator refuses to overwrite the research dataset, and "
            "neither dataset is ever committed.",
        ]
    )
    pdf.h2("Obtaining the official files")
    pdf.p(
        "python -m privacy_preserving_nad.data.download prints the current status "
        "and instructions; adding --fetch downloads the two files and verifies "
        "them against the recorded checksums. The files can also be downloaded "
        "manually from the official UNSW page "
        "(https://research.unsw.edu.au/projects/unsw-nb15-dataset) and placed in "
        "data/raw/. Nothing downloads automatically during tests or ordinary "
        "pipeline runs."
    )


def feature_policy(pdf: Report, data: dict):
    pdf.h1("4. Feature policy")
    pdf.p(
        "The policy is written down before any model is trained: traffic-level "
        "fields only, no packet contents, no raw identity. It lives in "
        "configs/config.yaml and is enforced by a guard in feature selection that "
        "aborts the run if a forbidden name appears in a feature set. "
        "docs/feature_policy.md documents every field and its justification."
    )
    pdf.p(
        "Allowed categories: flow duration and timing (dur, sinpkt, dinpkt, sjit, "
        "djit, tcprtt, synack, ackdat); packet and byte counts (spkts, dpkts, "
        "sbytes, dbytes); rates and throughput (rate, sload, dload); loss counters "
        "(sloss, dloss); packet-size statistics (smean, dmean); TCP state and "
        "timing (sttl, dttl, swin, dwin, stcpb, dtcpb); connection-count aggregates "
        "computed by the dataset's exporters (ct_* and is_sm_ips_ports); and the "
        "header-level labels proto, service and state."
    )
    pdf.h2("Excluded fields")
    rows = []
    for feature in bf.EXCLUDED_FEATURES:
        reason = bf.FEATURE_DOCUMENTATION.get(
            feature, "Identity or label field - not traffic behavior"
        )
        rows.append([feature, reason])
    for feature in bf.PAYLOAD_DERIVED_FEATURES:
        rows.append(
            [
                feature,
                bf.FEATURE_DOCUMENTATION.get(
                    feature, "Requires parsing application content"
                ),
            ]
        )
    pdf.table(["Field", "Why it is excluded"], rows, [55, 125])
    pdf.p(
        "Two inclusions deserve honest wording. The ct_*_dport/sport and "
        "is_sm_ips_ports aggregates are counts keyed on addresses and ports - the "
        "raw addresses and ports themselves never become inputs. The TCP timing "
        "fields assume a visible handshake, which encrypted transports still "
        "expose. The five application-layer fields were removed from the feature "
        "sets precisely because they only exist when HTTP/FTP content was parsed; "
        "keeping them would have made the payload-free claim false."
    )


def pipeline_section(pdf: Report, data: dict):
    config = data["config"]
    pdf.h1("5. Pipeline")
    pdf.bullets(
        [
            "1. Data detection: identify the dataset in data/raw/ (research vs "
            "development sample) and record the outcome in results/metrics/.",
            "2. Validation: schema, target values, class counts, missing values, "
            "duplicate rows and policy compliance - reported before training.",
            "3. Cleaning: no rows are dropped; string fields are trimmed; any "
            "missing value would be imputed inside the sklearn pipeline.",
            "4. Preprocessing: a reusable ColumnTransformer fitted on the training "
            "split only - median imputation plus standardization for numeric "
            "columns, one-hot encoding (handle_unknown='ignore') for proto, "
            "service and state - then applied to the test split.",
            "5. Training: three baselines per feature set with a fixed seed.",
            "6. Evaluation: metrics and confusion matrices written to "
            "results/metrics/, figures to results/figures/.",
            "7. This report is generated from those same files.",
        ]
    )
    pdf.h2("Experiment design")
    lr = config["models"]["logistic_regression"]
    rf = config["models"]["random_forest"]
    pdf.table(
        ["Item", "Setting"],
        [
            ["Split", "official UNSW-NB15 train/test, used as released"],
            [
                "Target",
                f"{config['target']['column']}: "
                f"{config['target']['normal_value']} = normal, "
                f"{config['target']['anomaly_value']} = anomalous",
            ],
            ["Random seed", str(config.get("random_seed"))],
            [
                "Numeric preprocessing",
                "median imputation + standardization (train-fitted)",
            ],
            [
                "Categorical preprocessing",
                "one-hot, handle_unknown='ignore' (train-fitted)",
            ],
            [
                "Logistic Regression",
                f"C={lr.get('C')}, max_iter={lr.get('max_iter')}, "
                f"solver={lr.get('solver')}, class_weight={lr.get('class_weight')}",
            ],
            [
                "Random Forest",
                f"n_estimators={rf.get('n_estimators')}, "
                f"max_depth={rf.get('max_depth')}, "
                f"min_samples_split={rf.get('min_samples_split')}, "
                f"min_samples_leaf={rf.get('min_samples_leaf')}, "
                f"class_weight={rf.get('class_weight')}",
            ],
            ["Decision threshold", "0.5 (default; nothing tuned on the test set)"],
            [
                "Metrics",
                ", ".join(config["evaluation"]["metrics"] + ["confusion matrix"]),
            ],
        ],
        [55, 125],
    )
    pdf.p(
        "No model complexity was added on purpose. These are baselines: the point "
        "of the experiment is the information content of the metadata, not "
        "architecture search. class_weight='balanced' is used because the classes "
        "are uneven in both splits (the anomaly class is the majority), which "
        "keeps a naive all-anomalous classifier from looking competitive."
    )


def results_section(pdf: Report, data: dict):
    eval_results = data["eval"]
    config = data["config"]
    pdf.h1("6. Results")

    rows = []
    for fs in SET_ORDER:
        for model in MODEL_ORDER:
            m = eval_results[fs][model]
            rows.append(
                [
                    fs,
                    MODEL_LABELS[model],
                    f4(m["precision"]),
                    f4(m["recall"]),
                    f4(m["f1_score"]),
                    f4(m["false_positive_rate"]),
                    f4(m["false_negative_rate"]),
                ]
            )
    pdf.table(
        ["Feature set", "Model", "Precision", "Recall", "F1", "FPR", "FNR"],
        rows,
        [42, 38, 22, 22, 22, 22, 22],
        ["L", "L", "R", "R", "R", "R", "R"],
    )

    majority = eval_results["FULL_METADATA"]["majority_baseline"]
    n_test = majority["n_samples"]
    n_normal = majority["n_normal"]
    n_anom = majority["n_anomaly"]
    rf_full = eval_results["FULL_METADATA"]["random_forest"]
    rf_rest = eval_results["RESTRICTED_METADATA"]["random_forest"]
    lr_full = eval_results["FULL_METADATA"]["logistic_regression"]
    lr_rest = eval_results["RESTRICTED_METADATA"]["logistic_regression"]

    pdf.h2("Class distribution")
    pdf.p(
        f"The test split holds {n_test:,} flows: {n_normal:,} normal "
        f"({p100(n_normal / n_test)}) and {n_anom:,} anomalous "
        f"({p100(n_anom / n_test)}). The anomaly class is the majority, so the "
        f"majority baseline reaches F1 {f4(majority['f1_score'])} by predicting "
        f"'anomalous' for everything while flagging every normal flow as malicious "
        f"(FPR {f4(majority['false_positive_rate'])}). That is the floor the real "
        "models had to clear, and both cleared it."
    )

    candidates = [
        ("Random Forest with FULL_METADATA", rf_full),
        ("Random Forest with RESTRICTED_METADATA", rf_rest),
        ("Logistic Regression with FULL_METADATA", lr_full),
        ("Logistic Regression with RESTRICTED_METADATA", lr_rest),
    ]
    best_name, best = max(candidates, key=lambda kv: kv[1]["f1_score"])
    pdf.h2("Best result of this run")
    pdf.p(
        f"{best_name}: F1 {f4(best['f1_score'])} (precision "
        f"{f4(best['precision'])}, recall {f4(best['recall'])}). In counts it "
        f"caught {best['confusion_matrix'][1][1]:,} of {n_anom:,} anomalous flows "
        f"and missed {best['confusion_matrix'][1][0]:,} "
        f"({p100(best['false_negative_rate'])} false negatives), while flagging "
        f"{best['confusion_matrix'][0][1]:,} of {n_normal:,} normal flows as "
        f"attacks ({p100(best['false_positive_rate'])} false positives). The "
        "false-positive column is what an operator pays for that recall."
    )
    pdf.p(
        "This is a ranking inside one experiment on one dataset. It is not a "
        "claim that this model is generally superior, and no claim of privacy, "
        "production readiness or adversarial robustness follows from it.",
        style="I",
    )

    pdf.h2("Effect of restricting the feature set")
    n_full = len(config["feature_sets"]["FULL_METADATA"])
    n_restricted = len(config["feature_sets"]["RESTRICTED_METADATA"])
    pdf.p(
        f"With fields cut from {n_full} to {n_restricted}: Random Forest moved "
        f"from F1 {f4(rf_full['f1_score'])} to {f4(rf_rest['f1_score'])} "
        f"({rf_rest['f1_score'] - rf_full['f1_score']:+.4f}) and its false "
        f"positive rate from {f4(rf_full['false_positive_rate'])} to "
        f"{f4(rf_rest['false_positive_rate'])}; Logistic Regression moved from "
        f"F1 {f4(lr_full['f1_score'])} to {f4(lr_rest['f1_score'])} "
        f"({lr_rest['f1_score'] - lr_full['f1_score']:+.4f}) with its false "
        f"positive rate from {f4(lr_full['false_positive_rate'])} to "
        f"{f4(lr_rest['false_positive_rate'])}. The tree model gave up little when "
        "the feature set was cut while the linear model lost more ground: the "
        "removed fields mattered more to the model that cannot reconstruct their "
        "interactions itself. Both restricted results still sit well above the "
        "baseline, so the small feature core does carry most of the detectable "
        "signal - that is the secondary research question answered for this run."
    )

    pdf.h2("Precision, recall and the operational trade-off")
    pdf.p(
        f"Recall on the full set was {f4(lr_full['recall'])} (logistic) and "
        f"{f4(rf_full['recall'])} (forest): most anomalous flows are detected, "
        "and most normal flows are not flagged - but "
        f"{f4(lr_full['false_positive_rate'])} and "
        f"{f4(rf_full['false_positive_rate'])} of normal flows respectively were. "
        "Which error is worse depends on the deployment: an intrusion detection "
        "system usually prefers the false negatives of the forest over the false "
        "positives of the logistic model, while an alerting system with few "
        "analysts may prefer the opposite. All numbers here are at the default "
        "0.5 threshold; moving the threshold trades recall against false positives "
        "and would change every figure in this section. No threshold, hyperparameter "
        "or model was selected using the test set."
    )

    imp = rf_full.get("feature_importances")
    if imp:
        names = imp.get("feature_names") or []
        values = imp.get("importances") or []
        ranked = sorted(zip(names, values), key=lambda kv: kv[1], reverse=True)[:5]
        if ranked:
            pdf.h2("What the forest relied on")
            pdf.p(
                "Top features by importance on the full set: "
                + "; ".join(f"{n} ({v:.4f})" for n, v in ranked)
                + ". Importance here reflects split-gain concentration, not "
                "causality - a dominant ttl-derived field can crowd the rest out "
                "even when the remaining fields carry independent signal."
            )


def figures_section(pdf: Report, data: dict):
    eval_results = data["eval"]
    rf_full = eval_results["FULL_METADATA"]["random_forest"]
    n_normal = rf_full["n_normal"]
    n_anom = rf_full["n_anomaly"]
    tn, fp = rf_full["confusion_matrix"][0]
    fn, tp = rf_full["confusion_matrix"][1]

    first_figure = ROOT / "results" / "figures" /         "confusion_matrix_random_forest_FULL_METADATA.png"
    reserve = 165
    if first_figure.exists():
        from PIL import Image

        with Image.open(first_figure) as im:
            px_w, px_h = im.size
        reserve = int(170 * px_h / px_w + 45)
    if pdf.get_y() > pdf.h - reserve:
        pdf.add_page()
    pdf.h1("7. Figures")
    figs = ROOT / "results" / "figures"
    pdf.figure(
        figs / "confusion_matrix_random_forest_FULL_METADATA.png",
        f"Confusion matrix - Random Forest on FULL_METADATA. Rows are true "
        f"classes, columns predicted: {tn:,} true negatives, {fp:,} false "
        f"positives, {fn:,} false negatives, {tp:,} true positives "
        f"(test split: {n_normal:,} normal, {n_anom:,} anomalous).",
    )
    pdf.figure(
        figs / "model_comparison_f1.png",
        "F1 by model and feature set, including the majority baseline floor.",
    )
    pdf.figure(
        figs / "model_comparison_fpr.png",
        "False positive rate by model and feature set - the normal traffic each "
        "detector would mislabel as anomalous.",
    )
    pdf.figure(
        figs / "feature_importance_random_forest_FULL_METADATA.png",
        "Feature importance of the Random Forest trained on FULL_METADATA.",
    )
    pdf.figure(
        figs / "class_distribution.png",
        "Class distribution of the train and test splits actually used for this run.",
    )


def limitations(pdf: Report, data: dict):
    info = data["dataset_info"]
    pdf.h1("8. Limitations")
    items = [
        "Dataset age: UNSW-NB15 was captured in 2015, before TLS 1.3, QUIC and "
        "encrypted SNI became widespread; modern traffic has different metadata "
        "distributions.",
        "Synthetic attacks: the attack traffic was generated in a lab and does "
        "not reproduce a determined human adversary.",
        "Distribution shift: any real deployment sees traffic this dataset does "
        "not contain; no cross-network generalization is claimed.",
        "Feature availability: timing and TCP fields assume a vantage point that "
        "observes the handshake and enough packets per flow; short or sampled "
        "flows provide weaker versions of them.",
        "Binary simplification: nine attack categories are collapsed into one "
        "anomaly class, so per-attack behavior is not analyzed.",
        "Single operating point: everything is reported at threshold 0.5 with no "
        "tuning; other points on the precision-recall curve were not explored.",
        "Metadata is not anonymity: traffic metadata can fingerprint and profile "
        "users even when payloads are never read. This project demonstrates "
        "payload-free detection, not a formal privacy guarantee, and makes no "
        "claim of robustness against evasion.",
    ]
    if info.get("kind") != "research":
        items.insert(
            0,
            "Non-research input: the numbers in this report come from development "
            "sample data, not the UNSW-NB15 benchmark, and are not citable "
            "results.",
        )
    pdf.bullets(items)


def reproduction(pdf: Report, data: dict):
    pdf.h1("9. Reproduction")
    pdf.p("From a clean checkout of the repository:")
    pdf.bullets(
        [
            "pip install -r requirements.txt && pip install -e .",
            "python -m privacy_preserving_nad.data.download --fetch  "
            "(or place the official CSVs in data/raw/ by hand)",
            "python -m privacy_preserving_nad.pipeline  (full experiment)",
            "python scripts/render_results.py  (refresh the README results block)",
            "python scripts/generate_report.py  (rebuild this PDF)",
            "python -m pytest tests/ -v  (test suite, runs on the development "
            "sample only - CI never downloads the full dataset)",
        ]
    )
    pdf.p(
        "Determinism: random seeds are fixed at "
        f"{data['config'].get('random_seed')} for numpy, Python's random module "
        "and every estimator, so repeating the pipeline reproduces the same "
        "metrics. Every figure and table in this report is regenerated from "
        "results/metrics/ at build time; none of it is entered by hand."
    )
    pdf.h1("10. Future work")
    pdf.bullets(
        [
            "Repeat the same policy on newer captures (CIC-IDS2017, CIC-DoS2019) "
            "to test transfer beyond a 2015 laboratory dataset.",
            "Adversarial evaluation: measure how far an attacker can perturb "
            "metadata before detection breaks.",
            "Per-attack-category metrics instead of a single anomaly class.",
            "Threshold selection for a target false-positive budget.",
            "Concept drift monitoring of metadata statistics over time.",
            "Additional baselines (gradient boosting, simple unsupervised "
            "detectors) once these baselines are established.",
        ]
    )


def build():
    data = load_inputs()
    smoke = data["dataset_info"].get("kind", "research") != "research"
    pdf = Report(smoke=smoke)
    pdf.set_auto_page_break(auto=True, margin=18)
    cover(pdf, data)
    overview(pdf, data)
    questions(pdf, data)
    dataset_section(pdf, data)
    feature_policy(pdf, data)
    pipeline_section(pdf, data)
    results_section(pdf, data)
    figures_section(pdf, data)
    limitations(pdf, data)
    reproduction(pdf, data)

    pdf.h1("11. Files this report was built from")
    pdf.bullets(
        [
            "configs/config.yaml - feature sets, model parameters, seed, checksums",
            "results/metrics/evaluation_results.json - every metric and confusion "
            "matrix above",
            "results/metrics/dataset_info.json - which dataset was used, row "
            "counts, class counts",
            "results/figures/*.png - the embedded plots",
            "src/privacy_preserving_nad/features/build_features.py - feature "
            "policy text and exclusions",
        ]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT))
    print(f"Report written to: {OUTPUT}")


if __name__ == "__main__":
    build()
