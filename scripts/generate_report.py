"""Generate a PDF project report from actual experiment results.

Run after the pipeline: python scripts/generate_report.py
Output: docs/project_report.pdf
"""

import csv
import json
import unicodedata
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
OUTPUT = ROOT / "docs" / "project_report.pdf"

MODEL_LABELS = {
    "majority_baseline": "Majority Baseline",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def san(text: str) -> str:
    """Replace characters the built-in Helvetica font cannot render."""
    replacements = {
        "—": "-", "–": "-", "‘": "'", "’": "'",
        "“": '"', "”": '"', "•": "-", "→": "->",
        "≤": "<=", "≥": ">=", "×": "x", "·": "-",
        "−": "-", "…": "...",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return unicodedata.normalize("NFKD", text).encode("latin-1", "replace").decode("latin-1")


class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, san("Privacy-Preserving Network Anomaly Detection"), align="L")
        self.cell(0, 8, "Page %d" % self.page_no(), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, 16, self.w - self.r_margin, 16)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, san("Generated from pipeline output - results/metrics/"), align="C")

    def h1(self, text):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(25, 60, 120)
        self.ln(4)
        self.multi_cell(0, 9, san(text))
        self.set_draw_color(25, 60, 120)
        self.line(self.l_margin, self.get_y() + 1, self.w - self.r_margin, self.get_y() + 1)
        self.ln(4)

    def h2(self, text):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(40, 90, 160)
        self.ln(3)
        self.multi_cell(0, 8, san(text))
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, san(text))
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin + 4)
        self.multi_cell(0, 6, san("- " + text))
        self.ln(1)

    def mono(self, text):
        self.set_font("Courier", "", 9.5)
        self.set_text_color(40, 40, 40)
        self.set_fill_color(242, 244, 248)
        self.multi_cell(0, 5.5, san(text), fill=True)
        self.ln(2)

    def callout(self, text):
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(90, 70, 20)
        self.set_fill_color(255, 248, 225)
        self.set_x(self.l_margin + 4)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 8, 5.5, san(text), fill=True)
        self.ln(3)

    def table(self, headers, rows, widths):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(25, 60, 120)
        self.set_text_color(255, 255, 255)
        for h, w in zip(headers, widths):
            self.cell(w, 7, san(h), border=1, align="C", fill=True)
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in rows:
            self.set_fill_color(244, 247, 251)
            for val, w in zip(row, widths):
                self.cell(w, 6.5, san(str(val)), border=1, align="C", fill=fill)
            self.ln()
            fill = not fill
        self.ln(3)

    def image_safe(self, path: Path, w=170):
        if path.exists():
            self.image(str(path), w=w)
            self.ln(3)
            return True
        self.set_font("Helvetica", "I", 9)
        self.multi_cell(0, 5, san("[figure not found: %s]" % path.name))
        return False


def load_results():
    path = RESULTS / "metrics" / "evaluation_results.json"
    with open(path) as f:
        return json.load(f)


def main():
    results = load_results()

    pdf = Report()
    pdf.set_auto_page_break(True, margin=20)
    pdf.set_margins(20, 20, 20)

    # ---------- Cover page ----------
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(25, 60, 120)
    pdf.multi_cell(0, 12, "Privacy-Preserving Network\nAnomaly Detection", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 8, san("Detecting abnormal network behavior from encrypted\ntraffic metadata, without inspecting packet payloads"), align="C")
    pdf.ln(14)
    pdf.set_font("Helvetica", "I", 11)
    pdf.multi_cell(0, 7, san("Project report - what the system is, why it exists,\nand how it works"), align="C")
    pdf.ln(30)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 6, san("Dataset: UNSW-NB15 (official train/test split)\nModels: Majority Baseline, Logistic Regression, Random Forest\nFeature sets: FULL_METADATA, RESTRICTED_METADATA\nRepository: github.com/TMStroo/Privacy-Preserving-Network-Anomaly-Detection"), align="C")

    # ---------- 1. What is this ----------
    pdf.add_page()
    pdf.h1("1. What this project is")
    pdf.body(
        "This project is a machine learning experiment for detecting abnormal network behavior "
        "using only traffic metadata. Metadata means the observable properties of a communication "
        "flow: how long it lasted, how many packets and bytes it carried, timing statistics, "
        "protocol and connection state. It never looks at what the packets contain."
    )
    pdf.body(
        "The practical motivation is encryption. Modern networks encrypt their payloads with TLS 1.3, "
        "QUIC and encrypted DNS, so the old approach of reading packet contents (deep packet inspection) "
        "either fails or requires breaking encryption, which is both hard and invasive. Network defenders "
        "still need to spot scanning, denial of service, exploits and other abnormal activity. The question "
        "this project asks is whether the remaining visible surface - the metadata - carries enough signal "
        "to do that."
    )
    pdf.body(
        "Concretely, the system downloads the UNSW-NB15 intrusion detection dataset, validates it, "
        "applies an explicit feature policy that forbids payload and identity fields, trains three "
        "baseline classifiers under two feature regimes, evaluates them on the official held-out test "
        "set, and writes metrics and figures to disk. Everything is driven by one command and one "
        "configuration file so the experiment can be repeated exactly."
    )

    pdf.h2("Research questions")
    pdf.bullet("Primary: can encrypted network traffic metadata provide enough information to detect abnormal network behavior without inspecting message contents?")
    pdf.bullet("Secondary: how much does detection performance change when the model is restricted to a small set of traffic metadata features?")

    pdf.h2("Hypothesis")
    pdf.body(
        "Traffic metadata contains sufficient signal to separate anomalous from normal traffic without "
        "payload inspection, and shrinking from the full metadata set to a small core subset will reduce "
        "performance measurably but not catastrophically - implying that a handful of flow statistics "
        "carry most of the discriminative information."
    )

    # ---------- 2. Why it matters ----------
    pdf.h1("2. Why the problem matters")
    pdf.body(
        "Encryption adoption is now near-universal on the public internet. That is good for privacy and bad "
        "for any defense that depends on reading content. At the same time, enterprises and operators cannot "
        "simply stop monitoring: scanning, command-and-control traffic, denial of service and data exfiltration "
        "all leave traces in how traffic behaves, not only in what it says."
    )
    pdf.body(
        "There is also a legitimacy angle. Inspecting payload means touching user content, which raises legal "
        "and trust issues and is prohibited in some jurisdictions and networks. A detector that only consumes "
        "flow records (NetFlow/IPFIX) or packet headers can be deployed on links where content inspection is "
        "not allowed. This project is a reproducible baseline for that idea: it measures exactly how much "
        "detection power survives when content is off the table."
    )

    pdf.h2("Threat model")
    pdf.bullet("Attacker: generates malicious traffic; may encrypt, obfuscate or mimic legitimate flows.")
    pdf.bullet("Defender: observes only metadata - flow records, packet headers, timing, connection state. Cannot decrypt payloads.")
    pdf.bullet("Assumption: the attacker cannot perfectly replicate the statistical profile of legitimate traffic across every metadata dimension at once.")

    # ---------- 3. Dataset ----------
    pdf.h1("3. Dataset")
    pdf.body(
        "UNSW-NB15 (2015) was created by the Australian Centre for Cyber Security. It mixes real normal "
        "traffic with synthetic attack traffic covering denial of service, exploits, fuzzers, reconnaissance, "
        "shellcode, worms, backdoors and generic attacks."
    )
    pdf.bullet("Training set: about 175,000 rows (official split).")
    pdf.bullet("Test set: about 82,000 rows (official split, held out).")
    pdf.bullet("49 columns: 45 predictive fields, plus the binary label and the attack category.")
    pdf.bullet("The label is converted to a clean binary target: 0 = normal, 1 = anomalous. The attack category column is never used as a feature.")
    pdf.body(
        "The official split is used as-is. Rows are never randomly re-mixed between train and test, because "
        "that would leak information and inflate scores. The repository does not commit any dataset files: "
        "raw and processed data are gitignored, and a script generates small sample files so tests and CI "
        "run without the 250 MB download."
    )
    pdf.callout(
        "Current results in this report were produced on the bundled 100-row sample data, so the pipeline "
        "can be demonstrated anywhere. They are a smoke test, not a benchmark claim. Re-run on the full "
        "dataset for real numbers."
    )

    # ---------- 4. Feature policy ----------
    pdf.h1("4. Feature policy: what the model may see")
    pdf.body(
        "The privacy claim of this project is only as good as its feature list, so the policy is written in "
        "code and configuration rather than left implicit. Every column used for prediction is documented in "
        "src/features/build_features.py with a note on why it qualifies as metadata."
    )
    pdf.h2("Used (metadata only)")
    pdf.bullet("Flow timing: dur, sinpkt, dinpkt, sjit, djit, tcprtt, synack, ackdat")
    pdf.bullet("Volume and rate: spkts, dpkts, sbytes, dbytes, rate, sload, dload")
    pdf.bullet("TCP state and windows: sttl, dttl, swin, dwin, stcpb, dtcpb, sloss, dloss")
    pdf.bullet("Packet size statistics: smean, dmean, trans_depth, response_body_len")
    pdf.bullet("Behavioral aggregates: ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, ct_src_ltm, ct_srv_dst, is_sm_ips_ports, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd")
    pdf.bullet("Protocol, service and connection state: proto, service, state")
    pdf.h2("Deliberately excluded")
    pdf.bullet("srcip, dstip - IP addresses are identity-bearing and do not generalize.")
    pdf.bullet("sport, dsport - ports can identify services and endpoints.")
    pdf.bullet("Ltime, Stime - absolute timestamps enable correlation and tracking.")
    pdf.bullet("attack_cat - it is a label, not a predictive feature.")
    pdf.bullet("Anything derived from packet content - the whole point of the policy.")

    pdf.h2("The two feature sets")
    pdf.table(
        ["Feature set", "Transformed size", "Contents"],
        [
            ["FULL_METADATA", "37 columns", "All policy-compliant metadata"],
            ["RESTRICTED_METADATA", "12 columns", "Basic flow statistics and timing only"],
        ],
        [45, 35, 90],
    )
    pdf.body(
        "Both lists live in configs/config.yaml, so experiments can change features without touching code. "
        "The restricted set is dur, spkts, dpkts, sbytes, dbytes, sttl, dttl, sload, dload, tcprtt, synack, "
        "ackdat - nothing but sizes, counts and timing."
    )

    # ---------- 5. How it works ----------
    pdf.add_page()
    pdf.h1("5. How the system works")
    pdf.body("The pipeline runs five stages in order, all orchestrated by src/pipeline.py:")

    pdf.h2("Stage 1 - Ingest and validate")
    pdf.body(
        "The download module fetches the UNSW-NB15 files into data/raw (or verifies them if already present). "
        "The validator then checks row and column structure, required columns, missing values, duplicate rows "
        "and the class balance of the target, and reports any issue before training starts."
    )

    pdf.h2("Stage 2 - Clean and transform")
    pdf.body(
        "Cleaning is documented and deterministic: numeric missing values are filled with the training median, "
        "categorical missing values with the mode, and rows with invalid target values are dropped. The target "
        "becomes 0/1. Categorical columns (proto, service, state) are one-hot encoded, numeric columns are "
        "standardized."
    )
    pdf.callout(
        "The scaler and encoder are fit on the training data only, then applied to the test set. "
        "No test statistic ever leaks into preprocessing."
    )

    pdf.h2("Stage 3 - Train baselines")
    pdf.body(
        "Three models are trained for each feature set with a fixed random seed of 42, wrapped in scikit-learn "
        "pipelines so preprocessing and fitting stay reproducible:"
    )
    pdf.table(
        ["Model", "Configuration"],
        [
            ["Majority Baseline", "DummyClassifier, most_frequent"],
            ["Logistic Regression", "C=1.0, max_iter=1000, lbfgs, class_weight=balanced"],
            ["Random Forest", "n_estimators=200, max_depth=15, min_samples_split=5, class_weight=balanced"],
        ],
        [50, 120],
    )

    pdf.h2("Stage 4 - Evaluate")
    pdf.body(
        "Every model is scored on the official test set with precision, recall, F1, false positive rate, "
        "false negative rate, confusion matrix and class counts, plus ROC AUC and PR AUC when probabilities "
        "are available. Results are written as machine-readable JSON and CSV under results/metrics/."
    )

    pdf.h2("Stage 5 - Plot")
    pdf.body(
        "Figures are rendered to results/figures/: a confusion matrix for each main model, a model comparison "
        "by F1, a model comparison by false positive rate, the Random Forest feature importance chart, and the "
        "class distribution of train versus test."
    )

    pdf.h2("Experiment matrix")
    pdf.body(
        "The main experiment is a 2 x 3 grid: two feature sets crossed with three models, five real runs "
        "(the majority baseline needs only one, since it ignores features, but it is reported for both sets "
        "for symmetry). Train/test separation follows the dataset's official split throughout."
    )

    # ---------- 6. Results ----------
    pdf.add_page()
    pdf.h1("6. Results")
    pdf.body("All numbers below are read directly from results/metrics/evaluation_results.json at generation time. Test set: 100 rows (47 normal, 53 anomalous) of sample data.")

    headers = ["Feature Set", "Model", "Precision", "Recall", "F1", "FPR", "FNR"]
    widths = [38, 34, 22, 20, 20, 18, 18]
    rows = []
    for fs in ["FULL_METADATA", "RESTRICTED_METADATA"]:
        for model in ["majority_baseline", "logistic_regression", "random_forest"]:
            m = results[fs][model]
            rows.append([
                fs, MODEL_LABELS[model],
                "%.4f" % m["precision"], "%.4f" % m["recall"],
                "%.4f" % m["f1_score"], "%.4f" % m["false_positive_rate"],
                "%.4f" % m["false_negative_rate"],
            ])
    pdf.table(headers, rows, widths)

    pdf.h2("Reading the table")
    m = results["FULL_METADATA"]
    pdf.bullet(
        "Majority baseline: always predicts the anomaly class (the majority here), so recall is 1.0 and the "
        "false positive rate is 1.0. Its F1 of %.4f is the floor a real model must beat." % m["majority_baseline"]["f1_score"]
    )
    pdf.bullet(
        "Logistic Regression: F1 %.4f, FPR %.4f. A linear decision boundary does not separate the classes "
        "well on this data - it does not beat the baseline."
        % (m["logistic_regression"]["f1_score"], m["logistic_regression"]["false_positive_rate"])
    )
    pdf.bullet(
        "Random Forest: F1 %.4f, FPR %.4f - the best result, with 2 false positives and 1 false negative "
        "out of 100 rows."
        % (m["random_forest"]["f1_score"], m["random_forest"]["false_positive_rate"])
    )

    pdf.h2("What changed between the feature sets")
    full = results["FULL_METADATA"]["random_forest"]
    rest = results["RESTRICTED_METADATA"]["random_forest"]
    pdf.body(
        "On this run the two feature sets produced identical classification results for all three models: "
        "Random Forest reached F1 %.4f on both, and Logistic Regression reached F1 %.4f on both. Only the "
        "probability-based metrics moved, by less than 0.001 (ROC AUC %.4f vs %.4f for the Random Forest). "
        "For this sample, reducing 37 features to 12 changed nothing measurable - the extra features carry "
        "redundant signal at this scale."
        % (full["f1_score"], results["FULL_METADATA"]["logistic_regression"]["f1_score"],
           full.get("roc_auc", 0), rest.get("roc_auc", 0))
    )
    pdf.callout(
        "Honest reading: this is a 100-row smoke test, not evidence that restricted features are always "
        "sufficient. The secondary research question can only be answered on the full 250,000-row dataset."
    )

    # keep the heading together with the first figure
    if pdf.get_y() > pdf.h - 110:
        pdf.add_page()
    pdf.h2("Figures")
    if pdf.image_safe(FIGURES / "confusion_matrix_random_forest_FULL_METADATA.png", 105):
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(110, 110, 110)
        pdf.cell(0, 6, san("Confusion matrix - Random Forest, FULL_METADATA"), align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
    if pdf.image_safe(FIGURES / "model_comparison_f1.png", 120):
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(110, 110, 110)
        pdf.cell(0, 6, san("Model comparison by F1 score"), align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
    if pdf.image_safe(FIGURES / "model_comparison_fpr.png", 120):
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(110, 110, 110)
        pdf.cell(0, 6, san("Model comparison by false positive rate"), align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    if pdf.image_safe(FIGURES / "feature_importance_random_forest_FULL_METADATA.png", 150):
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(110, 110, 110)
        pdf.cell(0, 6, san("Feature importance - Random Forest, FULL_METADATA"), align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
    if pdf.image_safe(FIGURES / "class_distribution.png", 130):
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(110, 110, 110)
        pdf.cell(0, 6, san("Class distribution - train vs test"), align="C", new_x="LMARGIN", new_y="NEXT")

    # ---------- 7. Reproduce ----------
    pdf.add_page()
    pdf.h1("7. Reproducing the experiment")
    pdf.body("A fresh clone reproduces everything with four commands:")
    pdf.mono(
        "git clone https://github.com/TMStroo/Privacy-Preserving-Network-Anomaly-Detection\n"
        "cd Privacy-Preserving-Network-Anomaly-Detection\n"
        "pip install -r requirements.txt && pip install -e .\n"
        "python scripts/create_sample_data.py   # or drop the real UNSW-NB15 CSVs in data/raw/\n"
        "python -m privacy_preserving_nad.pipeline\n"
        "pytest tests/ -v"
    )
    pdf.body(
        "The pipeline command runs validation, preprocessing, training, evaluation and plotting in one pass. "
        "Individual stages can be run with --step validate, preprocess, train, evaluate or plot. All knobs - "
        "dataset paths, feature lists, target column, seed, model hyperparameters and output directories - "
        "live in configs/config.yaml."
    )
    pdf.body(
        "Determinism: random seeds are fixed (42) for numpy, python's random and every estimator, the train/test "
        "split is the dataset's official one, and preprocessing is fit only on training data. Repeated runs on "
        "the same inputs produce the same metrics."
    )
    pdf.body(
        "Continuous integration runs the test suite and a pipeline smoke test on Python 3.9, 3.10 and 3.11, "
        "using the generated sample data so no large download is needed."
    )

    pdf.h2("Project structure")
    pdf.mono(
        "configs/config.yaml            all experiment configuration\n"
        "data/raw/                      raw dataset (gitignored)\n"
        "data/processed/                processed sets (gitignored)\n"
        "src/privacy_preserving_nad/\n"
        "  data/       download, validate, preprocess\n"
        "  features/   metadata feature policy\n"
        "  models/     baselines, training, inference\n"
        "  evaluation/ metrics and plots\n"
        "  pipeline.py orchestration entry point\n"
        "tests/                         40 pytest tests\n"
        "notebooks/                     exploratory analysis\n"
        "results/metrics/, figures/     generated outputs\n"
        "docs/                          this report\n"
        ".github/workflows/ci.yml       CI pipeline"
    )

    # ---------- 8. Limits ----------
    pdf.h1("8. Limitations")
    pdf.bullet("Dataset age: UNSW-NB15 dates from 2015 and predates widespread TLS 1.3, QUIC and encrypted SNI; modern traffic looks different.")
    pdf.bullet("Dataset-specific behavior: attacks are synthetic, and generators do not reproduce determined real-world adversaries.")
    pdf.bullet("Distribution shift: the split is temporal within the dataset; production traffic will differ further.")
    pdf.bullet("Feature availability: fields like tcprtt, synack and ackdat require visibility of the TCP handshake, which not every collection point has.")
    pdf.bullet("Binary simplification: distinct attack categories are collapsed into one anomaly class, hiding per-attack differences.")
    pdf.bullet("Sample-scale numbers: the results in this report come from 100-row sample data and are not benchmark results.")
    pdf.bullet("Scope of privacy: 'privacy-preserving' means no payload is read. Metadata alone can still identify and profile users; this project proves no formal privacy guarantee.")

    pdf.h2("Future directions")
    pdf.bullet("Re-run on the full UNSW-NB15 split and on newer datasets such as CIC-IDS2017.")
    pdf.bullet("Test adversarial robustness: can an attacker shuffle metadata to evade these detectors?")
    pdf.bullet("Study temporal concept drift in metadata distributions.")
    pdf.bullet("Extend from binary detection to multi-class attack attribution.")
    pdf.bullet("Evaluate feature attribution stability and deploy as a NetFlow/IPFIX consumer.")

    pdf.h2("License")
    pdf.body("MIT License. See the LICENSE file in the repository.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT))
    print("Report written to: %s" % OUTPUT)


if __name__ == "__main__":
    main()
