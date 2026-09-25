"""Write README.md from a template file.

The V1 README is preserved in git history and at results/superseded/. This
script exists so the replacement goes through a normal, reviewable file write
rather than an in-place mutation of a file this session only partially read.
"""

import pathlib
import sys

ROOT = pathlib.Path(r"F:\projects\Privacy-Preserving-Network-Anomaly-Detection")
SRC = ROOT / "scripts" / "readme_template.md"
DST = ROOT / "README.md"

if not SRC.exists():
    raise SystemExit(f"template not found: {SRC}")

text = SRC.read_text(encoding="utf-8")

BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"
if BEGIN not in text or END not in text:
    raise SystemExit("template is missing the generated-results markers")

previous = DST.read_text(encoding="utf-8") if DST.exists() else ""
DST.write_text(text, encoding="utf-8")

print(f"wrote {DST} ({len(text):,} chars, was {len(previous):,})")
