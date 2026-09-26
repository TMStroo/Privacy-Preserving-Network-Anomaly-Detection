"""Verify the figures the README embeds.

A README image that 404s, or that is a zero-byte or truncated file, renders as
a broken box, and nothing else in the repository would notice. This reads every
image link out of README.md, resolves it relative to the repository root, and
checks the file is a real PNG with plausible dimensions and no truncation.

Run:  python scripts/check_readme_figures.py
"""

from __future__ import annotations

import pathlib
import re
import struct
import sys

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")

# Markdown image: ![alt](target). The alt text is required to be descriptive,
# so an empty one is a failure rather than a style note.
LINK = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

problems: list[str] = []
seen: list[tuple[str, str]] = []

for alt, target in LINK.findall(README):
    seen.append((alt, target))

    if not alt.strip():
        problems.append("image with empty alt text")
    if len(alt) < 40:
        problems.append(f"alt text too short to be descriptive ({len(alt)} chars): {target}")
    if target.startswith(("http://", "https://", "//")):
        problems.append(f"external image URL, must be repository-relative: {target}")
    if "://" in target:
        problems.append(f"external image URL: {target}")
    if target.startswith("<") or target.startswith(" "):
        problems.append(f"malformed link target: {target!r}")

    path = (ROOT / target).resolve()
    # A link that escapes the repository would break for anyone who clones it.
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        problems.append(f"link points outside the repository: {target}")
        continue

    if not path.is_file():
        problems.append(f"referenced file does not exist: {target}")
        continue

    size = path.stat().st_size
    if size == 0:
        problems.append(f"file is empty: {target}")
        continue

    with open(path, "rb") as handle:
        head = handle.read(8)
    if head != b"\x89PNG\r\n\x1a\n":
        problems.append(f"not a PNG (bad signature): {target}")
        continue

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
    except Exception as error:  # noqa: BLE001 - any decode failure is a finding
        problems.append(f"PNG will not decode: {target}: {error}")
        continue

    if width < 200 or height < 150:
        problems.append(f"implausibly small image {width}x{height}: {target}")
    if width < height:
        problems.append(f"portrait image will render poorly in a README column: {target}")

    # A truncated PNG still has a valid signature, so check the trailer too.
    # The last twelve bytes are the IEND chunk: a four-byte zero length, the
    # four-byte type, then the CRC. An earlier version of this check compared
    # only the final eight and so failed on every valid PNG.
    with open(path, "rb") as handle:
        handle.seek(-12, 2)
        trailer = handle.read(12)
    if trailer != b"\x00\x00\x00\x00IEND" + struct.pack(">I", 0xAE426082):
        problems.append(f"PNG appears truncated (no IEND): {target}")

    print(f"  ok  {target:52s} {width}x{height} {mode:5s} {size:>8,} bytes  alt={len(alt)} chars")

# Every PNG directly in docs/figures should be referenced, or the directory is
# carrying dead weight a reader will eventually trip over. The panels/ subfolder
# holds the per-dataset source images the composed figures are built from, so
# those are inputs rather than orphans.
FIGS = ROOT / "docs" / "figures"
referenced = {t for _, t in seen}
if FIGS.is_dir():
    for image in sorted(FIGS.glob("*.png")):
        rel = f"docs/figures/{image.name}"
        if rel not in referenced:
            problems.append(f"file in docs/figures is never referenced by the README: {rel}")

print(f"\n{len(seen)} images referenced by README.md")
if problems:
    print(f"{len(problems)} problems:")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)
print("every referenced image exists, is a valid non-empty PNG, and is descriptive")
