"""Write README.md from a checked-in source file.

The README is rendered from results/experiments between releases, so it is
rewritten by a script rather than by an editor. Keeping the text in a separate
file also means the content is reviewable as content, and a regeneration never
has to reconstruct prose that a tool call may have truncated.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "readme_source.md"
TARGET = ROOT / "README.md"

text = SOURCE.read_text(encoding="utf-8")
TARGET.write_text(text, encoding="utf-8")
print(f"README.md written: {len(text.splitlines())} lines, {len(text):,} chars")
