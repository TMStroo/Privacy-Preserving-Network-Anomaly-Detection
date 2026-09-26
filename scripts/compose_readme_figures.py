"""Build the two cross-dataset comparison figures the README shows.

Three of the four figures the README embeds exist in a run directory as a
single PNG. The other two compare UNSW-NB15 with UGR'16, and the project's
figure code produces one panel per run, so there is no single image of the
comparison. Rather than draw a new chart from the metrics -- which would mean
a second code path producing numbers that could drift from the first -- these
panels are the existing rendered PNGs placed side by side and labelled.

Composing the rendered figures rather than re-plotting is deliberate: the
values in the README still come from the artifacts, and the panels here are
copied byte-for-byte from the run that produced them. Only the arrangement is
new.

Run:  python scripts/compose_readme_figures.py
"""

from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGS = ROOT / "docs" / "figures"
# The per-dataset panels the composed comparisons are built from. They are
# copied out of the run directories, not re-plotted, so the values in the
# comparison are the same bytes the run produced.
PANELS = FIGS / "panels"

# White, so the composed PNG matches the panels it joins. The panels are RGBA
# with an opaque white background already; compositing on white keeps the
# seam invisible if a panel ever arrives transparent.
BACKGROUND = (255, 255, 255, 255)
LABEL_COLOUR = (26, 26, 26, 255)
MARGIN = 18
HEADER = 46
GAP = 16
# The four-line strategy labels in the source panels wrap down to the last few
# pixels of their own canvas. The panels are not rescaled or cropped, so the
# composed image adds a little extra space underneath to keep that final line
# clear of the edge of the file the README renders.
FOOTER = 14


def _font(size: int):
    """A bold sans font, falling back to PIL's default if none is installed.

    The composed figure is documentation, so a missing system font must not
    stop it being generated. The default bitmap font is smaller and less
    polished but still legible.
    """
    candidates = [
        pathlib.Path("C:/Windows/Fonts/segoeuib.ttf"),
        pathlib.Path("C:/Windows/Fonts/arialbd.ttf"),
        pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    return ImageFont.load_default()


def compose_side_by_side(
    left: pathlib.Path,
    right: pathlib.Path,
    out_path: pathlib.Path,
    left_label: str,
    right_label: str,
    title: str,
) -> None:
    """Place two rendered figures next to each other under one title.

    The panels are never rescaled: a different size would resample the text
    in the chart itself, and the composed image has to stay readable at the
    width a README renders it. The canvas is sized to the taller panel and
    the shorter one is centred vertically against it.
    """
    left_image = Image.open(left).convert("RGBA")
    right_image = Image.open(right).convert("RGBA")

    width = MARGIN * 3 + left_image.width + right_image.width
    height = HEADER + MARGIN * 2 + FOOTER + max(left_image.height, right_image.height)

    canvas = Image.new("RGBA", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (MARGIN * 2, MARGIN - 2),
        title,
        font=_font(21),
        fill=LABEL_COLOUR,
    )

    # Each panel keeps its own suptitle from the run that produced it, so the
    # dataset name is repeated here only where it is not already visible.
    label_font = _font(17)
    top = HEADER
    draw.text(
        (MARGIN * 2, top + 4),
        left_label,
        font=label_font,
        fill=LABEL_COLOUR,
    )
    draw.text(
        (MARGIN * 2 + left_image.width + GAP, top + 4),
        right_label,
        font=label_font,
        fill=LABEL_COLOUR,
    )

    left_y = top + 26 + MARGIN
    right_y = left_y + (left_image.height - right_image.height) // 2
    canvas.alpha_composite(left_image, (MARGIN * 2, left_y))
    canvas.alpha_composite(right_image, (MARGIN * 2 + left_image.width + GAP, right_y))

    canvas.convert("RGB").save(out_path, "PNG", optimize=True)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    PANELS.mkdir(parents=True, exist_ok=True)

    compose_side_by_side(
        PANELS / "unsw_adaptation_strategies.png",
        PANELS / "ugr16_adaptation_strategies.png",
        FIGS / "adaptation_unsw_vs_ugr16.png",
        "UNSW-NB15  (forward period has FEWER attacks than the recent history)",
        "UGR'16  (forward period has MORE attacks than the recent history)",
        "The same adaptation strategy helps one dataset and hurts the other",
    )

    compose_side_by_side(
        PANELS / "unsw_drift_alert_rates.png",
        PANELS / "ugr16_drift_alert_rates.png",
        FIGS / "drift_alert_rates_comparison.png",
        "UNSW-NB15",
        "UGR'16",
        "The four drift detectors rank the two datasets in opposite orders",
    )

    for name in (
        "adaptation_unsw_vs_ugr16.png",
        "drift_alert_rates_comparison.png",
        "unsw_backtest_vs_forward.png",
        "target_fpr_ablation.png",
    ):
        path = FIGS / name
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing or empty: {path}")
        print(f"{name:38s} {Image.open(path).size} {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
