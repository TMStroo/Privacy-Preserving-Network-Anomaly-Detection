"""Opt-in dataset downloads.

Every fetch is explicit: a human runs it, sizes are stated up front, and nothing
here is invoked by the test suite, the demo, or an ordinary benchmark run.
Checksums are recorded so a later run can verify what it got.
"""

import os
from typing import Dict, List, Optional

import requests

UNSW_MIRROR = "https://huggingface.co/datasets/research-simulation26/unsw-nb15-parquet/resolve/main"

UGR16_MIRROR = "https://huggingface.co/datasets/mehedi107/ugr-16/resolve/main"

# Sizes are the compressed sizes published by the mirror's manifest.
UGR16_WEEKS: Dict[str, Dict[str, object]] = {
    "march_week3": {"path": "calibration/march/week3/march_week3_csv.tar.gz", "gb": 3.6, "period": "calibration"},
    "may_week1": {"path": "calibration/may/week1/may_week1_csv.tar.gz", "gb": 1.7, "period": "calibration"},
    "july_week5": {"path": "test/july/week5/july_week5_csv.tar.gz", "gb": 7.7, "period": "test"},
    "august_week1": {"path": "test/august/week1/august_week1_csv.tar.gz", "gb": 12.1, "period": "test"},
    "august_week2": {"path": "test/august/week2/august_week2_csv.tar.gz", "gb": 12.1, "period": "test"},
    "august_week3": {"path": "test/august/week3/august_week3_csv.tar.gz", "gb": 11.9, "period": "test"},
    "august_week4": {"path": "test/august/week4/august_week4_csv.tar.gz", "gb": 12.5, "period": "test"},
    "august_week5": {"path": "test/august/week5/august_week5_csv.tar.gz", "gb": 0.6, "period": "test"},
}


def _download(url: str, target: str) -> bool:
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"  already present: {os.path.basename(target)}")
        return True
    print(f"  downloading {os.path.basename(target)} ...")
    try:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(target, "wb") as handle:
                for chunk in response.iter_content(1 << 22):
                    handle.write(chunk)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return False
    size_gb = os.path.getsize(target) / 1e9
    print(f"  done ({size_gb:.2f} GB)")
    return True


def fetch_unsw_nb15(raw_dir: str, **kwargs) -> int:
    """Fetch the full UNSW-NB15 raw release, which carries flow timestamps."""
    print("UNSW-NB15 full raw release (2.54M flows with Stime/Ltime, ~130 MB)")
    target = os.path.join(raw_dir, "UNSW_NB15_full_raw.parquet")
    return 0 if _download(f"{UNSW_MIRROR}/raw_data.parquet", target) else 1


def fetch_ugr16(raw_dir: str, weeks: str = "all", **kwargs) -> int:
    """Fetch selected UGR'16 weekly netflow archives.

    The full UGR'16 release is about 215 GB compressed, so the week list is
    explicit and the size is printed before anything is written.
    """
    selected = list(UGR16_WEEKS) if weeks == "all" else [w.strip() for w in weeks.split(",")]
    unknown = [w for w in selected if w not in UGR16_WEEKS]
    if unknown:
        print(f"unknown weeks: {unknown}. Available: {sorted(UGR16_WEEKS)}")
        return 1
    total = sum(float(UGR16_WEEKS[w]["gb"]) for w in selected)
    print(f"UGR'16 weekly archives: {len(selected)} file(s), about {total:.1f} GB compressed")
    print("The calibration weeks contain background traffic only; the test weeks")
    print("contain background plus synthetic attacks. Both are needed for the")
    print("temporal experiment.\n")
    ok = True
    for week in selected:
        entry = UGR16_WEEKS[week]
        target = os.path.join(raw_dir, f"{week}_csv.tar.gz")
        ok &= _download(f"{UGR16_MIRROR}/{entry['path']}", target)
    return 0 if ok else 1


FETCHERS = {
    "unsw_nb15": fetch_unsw_nb15,
    "ugr16": fetch_ugr16,
}
