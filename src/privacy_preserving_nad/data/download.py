"""Explicit, opt-in fetch of the official UNSW-NB15 research dataset.

Nothing in the test suite or the normal pipeline calls the download: fetching
tens of megabytes only happens when this module is run on purpose:

    python -m privacy_preserving_nad.data.download           # status + instructions
    python -m privacy_preserving_nad.data.download --fetch   # download + verify

The official release comes from UNSW (research.unsw.edu.au); the configured
mirror is a byte-identical copy, verified against the sha256 values recorded
in configs/config.yaml before any file is accepted.
"""

import argparse
from pathlib import Path
from typing import Dict

import requests

from privacy_preserving_nad.data.dataset import (
    count_rows,
    detect_dataset,
    load_config,
    print_dataset_banner,
    sha256_of,
)


def fetch_file(url: str, dest: Path) -> None:
    print(f"  Fetching {dest.name} ...")
    response = requests.get(url, stream=True, timeout=(15, 120))
    response.raise_for_status()

    tmp = dest.with_suffix(dest.suffix + ".part")
    downloaded = 0
    with open(tmp, "wb") as fh:
        for chunk in response.iter_content(1 << 20):
            fh.write(chunk)
            downloaded += len(chunk)
            print(f"\r    {downloaded / 1e6:,.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def verify_file(path: Path, expected_rows: int, expected_sha256: str) -> Dict:
    rows = count_rows(path)
    if rows is None:
        return {"file": path.name, "status": "missing"}

    if rows != expected_rows:
        return {"file": path.name, "status": "wrong-row-count", "rows": rows,
                "expected_rows": expected_rows}

    digest = sha256_of(path)
    ok = digest == expected_sha256
    return {"file": path.name,
            "status": "ok" if ok else "checksum-mismatch",
            "rows": rows, "sha256": digest}


def fetch_research_dataset(config_path: str = "configs/config.yaml") -> bool:
    config = load_config(config_path)
    raw_dir = Path(config["dataset"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    spec = config["dataset"]["research_dataset"]
    sources = config["dataset"]["sources"]

    targets = [
        (config["dataset"]["train_file"], sources["mirror_train"],
         spec["train_rows"], spec["train_sha256"]),
        (config["dataset"]["test_file"], sources["mirror_test"],
         spec["test_rows"], spec["test_sha256"]),
    ]

    all_ok = True
    for filename, url, rows, sha in targets:
        dest = raw_dir / filename
        result = verify_file(dest, rows, sha)
        if result["status"] == "ok":
            print(f"  {filename}: already present and verified ({rows:,} rows)")
            continue

        fetch_file(url, dest)
        result = verify_file(dest, rows, sha)
        if result["status"] != "ok":
            all_ok = False
            print(f"  {filename}: verification failed ({result['status']})")
        else:
            print(f"  {filename}: verified ({rows:,} rows, sha256 ok)")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="UNSW-NB15 research dataset helper")
    parser.add_argument("--fetch", action="store_true",
                        help="download the official split into data/raw/ and verify it")
    args = parser.parse_args()

    if args.fetch:
        ok = fetch_research_dataset()
        print("\nResearch dataset ready." if ok else
              "\nDownload finished but verification failed; re-run or fetch manually.")
        return

    print("UNSW-NB15 research dataset - status")
    print("=" * 60)
    print("Official release (source of record):")
    print("  https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    print()
    print("Manual download: place these files in data/raw/")
    print("  UNSW_NB15_training-set.csv   (175,341 rows)")
    print("  UNSW_NB15_testing-set.csv    (82,332 rows)")
    print()
    print("Or fetch the verified copy programmatically:")
    print("  python -m privacy_preserving_nad.data.download --fetch")
    print()
    print_dataset_banner(detect_dataset())


if __name__ == "__main__":
    main()
