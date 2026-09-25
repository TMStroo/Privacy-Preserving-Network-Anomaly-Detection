"""Detection and integrity checks for the data living in data/raw/.

The repository distinguishes two things that share the same filenames:

- the research dataset: the official UNSW-NB15 split (175,341 train rows,
  82,332 test rows), verified by row count and sha256
- a development sample written by scripts/create_sample_data.py, used only
  for tests and pipeline smoke checks

Nothing here downloads data; see data/download.py for fetching the research
dataset explicitly.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import yaml


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def count_rows(path: Path) -> Optional[int]:
    """Count data rows (total lines minus header) without loading the file."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        lines = sum(1 for _ in f)
    return max(lines - 1, 0)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_dataset(config: Optional[Dict] = None, config_path: str = "configs/config.yaml") -> Dict:
    """Classify what is currently in data/raw/."""
    if config is None:
        config = load_config(config_path)

    raw_dir = Path(config["dataset"]["raw_dir"])
    spec = config["dataset"]["research_dataset"]

    train_rows = count_rows(raw_dir / config["dataset"]["train_file"])
    test_rows = count_rows(raw_dir / config["dataset"]["test_file"])

    info = {
        "raw_dir": str(raw_dir),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "research_train_rows": spec["train_rows"],
        "research_test_rows": spec["test_rows"],
        "train_sha256": None,
        "test_sha256": None,
        "checksums_match": None,
        "kind": "missing",
    }

    if train_rows is None and test_rows is None:
        info["kind"] = "missing"
    elif train_rows == spec["train_rows"] and test_rows == spec["test_rows"]:
        info["train_sha256"] = sha256_of(raw_dir / config["dataset"]["train_file"])
        info["test_sha256"] = sha256_of(raw_dir / config["dataset"]["test_file"])
        match = info["train_sha256"] == spec["train_sha256"] and info["test_sha256"] == spec["test_sha256"]
        info["checksums_match"] = match
        info["kind"] = "research" if match else "unexpected-content"
    elif train_rows is not None and test_rows is not None:
        info["kind"] = "development-sample"
    else:
        info["kind"] = "incomplete"

    return info


def is_research_dataset(info: Dict) -> bool:
    return info["kind"] == "research"


def print_dataset_banner(info: Dict) -> None:
    print("\n" + "=" * 60)
    print("DATASET STATUS")
    print("=" * 60)

    kind = info["kind"]
    if kind == "research":
        print(f"  Research dataset found: {info['train_rows']:,} train rows / "
              f"{info['test_rows']:,} test rows (official UNSW-NB15 split)")
        print("  Checksums match the recorded integrity values.")
        print("  This run produces research results.")
    elif kind == "development-sample":
        print(f"  Development sample found: {info['train_rows']:,} train rows / "
              f"{info['test_rows']:,} test rows")
        print("  >>> SMOKE TEST ONLY - these are NOT UNSW-NB15 benchmark results.")
        print("  >>> Place the official split in data/raw/ for real experiments;")
        print("  >>> see README.md, 'Dataset'.")
    elif kind == "unexpected-content":
        print("  Files match the official row counts but NOT the recorded checksums.")
        print("  The files may be corrupted or re-encoded. Re-download them; see README.md.")
    elif kind == "incomplete":
        print("  Only one of the two dataset files is present.")
        print("  Both are required. See README.md, 'Dataset'.")
    else:
        print("  No dataset found in data/raw/.")
        print("  For research results, fetch the official UNSW-NB15 split:")
        print("    python -m privacy_preserving_nad.data.download")
        print("  For a smoke test, write a development sample:")
        print("    python scripts/create_sample_data.py")


def save_dataset_info(info: Dict, path: str = "results/metrics/dataset_info.json") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    return out


def main():
    info = detect_dataset()
    print_dataset_banner(info)
    save_dataset_info(info)
    print(f"\nSaved: results/metrics/dataset_info.json")


if __name__ == "__main__":
    main()
