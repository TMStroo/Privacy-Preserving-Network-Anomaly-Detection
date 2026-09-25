"""Data download module for UNSW-NB15 dataset."""

import os
import hashlib
import requests
from pathlib import Path
from typing import Dict, Optional
import yaml


class UNSWNB15Downloader:
    """Handles downloading and verifying UNSW-NB15 dataset files."""

    # Known file sizes for validation (in bytes)
    EXPECTED_SIZES = {
        "UNSW_NB15_training-set.csv": 189690000,  # ~180 MB
        "UNSW_NB15_testing-set.csv": 73600000,    # ~70 MB
        "UNSW_NB15_features.csv": 5000,           # ~5 KB
    }

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize downloader with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = Path(self.config["dataset"]["raw_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.files = {
            "train": self.config["dataset"]["train_file"],
            "test": self.config["dataset"]["test_file"],
            "features": self.config["dataset"]["features_file"],
        }

    def _download_file(self, url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
        """Download a file with progress indication."""
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(f"\r  Downloading {dest_path.name}: {pct:.1f}%", end="", flush=True)

            print(f"\r  Downloaded {dest_path.name}: {downloaded / 1e6:.1f} MB")
            return True

        except Exception as e:
            print(f"\n  Error downloading {dest_path.name}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False

    def _verify_file(self, file_path: Path) -> bool:
        """Verify file exists and has expected size."""
        if not file_path.exists():
            print(f"  Missing: {file_path.name}")
            return False

        actual_size = file_path.stat().st_size
        expected_size = self.EXPECTED_SIZES.get(file_path.name)

        if expected_size and actual_size < expected_size * 0.9:
            print(f"  Size mismatch: {file_path.name} ({actual_size} bytes, expected ~{expected_size})")
            return False

        print(f"  Verified: {file_path.name} ({actual_size / 1e6:.1f} MB)")
        return True

    def download_all(self, force: bool = False) -> Dict[str, bool]:
        """Download all required dataset files."""
        results = {}

        print("Downloading UNSW-NB15 dataset...")
        print(f"Destination: {self.raw_dir}")

        # Note: The official UNSW-NB15 dataset is hosted on CloudStor
        # which requires manual download. We provide the URLs and instructions.
        print("\nNOTE: UNSW-NB15 dataset must be downloaded manually from:")
        print("  https://www.unsw.adfa.edu.au/australian-centre-for-cyber-security/cybersecurity/ADFA-NB15-Datasets/")
        print("\nRequired files:")
        for key, filename in self.files.items():
            filepath = self.raw_dir / filename
            print(f"  - {filename}")

        # Check existing files
        all_present = True
        for filename in self.files.values():
            filepath = self.raw_dir / filename
            if not self._verify_file(filepath):
                all_present = False

        if all_present and not force:
            print("\nAll files already present and verified.")
            results["all"] = True
            return results

        print("\nPlease download the files manually and place them in:", self.raw_dir)
        print("Then run this script again to verify.")

        results["all"] = all_present
        return results

    def verify_all(self) -> bool:
        """Verify all dataset files are present and valid."""
        all_valid = True
        for filename in self.files.values():
            filepath = self.raw_dir / filename
            if not self._verify_file(filepath):
                all_valid = False
        return all_valid


def main():
    """Main entry point for downloading dataset."""
    downloader = UNSWNB15Downloader()
    downloader.download_all()


if __name__ == "__main__":
    main()