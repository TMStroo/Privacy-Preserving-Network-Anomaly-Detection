"""Immutable per-experiment output directories.

Each run writes into ``results/experiments/<experiment_id>/`` and refuses to
overwrite an existing one, so a later run can never quietly replace the evidence
behind a claim already written in the README or the report.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

EXPERIMENT_FILES = [
    "config.yaml",
    "metadata.json",
    "metrics.json",
    "drift_events.csv",
    "feature_importance.csv",
    "adaptation.json",
    "predictions.parquet",
    "failure_analysis.json",
    "calibration.json",
]


def experiment_id(kind: str, dataset: str, tag: Optional[str] = None, now: Optional[datetime] = None) -> str:
    """Build a run identifier. A short random suffix keeps two runs started in
    the same second from colliding, which matters because experiment directories
    are immutable and a collision would abort the second run."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    parts = [stamp, kind, dataset]
    if tag:
        parts.append(tag)
    parts.append(uuid.uuid4().hex[:6])
    return "_".join(parts)


def git_commit(root: str = ".") -> str:
    """The commit a run came from, or an honest 'unknown'.

    A container image deliberately does not carry the .git directory, because
    that would put the remote URL and any other repository metadata into a
    distributable artifact. The commit is passed in as a build argument
    instead, and this reads it before falling back to asking git directly.
    """
    override = os.environ.get("DRIFTGUARD_GIT_COMMIT", "").strip()
    if override and override != "unknown":
        return override
    # A run is not reproducible without its commit, so a transient failure here
    # is retried rather than recorded. git rev-parse fails while another process
    # holds the index lock, which is exactly what happens when a long benchmark
    # starts at the same moment as a commit.
    for attempt in range(3):
        for candidate in (root, os.getcwd(), _package_root()):
            if not candidate or not os.path.isdir(candidate):
                continue
            try:
                out = subprocess.run(
                    ["git", "-C", candidate, "rev-parse", "HEAD"],
                    capture_output=True, text=True, timeout=15,
                )
            except Exception:
                # git may be absent entirely, which is the normal case inside the
                # container image. That is not a transient failure, so it moves on
                # to the next candidate instead of consuming a retry.
                continue
            commit = out.stdout.strip()
            if len(commit) == 40:
                return commit
        time.sleep(0.5 * (attempt + 1))
    return "unknown"


def _package_root() -> str:
    """The repository that contains the installed package, if there is one."""
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        here = os.path.dirname(here)
        if os.path.isdir(os.path.join(here, ".git")):
            return here
    return ""


def file_checksums(paths: List[str]) -> Dict[str, str]:
    checksums = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        checksums[os.path.basename(path)] = digest.hexdigest()
    return checksums


def schema_hash(columns: List[str]) -> str:
    """Stable fingerprint of a feature schema, so two runs are comparable."""
    payload = "\n".join(sorted(columns)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def package_versions() -> Dict[str, str]:
    """Installed versions of everything that can move a number.

    The import name and the distribution name differ for two of these
    (``scikit-learn`` installs as ``sklearn``, ``PyYAML`` as ``yaml``), so the
    module is imported by its real name and the distribution metadata is the
    fallback. Getting this wrong records "not installed" for a library the run
    plainly used.
    """
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name, module_name in [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("PyYAML", "yaml"),
    ]:
        version = "unknown"
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", None) or "unknown"
        except Exception:
            pass
        if version == "unknown":
            try:
                from importlib.metadata import version as dist_version

                version = dist_version(name)
            except Exception:
                version = "not installed"
        versions[name] = version
    return versions


class ExperimentRun:
    """A single experiment's output directory. Write once, never overwrite."""

    def __init__(self, experiment_id: str, root: str = "results/experiments", allow_existing: bool = False):
        self.experiment_id = experiment_id
        self.path = Path(root) / experiment_id
        if self.path.exists() and not allow_existing:
            raise FileExistsError(
                f"experiment '{experiment_id}' already exists at {self.path}. "
                "Experiments are immutable; choose a new id or pass allow_existing=True deliberately."
            )
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "figures").mkdir(exist_ok=True)

    def write_json(self, name: str, payload) -> Path:
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return target

    def read_json(self, name: str):
        target = self.path / name
        if not target.exists():
            return None
        with open(target, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_text(self, name: str, text: str) -> Path:
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def write_table(self, name: str, frame) -> Path:
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
        return target

    def figure_path(self, name: str) -> Path:
        return self.path / "figures" / name

    def metadata(
        self,
        dataset: str,
        dataset_checksums: Dict[str, str],
        features: List[str],
        seed: int,
        config: Dict,
        split_description: Optional[Dict] = None,
    ) -> Dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "dataset": dataset,
            "dataset_checksums": dataset_checksums,
            "feature_schema_hash": schema_hash(features),
            "feature_count": len(features),
            "random_seed": int(seed),
            "package_versions": package_versions(),
            "temporal_split": split_description or {},
            "config": config,
        }

    def summary(self) -> Dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "path": str(self.path),
            "files": sorted(p.name for p in self.path.iterdir() if p.is_file()),
        }


def list_experiments(root: str = "results/experiments") -> List[Dict[str, object]]:
    base = Path(root)
    if not base.exists():
        return []
    out = []
    for directory in sorted(base.iterdir()):
        if not directory.is_dir():
            continue
        meta = directory / "metadata.json"
        entry = {"experiment_id": directory.name}
        if meta.exists():
            with open(meta, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            entry["dataset"] = data.get("dataset")
            entry["git_commit"] = data.get("git_commit")
            entry["created_utc"] = data.get("created_utc")
        out.append(entry)
    return out
