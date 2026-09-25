"""Produce the whole evidence set in one detached run.

The three stages run sequentially on purpose. Gradient boosting on 2.5M rows
uses every core, so running two benchmarks at once would make each about twice
as slow for no gain. Each stage writes its own immutable experiment directory,
so a failure in a later stage never invalidates an earlier one.

DRIFTGUARD_RAW_DIR points at the external data directory; nothing is downloaded.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = sys.executable

STAGES = [
    ("UNSW-NB15 temporal benchmark", ["-m", "driftguard", "benchmark", "--config", "configs/benchmark.yaml"]),
    ("UGR'16 subset benchmark", ["-m", "driftguard", "benchmark", "--config", "configs/ugr16_subset.yaml"]),
    ("Controlled shifts on UNSW-NB15", ["-m", "driftguard", "shift", "--config", "configs/shifts.yaml"]),
]


def main() -> int:
    os.chdir(REPO)
    started = time.time()
    for index, (label, argv) in enumerate(STAGES, start=1):
        print(f"\n{'=' * 70}\n[{index}/{len(STAGES)}] {label}\n{'=' * 70}", flush=True)
        t0 = time.time()
        # Each stage is its own process so one crash cannot corrupt another's
        # interpreter state.
        result = subprocess.run([PY, *argv], cwd=str(REPO), env=os.environ.copy())
        print(f"--- {label} finished rc={result.returncode} in {time.time() - t0:.0f}s", flush=True)
        if result.returncode != 0:
            print(f"STAGE FAILED: {label}", flush=True)
            return result.returncode
    print(f"\nall stages finished in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
