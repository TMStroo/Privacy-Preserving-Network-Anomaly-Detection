#!/usr/bin/env python
"""Write a small DEVELOPMENT SAMPLE of UNSW-NB15-shaped data for tests and CI.

This is not the research dataset. It carries the official 45-column schema so
the pipeline is exercised on the same shape as the real split, but its rows are
synthetic. The pipeline reports it as a development sample; benchmark numbers
require the official files (see README, 'Dataset').

Never runs over the research dataset: if the official files are already in
place, this script refuses to touch them.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_TRAIN_ROWS = 175341
RESEARCH_TEST_ROWS = 82332
SEED = 42

COLUMNS = [
    "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes",
    "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth",
    "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports", "attack_cat", "label",
]


def make_split(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    attack_cat = np.where(label == 1, "Exploits", "Normal")

    spkts = rng.integers(1, 60, n)
    dpkts = np.maximum(1, rng.integers(0, 40, n))
    dur = rng.exponential(5.0, n) + 0.001

    df = pd.DataFrame({
        "id": np.arange(n),
        "dur": dur,
        "proto": rng.choice(["tcp", "udp", "icmp"], n, p=[0.7, 0.25, 0.05]),
        "service": rng.choice(["http", "dns", "ftp", "smtp", "-"], n, p=[0.4, 0.2, 0.1, 0.1, 0.2]),
        "state": rng.choice(["FIN", "CON", "REQ", "RST", "INT"], n),
        "spkts": spkts,
        "dpkts": dpkts,
        "sbytes": spkts * rng.integers(60, 900, n),
        "dbytes": dpkts * rng.integers(60, 900, n),
        "rate": (spkts + dpkts) / dur,
        "sttl": rng.integers(32, 255, n),
        "dttl": rng.integers(32, 255, n),
        "sload": rng.integers(100, 100000, n),
        "dload": rng.integers(100, 100000, n),
        "sloss": rng.integers(0, 4, n),
        "dloss": rng.integers(0, 4, n),
        "sinpkt": rng.exponential(0.5, n),
        "dinpkt": rng.exponential(0.5, n),
        "sjit": rng.exponential(0.2, n),
        "djit": rng.exponential(0.2, n),
        "swin": rng.integers(1024, 65536, n),
        "stcpb": rng.integers(0, 2**31, n),
        "dtcpb": rng.integers(0, 2**31, n),
        "dwin": rng.integers(1024, 65536, n),
        "tcprtt": rng.exponential(0.08, n),
        "synack": rng.exponential(0.04, n),
        "ackdat": rng.exponential(0.04, n),
        "smean": rng.integers(60, 1400, n),
        "dmean": rng.integers(60, 1400, n),
        "trans_depth": rng.integers(0, 8, n),
        "response_body_len": rng.integers(0, 4000, n),
        "ct_srv_src": rng.integers(1, 30, n),
        "ct_state_ttl": rng.integers(1, 30, n),
        "ct_dst_ltm": rng.integers(1, 30, n),
        "ct_src_dport_ltm": rng.integers(1, 30, n),
        "ct_dst_sport_ltm": rng.integers(1, 30, n),
        "ct_dst_src_ltm": rng.integers(1, 30, n),
        "is_ftp_login": rng.integers(0, 2, n),
        "ct_ftp_cmd": rng.integers(0, 5, n),
        "ct_flw_http_mthd": rng.integers(0, 5, n),
        "ct_src_ltm": rng.integers(1, 30, n),
        "ct_srv_dst": rng.integers(1, 30, n),
        "is_sm_ips_ports": rng.integers(0, 2, n),
        "attack_cat": attack_cat,
        "label": label,
    })
    return df[COLUMNS]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Write a DEVELOPMENT SAMPLE of UNSW-NB15-shaped data."
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Destination directory (default: data/raw)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw_dir / "UNSW_NB15_training-set.csv"
    test_path = raw_dir / "UNSW_NB15_testing-set.csv"

    for path, expected in [(train_path, RESEARCH_TRAIN_ROWS), (test_path, RESEARCH_TEST_ROWS)]:
        if path.exists():
            with open(path, "rb") as f:
                rows = sum(1 for _ in f) - 1
            if rows == expected:
                print(f"Refusing to overwrite the research dataset: {path} ({rows:,} rows).")
                print("Delete the file yourself if you really want the development sample.")
                sys.exit(1)

    make_split(100, SEED).to_csv(train_path, index=False)
    make_split(60, SEED + 1).to_csv(test_path, index=False)

    print("DEVELOPMENT SAMPLE written - this is not the research dataset.")
    print(f"  {train_path}: 100 rows (synthetic)")
    print(f"  {test_path}: 60 rows (synthetic)")
    print("Benchmark results require the official UNSW-NB15 split; see README.md.")


if __name__ == "__main__":
    main()
