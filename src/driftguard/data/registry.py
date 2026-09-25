"""Dataset adapters and the registry describing each one's provenance."""

import io
import json
from typing import Dict

import pandas as pd

from driftguard.data.schema import (
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    FlowFrame,
    add_derived_features,
)


class DatasetAdapter:
    """Base adapter. Subclasses implement ``load`` for one concrete dataset."""

    name = "base"
    files: tuple = ()

    def load(self, raw_dir: str, **kwargs) -> FlowFrame:
        raise NotImplementedError

    def missing_files(self, raw_dir: str) -> list:
        import os

        return [f for f in self.files if not os.path.exists(os.path.join(raw_dir, f))]


class Unswnb15Adapter(DatasetAdapter):
    """UNSW-NB15, mapped from the full raw release that carries flow timestamps.

    The official published train/test CSVs ship without Stime/Ltime, so temporal
    evaluation is impossible from them. The full raw release keeps the start
    time of every flow, which is what this project needs. Column names differ
    in case between the two exports, so the mapping is explicit and
    case-insensitive.
    """

    name = "unsw_nb15"
    files = ("UNSW_NB15_full_raw.parquet",)

    COLUMN_MAP = {
        "stime": TIMESTAMP_COLUMN,
        "dur": "flow_duration",
        "spkts": "forward_packets",
        "dpkts": "backward_packets",
        "sbytes": "forward_bytes",
        "dbytes": "backward_bytes",
        "label": TARGET_COLUMN,
        "tcp_flags": "tcp_flags",
    }

    def load(self, raw_dir: str, max_rows: int = None, **kwargs) -> FlowFrame:
        import os

        path = os.path.join(raw_dir, self.files[0])
        frame = pd.read_parquet(path)
        if max_rows:
            frame = frame.head(max_rows)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        frame = frame.rename(columns={k: v for k, v in self.COLUMN_MAP.items() if k in frame.columns})

        required = ["flow_duration", "forward_packets", "backward_packets", "forward_bytes", "backward_bytes"]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"UNSW-NB15 export is missing expected columns: {missing}")

        frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], unit="s", errors="coerce")
        frame = frame.dropna(subset=[TIMESTAMP_COLUMN])
        if TARGET_COLUMN in frame.columns:
            frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
        keep = [c for c in frame.columns if c in {TIMESTAMP_COLUMN, TARGET_COLUMN} | set(
            ["flow_duration", "forward_packets", "backward_packets", "forward_bytes", "backward_bytes", "tcp_flags"]
        )]
        frame = add_derived_features(frame[keep])
        return FlowFrame(
            self.name,
            frame,
            [os.path.basename(path)],
            {"kind": "bidirectional flow capture with timestamps", "source_release": "UNSW-NB15 full raw"},
        )


class Ugr16Adapter(DatasetAdapter):
    """UGR'16, mapped from the netflow v9 weekly archives.

    Each weekly archive holds a single headerless CSV with twelve netflow
    fields plus a label. The export is unidirectional: it carries total flow
    duration, packet count and byte count, but no forward/backward split and no
    per-direction counters. Those common-schema fields are therefore left out
    rather than approximated, which is why UGR'16 supports temporal and
    cross-dataset experiments but not the direction-based ones.
    """

    name = "ugr16"

    # The full UGR'16 release is 23 weekly archives totalling roughly 215 GB.
    # This project uses a documented subset, and results are reported as
    # subset results - never as full-dataset UGR'16 numbers.
    #
    # Two calibration weeks (background traffic only, no attacks) and five test
    # weeks (background plus attacks). The subset keeps a contiguous run of
    # test weeks so the forward test is genuinely later traffic:
    #
    #   calibration : march_week3, may_week1
    #   test        : july_week5, august_week1, august_week2, august_week3
    #
    # august_week4 and august_week5 are listed but only august_week5 is used;
    # see SUBSET_NOTES for why the run stops at week 3.
    files = (
        "march_week3_csv.tar.gz",
        "may_week1_csv.tar.gz",
        "july_week5_csv.tar.gz",
        "august_week1_csv.tar.gz",
        "august_week2_csv.tar.gz",
        "august_week3_csv.tar.gz",
    )

    CALIBRATION_FILES = ("march_week3_csv.tar.gz", "may_week1_csv.tar.gz")
    TEST_FILES = (
        "july_week5_csv.tar.gz",
        "august_week1_csv.tar.gz",
        "august_week2_csv.tar.gz",
        "august_week3_csv.tar.gz",
    )

    SUBSET_NOTES = {
        "full_release_size": "~215 GB across 23 weekly archives",
        "subset_size": "~40 GB across 6 weekly archives",
        "weeks_used": list(TEST_FILES),
        "calibration_weeks": list(CALIBRATION_FILES),
        "why": (
            "A contiguous run of test weeks is needed so the forward test is "
            "later traffic. August weeks 1-3 give three consecutive weeks with "
            "both normal and attack flows, with july_week5 as the earliest "
            "point and the two calibration weeks providing background-only "
            "traffic. This covers roughly one month, not the dataset's full "
            "six-month span."
        ),
        "limitations": [
            "This is a subset, not the full UGR'16 release. No claim is made "
            "about the weeks that were not downloaded.",
            "The calibration weeks contain background traffic only, so no model "
            "is trained on them; they exist to characterise normal behaviour.",
            "UGR'16 attacks are produced by replaying malware and tools against "
            "a real ISP link, so attack realism is bounded by that setup.",
            "The netflow export has no forward/backward direction split, which "
            "removes a set of features UNSW-NB15 provides.",
        ],
    }

    # Positional schema of the headerless netflow export, in file order.
    # Read from the actual files, which are comma separated and carry a leading
    # blank line. A real row from this subset:
    #   2016-05-01 00:03:06,4.236,165.131.105.128,42.219.156.211,56428,80,TCP,.AP.S.,0,0,5,677,background
    # gives thirteen fields: datetime, duration, src, dst, sport, dport,
    # protocol, flags, tos, tos_field, packets, bytes, label. UGR'16's published
    # netflow description also lists a 14th attack-name column, which the
    # archives in this subset do not carry - the reader below names fourteen and
    # lets pandas fill the absent one, so a future archive that does include it
    # is read correctly without a schema change.
    NETFLOW_COLUMNS = [
        "timestamp",
        "duration",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "flags",
        "tos",
        "tos_field",
        "packets",
        "bytes",
        "label",
        "attack",
    ]

    LABEL_MAP = {"background": 0, "blacklist": 1}

    def load(self, raw_dir: str, max_rows_per_file: int = None, **kwargs) -> FlowFrame:
        import os
        import tarfile

        present = [f for f in self.files if os.path.exists(os.path.join(raw_dir, f))]
        if not present:
            raise FileNotFoundError(
                f"No UGR'16 weekly archives found in {raw_dir}. "
                "Fetch them with: python -m driftguard data fetch --dataset ugr16"
            )
        frames = []
        used = []
        for fname in present:
            frames.append(self._read_archive(os.path.join(raw_dir, fname), max_rows_per_file))
            used.append(fname)

        frame = pd.concat(frames, ignore_index=True)
        frame = add_derived_features(frame)
        return FlowFrame(
            self.name,
            frame,
            used,
            {
                "kind": "netflow v9 weekly archives, unidirectional",
                "source_release": "UGR'16 calibration (Mar-Jun 2016) and test (Jul-Aug 2016)",
            },
        )

    @staticmethod
    def _open_archive(path: str):
        import tarfile

        return tarfile.open(path, "r|gz")

    def _read_archive(self, path: str, max_rows: int) -> pd.DataFrame:
        # Stream rather than calling getmembers(): a weekly archive holds one
        # 12 GB CSV, and building the full member list first means
        # decompressing the whole thing before a single row is read.
        with self._open_archive(path) as tar:
            member = next((m for m in tar if m.isfile()), None)
            if member is None:
                raise ValueError(f"Archive contains no data file: {path}")
            handle = tar.extractfile(member)
            # A `r|gz` stream is not seekable and read_csv insists on seeking, so
            # lines are collected here and handed over as one string. The limit
            # counts *data* rows: the exports open with a blank line, and a
            # chunked read can hand back a trailing partial line, so one extra
            # line is read and the newest complete one is kept.
            buffer: List[str] = []
            for line in handle:
                text = line.decode("utf-8", "replace")
                if not text.endswith("\n"):
                    text += "\n"  # complete the last line of a chunked read
                buffer.append(text)
                data_lines = sum(1 for b in buffer if b.strip())
                if max_rows and data_lines > max_rows:
                    break
            # The exports open with a blank line, so the first entry is dropped
            # when it holds no data at all.
            if buffer and not buffer[0].strip():
                buffer = buffer[1:]
            if max_rows:
                buffer = buffer[:max_rows]
            # These archives carry thirteen fields. The reader is given all
            # fourteen names so an archive that does include the documented
            # attack-name column still lines up, and index_col=False keeps the
            # short rows from being treated as having an unnamed first column.
            frame = pd.read_csv(
                io.StringIO("".join(buffer)),
                names=self.NETFLOW_COLUMNS,
                header=None,
                index_col=False,
                dtype={"src_ip": str, "dst_ip": str, "protocol": str,
                       "flags": str, "label": str, "src_port": str,
                       "dst_port": str, "tos_field": str},
            )
        if frame.empty:
            raise ValueError(f"No rows parsed from {path}")
        return self._to_common(frame)

    def _to_common(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        # The export writes "YYYY-MM-DD HH:MM:SS"; naming the format avoids a
        # per-element fallback that is both slow and inconsistent.
        out[TIMESTAMP_COLUMN] = pd.to_datetime(
            frame["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        out["flow_duration"] = pd.to_numeric(frame["duration"], errors="coerce")
        out["total_packets"] = pd.to_numeric(frame["packets"], errors="coerce")
        out["total_bytes"] = pd.to_numeric(frame["bytes"], errors="coerce")
        out["protocol"] = frame["protocol"].astype(str).str.strip()
        out["tcp_flags"] = frame["flags"].astype(str).str.strip()
        out["tos"] = pd.to_numeric(frame["tos"], errors="coerce")
        out[TARGET_COLUMN] = frame["label"].astype(str).str.strip().str.lower().map(self.LABEL_MAP)
        return out.dropna(subset=[TIMESTAMP_COLUMN, TARGET_COLUMN])


class SyntheticAdapter(DatasetAdapter):
    """Deterministic synthetic flows with an inserted distribution shift.

    This exists so the demo and the test suite can exercise the whole pipeline
    without a multi-gigabyte download. Its output is generated data with a
    deliberately planted shift, and is never presented as a research result.
    """

    name = "synthetic"
    files = ()

    def load(self, raw_dir: str, rows: int = 40000, days: int = 20,
             shift_at_fraction: float = 0.6, shift_kind: str = "packet_size",
             shift_magnitude: float = 3.0, **kwargs) -> FlowFrame:
        from driftguard.data.synthetic import generate_flows

        frame = generate_flows(
            rows=rows,
            days=days,
            shift_at_fraction=shift_at_fraction,
            shift_kind=shift_kind,
            shift_magnitude=shift_magnitude,
            seed=int(kwargs.get("seed", 7)),
        )
        return FlowFrame(
            self.name,
            frame,
            (),
            {"kind": "synthetic demo fixture - NOT a research dataset",
             "shift": {"at_fraction": shift_at_fraction, "kind": shift_kind, "magnitude": shift_magnitude}},
        )


ADAPTERS = {
    Unswnb15Adapter.name: Unswnb15Adapter,
    Ugr16Adapter.name: Ugr16Adapter,
    SyntheticAdapter.name: SyntheticAdapter,
}


def get_adapter(name: str) -> DatasetAdapter:
    if name not in ADAPTERS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(ADAPTERS)}")
    return ADAPTERS[name]()


def dataset_record(name: str) -> Dict:
    """Provenance record for one dataset, used by the report and README renderer."""
    here = __file__
    import os

    catalog_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(here))), "data", "dataset_catalog.json")
    if not os.path.exists(catalog_path):
        return {"name": name, "available": False, "reason": "catalog not found"}
    with open(catalog_path, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    for record in catalog.get("datasets", []):
        if record.get("name") == name:
            return record
    return {"name": name, "available": False, "reason": "not in catalog"}


def all_records() -> list:
    here = __file__
    import os

    catalog_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(here))), "data", "dataset_catalog.json")
    if not os.path.exists(catalog_path):
        return []
    with open(catalog_path, "r", encoding="utf-8") as handle:
        return json.load(handle).get("datasets", [])
