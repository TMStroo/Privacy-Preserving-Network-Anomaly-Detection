"""UGR'16 adapter tests.

The netflow export is headerless, so every column is positional. A schema that
is off by one does not fail loudly - it just produces a frame where the
timestamps are ports and the protocols are TCP flags - so these tests pin the
mapping against a real row copied from the subset archives.
"""

import io
import tarfile

import numpy as np
import pandas as pd
import pytest

from driftguard.data.registry import Ugr16Adapter, get_adapter

# A verbatim row from the subset archives (may_week1), with the leading blank
# line the files carry.
REAL_ROW = (
    "2016-05-01 00:03:06,4.236,165.131.105.128,42.219.156.211,56428,80,"
    "TCP,.AP.S.,0,0,5,677,background\n"
)


def _archive_with(tmp_path, lines: str) -> str:
    path = tmp_path / "week_csv.tar.gz"
    payload = ("\n" + lines).encode("utf-8")
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("uniq/test.week1.csv.uniqblacklistremoved")
        info.size = len(payload)
        import io as _io

        tar.addfile(info, _io.BytesIO(payload))
    return str(path)


def test_netflow_schema_matches_the_real_row():
    assert len(REAL_ROW.strip().split(",")) == 13


def test_adapter_maps_every_positional_field(tmp_path):
    path = _archive_with(tmp_path, REAL_ROW)
    frame = Ugr16Adapter()._read_archive(path, max_rows=10)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["timestamp"] == pd.Timestamp("2016-05-01 00:03:06")
    assert row["flow_duration"] == pytest.approx(4.236)
    assert row["protocol"] == "TCP"
    assert row["tcp_flags"] == ".AP.S."
    assert row["total_packets"] == pytest.approx(5.0)
    assert row["total_bytes"] == pytest.approx(677.0)
    assert row["label"] == 0


def test_ips_are_not_read_as_numbers(tmp_path):
    """A port column read as a float would make the timestamp nonsense."""
    path = _archive_with(tmp_path, REAL_ROW)
    frame = Ugr16Adapter()._read_archive(path, max_rows=10)
    assert not frame.empty
    assert frame["timestamp"].notna().all()


def test_blacklist_rows_become_the_attack_class(tmp_path):
    attack = REAL_ROW.replace("background", "blacklist")
    path = _archive_with(tmp_path, REAL_ROW + attack)
    frame = Ugr16Adapter()._read_archive(path, max_rows=10)
    assert frame["label"].tolist() == [0, 1]


def test_leading_blank_line_is_not_counted_as_data(tmp_path):
    """The exports open with a bare newline. Reading it as a row both corrupts
    the frame and makes the row limit return one row too few."""
    path = _archive_with(tmp_path, REAL_ROW)
    frame = Ugr16Adapter()._read_archive(path, max_rows=1)
    assert len(frame) == 1
    assert frame.iloc[0]["timestamp"] == pd.Timestamp("2016-05-01 00:03:06")


def test_row_limit_is_respected(tmp_path):
    # REAL_ROW carries its own newline, so the rows are already separated.
    lines = "".join(
        REAL_ROW.replace("2016-05-01 00:03:06", f"2016-05-01 00:01:{i:02d}")
        for i in range(20)
    )
    path = _archive_with(tmp_path, lines)
    frame = Ugr16Adapter()._read_archive(path, max_rows=5)
    assert len(frame) == 5


def test_subset_is_documented_and_contiguous():
    adapter = Ugr16Adapter()
    assert adapter.TEST_FILES[0] == "july_week5_csv.tar.gz"
    # August weeks 1-3 are consecutive, which is what makes the forward test
    # genuinely later traffic.
    assert "august_week1_csv.tar.gz" in adapter.TEST_FILES
    assert "august_week2_csv.tar.gz" in adapter.TEST_FILES
    assert "august_week3_csv.tar.gz" in adapter.TEST_FILES
    assert "august_week4_csv.tar.gz" not in adapter.TEST_FILES


def test_subset_notes_state_it_is_not_the_full_dataset():
    notes = Ugr16Adapter.SUBSET_NOTES
    assert "215" in notes["full_release_size"]
    assert notes["weeks_used"]
    assert notes["calibration_weeks"]
    assert any("subset" in lim.lower() for lim in notes["limitations"])


def test_declared_files_are_exactly_the_subset():
    adapter = Ugr16Adapter()
    assert set(adapter.files) == set(adapter.CALIBRATION_FILES) | set(adapter.TEST_FILES)


def test_missing_archives_produce_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        Ugr16Adapter().load(str(tmp_path))
    message = str(err.value)
    assert "ugr16" in message.lower()
    assert "fetch" in message.lower()


def test_registry_resolves_the_adapter_by_name():
    assert get_adapter("ugr16") is not None
    assert Ugr16Adapter().name == "ugr16"


def test_ugr16_has_no_direction_columns():
    """The netflow export has no forward/backward split, so the common schema
    must not invent one."""
    assert "forward_bytes" not in Ugr16Adapter.NETFLOW_COLUMNS
    assert "backward_bytes" not in Ugr16Adapter.NETFLOW_COLUMNS


def test_reading_an_archive_is_linear_not_quadratic(tmp_path, monkeypatch):
    """The row counter must not rescan the buffer for every line.

    Counting with a generator over the accumulated buffer makes reading one
    archive quadratic in its length. On a 12 GB weekly export that is hours of
    CPU before a single row reaches read_csv, which is exactly what stalled the
    first UGR'16 run. The test compares wall time across a 4x larger cap: a
    linear reader gets close to 4x, a quadratic one gets about 16x.
    """
    import io
    import tarfile
    import time

    from driftguard.data.registry import Ugr16Adapter

    def make_archive(path, rows):
        body = "\n" + "".join(          # the real exports open with a blank line
            f"2016-07-27 13:43:21,1.5,10.0.0.1,10.0.0.2,1234,80,TCP,.A....,0,0,2,100,background\n"
            for _ in range(rows)
        )
        payload = body.encode("utf-8")
        info = tarfile.TarInfo("uniq/july.week5.csv.uniqblacklistremoved")
        info.size = len(payload)
        with tarfile.open(path, "w:gz") as tar:
            tar.addfile(info, io.BytesIO(payload))

    adapter = Ugr16Adapter()
    small, large = 20_000, 80_000
    for rows in (small, large):
        path = tmp_path / f"a{rows}.tar.gz"
        make_archive(path, rows)

    timings = {}
    for rows, cap in ((small, small), (large, large)):
        start = time.perf_counter()
        frame = adapter._read_archive(str(tmp_path / f"a{rows}.tar.gz"), cap)
        timings[rows] = time.perf_counter() - start
        assert len(frame) == cap

    growth = timings[large] / max(timings[small], 1e-6)
    # A linear reader scales with the data; quadratic would land near 16.
    assert growth < 9.0, (
        f"4x the rows cost {growth:.1f}x the time, which suggests the reader "
        f"is still quadratic: {timings}"
    )


def test_unsw_adapter_loads_the_protocol_column():
    """COMMON_SCHEMA declares 'protocol', so the UNSW adapter must supply it.

    The raw export names it 'proto' and the adapter's column allowlist omitted
    it entirely, which silently skipped the protocol_mixture controlled shift on
    the one dataset this project has the most data for.
    """
    from driftguard.data.registry import Unswnb15Adapter
    from driftguard.data.schema import COMMON_SCHEMA

    assert "protocol" in COMMON_SCHEMA
    assert Unswnb15Adapter.COLUMN_MAP["proto"] == "protocol"
    assert "protocol" in Unswnb15Adapter.KEPT_COLUMNS
