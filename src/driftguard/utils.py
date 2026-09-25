"""Small shared helpers.

Configuration values arrive from YAML, where a duration may be written as "6h"
or as a bare number. A bare number is read as seconds: the default unit for
every duration in this project.
"""

import os
from typing import Any, Dict, Union

import pandas as pd


def as_timedelta(value: Any) -> pd.Timedelta:
    """Interpret a config value as a duration, defaulting bare numbers to seconds."""
    if isinstance(value, pd.Timedelta):
        return value
    if isinstance(value, (int, float)):
        return pd.Timedelta(seconds=float(value))
    text = str(value).strip()
    try:
        return pd.Timedelta(text)
    except ValueError as exc:
        raise ValueError(
            f"cannot read {value!r} as a duration; write it as '6h', '30m' or a number of seconds"
        ) from exc


def iso_or_none(value: Union[pd.Timestamp, None]) -> Union[str, None]:
    return None if value is None else pd.Timestamp(value).isoformat()


def resolve_raw_dir(config: Dict) -> str:
    """Where the raw dataset lives, honouring the DRIFTGUARD_RAW_DIR override.

    The committed configs point at ``data/raw`` so a checkout is self-contained,
    but the full datasets are far too large to live in a repository. One env
    var moves the location for every entry point, which is why this lives here
    rather than being re-implemented in each command.
    """
    return os.environ.get("DRIFTGUARD_RAW_DIR") or config["dataset"]["raw_dir"]
