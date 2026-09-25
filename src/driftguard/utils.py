"""Small shared helpers.

Configuration values arrive from YAML, where a duration may be written as "6h"
or as a bare number. A bare number is read as seconds: the default unit for
every duration in this project.
"""

from typing import Any, Union

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
