"""Shared optimization helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def detect_engine(df: Any) -> str:
    """Detect the DataFrame engine from an object instance."""
    if "dask" in str(type(df)).lower():
        return "dask"

    try:
        import polars as pl
    except ImportError:
        pl = None
    if pl is not None and isinstance(df, pl.DataFrame):
        return "polars"

    if isinstance(df, pd.DataFrame):
        return "pandas"

    raise TypeError(f"Unsupported dataframe type: {type(df)!r}")

