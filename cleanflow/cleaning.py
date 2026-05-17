"""Simple compatibility cleaning helpers from the former data-optimizer repo."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def drop_na(df: pd.DataFrame, axis: int = 0, how: str = "any", subset: Iterable[str] | None = None) -> pd.DataFrame:
    """Drop missing values using pandas semantics."""
    return df.dropna(axis=axis, how=how, subset=subset)


def fill_na(df: pd.DataFrame, numeric_fill: float | int = 0, categorical_fill: str = "unknown") -> pd.DataFrame:
    """Fill missing values by broad dtype family."""
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].fillna(numeric_fill)
        else:
            out[column] = out[column].fillna(categorical_fill)
    return out

