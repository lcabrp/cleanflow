"""Pandas-specific memory optimization."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ...logging_utils import log


def optimize(
    df: pd.DataFrame,
    logger: logging.Logger | None = None,
    for_parquet: bool = False,
    cat_threshold: float = 0.5,
    cat_max_unique: int = 1000,
    downcast_float: bool = True,
    require_float_notnull: bool = True,
) -> pd.DataFrame:
    """Reduce pandas memory usage while preserving values.

    The logic is intentionally conservative: integer ranges are shrunk, floats
    are downcast only when allowed, and low-cardinality object columns become
    categoricals unless the caller is preparing Parquet output.
    """
    out = df.copy()
    start_mem = out.memory_usage(deep=True).sum() / 1024**2

    for column in out.columns:
        series = out[column]
        dtype = series.dtype

        if pd.api.types.is_integer_dtype(dtype):
            out[column] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(dtype) and downcast_float:
            if (not require_float_notnull) or series.notnull().all():
                out[column] = pd.to_numeric(series, downcast="float")
        elif not for_parquet and _is_categorical_candidate(dtype):
            unique = series.nunique(dropna=False)
            total = max(len(series), 1)
            if unique <= cat_max_unique or unique / total < cat_threshold:
                out[column] = series.astype("category")

    end_mem = out.memory_usage(deep=True).sum() / 1024**2
    log(logger, f"Pandas memory: {start_mem:.2f} -> {end_mem:.2f} MB")
    return out


def smallest_int_dtype(min_value: int, max_value: int, nullable: bool = False) -> str:
    """Return the smallest integer dtype name that can hold a value range."""
    if min_value >= 0:
        for dtype in ("UInt8", "UInt16", "UInt32", "UInt64") if nullable else ("uint8", "uint16", "uint32", "uint64"):
            info = np.iinfo(dtype.lower() if not nullable else dtype[1:].lower())
            if max_value <= info.max:
                return dtype
    for dtype in ("Int8", "Int16", "Int32", "Int64") if nullable else ("int8", "int16", "int32", "int64"):
        info = np.iinfo(dtype[1:].lower() if nullable else dtype)
        if min_value >= info.min and max_value <= info.max:
            return dtype
    return "Int64" if nullable else "int64"


def _is_categorical_candidate(dtype: pd.api.extensions.ExtensionDtype | np.dtype) -> bool:
    """Return True for text-like columns worth considering for category.

    Pandas 3 can infer plain text as the dedicated ``str`` dtype instead of
    ``object``. Treating both the same keeps optimization behavior stable
    across pandas versions.
    """
    return pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
