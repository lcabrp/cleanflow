"""Analyze and apply DataFrame dtype optimizations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..logging_utils import log
from .utils import detect_engine


def analyze_optimization(
    df: Any,
    cat_threshold: float = 0.5,
    cat_max_unique: int = 1000,
    downcast_float: bool = True,
    require_float_notnull: bool = True,
    for_parquet: bool = False,
    verbose: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, list[str]]:
    """Recommend dtype changes without mutating the DataFrame."""
    engine = detect_engine(df)
    if engine == "pandas":
        return _analyze_pandas(df, cat_threshold, cat_max_unique, downcast_float, require_float_notnull, for_parquet, verbose, logger)
    if engine == "polars":
        return _analyze_polars(df, logger)
    raise ValueError(f"Unsupported engine: {engine}")


def _analyze_pandas(
    df: pd.DataFrame,
    cat_threshold: float,
    cat_max_unique: int,
    downcast_float: bool,
    require_float_notnull: bool,
    for_parquet: bool,
    verbose: bool,
    logger: logging.Logger | None,
) -> dict[str, list[str]]:
    type_map: dict[str, list[str]] = {}
    row_count = max(len(df), 1)

    for column in df.columns:
        series = df[column]
        dtype = series.dtype

        if pd.api.types.is_integer_dtype(dtype):
            min_value = series.min()
            max_value = series.max()
            has_na = series.isnull().any()
            rec = _smallest_int_dtype(int(min_value), int(max_value), nullable=has_na)
            type_map.setdefault(rec, []).append(column)
            if verbose:
                log(logger, f"[{column}] integer -> {rec}")

        elif pd.api.types.is_float_dtype(dtype):
            rec = "float64"
            if downcast_float and ((not require_float_notnull) or series.notnull().all()):
                min_value = series.min()
                max_value = series.max()
                if min_value >= np.finfo(np.float32).min and max_value <= np.finfo(np.float32).max:
                    rec = "float32"
            if rec != str(dtype):
                type_map.setdefault(rec, []).append(column)
            if verbose:
                log(logger, f"[{column}] float -> {rec}")

        elif _is_categorical_candidate(dtype) and not for_parquet:
            unique = series.nunique(dropna=False)
            if unique <= cat_max_unique or unique / row_count <= cat_threshold:
                type_map.setdefault("category", []).append(column)
                if verbose:
                    log(logger, f"[{column}] object unique={unique} -> category")

    return type_map


def _analyze_polars(df: Any, logger: logging.Logger | None) -> dict[str, list[str]]:
    import polars as pl

    numeric_types = {pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8, pl.Float64, pl.Float32}
    columns = [column for column in df.columns if df[column].dtype in numeric_types]
    log(logger, f"Polars recommendation: shrink_dtype for {len(columns)} columns")
    return {"shrink_dtype": columns} if columns else {}


def apply_optimization(df: Any, type_map: dict[str, list[str]], logger: logging.Logger | None = None) -> Any:
    """Apply dtype recommendations returned by analyze_optimization()."""
    engine = detect_engine(df)
    if engine == "polars":
        if "shrink_dtype" in type_map:
            return df.select(df[col].shrink_dtype().alias(col) if col in type_map["shrink_dtype"] else df[col] for col in df.columns)
        return df

    out = df.copy()
    for dtype, columns in type_map.items():
        for column in columns:
            if column not in out.columns:
                log(logger, f"Column not found during optimization: {column}")
                continue
            try:
                out[column] = out[column].astype(dtype)
            except Exception as exc:
                log(logger, f"Could not convert {column} to {dtype}: {exc}")
    return out


def optimization_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    type_map: dict[str, list[str]],
    show_per_column: bool = True,
) -> tuple[float, float, float]:
    """Print and return before/after memory usage metrics."""
    before = df_before.memory_usage(deep=True)
    after = df_after.memory_usage(deep=True)
    total_before = before.sum()
    total_after = after.sum()
    reduction = (total_before - total_after) / total_before * 100 if total_before else 0

    print("=" * 70)
    print("CleanFlow Memory Optimization Report")
    print("=" * 70)
    if show_per_column:
        print(f"{'Column':<25} {'Before':>12} {'After':>12} {'Dtype Before':<15} {'Dtype After':<15}")
        print("-" * 85)
        for column in df_before.columns:
            print(
                f"{column:<25} {before[column]:>12,} {after[column]:>12,} "
                f"{str(df_before[column].dtype):<15} {str(df_after[column].dtype):<15}"
            )
    print(f"Total before: {total_before / 1024**2:.2f} MB")
    print(f"Total after:  {total_after / 1024**2:.2f} MB")
    print(f"Reduction:    {reduction:.1f}%")
    if type_map:
        print("Conversions:")
        for dtype, columns in type_map.items():
            print(f"  {dtype}: {len(columns)} columns")
    return total_before / 1024**2, total_after / 1024**2, reduction


def _smallest_int_dtype(min_value: int, max_value: int, nullable: bool = False) -> str:
    if min_value >= 0:
        candidates = ("UInt8", "UInt16", "UInt32", "UInt64") if nullable else ("uint8", "uint16", "uint32", "uint64")
        for dtype in candidates:
            info = np.iinfo(dtype[1:].lower() if nullable else dtype)
            if max_value <= info.max:
                return dtype

    candidates = ("Int8", "Int16", "Int32", "Int64") if nullable else ("int8", "int16", "int32", "int64")
    for dtype in candidates:
        info = np.iinfo(dtype[1:].lower() if nullable else dtype)
        if min_value >= info.min and max_value <= info.max:
            return dtype
    return "Int64" if nullable else "int64"


def _is_categorical_candidate(dtype: pd.api.extensions.ExtensionDtype | np.dtype) -> bool:
    """Return True for pandas object or string columns.

    Pandas 3 may infer text columns as ``str`` instead of ``object``. Keeping
    the optimizer aware of both dtypes prevents version-specific behavior.
    """
    return pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
