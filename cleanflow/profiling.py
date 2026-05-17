"""Lightweight dataset profiling utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def _efficient_profile(df: pd.DataFrame, deep: bool = True, include_numeric_summary: bool = True) -> tuple[dict[str, Any], pd.DataFrame]:
    dtypes = df.dtypes.astype(str)
    null_counts = df.isna().sum()
    non_null_counts = df.notna().sum()
    unique_counts = df.nunique(dropna=False)
    memory_per_col = df.memory_usage(index=False, deep=deep) / 1024**2
    row_count = max(len(df), 1)

    profile = {
        "rows": len(df),
        "columns": df.shape[1],
        "memory_mb": float(memory_per_col.sum()),
        "dtypes": dtypes.to_dict(),
        "missing_values": null_counts.to_dict(),
    }
    if include_numeric_summary:
        profile["numeric_summary"] = df.describe(include="number").to_dict()

    overview = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": dtypes,
            "non_nulls": non_null_counts,
            "nulls": null_counts,
            "missing_pct": (null_counts / row_count * 100).round(2),
            "unique_count": unique_counts,
            "memory_mb": memory_per_col,
        }
    ).sort_values("memory_mb", ascending=False)
    return profile, overview


def profile_dataframe(df: pd.DataFrame, deep: bool = True, include_numeric_summary: bool = True) -> dict[str, Any]:
    """Return dataset-level profile metadata."""
    profile, _ = _efficient_profile(df, deep=deep, include_numeric_summary=include_numeric_summary)
    return profile


def dataset_overview(df: pd.DataFrame, deep: bool = True) -> pd.DataFrame:
    """Return a compact per-column profile table."""
    _, overview = _efficient_profile(df, deep=deep, include_numeric_summary=False)
    return overview


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for quick file profiling."""
    from .io import load_dataset

    parser = argparse.ArgumentParser(description="Profile a CSV or Parquet dataset.")
    parser.add_argument("path", help="Path to CSV or Parquet file.")
    parser.add_argument("--fast", action="store_true", help="Skip deep memory scan and numeric summary.")
    parser.add_argument("--sample", type=int, default=None, help="Load only the first N rows.")
    parser.add_argument("--engine", choices=["pandas", "polars"], default="pandas", help="Loading engine.")
    args = parser.parse_args(argv)

    path = Path(args.path)
    if args.sample and path.suffix.lower() == ".csv":
        df = pd.read_csv(path, nrows=args.sample, low_memory=False)
    else:
        loaded = load_dataset(path, engine=args.engine)
        df = loaded.to_pandas() if hasattr(loaded, "to_pandas") else loaded

    profile, overview = _efficient_profile(df, deep=not args.fast, include_numeric_summary=not args.fast)
    print("=" * 60)
    print("Dataset Profile")
    print("=" * 60)
    print(f"Rows: {profile['rows']:,}")
    print(f"Columns: {profile['columns']:,}")
    print(f"Memory: {profile['memory_mb']:.2f} MB")
    print(f"Missing values: {sum(profile['missing_values'].values()):,}")
    print("\nColumn overview:")
    print(overview.head(20).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
