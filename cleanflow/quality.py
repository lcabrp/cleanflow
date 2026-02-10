"""
Data quality analysis and reporting.

Provides comprehensive quality checks including completeness scoring,
suspicious value detection, and feature engineering helpers.
Inspired by automate_5_steps.py and KDNuggets technique #9.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cleanflow")


def check_quality(df: pd.DataFrame) -> dict:
    """Analyze a DataFrame and return a comprehensive health report.

    Returns a dict with:
        - total_rows, total_columns
        - missing_values (per column)
        - duplicate_rows
        - memory_usage_mb
        - completeness_percentage (overall)
        - column_completeness (per column)
        - dtypes summary
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    completeness = round(((total_cells - total_missing) / total_cells) * 100, 2) if total_cells > 0 else 0

    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing_values": int(total_missing),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
        "completeness_percentage": completeness,
        "column_completeness": {
            col: round((1 - df[col].isna().mean()) * 100, 2)
            for col in df.columns
        },
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    logger.info(f"Quality Check: {completeness}% complete, {report['duplicate_rows']} duplicates.")
    return report


def quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """Score each row by its completeness.

    Adds a 'quality_score' (0-10) and 'quality_category' (Poor/Average/Good).
    Inspired by KDNuggets technique #9: Feature Engineering from Dirty Data.

    Returns the DataFrame with two new columns appended.
    """
    df = df.copy()
    n_cols = len(df.columns)
    if n_cols == 0:
        df["quality_score"] = 0
        df["quality_category"] = "Poor"
        return df

    df["quality_score"] = df.notna().sum(axis=1) / n_cols * 10
    df["quality_category"] = pd.cut(
        df["quality_score"],
        bins=[0, 4, 7, 10],
        labels=["Poor", "Average", "Good"],
        include_lowest=True,
    )
    return df


def detect_suspicious(
    df: pd.DataFrame,
    round_modulus: int = 10000,
    age_range: tuple = (0, 120),
    rating_range: tuple = (1, 5),
) -> pd.DataFrame:
    """Flag suspicious values in numeric columns.

    Looks for:
        - Perfectly round values (e.g., income divisible by round_modulus)
        - Values outside expected ranges for age-like and rating-like columns

    Returns a DataFrame with boolean flag columns (e.g., 'income_suspiciously_round').
    """
    flags = pd.DataFrame(index=df.index)
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        col_lower = col.lower()

        # Flag suspiciously round values
        if any(keyword in col_lower for keyword in ("income", "salary", "price", "amount", "revenue")):
            flags[f"{col}_suspiciously_round"] = (df[col] % round_modulus == 0).astype(int)

        # Flag out-of-range ages
        if "age" in col_lower:
            lo, hi = age_range
            flags[f"{col}_out_of_range"] = ((df[col] < lo) | (df[col] > hi)).astype(int)

        # Flag out-of-range ratings
        if "rating" in col_lower:
            lo, hi = rating_range
            flags[f"{col}_out_of_range"] = ((df[col] < lo) | (df[col] > hi)).astype(int)

    return flags


def add_missing_indicators(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Add binary indicator columns for missing values.

    KDNuggets technique #9: Sometimes the *pattern* of missingness is informative.

    Args:
        df: Input DataFrame.
        columns: Columns to create indicators for. Defaults to all columns.

    Returns:
        DataFrame with '{col}_is_missing' columns appended.
    """
    df = df.copy()
    cols = columns or list(df.columns)
    for col in cols:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isna().astype(int)
    return df