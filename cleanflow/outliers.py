"""
Outlier detection and treatment.

Detects outliers using multiple statistical methods and applies
configurable treatment strategies. Adapted from Bala Priya C's
OutlierDetector and KDNuggets technique #4.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Literal, Tuple, Any
from dataclasses import dataclass

from .base import BaseTransformer


@dataclass
class OutlierReport:
    """Report for a single column's outlier detection and treatment."""
    column: str
    method: str
    lower_bound: float
    upper_bound: float
    outlier_count: int
    outlier_pct: float
    treatment: str


class OutlierHandler(BaseTransformer):
    """Detects and treats outliers in numeric columns.

    Detection methods:
        - IQR (Interquartile Range) — robust to non-normal distributions
        - Z-score — assumes normal distribution
        - Modified Z-score (MAD-based) — robust alternative to Z-score
        - Percentile — clip at user-defined percentile bounds

    Treatment strategies:
        - remove: drop outlier rows
        - cap: clip values to bounds
        - winsorize: same as cap (KDNuggets terminology)
        - flag: add boolean indicator columns
        - impute_mean / impute_median: replace outliers with column mean/median

    Example::

        handler = OutlierHandler()
        handler.detect(df, method="iqr", threshold=1.5)
        cleaned_df = handler.transform(df, strategy="cap")
        print(handler.get_report())
    """

    def __init__(self):
        self.outlier_masks: Dict[str, pd.Series] = {}
        self.bounds: Dict[str, Tuple[float, float]] = {}
        self._reports: List[OutlierReport] = []

    # -- detection -----------------------------------------------------------

    def detect(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: Literal["iqr", "zscore", "modified_zscore", "percentile"] = "iqr",
        threshold: float = 1.5,
        percentile_range: Tuple[float, float] = (0.01, 0.99),
    ) -> "OutlierHandler":
        """Detect outliers in numeric columns.

        Args:
            df: Input DataFrame.
            columns: Columns to check. None = all numeric columns.
            method: Detection method.
            threshold: Sensitivity — IQR multiplier or Z-score cutoff.
            percentile_range: Lower/upper percentile for 'percentile' method.

        Returns:
            self (for chaining).
        """
        self.outlier_masks = {}
        self.bounds = {}

        if columns is None:
            columns = df.select_dtypes(include="number").columns.tolist()

        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue

            series = df[col].dropna()
            if len(series) < 3:
                continue

            if method == "iqr":
                lower, upper, mask = self._detect_iqr(df[col], threshold)
            elif method == "zscore":
                lower, upper, mask = self._detect_zscore(df[col], threshold)
            elif method == "modified_zscore":
                lower, upper, mask = self._detect_modified_zscore(df[col], threshold)
            elif method == "percentile":
                lower, upper, mask = self._detect_percentile(df[col], percentile_range)
            else:
                raise ValueError(f"Unknown detection method: {method}")

            self.outlier_masks[col] = mask
            self.bounds[col] = (lower, upper)

        return self

    # -- treatment -----------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame,
        strategy: Literal["remove", "cap", "winsorize", "flag", "impute_mean", "impute_median"] = "cap",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Treat detected outliers.

        Args:
            df: Input DataFrame (same one used for detect).
            strategy: Treatment strategy to apply.
            columns: Columns to treat. None = all detected columns.

        Returns:
            Treated DataFrame.
        """
        df = df.copy()
        self._reports = []
        treat_cols = columns or list(self.outlier_masks.keys())

        for col in treat_cols:
            if col not in self.outlier_masks:
                continue

            mask = self.outlier_masks[col]
            lower, upper = self.bounds[col]
            outlier_count = int(mask.sum())
            outlier_pct = round(outlier_count / len(df) * 100, 2) if len(df) > 0 else 0

            if strategy in ("cap", "winsorize"):
                df[col] = df[col].clip(lower=lower, upper=upper)
            elif strategy == "remove":
                df = df[~mask]
            elif strategy == "flag":
                df[f"{col}_is_outlier"] = mask.astype(int)
            elif strategy == "impute_mean":
                col_mean = df.loc[~mask, col].mean()
                df.loc[mask, col] = col_mean
            elif strategy == "impute_median":
                col_median = df.loc[~mask, col].median()
                df.loc[mask, col] = col_median

            self._reports.append(OutlierReport(
                column=col,
                method=list(self.outlier_masks.keys())[0] if self.outlier_masks else "unknown",
                lower_bound=lower,
                upper_bound=upper,
                outlier_count=outlier_count,
                outlier_pct=outlier_pct,
                treatment=strategy,
            ))

        if strategy == "remove":
            df = df.reset_index(drop=True)

        return df

    def get_report(self) -> pd.DataFrame:
        """Return outlier treatment report."""
        if not self._reports:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "column": r.column,
                "lower_bound": round(r.lower_bound, 4),
                "upper_bound": round(r.upper_bound, 4),
                "outlier_count": r.outlier_count,
                "outlier_pct": r.outlier_pct,
                "treatment": r.treatment,
            }
            for r in self._reports
        ])

    # -- detection methods ---------------------------------------------------

    @staticmethod
    def _detect_iqr(series: pd.Series, threshold: float):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (series < lower) | (series > upper)
        return lower, upper, mask

    @staticmethod
    def _detect_zscore(series: pd.Series, threshold: float):
        mean = series.mean()
        std = series.std()
        if std == 0:
            return mean, mean, pd.Series(False, index=series.index)
        z = (series - mean) / std
        mask = z.abs() > threshold
        lower = mean - threshold * std
        upper = mean + threshold * std
        return lower, upper, mask

    @staticmethod
    def _detect_modified_zscore(series: pd.Series, threshold: float):
        median = series.median()
        mad = np.median(np.abs(series.dropna() - median))
        if mad == 0:
            return median, median, pd.Series(False, index=series.index)
        modified_z = 0.6745 * (series - median) / mad
        mask = modified_z.abs() > threshold
        lower = median - threshold * mad / 0.6745
        upper = median + threshold * mad / 0.6745
        return lower, upper, mask

    @staticmethod
    def _detect_percentile(series: pd.Series, pct_range: Tuple[float, float]):
        lower = series.quantile(pct_range[0])
        upper = series.quantile(pct_range[1])
        mask = (series < lower) | (series > upper)
        return lower, upper, mask
