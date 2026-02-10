"""
Core transformers for data type standardization and missing value handling.

Adapted from Bala Priya C's modular cleaning scripts and the KDNuggets
"10 Essential Data Cleaning Techniques" article.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Literal, Optional, Any
from dataclasses import dataclass

from .base import BaseTransformer, FitTransformer


# ---------------------------------------------------------------------------
# Data Type Standardizer
# ---------------------------------------------------------------------------

@dataclass
class ConversionResult:
    """Record of a single column type conversion."""
    column: str
    original_dtype: str
    new_dtype: str
    success_count: int
    failed_count: int
    failed_values: List[Any]


class DataStandardizer(BaseTransformer):
    """Detects and converts columns to their intended data types.

    Handles: numeric strings (with currency/commas), dates (multiple formats),
    booleans, and percentages. Inspired by DataTypeFixer from Bala Priya C
    tutorials and KDNuggets technique #5.

    Example::

        standardizer = DataStandardizer()
        cleaned_df = standardizer.transform(df)
        print(standardizer.get_report())
    """

    DATE_FORMATS = [
        "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
        "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S",
        "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y",
    ]

    BOOL_MAP = {
        "true": True, "false": False,
        "yes": True, "no": False,
        "1": True, "0": False,
        "t": True, "f": False,
        "y": True, "n": False,
    }

    def __init__(self, auto_detect: bool = True, coerce_errors: bool = True):
        self.auto_detect = auto_detect
        self.coerce_errors = coerce_errors
        self._results: List[ConversionResult] = []

    # -- public API ----------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-detect and convert column types."""
        df = df.copy()
        self._results = []

        for col in df.columns:
            if not pd.api.types.is_string_dtype(df[col]):
                continue  # only attempt conversion on string/object columns

            original_dtype = str(df[col].dtype)
            converted, result = self._try_convert(df[col], col)
            if result is not None:
                df[col] = converted
                result.original_dtype = original_dtype
                self._results.append(result)

        return df

    def get_report(self) -> pd.DataFrame:
        """Return conversion report."""
        if not self._results:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "column": r.column,
                "original_dtype": r.original_dtype,
                "new_dtype": r.new_dtype,
                "success_count": r.success_count,
                "failed_count": r.failed_count,
            }
            for r in self._results
        ])

    # -- internal helpers ----------------------------------------------------

    def _try_convert(self, series: pd.Series, col_name: str):
        """Try converting a series through multiple type detections."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return series, None

        sample = non_null.head(50).astype(str).str.strip().str.lower()

        # 1. Boolean detection
        if self._looks_boolean(sample):
            return self._to_boolean(series, col_name)

        # 2. Percentage detection (e.g., "45%", "12.5%")
        if self._looks_percentage(sample):
            return self._to_percentage(series, col_name)

        # 3. Numeric detection (handles $, commas)
        if self._looks_numeric(sample):
            return self._to_numeric(series, col_name)

        # 4. Date detection
        if self._looks_date(sample):
            return self._to_datetime(series, col_name)

        return series, None

    def _looks_boolean(self, sample: pd.Series) -> bool:
        return sample.isin(self.BOOL_MAP.keys()).mean() > 0.8

    def _looks_percentage(self, sample: pd.Series) -> bool:
        return sample.str.match(r"^-?\d+\.?\d*\s*%$").mean() > 0.5

    def _looks_numeric(self, sample: pd.Series) -> bool:
        cleaned = sample.str.replace(r"[$,€£¥]", "", regex=True).str.strip()
        try:
            pd.to_numeric(cleaned, errors="raise")
            return True
        except (ValueError, TypeError):
            return cleaned.apply(self._is_numeric_str).mean() > 0.5

    @staticmethod
    def _is_numeric_str(val: str) -> bool:
        try:
            float(val.replace(",", ""))
            return True
        except (ValueError, AttributeError):
            return False

    def _looks_date(self, sample: pd.Series) -> bool:
        parsed = 0
        for val in sample.head(10):
            if self._try_parse_date(val) is not None:
                parsed += 1
        return parsed / min(len(sample), 10) > 0.5

    def _try_parse_date(self, value: str):
        from datetime import datetime
        for fmt in self.DATE_FORMATS:
            try:
                return datetime.strptime(value.strip(), fmt)
            except (ValueError, TypeError):
                continue
        return None

    def _to_boolean(self, series: pd.Series, col_name: str):
        mapping = self.BOOL_MAP
        converted = series.map(
            lambda x: mapping.get(str(x).strip().lower()) if pd.notna(x) else np.nan
        )
        success = converted.notna().sum() - series.isna().sum()
        failed = series.notna().sum() - success
        result = ConversionResult(
            column=col_name, original_dtype="", new_dtype="boolean",
            success_count=int(max(success, 0)), failed_count=int(max(failed, 0)),
            failed_values=[],
        )
        return converted.astype("boolean"), result

    def _to_percentage(self, series: pd.Series, col_name: str):
        def parse_pct(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip().replace("%", "")
            try:
                return float(s) / 100.0
            except ValueError:
                return np.nan

        converted = series.apply(parse_pct)
        failed_mask = series.notna() & converted.isna()
        result = ConversionResult(
            column=col_name, original_dtype="", new_dtype="float64",
            success_count=int(converted.notna().sum()),
            failed_count=int(failed_mask.sum()),
            failed_values=series[failed_mask].head(5).tolist(),
        )
        return converted, result

    def _to_numeric(self, series: pd.Series, col_name: str):
        cleaned = series.astype(str).str.replace(r"[$,€£¥]", "", regex=True).str.strip()
        errors = "coerce" if self.coerce_errors else "raise"
        converted = pd.to_numeric(cleaned, errors=errors)

        # Restore NaN from original
        converted[series.isna()] = np.nan

        failed_mask = series.notna() & converted.isna()
        result = ConversionResult(
            column=col_name, original_dtype="", new_dtype=str(converted.dtype),
            success_count=int(converted.notna().sum()),
            failed_count=int(failed_mask.sum()),
            failed_values=series[failed_mask].head(5).tolist(),
        )
        return converted, result

    def _to_datetime(self, series: pd.Series, col_name: str):
        errors = "coerce" if self.coerce_errors else "raise"
        converted = pd.to_datetime(series, errors=errors)

        failed_mask = series.notna() & converted.isna()
        result = ConversionResult(
            column=col_name, original_dtype="", new_dtype="datetime64[ns]",
            success_count=int(converted.notna().sum()),
            failed_count=int(failed_mask.sum()),
            failed_values=series[failed_mask].head(5).tolist(),
        )
        return converted, result


# ---------------------------------------------------------------------------
# Missing Value Handler
# ---------------------------------------------------------------------------

@dataclass
class MissingReport:
    """Record of a single column's missing-value imputation."""
    column: str
    missing_count: int
    missing_pct: float
    strategy_used: str
    fill_value: Any


class MissingValueHandler(FitTransformer):
    """Analyzes and handles missing values with intelligent imputation.

    Supports: mean, median, mode, zero, unknown, interpolate, ffill, bfill,
    drop_rows, drop_column, and flag_missing strategies. Auto-recommends
    strategies based on data type and distribution skewness.

    Adapted from Bala Priya C's MissingValueHandler and KDNuggets technique #1.

    Example::

        handler = MissingValueHandler(default_numeric="median")
        cleaned_df = handler.fit_transform(df)
        print(handler.get_report())
    """

    def __init__(
        self,
        strategies: Optional[Dict[str, str]] = None,
        default_numeric: Literal["mean", "median", "zero"] = "median",
        default_categorical: Literal["mode", "unknown"] = "mode",
        drop_threshold: float = 0.7,
    ):
        self.strategies = strategies or {}
        self.default_numeric = default_numeric
        self.default_categorical = default_categorical
        self.drop_threshold = drop_threshold
        self._report: List[MissingReport] = []
        self._fitted = False

    # -- analysis (pre-fit) --------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze missing-value patterns and recommend strategies."""
        analysis = []
        for col in df.columns:
            missing = df[col].isna().sum()
            total = len(df)
            analysis.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": missing,
                "missing_pct": round(missing / total * 100, 2) if total > 0 else 0,
                "unique_values": df[col].nunique(),
                "recommended_strategy": self._recommend_strategy(df, col),
            })
        return pd.DataFrame(analysis)

    # -- fit / transform -----------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "MissingValueHandler":
        """Learn imputation values from the DataFrame."""
        self._fit_values: Dict[str, Any] = {}
        self._fit_strategies: Dict[str, str] = {}

        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                continue

            strategy = self._pick_strategy(df, col)
            self._fit_strategies[col] = strategy
            self._fit_values[col] = self._compute_fill_value(df, col, strategy)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation using learned values."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform(). Use fit_transform() for convenience.")

        df = df.copy()
        self._report = []

        for col, strategy in self._fit_strategies.items():
            if col not in df.columns:
                continue

            missing_count = df[col].isna().sum()
            if missing_count == 0:
                continue

            fill_value = self._apply_strategy(df, col, strategy)

            self._report.append(MissingReport(
                column=col,
                missing_count=missing_count,
                missing_pct=round(missing_count / len(df) * 100, 2),
                strategy_used=strategy,
                fill_value=fill_value,
            ))

        return df

    def get_report(self) -> pd.DataFrame:
        """Return imputation report."""
        if not self._report:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "column": r.column,
                "missing_count": r.missing_count,
                "missing_pct": r.missing_pct,
                "strategy": r.strategy_used,
                "fill_value": r.fill_value,
            }
            for r in self._report
        ])

    # -- internal helpers ----------------------------------------------------

    def _recommend_strategy(self, df: pd.DataFrame, col: str) -> str:
        """Recommend handling strategy based on column characteristics."""
        dtype = df[col].dtype
        missing_pct = df[col].isna().mean()

        if missing_pct > 0.7:
            return "drop_column"
        elif missing_pct > 0.5:
            return "flag_missing"
        elif pd.api.types.is_bool_dtype(dtype):
            # Boolean columns should use mode, not mean/median
            return "mode"
        elif pd.api.types.is_numeric_dtype(dtype):
            n_valid = df[col].notna().sum()
            if n_valid > 2:
                skew = df[col].skew()
                return "median" if abs(skew) > 1 else "mean"
            return "median"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "interpolate"
        else:
            return "mode"

    def _pick_strategy(self, df: pd.DataFrame, col: str) -> str:
        """Pick strategy: user-specified > threshold-based > auto."""
        if col in self.strategies:
            return self.strategies[col]

        missing_pct = df[col].isna().mean()
        if missing_pct > self.drop_threshold:
            return "drop_column"

        # Check for boolean BEFORE numeric (since is_numeric_dtype returns True for bool)
        if pd.api.types.is_bool_dtype(df[col]):
            return "mode"
        elif pd.api.types.is_numeric_dtype(df[col]):
            return self.default_numeric
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            return "interpolate"
        else:
            return self.default_categorical

    def _compute_fill_value(self, df: pd.DataFrame, col: str, strategy: str) -> Any:
        """Compute the fill value during fit."""
        if strategy == "mean":
            return df[col].mean()
        elif strategy == "median":
            return df[col].median()
        elif strategy == "mode":
            modes = df[col].mode()
            if modes.empty:
                return "UNKNOWN"
            mode_value = modes.iloc[0]
            
            # Handle boolean columns: mode() returns numpy.bool
            # We need to convert to Python bool for pandas BooleanDtype compatibility
            if pd.api.types.is_bool_dtype(df[col]):
                # Convert to Python bool (not numpy bool_)
                return bool(int(mode_value)) if not pd.isna(mode_value) else pd.NA
            return mode_value
        elif strategy == "zero":
            return 0
        elif strategy == "unknown":
            return "UNKNOWN"
        else:
            return None  # strategies like drop/interpolate/ffill don't need a pre-computed value

    def _apply_strategy(self, df: pd.DataFrame, col: str, strategy: str) -> Any:
        """Apply the imputation strategy in-place on df, return the fill value used."""
        fill_value = self._fit_values.get(col)

        if strategy == "drop_column":
            df.drop(columns=[col], inplace=True)
            return "DROPPED"
        elif strategy == "drop_rows":
            df.dropna(subset=[col], inplace=True)
            return "ROWS_DROPPED"
        elif strategy in ("mean", "median", "mode", "zero", "unknown"):
            # Special handling for boolean columns
            if pd.api.types.is_bool_dtype(df[col]):
                # For boolean columns, ensure fill_value is compatible
                if fill_value not in (True, False, pd.NA, None):
                    # Convert numeric to bool
                    fill_value = bool(int(fill_value)) if not pd.isna(fill_value) else pd.NA
                df[col] = df[col].fillna(fill_value)
            else:
                df[col] = df[col].fillna(fill_value)
            return fill_value
        elif strategy == "interpolate":
            df[col] = df[col].interpolate(method="linear")
            return "INTERPOLATED"
        elif strategy == "ffill":
            df[col] = df[col].ffill()
            return "FORWARD_FILL"
        elif strategy == "bfill":
            df[col] = df[col].bfill()
            return "BACKWARD_FILL"
        elif strategy == "flag_missing":
            df[f"{col}_is_missing"] = df[col].isna().astype(int)
            return "FLAGGED"

        return fill_value