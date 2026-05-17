"""
CleanFlow — Modular Python library for automated data cleaning.

All transformers follow the analyze → transform → report pattern.
"""

from .pipeline import AutomatedCleaner
from .quality import check_quality, quality_score, detect_suspicious, add_missing_indicators
from .transformers import DataStandardizer, MissingValueHandler
from .duplicates import DuplicateHandler
from .outliers import OutlierHandler
from .text import TextCleaner
from .categories import CategoryStandardizer
from .cleaning import drop_na, fill_na
from .io import load_dataset, estimate_csv_memory
from .profiling import dataset_overview, profile_dataframe
from .optimization import (
    optimize_dataset,
    analyze_optimization,
    apply_optimization,
    optimization_report,
    convert_to_parquet,
    convert_to_parquet_optimized,
)
from .features import (
    FeatureScaler,
    NumericalTransformer,
    DateFeatureExtractor,
    MissingIndicator,
    FeatureSelector
)
from .base import BaseTransformer, FitTransformer

__version__ = "0.3.0"
__all__ = [
    # Pipeline
    "AutomatedCleaner",
    # Quality
    "check_quality",
    "quality_score",
    "detect_suspicious",
    "add_missing_indicators",
    # Transformers
    "DataStandardizer",
    "MissingValueHandler",
    "DuplicateHandler",
    "OutlierHandler",
    "TextCleaner",
    "CategoryStandardizer",
    # File IO / profiling / optimization
    "load_dataset",
    "estimate_csv_memory",
    "profile_dataframe",
    "dataset_overview",
    "drop_na",
    "fill_na",
    "optimize_dataset",
    "analyze_optimization",
    "apply_optimization",
    "optimization_report",
    "convert_to_parquet",
    "convert_to_parquet_optimized",
    # Feature Engineering
    "FeatureScaler",
    "NumericalTransformer",
    "DateFeatureExtractor",
    "MissingIndicator",
    "FeatureSelector",
    # Base Transformers
    "BaseTransformer",
    "FitTransformer",
]
