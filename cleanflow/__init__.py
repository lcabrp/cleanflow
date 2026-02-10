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
from .features import (
    FeatureScaler,
    NumericalTransformer,
    DateFeatureExtractor,
    MissingIndicator,
    FeatureSelector
)
from .base import BaseTransformer, FitTransformer

__version__ = "0.2.0"
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
    # Feature Engineering
    "FeatureScaler",
    "DateFeatureExtractor",
    "MissingIndicator",
    # Base Transformers
    "BaseTransformer",
    "FitTransformer",
]