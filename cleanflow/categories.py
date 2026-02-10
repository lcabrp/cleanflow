"""
Category standardization and grouping.

Handles inconsistent categorical values, maps synonyms to canonical forms,
and groups rare categories. Inspired by KDNuggets technique #6.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .base import BaseTransformer


@dataclass
class CategoryReport:
    """Report for a single column's category standardization."""
    column: str
    action: str
    original_unique: int
    final_unique: int
    rows_modified: int


class CategoryStandardizer(BaseTransformer):
    """Standardizes and consolidates categorical values.

    Features:
        - Value mapping: map inconsistent spellings to canonical forms
        - Category grouping: merge related categories into broader groups
        - Rare category handling: group infrequent values into 'Other'
        - Case normalization

    Example::

        standardizer = CategoryStandardizer()
        standardizer.add_mapping("country", {
            "US": "USA", "U.S.A.": "USA", "United States": "USA"
        })
        standardizer.add_mapping("gender", {
            "M": "Male", "m": "Male", "F": "Female", "f": "Female"
        })
        cleaned_df = standardizer.transform(df)
        print(standardizer.get_report())
    """

    def __init__(
        self,
        mappings: Optional[Dict[str, Dict[str, str]]] = None,
        groupings: Optional[Dict[str, Dict[str, str]]] = None,
        rare_threshold: Optional[int] = None,
        rare_label: str = "Other",
        normalize_case: bool = False,
    ):
        """
        Args:
            mappings: Dict of {column: {old_value: new_value}} for direct mapping.
            groupings: Dict of {column: {old_category: group_name}} for grouping.
            rare_threshold: Minimum count for a category to keep its own label.
                Categories below this count are replaced with rare_label.
                None = disabled.
            rare_label: Label for rare categories (default "Other").
            normalize_case: If True, lowercase all values before mapping.
        """
        self.mappings = mappings or {}
        self.groupings = groupings or {}
        self.rare_threshold = rare_threshold
        self.rare_label = rare_label
        self.normalize_case = normalize_case
        self._reports: List[CategoryReport] = []

    def add_mapping(self, column: str, mapping: Dict[str, str]) -> "CategoryStandardizer":
        """Add a value mapping for a column. Returns self for chaining."""
        self.mappings[column] = mapping
        return self

    def add_grouping(self, column: str, grouping: Dict[str, str]) -> "CategoryStandardizer":
        """Add a category grouping for a column. Returns self for chaining."""
        self.groupings[column] = grouping
        return self

    # -- transform -----------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured standardizations."""
        df = df.copy()
        self._reports = []

        # 1. Apply case normalization
        if self.normalize_case:
            # Only apply to object/string columns
            for col in df.select_dtypes(include=["object", "string"]).columns:
                # specific check to avoid boolean columns that might be caught
                if pd.api.types.is_bool_dtype(df[col]):
                    continue
                    
                original_unique = df[col].nunique()
                original = df[col].copy()
                df[col] = df[col].str.lower()
                
                # Safe comparison handling NaNs
                v1 = original.fillna("").astype(str)
                v2 = df[col].fillna("").astype(str)
                modified = (v1 != v2).sum()
                
                if modified > 0:
                    self._reports.append(CategoryReport(
                        column=col, action="normalize_case",
                        original_unique=original_unique,
                        final_unique=df[col].nunique(),
                        rows_modified=int(modified),
                    ))

        # 2. Apply value mappings
        for col, mapping in self.mappings.items():
            if col not in df.columns:
                continue
            
            # Skip if boolean to avoid dtype issues
            if pd.api.types.is_bool_dtype(df[col]):
                continue

            original_unique = df[col].nunique()
            original = df[col].copy()

            # If normalize_case is on, also lowercase the mapping keys
            if self.normalize_case:
                mapping = {str(k).lower(): v for k, v in mapping.items()}

            df[col] = df[col].replace(mapping)
            
            # Safe comparison
            v1 = original.fillna("").astype(str)
            v2 = df[col].fillna("").astype(str)
            modified = (v1 != v2).sum()
            
            self._reports.append(CategoryReport(
                column=col, action="value_mapping",
                original_unique=original_unique,
                final_unique=df[col].nunique(),
                rows_modified=int(modified),
            ))

        # 3. Apply category groupings
        for col, grouping in self.groupings.items():
            if col not in df.columns:
                continue
            
            if pd.api.types.is_bool_dtype(df[col]):
                continue

            original_unique = df[col].nunique()
            original = df[col].copy()
            df[col] = df[col].replace(grouping)
            
            v1 = original.fillna("").astype(str)
            v2 = df[col].fillna("").astype(str)
            modified = (v1 != v2).sum()

            self._reports.append(CategoryReport(
                column=col, action="category_grouping",
                original_unique=original_unique,
                final_unique=df[col].nunique(),
                rows_modified=int(modified),
            ))

        # 4. Handle rare categories
        if self.rare_threshold is not None:
            for col in df.select_dtypes(include=["object", "string"]).columns:
                if pd.api.types.is_bool_dtype(df[col]):
                    continue

                value_counts = df[col].value_counts()
                rare_values = value_counts[value_counts < self.rare_threshold].index
                if len(rare_values) == 0:
                    continue
                original_unique = df[col].nunique()
                original = df[col].copy()
                df[col] = df[col].apply(
                    lambda x: self.rare_label if x in rare_values else x
                )
                
                v1 = original.fillna("").astype(str)
                v2 = df[col].fillna("").astype(str)
                modified = (v1 != v2).sum()

                self._reports.append(CategoryReport(
                    column=col, action="rare_grouping",
                    original_unique=original_unique,
                    final_unique=df[col].nunique(),
                    rows_modified=int(modified),
                ))

        return df

    def get_report(self) -> pd.DataFrame:
        """Return category standardization report."""
        if not self._reports:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "column": r.column,
                "action": r.action,
                "unique_before": r.original_unique,
                "unique_after": r.final_unique,
                "rows_modified": r.rows_modified,
            }
            for r in self._reports
        ])
