"""
Automated data cleaning pipeline.

Orchestrates all transformers in a sensible order and produces
a comprehensive cleaning report. Inspired by automate_5_steps.py
from Bala Priya C's tutorials.
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Union

from .quality import check_quality
from .transformers import DataStandardizer, MissingValueHandler
from .duplicates import DuplicateHandler
from .outliers import OutlierHandler
from .text import TextCleaner
from .categories import CategoryStandardizer
from .features import (
    NumericalTransformer, 
    DateFeatureExtractor, 
    MissingIndicator, 
    FeatureSelector
)


class AutomatedCleaner:
    """Automates the full data cleaning lifecycle.

    Default pipeline order:
        1. Initial quality check
        2. Type standardization (DataStandardizer)
        3. Date feature extraction (DateFeatureExtractor) — if configured
        4. Text cleaning (TextCleaner) — if column_types provided
        5. Category standardization (CategoryStandardizer) — if configured
        6. Duplicate removal (DuplicateHandler) — if duplicate_subset provided
        7. Outlier handling (OutlierHandler) — if enabled
        8. Missing indicators (MissingIndicator) — if enabled
        9. Missing value imputation (MissingValueHandler)
        10. Feature transformation & scaling (NumericalTransformer) — if configured
        11. Feature selection (FeatureSelector) — if enabled
        12. Final quality check + validation

    Any step can be skipped via skip_steps.
    """

    def __init__(
        self,
        # Transformer instances (pass custom-configured ones or use defaults)
        standardizer: Optional[DataStandardizer] = None,
        missing_handler: Optional[MissingValueHandler] = None,
        text_cleaner: Optional[TextCleaner] = None,
        category_standardizer: Optional[CategoryStandardizer] = None,
        duplicate_handler: Optional[DuplicateHandler] = None,
        outlier_handler: Optional[OutlierHandler] = None,
        numerical_transformer: Optional[NumericalTransformer] = None,
        date_extractor: Optional[DateFeatureExtractor] = None,
        missing_indicator: Optional[MissingIndicator] = None,
        feature_selector: Optional[FeatureSelector] = None,
        # Convenience config for duplicates
        duplicate_subset: Optional[List[str]] = None,
        duplicate_survivorship: str = "most_complete",
        # Convenience config for outliers
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        outlier_strategy: str = "cap",
        outlier_columns: Optional[List[str]] = None,
        # Convenience config for text cleaning
        column_types: Optional[Dict[str, str]] = None,
        # Convenience config for feature engineering
        scale_method: Optional[str] = None,  # "standard", "minmax", "robust"
        auto_transform_numerics: bool = False, # Log/Box-Cox/Yeo-Johnson
        extract_date_features: Union[bool, List[str]] = False,
        add_missing_indicators: bool = False,
        drop_low_variance: bool = False,
        variance_threshold: float = 0.0,
        drop_correlated: bool = False,
        correlation_threshold: float = 0.95,
        # Steps to skip
        skip_steps: Optional[List[str]] = None,
    ):
        self.standardizer = standardizer or DataStandardizer()
        self.missing_handler = missing_handler or MissingValueHandler()
        self.text_cleaner = text_cleaner or TextCleaner()
        self.category_standardizer = category_standardizer
        self.duplicate_handler = duplicate_handler or DuplicateHandler()
        self.outlier_handler = outlier_handler or OutlierHandler()
        
        # New transformers
        self.numerical_transformer = numerical_transformer or NumericalTransformer(
            auto_transform=auto_transform_numerics,
            scaling_method=scale_method or "standard" if scale_method else None
        )
        
        # Configure date extractor
        date_feats = None
        if isinstance(extract_date_features, list):
            date_feats = extract_date_features
        elif extract_date_features is True:
            # Default set if True
            date_feats = ["year", "month", "day", "is_weekend"]
            
        self.date_extractor = date_extractor or DateFeatureExtractor(features=date_feats)
        
        self.missing_indicator = missing_indicator or MissingIndicator()
        
        self.feature_selector = feature_selector or FeatureSelector(
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold
        )

        self.duplicate_subset = duplicate_subset
        self.duplicate_survivorship = duplicate_survivorship
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.outlier_strategy = outlier_strategy
        self.outlier_columns = outlier_columns
        self.column_types = column_types
        
        self.scale_method = scale_method
        self.auto_transform_numerics = auto_transform_numerics
        self.extract_date_features = extract_date_features
        self.add_missing_indicators = add_missing_indicators
        self.drop_low_variance = drop_low_variance
        self.drop_correlated = drop_correlated

        self.skip_steps = set(skip_steps or [])

    def clean(self, df: pd.DataFrame) -> tuple:
        """Run the full cleaning pipeline.

        Args:
            df: Input DataFrame to clean.

        Returns:
            Tuple of (cleaned_df, report_dict).
        """
        report: Dict[str, Any] = {}
        original_shape = df.shape

        # Step 1: Initial quality check
        report["initial_quality"] = check_quality(df)

        # Step 2: Type standardization
        if "standardize" not in self.skip_steps:
            df = self.standardizer.transform(df)
            std_report = self.standardizer.get_report()
            if not std_report.empty:
                report["type_conversions"] = std_report.to_dict("records")
                
        # Step 3: Date feature extraction (NEW)
        if "date_features" not in self.skip_steps and self.extract_date_features:
            df = self.date_extractor.transform(df)
            date_report = self.date_extractor.get_report()
            if not date_report.empty:
                report["date_features"] = date_report.to_dict("records")

        # Step 4: Text cleaning
        if "text" not in self.skip_steps and self.column_types:
            df = self.text_cleaner.clean_all(df, self.column_types)
            text_report = self.text_cleaner.get_report()
            if not text_report.empty:
                report["text_cleaning"] = text_report.to_dict("records")

        # Step 5: Category standardization
        if "categories" not in self.skip_steps and self.category_standardizer is not None:
            df = self.category_standardizer.transform(df)
            cat_report = self.category_standardizer.get_report()
            if not cat_report.empty:
                report["category_standardization"] = cat_report.to_dict("records")

        # Step 6: Duplicate removal
        if "duplicates" not in self.skip_steps and self.duplicate_subset:
            self.duplicate_handler.find_exact(df, subset=self.duplicate_subset)
            df = self.duplicate_handler.transform(df, survivorship=self.duplicate_survivorship)
            dup_report = self.duplicate_handler.get_report()
            if not dup_report.empty:
                report["duplicate_handling"] = dup_report.to_dict("records")

        # Step 7: Outlier handling
        if "outliers" not in self.skip_steps and self.outlier_columns:
            self.outlier_handler.detect(
                df,
                columns=self.outlier_columns,
                method=self.outlier_method,
                threshold=self.outlier_threshold,
            )
            df = self.outlier_handler.transform(df, strategy=self.outlier_strategy)
            outlier_report = self.outlier_handler.get_report()
            if not outlier_report.empty:
                report["outlier_handling"] = outlier_report.to_dict("records")
                
        # Step 8: Missing indicators
        if "missing_indicators" not in self.skip_steps and self.add_missing_indicators:
            self.missing_indicator.fit(df)
            df = self.missing_indicator.transform(df)
            ind_report = self.missing_indicator.get_report()
            if not ind_report.empty:
                report["missing_indicators"] = ind_report.to_dict("records")

        # Step 9: Missing value handling
        if "missing" not in self.skip_steps:
            df = self.missing_handler.fit_transform(df)
            miss_report = self.missing_handler.get_report()
            if not miss_report.empty:
                report["missing_value_handling"] = miss_report.to_dict("records")
                
        # Step 10: Feature transformation & scaling
        if "scaling" not in self.skip_steps and (self.scale_method or self.auto_transform_numerics):
            df = self.numerical_transformer.fit_transform(df)
            scale_report = self.numerical_transformer.get_report()
            if not scale_report.empty:
                report["feature_scaling"] = scale_report.to_dict("records")
            
        # Step 11: Feature Selection
        if "selection" not in self.skip_steps and (self.drop_low_variance or self.drop_correlated):
            self.feature_selector.fit(df)
            df = self.feature_selector.transform(df)
            sel_report = self.feature_selector.get_report()
            if not sel_report.empty:
                report["feature_selection"] = sel_report.to_dict("records")

        # Step 12: Final quality check + validation
        report["final_quality"] = check_quality(df)
        report["validation"] = {
            "rows_before": original_shape[0],
            "rows_after": len(df),
            "columns_before": original_shape[1],
            "columns_after": len(df.columns),
            "data_loss_pct": round(
                (1 - len(df) / original_shape[0]) * 100, 2
            ) if original_shape[0] > 0 else 0,
            "missing_before": report["initial_quality"]["total_missing_values"],
            "missing_after": report["final_quality"]["total_missing_values"],
        }

        return df, report