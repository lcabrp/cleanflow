"""Tests for CleanFlow v2 — all transformers and pipeline."""

import pandas as pd
import pytest

from cleanflow import (
    AutomatedCleaner,
    check_quality,
    quality_score,
    detect_suspicious,
    add_missing_indicators,
    DataStandardizer,
    MissingValueHandler,
    DuplicateHandler,
    OutlierHandler,
    TextCleaner,
    CategoryStandardizer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Basic sample DataFrame for testing."""
    return pd.DataFrame({
        "name": ["Alice", "  bob wilson  ", "Charlie", None, "Diana"],
        "age": [25, 30, None, 28, 999],
        "income": ["$50,000", "60000", None, "$55,000", "70000"],
        "gender": ["Female", "m", "Male", "F", "female"],
        "email": ["alice@test.com", "BOB@Test.Com", "bad-email", None, "diana@test.com"],
        "country": ["USA", "US", "United States", "U.S.A.", "USA"],
        "rating": [4.5, 3.8, 4.2, 6.0, 3.5],
    })


@pytest.fixture
def dup_df():
    """DataFrame with exact and near-duplicates."""
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Alice", "Charlie", "alice"],
        "email": ["a@t.com", "b@t.com", "a@t.com", "c@t.com", "a@t.com"],
        "age": [25, 30, 25, 35, 25],
        "score": [None, 4.0, 3.5, 4.2, 3.8],
    })


@pytest.fixture
def outlier_df():
    """DataFrame with clear outliers."""
    return pd.DataFrame({
        "age": [25, 30, 35, 28, 999, 32, 45, 38, -5, 29],
        "income": [50000, 60000, 55000, 52000, 500000, 58000, 65000, 62000, 48000, 51000],
    })


# ---------------------------------------------------------------------------
# DataStandardizer
# ---------------------------------------------------------------------------

class TestDataStandardizer:
    def test_numeric_conversion(self, sample_df):
        std = DataStandardizer()
        result = std.transform(sample_df)
        # income should be converted from "$50,000" strings to numeric
        assert pd.api.types.is_numeric_dtype(result["income"])

    def test_report_generated(self, sample_df):
        std = DataStandardizer()
        std.transform(sample_df)
        report = std.get_report()
        assert not report.empty
        assert "column" in report.columns

    def test_no_change_on_numeric(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        std = DataStandardizer()
        result = std.transform(df)
        pd.testing.assert_frame_equal(result, df)
        assert std.get_report().empty


# ---------------------------------------------------------------------------
# MissingValueHandler
# ---------------------------------------------------------------------------

class TestMissingValueHandler:
    def test_median_imputation(self):
        df = pd.DataFrame({"x": [1, 2, None, 4, 5]})
        handler = MissingValueHandler(default_numeric="median")
        result = handler.fit_transform(df)
        assert result["x"].isna().sum() == 0
        assert result["x"].iloc[2] == 3.0  # median of [1,2,4,5]

    def test_mode_imputation(self):
        df = pd.DataFrame({"cat": ["a", "a", "b", None, "a"]})
        handler = MissingValueHandler(default_categorical="mode")
        result = handler.fit_transform(df)
        assert result["cat"].isna().sum() == 0
        assert result["cat"].iloc[3] == "a"

    def test_drop_threshold(self):
        df = pd.DataFrame({"x": [None, None, None, None, 1.0]})
        handler = MissingValueHandler(drop_threshold=0.7)
        result = handler.fit_transform(df)
        assert "x" not in result.columns

    def test_analyze(self):
        df = pd.DataFrame({"x": [1, None, 3], "y": ["a", None, "b"]})
        handler = MissingValueHandler()
        analysis = handler.analyze(df)
        assert len(analysis) == 2
        assert "recommended_strategy" in analysis.columns

    def test_report(self):
        df = pd.DataFrame({"x": [1, None, 3]})
        handler = MissingValueHandler()
        handler.fit_transform(df)
        report = handler.get_report()
        assert not report.empty
        assert "strategy" in report.columns

    def test_custom_strategy(self):
        df = pd.DataFrame({"x": [1, None, 3], "y": ["a", None, "b"]})
        handler = MissingValueHandler(strategies={"x": "zero", "y": "unknown"})
        result = handler.fit_transform(df)
        assert result["x"].iloc[1] == 0
        assert result["y"].iloc[1] == "UNKNOWN"


# ---------------------------------------------------------------------------
# DuplicateHandler
# ---------------------------------------------------------------------------

class TestDuplicateHandler:
    def test_exact_duplicates(self, dup_df):
        handler = DuplicateHandler()
        handler.find_exact(dup_df, subset=["name", "email"])
        assert len(handler.duplicate_groups) > 0

    def test_most_complete_survivorship(self, dup_df):
        handler = DuplicateHandler()
        handler.find_exact(dup_df, subset=["name", "email"])
        result = handler.transform(dup_df, survivorship="most_complete")
        # Should have fewer rows
        assert len(result) < len(dup_df)

    def test_no_duplicates(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        handler = DuplicateHandler()
        handler.find_exact(df)
        result = handler.transform(df)
        assert len(result) == 3

    def test_report(self, dup_df):
        handler = DuplicateHandler()
        handler.find_exact(dup_df, subset=["name", "email"])
        handler.transform(dup_df)
        report = handler.get_report()
        assert not report.empty

    def test_fuzzy_matching(self):
        df = pd.DataFrame({
            "name": ["John Smith", "Jon Smith", "Alice Brown", "Charlie Davis"],
            "age": [30, 30, 25, 40],
        })
        handler = DuplicateHandler()
        handler.find_fuzzy(df, match_columns=["name"], threshold=0.8)
        assert len(handler.duplicate_groups) > 0


# ---------------------------------------------------------------------------
# OutlierHandler
# ---------------------------------------------------------------------------

class TestOutlierHandler:
    def test_iqr_detection(self, outlier_df):
        handler = OutlierHandler()
        handler.detect(outlier_df, method="iqr")
        assert "age" in handler.outlier_masks
        assert handler.outlier_masks["age"].sum() > 0

    def test_cap_treatment(self, outlier_df):
        handler = OutlierHandler()
        handler.detect(outlier_df, columns=["age"], method="iqr")
        result = handler.transform(outlier_df, strategy="cap")
        assert result["age"].max() < 999

    def test_flag_treatment(self, outlier_df):
        handler = OutlierHandler()
        handler.detect(outlier_df, columns=["age"])
        result = handler.transform(outlier_df, strategy="flag")
        assert "age_is_outlier" in result.columns

    def test_zscore_detection(self, outlier_df):
        handler = OutlierHandler()
        handler.detect(outlier_df, method="zscore", threshold=2.0)
        assert "age" in handler.outlier_masks

    def test_report(self, outlier_df):
        handler = OutlierHandler()
        handler.detect(outlier_df, method="iqr")
        handler.transform(outlier_df, strategy="cap")
        report = handler.get_report()
        assert not report.empty
        assert "outlier_count" in report.columns


# ---------------------------------------------------------------------------
# TextCleaner
# ---------------------------------------------------------------------------

class TestTextCleaner:
    def test_strip_whitespace(self):
        assert TextCleaner.strip_whitespace("  hello   world  ") == "hello world"

    def test_titlecase(self):
        assert TextCleaner.titlecase("john smith") == "John Smith"

    def test_remove_html(self):
        assert TextCleaner.remove_html("<b>hello</b> &amp; world") == "hello & world"

    def test_clean_names_pipeline(self):
        df = pd.DataFrame({"name": ["  john SMITH  ", "JANE doe", "María García"]})
        cleaner = TextCleaner()
        result = cleaner.clean_names(df, "name")
        assert result["name"].iloc[0] == "John Smith"
        assert result["name"].iloc[1] == "Jane Doe"

    def test_clean_emails_pipeline(self):
        df = pd.DataFrame({"email": ["ALICE@Test.Com", "bad-email", None]})
        cleaner = TextCleaner()
        result = cleaner.clean_emails(df, "email")
        assert result["email"].iloc[0] == "alice@test.com"

    def test_clean_all(self):
        df = pd.DataFrame({
            "name": ["  ALICE  "],
            "email": ["ALICE@TEST.COM"],
        })
        cleaner = TextCleaner()
        result = cleaner.clean_all(df, {"name": "name", "email": "email"})
        assert result["name"].iloc[0] == "Alice"

    def test_report(self):
        df = pd.DataFrame({"name": ["  john  "]})
        cleaner = TextCleaner()
        cleaner.clean_names(df, "name")
        report = cleaner.get_report()
        assert not report.empty


# ---------------------------------------------------------------------------
# CategoryStandardizer
# ---------------------------------------------------------------------------

class TestCategoryStandardizer:
    def test_value_mapping(self):
        df = pd.DataFrame({"gender": ["M", "F", "m", "Male", "female"]})
        std = CategoryStandardizer(mappings={
            "gender": {"M": "Male", "m": "Male", "F": "Female", "female": "Female"}
        })
        result = std.transform(df)
        assert set(result["gender"].unique()) == {"Male", "Female"}

    def test_rare_category_grouping(self):
        df = pd.DataFrame({"job": ["dev", "dev", "dev", "qa", "pm"]})
        std = CategoryStandardizer(rare_threshold=2, rare_label="Other")
        result = std.transform(df)
        assert "Other" in result["job"].values

    def test_category_grouping(self):
        df = pd.DataFrame({"dept": ["IT", "Engineering", "Marketing"]})
        std = CategoryStandardizer(groupings={
            "dept": {"IT": "Technology", "Engineering": "Technology"}
        })
        result = std.transform(df)
        assert result["dept"].iloc[0] == "Technology"

    def test_chaining(self):
        std = CategoryStandardizer()
        result = std.add_mapping("x", {"a": "A"}).add_grouping("y", {"b": "B"})
        assert result is std

    def test_report(self):
        df = pd.DataFrame({"x": ["a", "b", "a"]})
        std = CategoryStandardizer(mappings={"x": {"a": "A"}})
        std.transform(df)
        report = std.get_report()
        assert not report.empty


# ---------------------------------------------------------------------------
# Quality functions
# ---------------------------------------------------------------------------

class TestQuality:
    def test_check_quality(self, sample_df):
        report = check_quality(sample_df)
        assert "total_rows" in report
        assert "completeness_percentage" in report
        assert report["total_rows"] == 5

    def test_quality_score(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, "x"]})
        result = quality_score(df)
        assert "quality_score" in result.columns
        assert "quality_category" in result.columns

    def test_detect_suspicious(self):
        df = pd.DataFrame({"income": [50000, 60000, 100000], "age": [25, 150, 30]})
        flags = detect_suspicious(df)
        assert "income_suspiciously_round" in flags.columns
        assert "age_out_of_range" in flags.columns
        assert flags["age_out_of_range"].iloc[1] == 1

    def test_add_missing_indicators(self):
        df = pd.DataFrame({"x": [1, None, 3]})
        result = add_missing_indicators(df)
        assert "x_is_missing" in result.columns
        assert result["x_is_missing"].sum() == 1


# ---------------------------------------------------------------------------
# AutomatedCleaner (end-to-end)
# ---------------------------------------------------------------------------

class TestAutomatedCleaner:
    def test_basic_pipeline(self, sample_df):
        cleaner = AutomatedCleaner()
        result, report = cleaner.clean(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert "initial_quality" in report
        assert "final_quality" in report
        assert "validation" in report

    def test_skip_steps(self, sample_df):
        cleaner = AutomatedCleaner(skip_steps=["standardize", "missing"])
        result, report = cleaner.clean(sample_df)
        assert "type_conversions" not in report
        assert "missing_value_handling" not in report

    def test_with_duplicates(self, dup_df):
        cleaner = AutomatedCleaner(
            duplicate_subset=["name", "email"],
            duplicate_survivorship="most_complete",
        )
        result, report = cleaner.clean(dup_df)
        assert len(result) < len(dup_df)
        assert "duplicate_handling" in report

    def test_with_outliers(self, outlier_df):
        cleaner = AutomatedCleaner(
            outlier_columns=["age", "income"],
            outlier_method="iqr",
            outlier_strategy="cap",
        )
        result, report = cleaner.clean(outlier_df)
        assert result["age"].max() < 999
        assert "outlier_handling" in report

    def test_validation_report(self, sample_df):
        cleaner = AutomatedCleaner()
        _, report = cleaner.clean(sample_df)
        v = report["validation"]
        assert "rows_before" in v
        assert "rows_after" in v
        assert "data_loss_pct" in v
        assert "missing_before" in v
        assert "missing_after" in v

    def test_full_pipeline_on_csv(self):
        """End-to-end test using the sample messy_data.csv."""
        import os
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "messy_data.csv")
        if not os.path.exists(csv_path):
            pytest.skip("messy_data.csv not found")

        df = pd.read_csv(csv_path)
        cat_std = CategoryStandardizer(mappings={
            "gender": {"M": "Male", "m": "Male", "F": "Female", "f": "Female", "female": "Female"},
            "country": {"US": "USA", "U.S.A.": "USA", "United States": "USA"},
        })
        cleaner = AutomatedCleaner(
            category_standardizer=cat_std,
            duplicate_subset=["name", "email"],
            outlier_columns=["age", "rating"],
            outlier_strategy="cap",
        )
        result, report = cleaner.clean(df)

        assert len(result) > 0
    def test_with_feature_engineering(self, sample_df):
        """Test the new feature engineering steps."""
        # Add a date column to sample_df
        df = sample_df.copy()
        df["join_date"] = ["2023-01-01", "2023-02-15", None, "2023-01-20", "2022-12-31"]
        
        cleaner = AutomatedCleaner(
            extract_date_features=True,
            add_missing_indicators=True,
            scale_method="minmax"
        )
        result, report = cleaner.clean(df)
        
        # Check Date Extraction
        assert "join_date_year" in result.columns
        assert "join_date_month" in result.columns
        assert "date_features" in report
        
        # Check Missing Indicators
        # 'age' has missing values in sample_df
        assert "age_is_missing" in result.columns
        assert "missing_indicators" in report
        
        # Check Scaling
        # 'rating' is numeric and should be scaled [0, 1]
        assert result["rating"].min() >= 0
        assert result["rating"].max() <= 1.0 + 1e-9  # Handle floating point epsilon
        assert "feature_scaling" in report

    def test_advanced_features(self):
        """Test integration of advanced features: cyclical dates, auto-transform, feature selection."""
        # Create a dataframe with specific characteristics
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01"]),
            "skewed": [1.0, 10.0, 100.0, 1000.0], # Highly skewed
            "constant": [1, 1, 1, 1], # Zero variance
            "correlated_a": [1, 2, 3, 4],
            "correlated_b": [2, 4, 6, 8], # Perfect correlation with a
        })
        
        cleaner = AutomatedCleaner(
            extract_date_features=["month_sin", "month_cos", "season"],
            auto_transform_numerics=True,
            drop_low_variance=True,
            variance_threshold=0.0,
            drop_correlated=True,
            correlation_threshold=0.95
        )
        
        result, report = cleaner.clean(df)
        
        # 1. Date Features
        assert "date_month_sin" in result.columns
        assert "date_season" in result.columns
        
        # 2. Skewed Feature (should be transformed)
        # NumericalTransformer usually keeps the same column name but transforms values
        # We can check if normality improved or if report says 'log'/'boxcox'
        if "feature_scaling" in report: # In pipeline, it reports under 'feature_scaling'
             # Find entry for 'skewed'
             transforms = {item['column']: item for item in report['feature_scaling']}
             assert transforms['skewed']['transformation'] in ['log', 'boxcox']
        
        # 3. Variance Filtering
        assert "constant" not in result.columns
        
        # 4. Correlation Filtering
        # Either 'correlated_a' or 'correlated_b' should be dropped
        assert not ("correlated_a" in result.columns and "correlated_b" in result.columns)
