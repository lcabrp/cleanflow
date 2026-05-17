"""Tests for advanced feature engineering transformers."""

import pandas as pd
import numpy as np
from cleanflow.features import NumericalTransformer, FeatureScaler, DateFeatureExtractor, MissingIndicator, FeatureSelector

# ---------------------------------------------------------------------------
# NumericalTransformer (formerly FeatureScaler)
# ---------------------------------------------------------------------------

def test_numerical_transformer_scaling():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": ["x", "y", "z", "w", "v"]
    })
    # Disable auto-transform to test just scaling
    transformer = NumericalTransformer(auto_transform=False, scaling_method="standard")
    result = transformer.fit_transform(df)
    
    # Check "a" is scaled
    assert np.isclose(result["a"].mean(), 0)
    assert np.isclose(result["a"].std(ddof=0), 1)
    
    # Check report
    report = transformer.get_report()
    assert not report.empty
    assert report.iloc[0]["column"] == "a"
    assert report.iloc[0]["transformation"] == "none"
    assert report.iloc[0]["scaling"] == "standard"

def test_numerical_transformer_log():
    # Create log-normal distribution (skewed)
    np.random.seed(42)
    data = np.random.lognormal(mean=0, sigma=1, size=100)
    df = pd.DataFrame({"skewed": data})
    
    transformer = NumericalTransformer(auto_transform=True, scaling_method=None)
    result = transformer.fit_transform(df)
    
    report = transformer.get_report()
    # It should likely pick 'log' or 'boxcox'
    transform_method = report.iloc[0]["transformation"]
    assert transform_method in ["log", "boxcox"]
    
    # Skewness should be reduced
    original_skew = df["skewed"].skew()
    new_skew = result["skewed"].skew()
    assert abs(new_skew) < abs(original_skew)

def test_feature_scaler_alias():
    # Verify backward compatibility (mostly)
    # Note: 'method' arg is gone, so users must update to 'scaling_method' 
    # or rely on default 'standard'.
    scaler = FeatureScaler(scaling_method="minmax", auto_transform=False)
    df = pd.DataFrame({"a": [0, 100]})
    result = scaler.fit_transform(df)
    assert result["a"].max() == 1.0

# ---------------------------------------------------------------------------
# DateFeatureExtractor
# ---------------------------------------------------------------------------

def test_date_extractor_cyclical():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-07-01"]), # Jan and July
    })
    # Jan is month 1, July is month 7. 
    # sin(2*pi*1/12) = 0.5, cos(2*pi*1/12) = 0.866
    # sin(2*pi*7/12) = -0.5, cos(2*pi*7/12) = -0.866
    
    extractor = DateFeatureExtractor(features=["month_sin", "month_cos"])
    result = extractor.transform(df)
    
    assert "date_month_sin" in result.columns
    assert "date_month_cos" in result.columns
    
    # Check range
    assert result["date_month_sin"].min() >= -1.0
    assert result["date_month_sin"].max() <= 1.0

def test_date_extractor_season():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15"]),
    })
    extractor = DateFeatureExtractor(features=["season"])
    result = extractor.transform(df)
    
    assert "date_season" in result.columns
    expected = ["Winter", "Spring", "Summer", "Fall"]
    assert result["date_season"].tolist() == expected

# ---------------------------------------------------------------------------
# FeatureSelector
# ---------------------------------------------------------------------------

def test_feature_selector_variance():
    df = pd.DataFrame({
        "constant": [1, 1, 1, 1, 1],
        "variable": [1, 2, 3, 4, 5]
    })
    selector = FeatureSelector(variance_threshold=0.0)
    result = selector.fit_transform(df)
    
    assert "constant" not in result.columns
    assert "variable" in result.columns
    
    report = selector.get_report()
    assert len(report) == 1
    assert report.iloc[0]["column"] == "constant"
    assert report.iloc[0]["reason"] == "low_variance"

def test_feature_selector_correlation():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [2, 4, 6, 8, 10], # Perfect correlation with a
        "c": [1, 3, 2, 5, 4]   # Independent
    })
    selector = FeatureSelector(correlation_threshold=0.95)
    result = selector.fit_transform(df)
    
    # Should drop 'b' (or 'a', implementation dependent, usually keeps first)
    # My implementation iterates columns, 'a' comes first, 'b' correlates with 'a', drop 'b'
    assert "a" in result.columns
    assert "b" not in result.columns
    assert "c" in result.columns
    
    report = selector.get_report()
    assert len(report) == 1
    assert report.iloc[0]["reason"] == "high_correlation"

# ---------------------------------------------------------------------------
# MissingIndicator
# ---------------------------------------------------------------------------

def test_missing_indicator():
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": ["x", "y", "z"]
    })
    indicator = MissingIndicator()
    indicator.fit(df)
    result = indicator.transform(df)
    
    assert "a_is_missing" in result.columns
    assert "b_is_missing" not in result.columns
    
    report = indicator.get_report()
    assert len(report) == 1
    assert report.iloc[0]["column"] == "a"
