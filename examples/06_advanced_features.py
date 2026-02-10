"""
Example 06: Advanced Feature Engineering
----------------------------------------
This script demonstrates the advanced feature engineering capabilities of CleanFlow,
inspired by modern KDNuggets data science techniques.

Concepts covered:
1. Automated Numerical Transformation (Log, Box-Cox, Yeo-Johnson)
2. Cyclical Date Features (sin/cos encoding)
3. Feature Selection (Low Variance, Correlation)
4. Missing Value Indicators
"""

import pandas as pd
import numpy as np
import cleanflow as cf
from cleanflow import AutomatedCleaner

def generate_complex_data(n_samples=1000):
    """Generates a dataset with complex feature engineering needs."""
    np.random.seed(42)
    
    # Dates
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="D")
    
    # Skewed Data (Log-normal distribution)
    skewed = np.random.lognormal(mean=0, sigma=1, size=n_samples)
    
    # Constant Data (Low Variance)
    constant = np.ones(n_samples)
    
    # Correlated Data
    base = np.random.normal(0, 1, n_samples)
    correlated_1 = base + np.random.normal(0, 0.01, n_samples) # Highly correlated
    correlated_2 = base # Perfect correlation
    
    # Missing Values
    missing_col = np.random.normal(0, 1, n_samples)
    mask = np.random.choice([True, False], size=n_samples, p=[0.1, 0.9])
    missing_col[mask] = np.nan
    
    df = pd.DataFrame({
        "date": dates,
        "revenue": skewed * 1000,           # Needs Log/Box-Cox
        "country_code": constant,           # Needs VarianceThreshold
        "metric_a": correlated_1,           # Needs CorrelationThreshold
        "metric_b": correlated_2,
        "user_score": missing_col           # Needs MissingIndicator
    })
    
    return df

def run_advanced_pipeline():
    print("Generating complex dataset...")
    df = generate_complex_data()
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    print("-" * 50)
    
    # Configure Advanced Cleaner
    cleaner = AutomatedCleaner(
        # 1. Date Features: Cyclical encoding is best for ML
        extract_date_features=["month_sin", "month_cos", "weekday_sin", "weekday_cos", "season"],
        
        # 2. Numerical Transformation: Auto-detect best transform (Log/Box-Cox) per column
        auto_transform_numerics=True,
        scale_method="standard",
        
        # 3. Missing Value Indicators: Flag missing values before imputation
        add_missing_indicators=True,
        
        # 4. Feature Selection: Remove useless features
        drop_low_variance=True,
        variance_threshold=0.0, # Remove constants
        drop_correlated=True,
        correlation_threshold=0.95
    )
    
    print("\nRunning Advanced Pipeline...")
    cleaned_df, report = cleaner.clean(df)
    
    print("-" * 50)
    print("PIPELINE REPORT SUMMARY")
    print("-" * 50)
    
    # Inspect Feature Selection
    if "feature_selection" in report:
        print("\n[Feature Selection] Dropped Columns:")
        for item in report["feature_selection"]:
            print(f"  - {item['column']} ({item['reason']})")
            
    # Inspect Transformations
    if "feature_scaling" in report:
        print("\n[Numerical Transformations]:")
        for item in report["feature_scaling"]:
            if item['transformation'] != 'none':
                print(f"  - {item['column']}: Applied {item['transformation']} transform")
    
    # Inspect Date Features
    new_date_feats = [c for c in cleaned_df.columns if "date_" in c]
    print(f"\n[Date Features] Created {len(new_date_feats)} features:")
    print(f"  {new_date_feats[:3]} ...")
    
    # Inspect Final Data
    print("\n" + "-" * 50)
    print(f"Final shape: {cleaned_df.shape}")
    print(f"Final columns: {cleaned_df.columns.tolist()}")
    
    # Verify skewness reduction
    orig_skew = df["revenue"].skew()
    new_skew = cleaned_df["revenue"].skew()
    print(f"\nSkewness of 'revenue': Original={orig_skew:.2f} -> Transformed={new_skew:.2f}")

if __name__ == "__main__":
    run_advanced_pipeline()
