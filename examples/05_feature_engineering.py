"""
CleanFlow Example 05: Feature Engineering
-----------------------------------------
This script demonstrates how to use the new feature engineering capabilities:
1. Date Feature Extraction (Year, Month, Weekday, etc.)
2. Missing Value Indicators (Preserving information about missingness)
3. Feature Scaling (Standard, MinMax, Robust)

These features correspond to KDNuggets Techniques #5, #7, and #9.
"""

import pandas as pd
from cleanflow import AutomatedCleaner

def main():
    # 1. Create a sample dataset with dates, missing values, and varying scales
    print("Creating sample dataset...")
    df = pd.DataFrame({
        "customer_id": range(1, 11),
        "signup_date": [
            "2023-01-01", "2023-02-14", "2023-03-30", None, "2022-12-25",
            "2023-01-15", "2023-06-01", "2023-07-04", None, "2023-09-10"
        ],
        "annual_income": [
            50000, 60000, 120000, 45000, None, 
            30000, 5000000, 75000, 62000, 58000
        ],  # High variance (one Millionaire)
        "credit_score": [
            700, 720, 800, 650, 600, 
            None, 850, 710, 690, 740
        ],   # Different scale (300-850)
        "age": [
            25, 30, 45, 22, 55, 
            None, 40, 33, 29, 27
        ]
    })
    
    print("Original DataFrame:")
    print(df.head())
    print("-" * 50)
    
    # 2. Configure AutomatedCleaner with feature engineering steps
    cleaner = AutomatedCleaner(
        # Enable Date Feature Extraction
        extract_date_features=["year", "month", "weekday", "is_month_start"],
        
        # Enable Missing Value Indicators
        add_missing_indicators=True,
        
        # Enable Feature Scaling (RobustScaler handles the millionaire outlier better)
        scale_method="robust",
        
        # Handle outliers normally
        outlier_columns=["annual_income"],
        outlier_method="iqr",
        outlier_strategy="cap"
    )
    
    # 3. Run the pipeline
    print("\nRunning pipeline with feature engineering...")
    cleaned_df, report = cleaner.clean(df)
    
    # 4. Inspect results
    print("-" * 50)
    print("Cleaned DataFrame (First 5 rows):")
    print(cleaned_df.head())
    
    print("\n--- Feature Engineering Report ---")
    
    if "date_features" in report:
        print(f"\nExtracted {len(report['date_features'])} date features:")
        for feat in report["date_features"]:
            # Correct keys: 'feature' and 'source'
            print(f"- {feat['feature']} (from {feat['source']})")
            
    if "missing_indicators" in report:
        print(f"\nCreated {len(report['missing_indicators'])} missing value indicators:")
        for ind in report["missing_indicators"]:
            # Correct keys: 'column' and 'indicator'
            print(f"- {ind['indicator']} (for {ind['column']})")
            
    if "feature_scaling" in report:
        print("\nFeature Scaling Applied:")
        for scale in report["feature_scaling"]:
            # Correct keys: 'column', 'transformation', 'scaling'
            print(f"- {scale['column']} (scaling: {scale['scaling']}, transform: {scale['transformation']})")
            
    # Verify we can feed this to a model (all numeric/bool)
    print("\nFinal Data Types:")
    print(cleaned_df.dtypes)

if __name__ == "__main__":
    main()
