"""
CleanFlow Example 03: Quality Analysis
--------------------------------------
This script demonstrates the standalone data quality tools.

It shows:
1. Generating a comprehensive quality report with 'check_quality()'
2. Scoring row completeness with 'quality_score()'
3. Detecting suspicious values (round numbers, out-of-range ages) with 'detect_suspicious()'
4. Adding missing value indicators for better modeling ('add_missing_indicators()')

Target Audience: Data Scientists performing EDA (Exploratory Data Analysis)
"""

import pandas as pd
from cleanflow import check_quality, quality_score, detect_suspicious, add_missing_indicators

def main():
    # 1. Load data
    try:
        df = pd.read_csv("data/tutorials/tutorial_basic.csv")
    except FileNotFoundError:
        print("Please run scripts/generate_tutorial_data.py first.")
        return

    # 2. Comprehensive Quality Check
    print("--- Detailed Quality Report ---")
    quality_report = check_quality(df)
    
    print(f"Total Rows: {quality_report['total_rows']}")
    print(f"Overall Completeness: {quality_report['completeness_percentage']}%")
    print(f"Total Duplicate Rows: {quality_report['duplicate_rows']}")
    print("\nMissing Values per Column:")
    for col, count in quality_report["missing_values"].items():
        if count > 0:
            print(f"  - {col}: {count} missing")

    # 3. Row-level Quality Scoring
    # Useful for filtering out low-quality records or prioritizing manual review
    print("\n--- Row Quality Scoring ---")
    scored_df = quality_score(df)
    
    # Let's see the distribution of quality categories
    print(scored_df["quality_category"].value_counts())
    
    # Identify poor quality rows
    poor_rows = scored_df[scored_df["quality_category"] == "Poor"]
    print(f"\nFound {len(poor_rows)} rows categorized as 'Poor' quality.")

    # 4. Suspicious Value Detection
    # Flags likely erroneous data like age > 120 or suspiciously round salary figures
    print("\n--- Suspicious Value Detection ---")
    suspicious_flags = detect_suspicious(df)
    
    # Calculate how many rows were flagged
    flagged_counts = suspicious_flags.sum()
    print("Flagged suspicious values:")
    print(flagged_counts[flagged_counts > 0])
    
    # 5. Missing Indicators
    # Feature engineering step: creates binary columns for missing values
    print("\n--- Missing Value Indicators ---")
    # Only create indicators for columns with missing values
    cols_with_missing = [c for c in df.columns if df[c].isna().any()]
    df_with_indicators = add_missing_indicators(df, columns=cols_with_missing)
    
    new_cols = [c for c in df_with_indicators.columns if c.endswith("_is_missing")]
    print(f"Added {len(new_cols)} indicator columns: {', '.join(new_cols)}")

if __name__ == "__main__":
    main()
