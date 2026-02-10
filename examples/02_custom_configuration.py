"""
CleanFlow Example 02: Custom Configuration
------------------------------------------
This script demonstrates how to customize the AutomatedCleaner pipeline.

It shows: 
1. Using 'CategoryStandardizer' to map values and group categories
2. Setting outlier detection parameters
3. Providing column type hints for better TextCleaner performance
4. Skipping specific steps in the pipeline

Target Audience: Intermediate users who need fine-grained control
"""

import pandas as pd
from cleanflow import AutomatedCleaner, CategoryStandardizer

def main():
    # 1. Load the data
    try:
        # Load 'active' as object to prevent pandas from auto-inferring types,
        # and to demonstrate CategoryStandardizer's mapping capabilities.
        df = pd.read_csv("data/tutorials/tutorial_basic.csv", dtype={"active": "object"})
    except FileNotFoundError:
        print("Please run scripts/generate_tutorial_data.py first.")
        return

    # 2. Configure CategoryStandardizer
    # We want to map "1"/"0"/"true" to "yes"/"no" for better consistency
    # Note: DataStandardizer handles booleans automatically, but this shows 
    # how to exert manual control over categories.
    cat_std = CategoryStandardizer(
        mappings={
            "active": {
                "1": "yes", "0": "no",
                "true": "yes", "false": "no"
            }
        },
        normalize_case=True  # Lowercase values before mapping
    )

    # 3. Initialize AutomatedCleaner with custom settings
    cleaner = AutomatedCleaner(
        # Inject our custom category standardizer
        category_standardizer=cat_std,
        
        # Configure outlier detection
        # Use simple capping at 1.5 * IQR for the 'age' column
        outlier_columns=["age"],
        outlier_method="iqr",
        outlier_threshold=1.5,
        outlier_strategy="cap",

        # Provide column type hints for text cleaning
        # This tells TextCleaner which pipeline to use for 'name'
        column_types={"name": "name"},

        # Skip steps we don't want (e.g. if we want to handle duplicates manually later)
        skip_steps=["duplicates"]
    )

    print("Cleaning with custom configuration...")
    cleaned_df, report = cleaner.clean(df)

    # 4. Verification
    print("\n--- Outlier Handling Report ---")
    if "outlier_handling" in report:
        for item in report["outlier_handling"]:
            print(f"Column '{item['column']}': {item['outlier_count']} outliers treated with '{item['treatment']}'")

    print("\n--- Category Standardization Report ---")
    if "category_standardization" in report:
        for item in report["category_standardization"]:
            print(f"Column '{item['column']}': {item['unique_before']} -> {item['unique_after']} unique values")

    print(f"\nResult shape: {cleaned_df.shape}")

if __name__ == "__main__":
    main()
