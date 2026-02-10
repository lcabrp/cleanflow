"""
CleanFlow Example 04: Optimization Workflow
-------------------------------------------
This script demonstrates the complete data pipeline: 
Loading -> Cleaning (CleanFlow) -> Optimizing (DataOptimizer) -> Saving (Parquet)

It shows how to combine CleanFlow with the complementary library 'data_optimizer'
for maximum efficiency.

Prerequisites:
    # If data_optimizer is in a sibling folder (e.g. ../data_optimizer):
    pip install -e ../data_optimizer
    pip install pyarrow  # for Parquet support

Target Audience: Data Engineers building production pipelines
"""

import pandas as pd
from cleanflow import AutomatedCleaner
try:
    from data_optimizer import load_dataset, optimize_dataset, convert_to_parquet_optimized
except ImportError:
    print("This tutorial requires the 'data_optimizer' library.")
    print("Install from sibling folder: pip install -e ../data_optimizer")
    print("Also install Parquet support: pip install pyarrow")
    exit()

def main():
    # 1. Efficient Loading
    # Use data_optimizer to load the data (supports Polars backend for speed)
    # Falling back to pandas if Polars is not available
    print("Loading data via data_optimizer...")
    try:
        # Generate some larger data if needed, but for tutorial we use the basic set
        input_file = "data/tutorials/tutorial_basic.csv"
        # Force pandas engine for compatibility with CleanFlow (which expects pandas DF)
        df = load_dataset(input_file, engine="pandas")
    except FileNotFoundError:
        print("Please run scripts/generate_tutorial_data.py first.")
        return

    print(f"Data loaded: {df.shape}")

    # 2. Cleaning with CleanFlow
    # Standard cleaning pipeline
    cleaner = AutomatedCleaner()
    print("\nCleaning data...")
    cleaned_df, report = cleaner.clean(df)
    
    # 3. Memory Optimization
    # Now that data is clean, optimize data types to save memory
    # (e.g. downcast integers, convert strings to categories if appropriate)
    print("\nOptimizing memory usage...")
    optimized_df = optimize_dataset(cleaned_df)
    
    # Calculate savings
    original_mem = df.memory_usage(deep=True).sum() / 1024**2
    cleaned_mem = cleaned_df.memory_usage(deep=True).sum() / 1024**2
    optimized_mem = optimized_df.memory_usage(deep=True).sum() / 1024**2
    
    print(f"Original Memory:  {original_mem:.2f} MB")
    print(f"Cleaned Memory:   {cleaned_mem:.2f} MB")
    print(f"Optimized Memory: {optimized_mem:.2f} MB")
    print(f"Total Reduction:  {(1 - optimized_mem/original_mem)*100:.1f}%")

    # 4. Save as Optimized Parquet
    # efficient storage with compression
    output_file = "data/tutorials/tutorial_cleaned.parquet"
    print(f"\nSaving to {output_file}...")
    
    # Simple pandas save (data_optimizer also has advanced saving tools)
    optimized_df.to_parquet(output_file, compression="zstd")
    
    print("Done! Workflow complete.")

if __name__ == "__main__":
    main()
