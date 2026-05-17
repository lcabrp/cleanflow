"""
CleanFlow Example 04: Optimization Workflow
-------------------------------------------
This script demonstrates the complete data pipeline: 
Loading -> Cleaning -> Optimizing -> Saving (Parquet)

The optimizer features from the former data-optimizer project now live inside
CleanFlow, so this example uses one package end to end.

Prerequisites:
    pip install -e ".[parquet]"

Target Audience: Data Engineers building production pipelines
"""

from cleanflow import AutomatedCleaner, load_dataset, optimize_dataset

def main():
    # 1. Efficient Loading
    print("Loading data via CleanFlow...")
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
    
    optimized_df.to_parquet(output_file, compression="zstd")
    
    print("Done! Workflow complete.")

if __name__ == "__main__":
    main()
