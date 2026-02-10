import pandas as pd
from cleanflow import AutomatedCleaner

# Load the data
df = pd.read_csv("data/tutorials/tutorial_basic.csv")

print("Original DataFrame:")
print(f"active dtype: {df['active'].dtype}")
print(f"active unique: {df['active'].unique()}")

# Create cleaner
cleaner = AutomatedCleaner()

# Manually step through the pipeline
print("\n=== Step 1: DataStandardizer ===")
df_step1 = cleaner.standardizer.transform(df.copy())
print(f"active dtype after standardizer: {df_step1['active'].dtype}")
print(f"active has NaN: {df_step1['active'].isna().sum()}")

print("\n=== Step 2: MissingValueHandler ===")
print("Fitting...")
cleaner.missing_handler.fit(df_step1)
print(f"Fit values: {cleaner.missing_handler._fit_values}")
print(f"Fit strategies: {cleaner.missing_handler._fit_strategies}")

if 'active' in cleaner.missing_handler._fit_values:
    fill_val = cleaner.missing_handler._fit_values['active']
    print(f"Fill value for 'active': {fill_val}, type: {type(fill_val)}")

print("\nTransforming...")
try:
    df_step2 = cleaner.missing_handler.transform(df_step1)
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
