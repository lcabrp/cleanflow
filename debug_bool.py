import pandas as pd
from cleanflow import DataStandardizer

# Load the data
df = pd.read_csv("data/tutorials/tutorial_basic.csv")

print("Before DataStandardizer:")
print(f"active dtype: {df['active'].dtype}")
print(f"active values: {df['active'].unique()}")
print(f"active has NaN: {df['active'].isna().any()}")

# Apply DataStandardizer
std = DataStandardizer()
df = std.transform(df)

print("\nAfter DataStandardizer:")
print(f"active dtype: {df['active'].dtype}")
print(f"active values (non-null): {df['active'].dropna().unique()}")
print(f"active has NaN: {df['active'].isna().any()}")

# Try to compute mode
mode_result = df['active'].mode()
print(f"\nMode result: {mode_result}")
print(f"Mode type: {type(mode_result.iloc[0])}")
print(f"Mode value: {mode_result.iloc[0]}")

# Try to fill
try:
    filled = df['active'].fillna(mode_result.iloc[0])
    print("\nFillna succeeded!")
except Exception as e:
    print(f"\nFillna failed: {e}")
    
    # Try with bool conversion
    try:
        bool_val = bool(int(mode_result.iloc[0]))
        print(f"Converted to bool: {bool_val}, type: {type(bool_val)}")
        filled = df['active'].fillna(bool_val)
        print("Fillna with bool() succeeded!")
    except Exception as e2:
        print(f"Fillna with bool() also failed: {e2}")
