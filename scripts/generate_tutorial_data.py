import pandas as pd
import numpy as np
import random
from pathlib import Path

def setup_data_dir():
    """Ensure data/tutorials directory exists."""
    data_dir = Path("data/tutorials")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def generate_basic_dataset(output_path):
    """Generate a dataset with mixed errors for basic cleaning tutorial."""
    np.random.seed(42)
    random.seed(42)
    n_rows = 200

    data = {
        "id": range(1, n_rows + 1),
        "name": [
            random.choice(["  John Smith  ", "jane doe", "Alice Brown", None, "Bob Wilson"]) 
            for _ in range(n_rows)
        ],
        "age": [
            random.choice([25, 30, 35, 40, 999, -5, None]) 
            for _ in range(n_rows)
        ],
        "salary": [
            random.choice(["$50,000", "60000", "$75,000", None, "100000"]) 
            for _ in range(n_rows)
        ],
        "join_date": [
            random.choice(["2023-01-01", "01/02/2023", "March 5, 2023", None]) 
            for _ in range(n_rows)
        ],
        "active": [
            random.choice(["yes", "no", "1", "0", "true", "false", None]) 
            for _ in range(n_rows)
        ]
    }
    
    df = pd.DataFrame(data)
    # Add some duplicate rows
    df = pd.concat([df, df.iloc[:10]]).reset_index(drop=True)
    
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} ({len(df)} rows)")

def generate_duplicates_dataset(output_path):
    """Generate a dataset with exact and fuzzy duplicates."""
    data = {
        "email": [
            "alice@test.com", "bob@test.com", "charlie@test.com", 
            "alice@test.com", "BOB@TEST.COM", "dave@test.com"
        ],
        "name": [
            "Alice Smith", "Bob Jones", "Charlie Brown",
            "Alice Smith", "Bob Jones", "Dave Wilson"
        ],
        "score": [85, 90, 88, 85, 90, 92],
        "city": [
            "New York", "Los Angeles", "Chicago",
            "New York", "L.A.", "Seattle"
        ]
    }
    # Replicate to make it larger
    df = pd.DataFrame(data)
    df = pd.concat([df] * 20, ignore_index=True)
    
    # Add some noise
    df.loc[10:15, "name"] = "Alice  Smith"  # fuzzy match
    
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} ({len(df)} rows)")

def generate_text_dataset(output_path):
    """Generate a dataset with messy text fields."""
    data = {
        "full_name": [
            "  john DOE  ", "Mary-Jane", "O'Connor, Tim", 
            "alice   white", "BOB"
        ] * 20,
        "phone": [
            "(555) 123-4567", "555.123.4567", "5551234567", 
            "+1-555-123-4567", "invalid"
        ] * 20,
        "address": [
            "123 Main St.", "456 Oak ave", "789 Pine BLVD.", 
            "101 Maple dr", "202 Elm Street"
        ] * 20,
        "notes": [
            "<p>Customer since 2020</p>", "VIP user (see: http://crm.com/123)",
            "  check  history  ", "no notes", "<b>Important!</b>"
        ] * 20
    }
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} ({len(df)} rows)")

if __name__ == "__main__":
    data_dir = setup_data_dir()
    generate_basic_dataset(data_dir / "tutorial_basic.csv")
    generate_duplicates_dataset(data_dir / "tutorial_duplicates.csv")
    generate_text_dataset(data_dir / "tutorial_text.csv")
