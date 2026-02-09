import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cleanflow")

def check_quality(df: pd.DataFrame) -> dict:
    """
    Analyzes the DataFrame and returns a health report.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    report = {
        "total_rows": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2)
    }
    
    # Calculate health metric
    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    report["completeness_percentage"] = round(((total_cells - total_missing) / total_cells) * 100, 2)
    
    logger.info(f"Quality Check Complete: {report['completeness_percentage']}% complete.")
    return report