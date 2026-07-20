# CleanFlow Data Cleaning Library — Copilot Instructions

This repo is a **modular Python library** for automated data cleaning. Primary focus: composable transformers following an **analyze → transform → report** pattern with 10+ essential data cleaning techniques.

**Package Name:** cleanflow  
**Purpose:** Reusable data cleaning transformers for pandas DataFrames  
**Tech Stack:** Python 3.14+, pandas 3.0+, numpy; optional SciPy via `cleanflow[features]` for Box-Cox/Yeo-Johnson transforms  
**Pattern:** Base classes (BaseTransformer, FitTransformer) for consistency

---

## Project Structure

```
cleanflow/
├── cleanflow/
│   ├── base.py                       # Abstract base classes
│   ├── transformers.py               # DataStandardizer, MissingValueHandler
│   ├── duplicates.py                 # DuplicateHandler (exact + fuzzy)
│   ├── outliers.py                   # OutlierHandler (4 methods, 6 treatments)
│   ├── text.py                       # TextCleaner (6 type-specific pipelines)
│   ├── categories.py                 # CategoryStandardizer
│   ├── quality.py                    # Quality analysis functions
│   ├── features.py                   # Feature engineering utilities
│   └── pipeline.py                   # AutomatedCleaner orchestrator
├── examples/
│   ├── 01_basic_cleaning.py          # Getting started
│   ├── 02_custom_configuration.py    # Configuration examples
│   ├── 03_quality_analysis.py        # Quality scoring
│   ├── 04_optimization_workflow.py   # Advanced workflows
│   ├── 05_feature_engineering.py     # Feature engineering
│   └── 06_advanced_features.py       # Complex scenarios
├── tests/                            # Unit tests
├── data/                             # Sample datasets
├── TECHNICAL.md                      # Architecture deep dive
└── pyproject.toml                    # Package metadata
```

---

## Core Architecture

### Base Classes

**Two abstract base classes for consistency:**

**1. BaseTransformer (Stateless)**
```python
from cleanflow.base import BaseTransformer

class MyTransformer(BaseTransformer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation (required)"""
        # Logic here
        return df
    
    def get_report(self) -> dict:
        """Report what was done (optional)"""
        return {"changes": self._changes}
```

**Used by:** DataStandardizer, DuplicateHandler, OutlierHandler, TextCleaner, CategoryStandardizer

**2. FitTransformer (Stateful)**
```python
from cleanflow.base import FitTransformer

class MyFitTransformer(FitTransformer):
    def fit(self, df: pd.DataFrame) -> "MyFitTransformer":
        """Learn parameters from data (required)"""
        self._learned_params = df.mean()
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned transformation (required)"""
        return df - self._learned_params
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method (inherited)"""
        return self.fit(df).transform(df)
```

**Used by:** MissingValueHandler (learns imputation values during fit)

### Why Two Base Classes?

- **BaseTransformer:** Each call to `transform()` is self-contained (e.g., outlier bounds computed fresh)
- **FitTransformer:** Learned values persist across calls (train/test split workflows)

---

## 10 Data Cleaning Transformers

### 1. DataStandardizer (Auto Type Detection)

**Converts string columns to proper types:**

```python
from cleanflow import DataStandardizer

df = pd.DataFrame({
    "price": ["$1,200.50", "$850.00", "$3,400.99"],
    "active": ["yes", "no", "yes"],
    "date": ["2024-01-15", "2024-02-20", "2024-03-10"],
    "rate": ["45%", "12.5%", "78%"],
})

std = DataStandardizer()
cleaned = std.transform(df)
print(cleaned.dtypes)
# price     float64
# active    boolean
# date      datetime64[ns]
# rate      float64
```

**Auto-detects:**
- Currency symbols: `$`, `€`, `£`, `¥`
- Thousands separators: `,`
- Percentages: `%`
- Booleans: yes/no, true/false, 1/0, y/n
- Dates: 12 common formats

### 2. MissingValueHandler (Intelligent Imputation)

**10+ strategies with auto-recommendation:**

```python
from cleanflow import MissingValueHandler

handler = MissingValueHandler(default_numeric="median")

# Analyze first (optional) - see recommendations
recommendations = handler.analyze(df)
print(recommendations)

# Fit and transform
cleaned = handler.fit_transform(df)
print(handler.get_report())
```

**Available strategies:**
- `mean`, `median`, `mode` - Central tendency
- `zero`, `unknown` - Fixed values
- `interpolate` - Linear interpolation
- `ffill`, `bfill` - Forward/backward fill
- `drop_rows`, `drop_column` - Removal
- `flag_missing` - Create indicator column

**Auto-recommendations:**
- Skewed numeric → median
- Symmetric numeric → mean
- Categorical → mode
- >70% missing → drop_column

### 3. DuplicateHandler (Exact + Fuzzy Matching)

**Exact duplicates:**
```python
from cleanflow import DuplicateHandler

handler = DuplicateHandler()

# Find exact matches
handler.find_exact(df, subset=["name", "email"])

# Resolve with survivorship rule
cleaned = handler.transform(df, survivorship="most_complete")
```

**Fuzzy duplicates:**
```python
# String similarity matching
handler.find_fuzzy(
    df,
    match_columns=["name"],
    threshold=0.85,  # SequenceMatcher ratio
    blocking_column="city"  # Optional performance boost
)

cleaned = handler.transform(df, survivorship="merge")
```

**Survivorship rules:**
- `first` - Keep first occurrence
- `last` - Keep last occurrence
- `most_complete` - Fewest NaNs
- `merge` - Combine non-null fields

### 4. OutlierHandler (4 Detection Methods, 6 Treatments)

**Detection methods:**
```python
from cleanflow import OutlierHandler

handler = OutlierHandler()

# IQR method (default)
handler.detect(df, column="price", method="iqr", threshold=1.5)

# Z-score method
handler.detect(df, column="price", method="zscore", threshold=3.0)

# Modified Z-score (MAD - more robust)
handler.detect(df, column="price", method="modified_zscore", threshold=3.5)

# Percentile method
handler.detect(df, column="price", method="percentile", 
               lower_percentile=1, upper_percentile=99)
```

**Treatment methods:**
```python
# Cap/Winsorize (replace with boundary values)
cleaned = handler.transform(df, treatment="cap")

# Remove outliers
cleaned = handler.transform(df, treatment="remove")

# Flag (add boolean column)
cleaned = handler.transform(df, treatment="flag")

# Impute with mean/median
cleaned = handler.transform(df, treatment="impute_median")
```

### 5. TextCleaner (6 Type-Specific Pipelines)

**Type-specific cleaning:**
```python
from cleanflow import TextCleaner

cleaner = TextCleaner()

# Names (Title Case, remove extra spaces)
df["name"] = cleaner.clean_names(df["name"])

# Addresses (standardize abbreviations)
df["address"] = cleaner.clean_addresses(df["address"])

# Descriptions (HTML/URL removal, unicode normalization)
df["description"] = cleaner.clean_descriptions(df["description"])

# Codes (uppercase, remove special chars)
df["product_code"] = cleaner.clean_codes(df["product_code"])

# Emails (lowercase, validate format)
df["email"] = cleaner.clean_emails(df["email"])

# Phones (extract digits, format)
df["phone"] = cleaner.clean_phones(df["phone"])
```

**Common operations:**
- HTML/URL removal
- Unicode normalization (NFD/NFC)
- Accent stripping
- Extra whitespace removal
- Case normalization

### 6. CategoryStandardizer (Mapping & Grouping)

**Value mapping:**
```python
from cleanflow import CategoryStandardizer

std = CategoryStandardizer()

# Map variations to standard values
mapping = {
    "USA": ["U.S.", "United States", "America"],
    "UK": ["United Kingdom", "Great Britain"],
}
df["country"] = std.map_categories(df["country"], mapping)

# Group rare categories
df["category"] = std.group_rare(df["category"], threshold=0.05, group_name="Other")

# Case normalization
df["status"] = std.normalize_case(df["status"], case="upper")
```

### 7-10. Feature Engineering, Quality Tools

**Scaling:**
```python
from cleanflow.features import scale_features

# Standard scaling (mean=0, std=1)
df_scaled = scale_features(df, method="standard", columns=["price", "quantity"])

# MinMax scaling (0-1 range)
df_scaled = scale_features(df, method="minmax", columns=["price"])

# Robust scaling (uses IQR, resistant to outliers)
df_scaled = scale_features(df, method="robust", columns=["price"])
```

**Date extraction:**
```python
from cleanflow.features import extract_date_features

df = extract_date_features(df, date_column="order_date", 
                           features=["year", "month", "day_of_week", "quarter"])
```

**Quality analysis:**
```python
from cleanflow.quality import check_quality, quality_score, detect_suspicious

# Comprehensive quality check
report = check_quality(df)
print(f"Completeness: {report['completeness_percentage']}%")
print(f"Duplicates: {report['duplicate_rows']}")

# Single quality score (0-100)
score = quality_score(df)

# Detect suspicious patterns
suspicious = detect_suspicious(df)
```

---

## AutomatedCleaner Pipeline

**Orchestrates all transformers:**

```python
from cleanflow import AutomatedCleaner

cleaner = AutomatedCleaner(
    handle_missing=True,
    remove_duplicates=True,
    handle_outliers=True,
    standardize_text=True,
    standardize_categories=True
)

cleaned_df, report = cleaner.clean(df)

# Report structure
print(report.keys())
# ['initial_quality', 'final_quality', 'type_conversions', 
#  'text_cleaning', 'category_standardization', 'duplicate_handling',
#  'outlier_handling', 'missing_value_handling', 'validation']

print(f"Data loss: {report['validation']['data_loss_pct']}%")
```

**Pipeline order:**
1. check_quality() → initial_quality
2. DataStandardizer → type_conversions
3. TextCleaner → text_cleaning
4. CategoryStandardizer → category_standardization
5. DuplicateHandler → duplicate_handling
6. OutlierHandler → outlier_handling
7. MissingValueHandler → missing_value_handling
8. check_quality() → final_quality

---

## Python/Package Conventions

### Import Patterns

**Public API (preferred):**
```python
from cleanflow import (
    AutomatedCleaner,
    DataStandardizer,
    MissingValueHandler,
    DuplicateHandler,
    OutlierHandler,
    TextCleaner,
    CategoryStandardizer,
)
```

**Avoid internal imports:**
```python
# ✗ DON'T - Internal modules
from cleanflow.base import BaseTransformer
from cleanflow.transformers import DataStandardizer  # Use top-level import
```

### Transformer Usage Pattern

**Standard workflow:**
```python
# 1. Instantiate
handler = MissingValueHandler(default_numeric="median")

# 2. Analyze (optional - for inspection)
analysis = handler.analyze(df)
print(analysis)

# 3. Fit (for FitTransformers only)
handler.fit(df)

# 4. Transform
cleaned = handler.transform(df)

# Or combined:
cleaned = handler.fit_transform(df)

# 5. Get report
report = handler.get_report()
```

### Configuration Patterns

**Transformer-specific config:**
```python
# Missing value handler
handler = MissingValueHandler(
    default_numeric="median",
    default_categorical="mode",
    drop_threshold=0.7,  # Drop columns with >70% missing
    fill_values={"specific_col": "custom_value"}
)

# Outlier handler
outlier = OutlierHandler(
    method="iqr",
    threshold=1.5,
    treatment="cap"
)

# Text cleaner
text = TextCleaner(
    strip_html=True,
    remove_urls=True,
    normalize_unicode=True,
    strip_accents=True
)
```

---

## Real-World Use Cases

### 1. Automated Data Ingestion (Airflow/Cron)

```python
# In Airflow DAG or cron job
from cleanflow import AutomatedCleaner
import pandas as pd

def clean_daily_data():
    # Read raw data
    df = pd.read_csv("s3://bucket/raw/daily_data.csv")
    
    # Clean
    cleaner = AutomatedCleaner()
    cleaned_df, report = cleaner.clean(df)
    
    # Save to processed location
    cleaned_df.to_parquet("s3://bucket/processed/daily_data.parquet")
    
    # Alert if quality degraded
    if report['final_quality']['completeness_percentage'] < 95:
        send_alert(report)
```

### 2. ML Preprocessing Pipeline

```python
from cleanflow import MissingValueHandler, OutlierHandler

# Split data
split_at = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]

# Fit on training data only
missing_handler = MissingValueHandler()
missing_handler.fit(X_train)

outlier_handler = OutlierHandler()
outlier_handler.detect(X_train, column="price")

# Transform both sets with learned parameters
X_train_clean = missing_handler.transform(X_train)
X_train_clean = outlier_handler.transform(X_train_clean)

X_test_clean = missing_handler.transform(X_test)
X_test_clean = outlier_handler.transform(X_test_clean)
```

### 3. Dataset Auditing

```python
from cleanflow.quality import check_quality, quality_score

# Quick assessment
df = pd.read_csv("third_party_data.csv")
score = quality_score(df)
print(f"Quality Score: {score}/100")

# Detailed report
report = check_quality(df)
print(f"Missing: {report['missing_values_pct']}%")
print(f"Duplicates: {report['duplicate_rows']} rows")
print(f"Suspicious: {len(report['suspicious_columns'])} columns")
```

### 4. Legacy Data Migration

```python
from cleanflow import CategoryStandardizer, TextCleaner

# Standardize messy manual entries
df["country"] = CategoryStandardizer().map_categories(
    df["country"],
    {"USA": ["U.S.", "United States", "America", "US"]}
)

# Clean names and addresses
df["name"] = TextCleaner().clean_names(df["name"])
df["address"] = TextCleaner().clean_addresses(df["address"])
```

---

## Testing Patterns

### Unit Test Structure

```python
import pytest
from cleanflow import DataStandardizer
import pandas as pd

def test_currency_conversion():
    """Test that currency strings convert to float"""
    df = pd.DataFrame({"price": ["$1,200.50", "$850.00"]})
    std = DataStandardizer()
    result = std.transform(df)
    
    assert result["price"].dtype == "float64"
    assert result["price"].iloc[0] == 1200.50

def test_boolean_detection():
    """Test boolean conversion"""
    df = pd.DataFrame({"active": ["yes", "no", "yes"]})
    std = DataStandardizer()
    result = std.transform(df)
    
    assert result["active"].dtype == "boolean"
    assert result["active"].iloc[0] == True
```

### Integration Tests

```python
def test_full_pipeline():
    """Test complete cleaning pipeline"""
    df = load_messy_dataset()
    
    cleaner = AutomatedCleaner()
    cleaned, report = cleaner.clean(df)
    
    # Validate improvements
    assert report['final_quality']['completeness_percentage'] > \
           report['initial_quality']['completeness_percentage']
    
    # Validate data integrity
    assert len(cleaned) > 0
    assert cleaned.isnull().sum().sum() < df.isnull().sum().sum()
```

---

## Development Workflows

### Adding New Transformer

1. **Create class extending BaseTransformer or FitTransformer:**
```python
# cleanflow/my_transformer.py
from cleanflow.base import BaseTransformer

class MyTransformer(BaseTransformer):
    def __init__(self, config_param=None):
        self.config_param = config_param
        self._report = {}
    
    def transform(self, df):
        # Implementation
        return df
    
    def get_report(self):
        return self._report
```

2. **Export in `__init__.py`:**
```python
from .my_transformer import MyTransformer
__all__ = [..., "MyTransformer"]
```

3. **Write tests:**
```python
def test_my_transformer():
    assert MyTransformer().transform(df) is not None
```

4. **Add example:**
```python
# examples/07_my_transformer.py
from cleanflow import MyTransformer
```

### Adding New Detection Method

```python
# cleanflow/outliers.py
class OutlierHandler(BaseTransformer):
    def detect(self, df, column, method="iqr", **kwargs):
        if method == "my_new_method":
            return self._detect_my_method(df, column, **kwargs)
        # ...existing methods...
    
    def _detect_my_method(self, df, column, threshold):
        # Implementation
        pass
```

---

## Performance Considerations

### Memory Efficiency

```python
# ✓ GOOD - Process in place when possible
handler.transform(df)  # Many transformers modify in place

# ✓ GOOD - Drop columns early
df = df.drop(columns=unnecessary_columns)

# ✗ AVOID - Multiple full copies
df1 = df.copy()
df2 = df.copy()
df3 = df.copy()
```

### Large Dataset Handling

```python
# For datasets > 1GB, process in chunks
chunk_size = 100_000
cleaned_chunks = []

for chunk in pd.read_csv("large_file.csv", chunksize=chunk_size):
    cleaner = AutomatedCleaner()
    cleaned_chunk, _ = cleaner.clean(chunk)
    cleaned_chunks.append(cleaned_chunk)

cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
```

---

## Common Pitfalls & Solutions

### FitTransformer Not Fitted

**Problem:** Calling `transform()` before `fit()`

**Solution:**
```python
# ✗ WRONG
handler = MissingValueHandler()
cleaned = handler.transform(df)  # Error!

# ✓ CORRECT
handler = MissingValueHandler()
handler.fit(df)
cleaned = handler.transform(df)

# ✓ OR USE fit_transform
cleaned = handler.fit_transform(df)
```

### Report Before Transform

**Problem:** Calling `get_report()` before transformation

**Solution:**
```python
# ✗ WRONG
handler = DuplicateHandler()
report = handler.get_report()  # Empty report

# ✓ CORRECT
handler = DuplicateHandler()
cleaned = handler.transform(df)
report = handler.get_report()  # Has data
```

---

## Documentation

- **README.md:** User guide, quick start, feature overview
- **TECHNICAL.md:** Architecture, algorithms, design decisions
- **examples/:** 6 example scripts covering all features
- **API docstrings:** All public methods documented

---

## References

- **pandas Documentation:** https://pandas.pydata.org/docs/
- **Data Quality:** ISO 8000 standards
- **Text Normalization:** Unicode consortium
