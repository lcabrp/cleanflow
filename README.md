# CleanFlow

Modular Python library for automated data cleaning. Implements 10 essential data cleaning techniques as composable transformers that follow a consistent **analyze → transform → report** pattern.

**Version:** 0.2.0 · **Python:** ≥3.9 · **License:** MIT

---

## Features

| Transformer | Techniques |
|---|---|
| **DataStandardizer** | Auto-detects and converts types: numerics (with `$`, `,`, `€`, `£`, `¥`), dates (12 formats), booleans, percentages |
| **MissingValueHandler** | 10+ strategies: mean, median, mode, zero, unknown, interpolate, ffill, bfill, drop, flag. Auto-recommends based on skewness and data type |
| **DuplicateHandler** | Exact and fuzzy matching (SequenceMatcher similarity). Survivorship rules: first, last, most_complete, merge |
| **OutlierHandler** | Detection: IQR, Z-score, Modified Z-score (MAD), Percentile. Treatment: cap/winsorize, remove, flag, impute_mean, impute_median |
| **TextCleaner** | Type-specific pipelines for names, addresses, descriptions, codes, emails, phones. HTML/URL removal, unicode normalization, accent stripping |
| **CategoryStandardizer** | Value mapping, category grouping, rare category handling, case normalization |
| **Feature Engineering** | Scaling (Standard, MinMax, Robust), Date Extraction (Year, Month, etc.), Missing Indicators |
| **Quality Tools** | `check_quality()`, `quality_score()`, `detect_suspicious()`, `add_missing_indicators()` |

---

## Installation

```bash
# From local source (editable)
pip install -e .

# Or with uv
uv sync
```

**Dependencies:** pandas ≥1.5, numpy ≥1.21, scikit-learn ≥1.0

---

## Real-World Scenarios

CleanFlow is designed for:

**1. Automated Data Ingestion**
   - **Scenario**: Use in an Airflow DAG or cron job to clean daily CSV dumps from valid sources.
   - **Benefit**: Ensures downstream dashboards (Tableau/PowerBI) never break due to bad data.

**2. Machine Learning Preprocessing**
   - **Scenario**: Preparing raw data for Scikit-Learn or XGBoost.
   - **Benefit**: Handles everything ML models hate: NaNs, outliers, skewness, and non-numeric dates.

**3. Dataset Auditing**
   - **Scenario**: Evaluating a 3rd-party dataset before purchase.
   - **Benefit**: `check_quality()` reveals missingness, duplicates, and inconsistencies in seconds.

**4. Legacy Data Migration**
   - **Scenario**: Standardizing old Excel files with messy manual entries ("USA", "U.S.", "United States").
   - **Benefit**: `CategoryStandardizer` and `TextCleaner` normalize inconsistencies automatically.

---


## Quick Start

```python
from cleanflow import AutomatedCleaner
import pandas as pd

df = pd.read_csv("data/messy_data.csv")
cleaner = AutomatedCleaner()
cleaned_df, report = cleaner.clean(df)

print(f"Completeness: {report['initial_quality']['completeness_percentage']}% → "
      f"{report['final_quality']['completeness_percentage']}%")
print(f"Data loss: {report['validation']['data_loss_pct']}%")
```

---

## Using Individual Transformers

### Data Type Standardization

Automatically detects and converts string columns to their proper types.

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

print(std.get_report())
```

### Missing Value Handling

Intelligent imputation with auto-recommendation based on data characteristics.

```python
from cleanflow import MissingValueHandler

handler = MissingValueHandler(default_numeric="median")

# Analyze first (optional) — see recommended strategies
print(handler.analyze(df))

# Then clean
cleaned = handler.fit_transform(df)
print(handler.get_report())
```

**Available strategies:** `mean`, `median`, `mode`, `zero`, `unknown`, `interpolate`, `ffill`, `bfill`, `drop_rows`, `drop_column`, `flag_missing`

Columns with >70% missing are automatically dropped. Skewed numeric columns default to `median`; symmetric ones to `mean`.

### Duplicate Detection & Resolution

```python
from cleanflow import DuplicateHandler

handler = DuplicateHandler()

# Exact matching
handler.find_exact(df, subset=["name", "email"])

# Or fuzzy matching (string similarity)
handler.find_fuzzy(df, match_columns=["name"], threshold=0.85,
                   blocking_column="city")  # optional blocking for performance

# Resolve with survivorship rule
cleaned = handler.transform(df, survivorship="most_complete")
print(handler.get_report())
```

**Survivorship rules:** `first`, `last`, `most_complete` (fewest NaNs), `merge` (combine non-null fields)

### Outlier Detection & Treatment

```python
from cleanflow import OutlierHandler

handler = OutlierHandler()
handler.detect(df, columns=["age", "income"], method="iqr", threshold=1.5)
cleaned = handler.transform(df, strategy="cap")
print(handler.get_report())
```

**Detection methods:** `iqr`, `zscore`, `modified_zscore` (MAD-based), `percentile`

**Treatment strategies:** `cap`/`winsorize`, `remove`, `flag`, `impute_mean`, `impute_median`

### Text Cleaning

Type-specific pipelines that apply the right operations for each kind of text.

```python
from cleanflow import TextCleaner

cleaner = TextCleaner()
cleaned = cleaner.clean_all(df, {
    "name": "name",           # strip → title case → remove accents
    "address": "address",     # strip → lowercase → expand abbreviations
    "bio": "description",     # strip → remove HTML → remove URLs → lowercase
    "sku": "code",            # strip → uppercase → remove spaces
    "email": "email",         # strip → lowercase → validate format
    "phone": "phone",         # extract digits → reformat (xxx) xxx-xxxx
})
print(cleaner.get_report())
```

### Category Standardization

```python
from cleanflow import CategoryStandardizer

std = CategoryStandardizer(
    mappings={
        "gender": {"M": "Male", "m": "Male", "F": "Female", "f": "Female"},
        "country": {"US": "USA", "U.S.A.": "USA", "United States": "USA"},
    },
    rare_threshold=5,       # group categories with <5 occurrences
    rare_label="Other",
    normalize_case=True,    # lowercase before mapping
)
cleaned = std.transform(df)
print(std.get_report())
```

---

## Full Pipeline with Options

```python
from cleanflow import AutomatedCleaner, CategoryStandardizer

cat_std = CategoryStandardizer(mappings={
    "gender": {"M": "Male", "m": "Male", "F": "Female", "f": "Female"},
    "country": {"US": "USA", "U.S.A.": "USA"},
})

cleaner = AutomatedCleaner(
    category_standardizer=cat_std,
    duplicate_subset=["name", "email"],
    duplicate_survivorship="most_complete",
    outlier_columns=["age", "income"],
    outlier_method="iqr",
    outlier_threshold=1.5,
    outlier_strategy="cap",
    column_types={"name": "name", "email": "email", "phone": "phone"},
    skip_steps=["text"],  # optionally skip any step
)
cleaned_df, report = cleaner.clean(df)
```

**Pipeline order:**
1. Initial quality check
2. Type standardization (`DataStandardizer`)
3. Text cleaning (`TextCleaner`) — if `column_types` provided
4. Category standardization (`CategoryStandardizer`) — if configured
5. Duplicate removal (`DuplicateHandler`) — if `duplicate_subset` provided
6. Outlier handling (`OutlierHandler`) — if `outlier_columns` provided
7. Missing value imputation (`MissingValueHandler`)
8. Final quality check + validation report

**Skip any step** via `skip_steps`: `"standardize"`, `"text"`, `"categories"`, `"duplicates"`, `"outliers"`, `"missing"`

---

## Data Quality Tools

Standalone utility functions for quality analysis.

```python
from cleanflow import check_quality, quality_score, detect_suspicious, add_missing_indicators

# Comprehensive health report
report = check_quality(df)
# Returns: total_rows, total_columns, missing_values, duplicate_rows,
#          memory_usage_mb, completeness_percentage, column_completeness, dtypes

# Per-row quality scoring (0-10 scale)
scored_df = quality_score(df)
# Adds: quality_score (float), quality_category (Poor/Average/Good)

# Flag suspicious values
flags = detect_suspicious(df)
# Detects: suspiciously round incomes, out-of-range ages/ratings

# Missing value indicators
flagged_df = add_missing_indicators(df, columns=["age", "income"])
# Adds: age_is_missing (0/1), income_is_missing (0/1)
```

---

## Advanced Feature Engineering

CleanFlow goes beyond cleaning to prepare data for machine learning, incorporating modern techniques.

```python
from cleanflow import NumericalTransformer, DateFeatureExtractor, FeatureSelector, MissingIndicator

# 1. Auto-Transform & Scale Numerics (Log, Box-Cox, Yeo-Johnson)
# Automatically selects best transformation to improve normality
num_trans = NumericalTransformer(auto_transform=True, scaling_method="standard")
scaled_df = num_trans.fit_transform(df)

# 2. Advanced Date Features (Cyclical Encoding)
# Creates month_sin, month_cos, season, is_weekend, etc.
date_ext = DateFeatureExtractor(features=["month_sin", "month_cos", "season", "weekday"])
dated_df = date_ext.transform(df)

# 3. Feature Selection (Filter methods)
# Remove constant columns and highly correlated features
selector = FeatureSelector(variance_threshold=0.0, correlation_threshold=0.95)
selected_df = selector.fit_transform(df)
```

You can enable these directly in `AutomatedCleaner`:

```python
cleaner = AutomatedCleaner(
    # Feature Engineering
    auto_transform_numerics=True,       # Try Log/Box-Cox
    extract_date_features=["month_sin", "month_cos", "season"],
    
    # Feature Selection
    drop_low_variance=True,
    drop_correlated=True,
    
    # Other
    add_missing_indicators=True,
    scale_method="standard"
)
```

---

## Using with data_optimizer

CleanFlow and [data_optimizer](../data_optimizer) are complementary libraries:

| Library | Focus |
|---|---|
| **data_optimizer** | Loading, memory optimization, profiling, Parquet conversion |
| **cleanflow** | Data quality — fixing types, missing values, duplicates, outliers, text |

### Installing data_optimizer

If the `data_optimizer` project is in a sibling folder (e.g. `../data_optimizer`), install it in editable mode:

```bash
# From the cleanflow project root
pip install -e ../data_optimizer

# Or with uv
uv pip install -e ../data_optimizer
```

You may also want to install optional backends for best performance:

```bash
pip install pyarrow fastparquet
```

### Combined Workflow

```python
from data_optimizer import load_dataset, optimize_dataset, convert_to_parquet_optimized
from cleanflow import AutomatedCleaner, CategoryStandardizer

# 1. LOAD — efficient loading via data_optimizer
df = load_dataset("data/sales_data.csv", engine="pandas")

# 2. CLEAN — fix data quality with cleanflow
cleaner = AutomatedCleaner(
    duplicate_subset=["Product_ID"],
    outlier_columns=["Price", "Quantity_Sold"],
    outlier_strategy="cap",
)
cleaned_df, report = cleaner.clean(df)

print(f"Missing: {report['validation']['missing_before']} → {report['validation']['missing_after']}")
print(f"Rows: {report['validation']['rows_before']} → {report['validation']['rows_after']}")

# 3. OPTIMIZE — reduce memory usage via data_optimizer
optimized_df = optimize_dataset(cleaned_df)

# 4. SAVE — compressed Parquet via data_optimizer
optimized_df.to_parquet("data/sales_cleaned.parquet", compression="zstd")
```

### Two-Step Optimization + Cleaning

```python
from data_optimizer import load_dataset, analyze_optimization, apply_optimization, optimization_report
from cleanflow import AutomatedCleaner, check_quality

# Load
df = load_dataset("data/users_02.csv")

# Check quality before cleaning
print("Before:", check_quality(df)["completeness_percentage"], "% complete")

# Clean
cleaner = AutomatedCleaner()
cleaned_df, _ = cleaner.clean(df)

# Analyze optimization opportunities
recommendations = analyze_optimization(cleaned_df, verbose=True)
optimized_df = apply_optimization(cleaned_df, recommendations)
before_mb, after_mb, reduction = optimization_report(cleaned_df, optimized_df, recommendations)

print(f"Memory: {before_mb:.1f} MB → {after_mb:.1f} MB ({reduction:.1f}% reduction)")
```

### Batch Processing Pipeline

```python
from pathlib import Path
from data_optimizer import load_dataset, optimize_dataset
from cleanflow import AutomatedCleaner

data_dir = Path("data")
cleaner = AutomatedCleaner(outlier_strategy="cap")

for csv_file in data_dir.glob("*.csv"):
    df = load_dataset(str(csv_file))
    cleaned_df, report = cleaner.clean(df)
    optimized_df = optimize_dataset(cleaned_df)

    output = csv_file.with_suffix(".parquet")
    optimized_df.to_parquet(str(output), compression="zstd")

    print(f"{csv_file.name}: {report['validation']['rows_before']} rows, "
          f"{report['initial_quality']['completeness_percentage']}% → "
          f"{report['final_quality']['completeness_percentage']}% complete")
```

---

## Project Structure

```
cleanflow/
├── cleanflow/
│   ├── __init__.py          # Public API exports
│   ├── base.py              # BaseTransformer, FitTransformer ABCs
│   ├── transformers.py      # DataStandardizer, MissingValueHandler
│   ├── duplicates.py        # DuplicateHandler (exact + fuzzy)
│   ├── outliers.py          # OutlierHandler (IQR, Z-score, etc.)
│   ├── text.py              # TextCleaner (6 pipelines)
│   ├── categories.py        # CategoryStandardizer
│   ├── quality.py           # Quality analysis utilities
│   └── pipeline.py          # AutomatedCleaner orchestrator
├── tests/
│   └── test_cleaning.py     # Comprehensive test suite
├── data/                    # Sample datasets
│   ├── messy_data.csv
│   ├── messy_data_02.csv
│   ├── users.csv
│   ├── users_02.csv
│   ├── sales_data.csv
│   └── product_details.csv
├── setup.py
├── pyproject.toml
├── requirements.txt
├── README.md                # This file
├── TECHNICAL.md             # Architecture & internals
└── CHANGELOG.md             # Version history
```

---

## Development

```bash
# Install in editable mode
pip install -e .

# Run tests
pytest tests/ -v

# Run a quick smoke test
python -c "from cleanflow import AutomatedCleaner; print('OK')"
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.