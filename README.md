# CleanFlow

Modular Python library for automated data cleaning, quality analysis, profiling, memory optimization, and CSV-to-Parquet workflows. CleanFlow now combines the cleaning pipeline from this repo with the best loading/profiling/optimization ideas from the former `data-optimizer` project.

**Version:** 0.3.0 · **Python:** ≥3.9 · **License:** MIT

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
| **Profiling & IO** | `load_dataset()`, `profile_dataframe()`, `dataset_overview()`, CSV/Parquet support |
| **Optimization** | `optimize_dataset()`, two-step dtype optimization, pandas/Polars/Dask backends, CSV→Parquet conversion |

---

## What CleanFlow Solves

CleanFlow is for projects where raw tabular data is useful but not yet safe to trust. It gives you a repeatable way to inspect, clean, document, and optimize pandas-style datasets before they reach dashboards, notebooks, machine learning models, or downstream storage.

Common problems it can fix or expose:

| Problem | CleanFlow support |
|---|---|
| Dirty types | Converts currency strings, percentages, dates, booleans, and numeric-looking text |
| Missing values | Profiles missingness, imputes by strategy, drops high-missing columns, or adds missing indicators |
| Duplicate records | Finds exact duplicates and fuzzy text duplicates, then applies survivorship rules |
| Extreme values | Detects and caps, removes, flags, or imputes outliers |
| Messy text | Normalizes names, emails, phones, addresses, descriptions, and codes |
| Category drift | Maps inconsistent labels, groups related categories, and folds rare values into `Other` |
| Unknown data quality | Produces before/after quality reports and row-level quality scores |
| Large CSV friction | Loads, profiles, downcasts, and converts CSV files to Parquet |
| Reviewability | Returns structured reports so future readers can see what changed and why |

CleanFlow is intentionally best for tabular datasets where rule-based cleaning is acceptable and reviewable. It is not a replacement for domain validation, database constraints, human data stewardship, or model-specific feature engineering decisions.

---

## Installation

```bash
# From local source (editable)
pip install -e .

# Or with uv
uv sync
```

**Core dependencies:** pandas ≥1.5, numpy ≥1.21, scikit-learn ≥1.0, scipy ≥1.10

Optional extras:

```bash
pip install "cleanflow[parquet]"  # PyArrow Parquet conversion
pip install "cleanflow[duckdb]"   # Out-of-core CSV -> Parquet
pip install "cleanflow[polars]"   # Faster optional loading/optimization backend
pip install "cleanflow[all]"      # All optional backends
```

---

## Real-World Scenarios

CleanFlow is useful in both small scripts and larger data workflows:

| Scenario | Why CleanFlow fits |
|---|---|
| Notebook exploration | Run `check_quality()` and `profile_dataframe()` before spending time on analysis |
| One-off CSV cleanup | Use `AutomatedCleaner` to standardize types, handle missing values, and export a cleaner file |
| Dashboard refreshes | Apply the same cleaning pipeline every day so Power BI, Tableau, or Streamlit inputs stay stable |
| ML preprocessing | Fit imputation/scaling/feature steps on training data and apply them consistently |
| Legacy spreadsheet migration | Normalize manual entries before loading records into a database or warehouse |
| Data handoff review | Share the cleaning report with reviewers so they can see data loss, duplicates, and conversions |
| Large file optimization | Downcast memory-heavy columns or convert CSV to Parquet before repeated analysis |

For more concrete project patterns, see [docs/USE_CASES.md](docs/USE_CASES.md).

---

## Choosing the Right Entry Point

| Need | Start with |
|---|---|
| "I just received a messy CSV" | `load_dataset()` → `AutomatedCleaner()` → `optimize_dataset()` |
| "I only need to know if this data is safe" | `check_quality()`, `profile_dataframe()`, `dataset_overview()` |
| "I want a controlled, reviewable dtype change" | `analyze_optimization()` → review → `apply_optimization()` |
| "I know the exact cleaning rules" | Individual transformers such as `TextCleaner`, `DuplicateHandler`, `OutlierHandler` |
| "I need a reusable pipeline" | Configure `AutomatedCleaner` once and keep the returned report |
| "The CSV is large and slow" | `convert_to_parquet_optimized()` or `convert_to_parquet()` |

---


## Quick Start

```python
from cleanflow import AutomatedCleaner, load_dataset, optimize_dataset

df = load_dataset("data/messy_data.csv")  # pandas by default
cleaner = AutomatedCleaner()
cleaned_df, report = cleaner.clean(df)
optimized_df = optimize_dataset(cleaned_df)

print(f"Completeness: {report['initial_quality']['completeness_percentage']}% → "
      f"{report['final_quality']['completeness_percentage']}%")
print(f"Data loss: {report['validation']['data_loss_pct']}%")
```

For a tiny project, this may be enough. For a production or portfolio project, keep the `report` next to your output dataset so reviewers can understand which rows, columns, and values changed.

### Profile and Optimize Only

Use these when your data is already clean enough and you just want the old `data-optimizer` workflow inside CleanFlow:

```python
from cleanflow import load_dataset, profile_dataframe, analyze_optimization, apply_optimization

df = load_dataset("large_data.csv")
profile = profile_dataframe(df)
print(profile["rows"], profile["memory_mb"])

recommendations = analyze_optimization(df)
df_optimized = apply_optimization(df, recommendations)
```

### Convert CSV to Parquet

```python
from cleanflow import convert_to_parquet_optimized

convert_to_parquet_optimized(
    "large_data.csv",
    "large_data.parquet",
    compression="zstd",
    engine="pandas",
)
```

For large files and fastest out-of-core conversion, install DuckDB and use:

```python
from cleanflow import convert_to_parquet

convert_to_parquet("large_data.csv", "large_data.parquet")
```

### CLI

```bash
cleanflow-profile data.csv
cleanflow-optimize data.csv
cleanflow-optimize data.csv --to-parquet data.parquet --optimize --compression zstd
```

The former `data-optimize` and `data-profile` ideas now live here as `cleanflow-optimize` and `cleanflow-profile`.

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

## Practical Workflows

### Clean, Optimize, Save

Use this when the source data is messy but still small enough to load into memory.

```python
from cleanflow import AutomatedCleaner, load_dataset, optimize_dataset

df = load_dataset("data/sales_data.csv")

cleaner = AutomatedCleaner(
    duplicate_subset=["Product_ID"],
    outlier_columns=["Price", "Quantity_Sold"],
    outlier_strategy="cap",
)
cleaned_df, report = cleaner.clean(df)
optimized_df = optimize_dataset(cleaned_df)

optimized_df.to_parquet("data/sales_cleaned.parquet", compression="zstd")

print(f"Rows: {report['validation']['rows_before']} -> {report['validation']['rows_after']}")
print(f"Missing: {report['validation']['missing_before']} -> {report['validation']['missing_after']}")
```

### Audit Before Cleaning

Use this when you need to explain data quality before making changes.

```python
from cleanflow import check_quality, dataset_overview, load_dataset, profile_dataframe

df = load_dataset("data/users_02.csv")

quality = check_quality(df)
profile = profile_dataframe(df)
overview = dataset_overview(df)

print(f"Completeness: {quality['completeness_percentage']}%")
print(f"Memory: {profile['memory_mb']} MB")
print(overview[["column", "dtype", "missing_pct", "unique_count"]])
```

### Reviewable Optimization

Use this when dtype changes need to be visible before they are applied.

```python
from cleanflow import analyze_optimization, apply_optimization, optimization_report

recommendations = analyze_optimization(df, verbose=True)
optimized_df = apply_optimization(df, recommendations)

before_mb, after_mb, reduction = optimization_report(df, optimized_df, recommendations)
print(f"Memory reduction: {reduction:.1f}%")
```

### Batch Processing

Use this when several recurring CSV files need the same treatment.

```python
from pathlib import Path
from cleanflow import AutomatedCleaner, load_dataset, optimize_dataset

cleaner = AutomatedCleaner(outlier_strategy="cap")

for csv_file in Path("data/raw").glob("*.csv"):
    df = load_dataset(csv_file)
    cleaned_df, report = cleaner.clean(df)
    optimized_df = optimize_dataset(cleaned_df)

    output = Path("data/clean") / csv_file.with_suffix(".parquet").name
    optimized_df.to_parquet(output, compression="zstd")

    print(
        f"{csv_file.name}: "
        f"{report['initial_quality']['completeness_percentage']}% -> "
        f"{report['final_quality']['completeness_percentage']}% complete"
    )
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
│   ├── io.py                # CSV/Parquet loading helpers
│   ├── profiling.py         # Dataset profiling helpers
│   ├── optimization/        # Memory optimization and Parquet conversion
│   ├── cli.py               # cleanflow-profile and cleanflow-optimize entry points
│   └── pipeline.py          # AutomatedCleaner orchestrator
├── docs/
│   └── USE_CASES.md         # Practical scenario guide
├── examples/
│   ├── 01_basic_cleaning.py
│   ├── 02_custom_configuration.py
│   ├── 03_quality_analysis.py
│   ├── 04_optimization_workflow.py
│   ├── 05_feature_engineering.py
│   └── 06_advanced_features.py
├── tests/
│   ├── test_cleaning.py
│   ├── test_features.py
│   └── test_optimization.py
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
