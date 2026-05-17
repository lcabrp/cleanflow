# CleanFlow Use Cases

This guide explains when CleanFlow is a good fit, what problems it can solve, and how to start with either a small script or a more deliberate project workflow.

CleanFlow works best with tabular datasets where the cleaning rules can be reviewed: CSV exports, spreadsheet-derived data, pandas DataFrames, dashboard inputs, analytics tables, and machine learning training data.

---

## When to Use CleanFlow

Use CleanFlow when you need to:

- Quickly understand the quality of an unfamiliar dataset.
- Standardize messy column types before analysis.
- Reduce repeated hand-cleaning in notebooks.
- Build a repeatable cleanup step for CSV files.
- Document what changed during cleaning.
- Prepare data for dashboards or machine learning.
- Reduce DataFrame memory usage before heavier processing.
- Convert frequently reused CSV files to Parquet.

Be careful when:

- The dataset has strict legal, medical, financial, or operational rules that must be enforced by domain-specific validation.
- Every imputation or outlier change needs manual approval.
- The data is unstructured text, nested JSON, images, logs, or streaming data.
- The main problem is schema enforcement in a database rather than DataFrame cleanup.

CleanFlow can still help in those projects, but it should sit beside domain validation instead of replacing it.

---

## Scenario 1: Quick Cleanup for a Notebook

Use this when you downloaded or received a CSV and want to start analysis without writing one-off cleaning code.

```python
from cleanflow import AutomatedCleaner, check_quality, load_dataset

df = load_dataset("data/customer_export.csv")
print(check_quality(df))

cleaner = AutomatedCleaner()
cleaned_df, report = cleaner.clean(df)

print(report["validation"])
print(cleaned_df.head())
```

Why this helps:

- Converts obvious string dates, booleans, percentages, and numeric text.
- Handles missing values with consistent defaults.
- Gives you a before/after quality report.
- Keeps the original DataFrame untouched.

---

## Scenario 2: Sales Dashboard Refresh

Use this when a dashboard depends on recurring CSV exports from a CRM, ERP, WMS, or e-commerce platform.

```python
from cleanflow import AutomatedCleaner, CategoryStandardizer, load_dataset, optimize_dataset

regions = CategoryStandardizer(
    mappings={
        "region": {
            "n": "North",
            "north": "North",
            "s": "South",
            "south": "South",
            "east ": "East",
            "west ": "West",
        }
    },
    normalize_case=True,
)

df = load_dataset("data/raw/sales_export.csv")

cleaner = AutomatedCleaner(
    category_standardizer=regions,
    duplicate_subset=["order_id"],
    duplicate_survivorship="most_complete",
    outlier_columns=["sales", "quantity"],
    outlier_strategy="cap",
)

cleaned_df, report = cleaner.clean(df)
optimized_df = optimize_dataset(cleaned_df)

optimized_df.to_parquet("data/curated/sales.parquet", compression="zstd")
print(report["validation"])
```

Problems addressed:

- Duplicate order rows.
- Inconsistent category labels.
- Extreme numeric values that break charts.
- Missing values that create dashboard blanks.
- Large CSVs that are slow to reload repeatedly.

---

## Scenario 3: Data Quality Audit Before Cleaning

Use this when you need to understand and explain the dataset before making changes.

```python
from cleanflow import check_quality, dataset_overview, load_dataset, profile_dataframe

df = load_dataset("data/vendor_sample.csv")

quality = check_quality(df)
profile = profile_dataframe(df)
overview = dataset_overview(df)

print(f"Rows: {quality['total_rows']}")
print(f"Columns: {quality['total_columns']}")
print(f"Completeness: {quality['completeness_percentage']}%")
print(f"Duplicates: {quality['duplicate_rows']}")
print(f"Memory: {profile['memory_mb']} MB")

print(overview.sort_values("missing_pct", ascending=False).head(10))
```

Problems addressed:

- Unknown missingness.
- Duplicate records.
- Suspicious dtype choices.
- High-cardinality or mostly-empty columns.
- Early decision-making before deeper analysis.

---

## Scenario 4: Machine Learning Preprocessing

Use this when the goal is to create a cleaner modeling table while keeping preprocessing choices visible.

```python
from cleanflow import AutomatedCleaner, FeatureSelector, MissingValueHandler, load_dataset

df = load_dataset("data/training.csv")

cleaner = AutomatedCleaner(
    missing_handler=MissingValueHandler(default_numeric="median", default_categorical="mode"),
    add_missing_indicators=True,
    auto_transform_numerics=True,
    scale_method="standard",
    extract_date_features=["year", "month", "weekday", "is_weekend"],
    drop_low_variance=True,
    drop_correlated=True,
    feature_selector=FeatureSelector(correlation_threshold=0.95),
)

model_df, report = cleaner.clean(df)
print(report.keys())
```

Problems addressed:

- Missing values in model features.
- Skewed numeric distributions.
- Date columns that need useful derived fields.
- Constant or highly correlated columns.
- Need for a single report that describes preprocessing.

For train/test workflows, prefer fitting individual stateful transformers on the training split and applying them to the test split. That avoids leakage and keeps model evaluation honest.

---

## Scenario 5: Reviewable Memory Optimization

Use this when dtype changes should be inspected before they are applied.

```python
from cleanflow import analyze_optimization, apply_optimization, load_dataset, optimization_report

df = load_dataset("data/events.csv")

recommendations = analyze_optimization(df, verbose=True)
print(recommendations)

optimized_df = apply_optimization(df, recommendations)
optimization_report(df, optimized_df, recommendations)
```

Problems addressed:

- Large DataFrames that consume more memory than needed.
- Integer columns stored as overly large dtypes.
- Float columns that can safely downcast.
- Repeated string labels that should become `category`.

---

## Scenario 6: Large CSV to Parquet

Use this when a CSV is reused often and read performance matters.

```python
from cleanflow import convert_to_parquet_optimized

convert_to_parquet_optimized(
    "data/raw/events.csv",
    "data/curated/events.parquet",
    compression="zstd",
    chunksize=500_000,
)
```

If DuckDB is installed, use the out-of-core path:

```python
from cleanflow import convert_to_parquet

convert_to_parquet("data/raw/events.csv", "data/curated/events.parquet")
```

Problems addressed:

- Slow repeated CSV reads.
- Oversized intermediate files.
- Data that is too large to comfortably rewrite in one pandas operation.

---

## Recommended Project Pattern

For reusable analytics or portfolio projects, keep these artifacts:

```text
project/
├── data/
│   ├── raw/
│   └── curated/
├── notebooks/
├── scripts/
│   └── clean_data.py
└── reports/
    └── cleaning_report.json
```

The important habit is to save both the cleaned output and the report. The output helps downstream consumers; the report helps your future self understand what the pipeline did.

```python
import json
from pathlib import Path

from cleanflow import AutomatedCleaner, load_dataset, optimize_dataset

raw_path = Path("data/raw/input.csv")
curated_path = Path("data/curated/input.parquet")
report_path = Path("reports/cleaning_report.json")

df = load_dataset(raw_path)
cleaned_df, report = AutomatedCleaner().clean(df)
optimized_df = optimize_dataset(cleaned_df)

curated_path.parent.mkdir(parents=True, exist_ok=True)
report_path.parent.mkdir(parents=True, exist_ok=True)

optimized_df.to_parquet(curated_path, compression="zstd")
report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
```

---

## Decision Guide

| Situation | Recommended approach |
|---|---|
| Need a fast health check | `check_quality()` and `profile_dataframe()` |
| Need a one-line cleanup | `AutomatedCleaner().clean(df)` |
| Need strict control | Configure individual transformers |
| Need explainability | Use reports from each transformer and the final `validation` block |
| Need memory savings | `analyze_optimization()` then `apply_optimization()` |
| Need recurring file processing | Put `load_dataset()` + `AutomatedCleaner` + save step in a script |
| Need very large file conversion | Use `convert_to_parquet()` with DuckDB or chunked `convert_to_parquet_optimized()` |

