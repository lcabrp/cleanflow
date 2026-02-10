# Technical Documentation

In-depth technical details about the internal architecture and algorithms of the CleanFlow library.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Base Classes](#base-classes)
3. [Data Type Standardization](#data-type-standardization)
4. [Missing Value Handling](#missing-value-handling)
5. [Duplicate Detection](#duplicate-detection)
6. [Outlier Detection](#outlier-detection)
7. [Text Cleaning](#text-cleaning)
8. [Category Standardization](#category-standardization)
9. [Quality Analysis](#quality-analysis)
10. [Pipeline Orchestration](#pipeline-orchestration)
11. [Design Decisions](#design-decisions)

---

## Architecture Overview

### Module Structure

```
cleanflow/
├── base.py              # Abstract base classes (BaseTransformer, FitTransformer)
├── transformers.py      # DataStandardizer, MissingValueHandler
├── duplicates.py        # DuplicateHandler (exact + fuzzy matching)
├── outliers.py          # OutlierHandler (4 detection methods, 6 treatments)
├── text.py              # TextCleaner (6 type-specific pipelines)
├── categories.py        # CategoryStandardizer (mapping, grouping, rare handling)
├── quality.py           # Standalone quality analysis functions
├── pipeline.py          # AutomatedCleaner (orchestrates all transformers)
└── __init__.py          # Public API exports
```

### Data Flow

```
Input DataFrame
  ↓
AutomatedCleaner.clean()
  ↓
┌──────────────────────────────────────────────────────┐
│ 1. check_quality()           → initial_quality       │
│ 2. DataStandardizer          → type_conversions      │
│ 3. TextCleaner               → text_cleaning         │
│ 4. CategoryStandardizer      → category_standardization│
│ 5. DuplicateHandler          → duplicate_handling     │
│ 6. OutlierHandler            → outlier_handling       │
│ 7. MissingValueHandler       → missing_value_handling │
│ 8. check_quality()           → final_quality          │
└──────────────────────────────────────────────────────┘
  ↓
(cleaned_df, report_dict)
```

---

## Base Classes

**File:** [`base.py`](cleanflow/base.py)

All transformers extend one of two abstract base classes:

### BaseTransformer

For stateless transformers that don't need to learn from data.

```python
class BaseTransformer(ABC):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...  # required
    def get_report(self) -> pd.DataFrame: ...                    # optional override
```

**Used by:** `DataStandardizer`, `DuplicateHandler`, `OutlierHandler`, `TextCleaner`, `CategoryStandardizer`

### FitTransformer

For transformers that learn parameters before transforming (sklearn-style).

```python
class FitTransformer(BaseTransformer):
    def fit(self, df: pd.DataFrame) -> "FitTransformer": ...     # required
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:   # convenience
        self.fit(df)
        return self.transform(df)
```

**Used by:** `MissingValueHandler` — learns imputation values (medians, modes) during `fit()`, then applies them in `transform()`.

### Why Two Base Classes?

- **BaseTransformer** — each call to `transform()` is self-contained (e.g., outlier bounds are computed fresh)
- **FitTransformer** — learned values persist across calls, enabling train/test split workflows where you fit on training data and transform test data with the same parameters

---

## Data Type Standardization

**File:** [`transformers.py`](cleanflow/transformers.py) · **Class:** `DataStandardizer`

### Detection Algorithm

For each string/object column, a sample of up to 50 non-null values is examined. Detection is attempted in priority order:

| Priority | Type | Detection Signal | Threshold |
|----------|------|-----------------|-----------|
| 1 | Boolean | Values match `{true, false, yes, no, 1, 0, t, f, y, n}` | >80% match |
| 2 | Percentage | Matches pattern `^-?\d+\.?\d*\s*%$` | >50% match |
| 3 | Numeric | After stripping `$,€£¥,` — parseable as float | >50% match |
| 4 | Date | Parseable by any of 12 date formats | >50% match |

### Supported Date Formats

```
%Y-%m-%d       %d-%m-%Y       %m-%d-%Y
%Y/%m/%d       %d/%m/%Y       %m/%d/%Y
%Y-%m-%d %H:%M:%S             %d-%m-%Y %H:%M:%S
%b %d, %Y      %B %d, %Y     %d %b %Y       %d %B %Y
```

### Conversion Report

Each conversion produces a `ConversionResult` dataclass:

| Field | Description |
|-------|-------------|
| `column` | Column name |
| `original_dtype` | Original pandas dtype |
| `new_dtype` | Converted dtype (e.g., `float64`, `boolean`, `datetime64[ns]`) |
| `success_count` | Values successfully converted |
| `failed_count` | Values that couldn't be converted (coerced to NaN) |
| `failed_values` | Sample of up to 5 failed values for debugging |

---

## Missing Value Handling

**File:** [`transformers.py`](cleanflow/transformers.py) · **Class:** `MissingValueHandler`

### Strategy Selection Logic

```
Column has missing values?
  ↓ yes
User specified strategy for this column?
  ↓ no
Missing % > drop_threshold (default 70%)?
  → yes: drop_column
  ↓ no
Is numeric?
  → yes: use default_numeric (default: "median")
       Note: if skewness > 1, auto-recommends median; else mean
Is datetime?
  → yes: interpolate
Else (categorical):
  → use default_categorical (default: "mode")
```

### Imputation Strategies

| Strategy | Applies To | Behavior |
|----------|-----------|----------|
| `mean` | Numeric | Fill with column mean |
| `median` | Numeric | Fill with column median (robust to skew) |
| `mode` | Any | Fill with most frequent value |
| `zero` | Numeric | Fill with 0 |
| `unknown` | Categorical | Fill with `"UNKNOWN"` |
| `interpolate` | Numeric/DateTime | Linear interpolation between neighbors |
| `ffill` | Any | Forward fill (propagate last valid value) |
| `bfill` | Any | Backward fill (propagate next valid value) |
| `drop_rows` | Any | Remove rows with missing values |
| `drop_column` | Any | Remove entire column |
| `flag_missing` | Any | Add `{col}_is_missing` binary indicator |

### Fit/Transform Separation

- **`fit()`** — scans each column, picks strategy, pre-computes fill values (mean, median, mode)
- **`transform()`** — applies pre-computed values; fails if not fitted
- **`fit_transform()`** — convenience: `fit()` then `transform()`

This separation enables fitting on training data and applying the same imputation to test data.

---

## Duplicate Detection

**File:** [`duplicates.py`](cleanflow/duplicates.py) · **Class:** `DuplicateHandler`

### Exact Matching

Uses `pd.DataFrame.duplicated(subset, keep=False)` to find all rows where the specified columns have identical values. Groups are formed by grouping on the subset columns.

**Complexity:** O(n log n) — pandas uses hash-based groupby.

### Fuzzy Matching

Pairwise comparison using `difflib.SequenceMatcher`:

```
For each pair of rows (i, j):
    similarity = average(SequenceMatcher(row_i[col], row_j[col]) for col in match_columns)
    if similarity >= threshold:
        group together
```

**Blocking optimization:** When `blocking_column` is provided, only rows sharing the same blocking value are compared. This reduces comparisons from O(n²) to O(n × b) where b = max block size.

**Complexity:** O(n²) without blocking, O(n × b) with blocking.

### Survivorship Rules

| Rule | Behavior |
|------|----------|
| `first` | Keep the first occurrence |
| `last` | Keep the last occurrence |
| `most_complete` | Keep the row with the fewest NaN values |
| `merge` | Start with most complete row, fill remaining NaNs from other duplicates |

---

## Outlier Detection

**File:** [`outliers.py`](cleanflow/outliers.py) · **Class:** `OutlierHandler`

### Detection Methods

#### IQR (Interquartile Range)
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Lower = Q1 - threshold × IQR
Upper = Q3 + threshold × IQR
Outlier if: value < Lower OR value > Upper
```
Default threshold: **1.5** (standard). Use 3.0 for extreme outliers only.

#### Z-Score
```
z = (value - mean) / std
Outlier if: |z| > threshold
```
Default threshold: **3.0**. Assumes normal distribution.

#### Modified Z-Score (MAD-based)
```
MAD = median(|values - median|)
modified_z = 0.6745 × (value - median) / MAD
Outlier if: |modified_z| > threshold
```
Default threshold: **3.5**. Robust to non-normal distributions and existing outliers.

#### Percentile
```
Lower = quantile(lower_pct)
Upper = quantile(upper_pct)
Outlier if: value < Lower OR value > Upper
```
Default range: **(0.01, 0.99)**. No distributional assumptions.

### Treatment Strategies

| Strategy | Behavior | Data Loss |
|----------|----------|-----------|
| `cap` / `winsorize` | Clip values to [lower, upper] bounds | None |
| `remove` | Drop outlier rows | Yes |
| `flag` | Add `{col}_is_outlier` binary column | None |
| `impute_mean` | Replace outliers with column mean (excluding outliers) | None |
| `impute_median` | Replace outliers with column median (excluding outliers) | None |

---

## Text Cleaning

**File:** [`text.py`](cleanflow/text.py) · **Class:** `TextCleaner`

### Built-in Pipelines

| Pipeline | Operations | Use Case |
|----------|-----------|----------|
| `clean_names` | strip → title case → remove accents | Person names |
| `clean_addresses` | strip → lowercase → expand abbreviations | Physical addresses |
| `clean_descriptions` | strip → remove HTML → remove URLs → lowercase | Free text, bios |
| `clean_codes` | strip → uppercase → remove spaces | SKUs, IDs, codes |
| `clean_emails` | strip → lowercase → validate format | Email addresses |
| `clean_phones` | extract digits → format as `(xxx) xxx-xxxx` | Phone numbers |

### Address Abbreviation Expansion

```
st. → street       ave. → avenue      blvd. → boulevard
dr. → drive        ln. → lane         rd. → road
ct. → court        pl. → place        apt. → apartment
```

### Static Operations (Composable)

All available as static methods for custom pipelines:

- `strip_whitespace()` — normalize internal spaces, trim edges
- `lowercase()` / `uppercase()` / `titlecase()`
- `remove_html()` — strip tags, decode entities
- `remove_special_chars(keep_chars="")` — whitelist approach
- `remove_urls()` / `remove_emails_from_text()`
- `normalize_unicode()` — NFKD normalization
- `remove_accents()` — strip combining characters

---

## Category Standardization

**File:** [`categories.py`](cleanflow/categories.py) · **Class:** `CategoryStandardizer`

### Processing Order

1. **Case normalization** (if `normalize_case=True`) — lowercase all string columns
2. **Value mapping** — apply `{old_value: new_value}` replacements per column
3. **Category grouping** — apply `{old_category: group_name}` replacements per column
4. **Rare category handling** (if `rare_threshold` set) — replace categories with count < threshold with `rare_label` (default `"Other"`)

### Key Design Note

If `normalize_case=True`, mapping keys are also lowercased before matching. This ensures mappings work regardless of input case.

---

## Quality Analysis

**File:** [`quality.py`](cleanflow/quality.py)

### `check_quality(df)` → dict

Returns a comprehensive health report:

| Key | Type | Description |
|-----|------|-------------|
| `total_rows` | int | Number of rows |
| `total_columns` | int | Number of columns |
| `missing_values` | dict | Per-column missing counts |
| `total_missing_values` | int | Total missing cells |
| `duplicate_rows` | int | Number of exact duplicate rows |
| `memory_usage_mb` | float | Deep memory usage in MB |
| `completeness_percentage` | float | `(total_cells - missing) / total_cells × 100` |
| `column_completeness` | dict | Per-column completeness % |
| `dtypes` | dict | Per-column dtype as string |

### `quality_score(df)` → DataFrame

Adds two columns:
- **`quality_score`** — `(non_null_values / total_columns) × 10` per row
- **`quality_category`** — `Poor` (0–4), `Average` (4–7), `Good` (7–10)

### `detect_suspicious(df)` → DataFrame

Flags numeric columns:
- **Income/salary/price columns** — flags values divisible by `round_modulus` (default 10,000) as suspiciously round
- **Age columns** — flags values outside `age_range` (default 0–120)
- **Rating columns** — flags values outside `rating_range` (default 1–5)

### `add_missing_indicators(df)` → DataFrame

Adds `{col}_is_missing` binary columns (0/1) for specified columns. Useful when the missingness pattern itself is informative for modeling.

---

## Pipeline Orchestration

**File:** [`pipeline.py`](cleanflow/pipeline.py) · **Class:** `AutomatedCleaner`

### Step Execution

Each step is conditional — it only runs if properly configured:

| Step | Runs When | Skip Key |
|------|-----------|----------|
| Type standardization | Always (unless skipped) | `"standardize"` |
| Text cleaning | `column_types` is provided | `"text"` |
| Category standardization | `category_standardizer` is provided | `"categories"` |
| Duplicate removal | `duplicate_subset` is provided | `"duplicates"` |
| Outlier handling | `outlier_columns` is provided | `"outliers"` |
| Missing value handling | Always (unless skipped) | `"missing"` |

### Report Structure

The returned `report` dict contains:

```python
{
    "initial_quality": { ... },          # check_quality() before cleaning
    "type_conversions": [ ... ],         # DataStandardizer report rows
    "text_cleaning": [ ... ],            # TextCleaner report rows
    "category_standardization": [ ... ], # CategoryStandardizer report rows
    "duplicate_handling": [ ... ],       # DuplicateHandler report rows
    "outlier_handling": [ ... ],         # OutlierHandler report rows
    "missing_value_handling": [ ... ],   # MissingValueHandler report rows
    "final_quality": { ... },            # check_quality() after cleaning
    "validation": {
        "rows_before": int,
        "rows_after": int,
        "columns_before": int,
        "columns_after": int,
        "data_loss_pct": float,
        "missing_before": int,
        "missing_after": int,
    }
}
```

Keys are only present if the corresponding step ran and produced non-empty results.

### Transformer Injection

All transformers can be passed as pre-configured instances:

```python
cleaner = AutomatedCleaner(
    standardizer=DataStandardizer(auto_detect=True, coerce_errors=False),
    missing_handler=MissingValueHandler(default_numeric="mean", drop_threshold=0.5),
    outlier_handler=OutlierHandler(),
    # ... etc
)
```

If not provided, defaults are instantiated with default parameters.

---

## Design Decisions

### 1. Analyze → Transform → Report

Every transformer produces a structured report of what it did. This enables:
- **Auditability** — know exactly what changed and why
- **Pipeline debugging** — inspect intermediate reports
- **Quality metrics** — before/after comparison via `check_quality()`

### 2. Copy-on-Write

All `transform()` methods start with `df = df.copy()`. The original DataFrame is never mutated. This prevents subtle bugs from in-place modification and enables safe pipeline composition.

### 3. Dataclass Reports

Each transformer uses `@dataclass` for internal report records (e.g., `ConversionResult`, `MissingReport`, `OutlierReport`). These provide type safety and clean serialization to DataFrames via `get_report()`.

### 4. Pipeline Order

The ordering is intentional:
1. **Standardize first** — fix types so downstream steps (outliers, imputation) can work on proper numerics
2. **Text/categories before duplicates** — normalize text so exact matching catches more true duplicates
3. **Duplicates before outliers** — remove duplicate rows before statistical analysis
4. **Outliers before missing values** — cap extreme values before computing medians/means for imputation
5. **Missing values last** — all other cleaning is done, impute what remains

### 5. Pandas-Native

CleanFlow operates on standard `pd.DataFrame` objects with no custom wrappers. This ensures compatibility with the entire pandas ecosystem and downstream tools like data_optimizer.

---

**Last Updated:** February 9, 2026
