# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-05-17

Merged the useful ideas from the former `data-optimizer` project into CleanFlow.

### Added
- File loading helpers: `load_dataset()` and `estimate_csv_memory()`.
- Profiling helpers: `profile_dataframe()` and `dataset_overview()`.
- Memory optimization helpers: `optimize_dataset()`, `analyze_optimization()`, `apply_optimization()`, and `optimization_report()`.
- Optional backend support for Polars, Dask, DuckDB, and PyArrow/Parquet.
- CLI entry points: `cleanflow-optimize` and `cleanflow-profile`.
- Compatibility helpers from data-optimizer: `drop_na()` and `fill_na()`.
- Tests for the merged optimization/profile/load workflow.

### Changed
- CleanFlow is now positioned as a data cleaning, quality, profiling, and optimization library.
- Packaging metadata now consistently targets Python 3.9+.
- Optional heavy dependencies moved behind extras instead of being required for basic cleaning.
- Documentation now includes problem framing, practical use cases, and current CleanFlow-only optimization workflows.
- `dataset_overview()` now includes explicit column names, missing percentages, and unique counts for easier audits.

## [0.2.0] - 2026-02-09

Complete refactoring into modular, composable transformers with a unified pipeline.

### Added
- **Base class hierarchy** (`base.py`)
  - `BaseTransformer` ABC with `transform()` / `get_report()` interface
  - `FitTransformer` ABC with `fit()` / `fit_transform()` for stateful transformers
- **DataStandardizer** (`transformers.py`)
  - Auto-detection of booleans, percentages, numerics (with currency symbols), and dates
  - 12 date format patterns supported
  - Currency symbol stripping (`$`, `€`, `£`, `¥`, `,`)
  - Configurable error coercion
- **MissingValueHandler** (`transformers.py`)
  - 10+ imputation strategies: mean, median, mode, zero, unknown, interpolate, ffill, bfill, drop_rows, drop_column, flag_missing
  - Auto-recommendation based on data type and distribution skewness
  - Configurable drop threshold (default 70%)
  - Fit/transform separation for train/test workflows
  - `analyze()` method for pre-imputation analysis
- **DuplicateHandler** (`duplicates.py`)
  - Exact matching on specified column subsets
  - Fuzzy matching using SequenceMatcher string similarity
  - Blocking column support for performance on large datasets
  - Survivorship rules: first, last, most_complete, merge
- **OutlierHandler** (`outliers.py`)
  - Detection: IQR, Z-score, Modified Z-score (MAD-based), Percentile
  - Treatment: cap/winsorize, remove, flag, impute_mean, impute_median
  - Configurable threshold and percentile range
- **TextCleaner** (`text.py`)
  - 6 type-specific pipelines: names, addresses, descriptions, codes, emails, phones
  - Address abbreviation expansion (st→street, ave→avenue, etc.)
  - HTML tag removal and entity decoding
  - URL and email removal from free text
  - Unicode normalization and accent stripping
  - Phone number extraction and reformatting
  - `clean_all()` batch method using column type mapping
- **CategoryStandardizer** (`categories.py`)
  - Value mapping with chaining API
  - Category grouping (merge related categories)
  - Rare category handling (threshold-based grouping to "Other")
  - Case normalization option
- **Quality utilities** (`quality.py`)
  - `check_quality()` — comprehensive DataFrame health report
  - `quality_score()` — per-row completeness scoring (0–10 scale)
  - `detect_suspicious()` — flag suspicious numeric values (round values, out-of-range)
  - `add_missing_indicators()` — binary missing-value columns
- **AutomatedCleaner** (`pipeline.py`)
  - 8-step cleaning pipeline with configurable step skipping
  - Initial/final quality comparison with validation report
  - Transformer injection (pass pre-configured instances)
  - Convenience parameters for common configurations
- **Comprehensive test suite** (`tests/test_cleaning.py`)
  - 30+ tests covering all transformers and pipeline
  - End-to-end test using sample CSV data
  - Fixtures for messy data, duplicates, and outliers
- **Documentation**
  - `README.md` — full project overview with examples
  - `TECHNICAL.md` — architecture and algorithm documentation
  - `CHANGELOG.md` — this file

### Changed
- Project restructured from flat scripts to modular package under `cleanflow/`
- All transformers follow consistent **analyze → transform → report** pattern
- All `transform()` methods use copy-on-write (never mutate input)

## [0.1.0] - Initial Release

### Added
- Original data cleaning scripts (flat structure)
- Basic missing value handling
- Simple duplicate removal
- Sample messy data CSV files
