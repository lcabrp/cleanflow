"""Tests for CleanFlow optimization and profiling utilities."""

import pandas as pd

from cleanflow import (
    analyze_optimization,
    apply_optimization,
    dataset_overview,
    fill_na,
    load_dataset,
    optimize_dataset,
    profile_dataframe,
)


def test_pandas_optimization_reduces_memory():
    df = pd.DataFrame(
        {
            "small_int": [1, 2, 3, 4],
            "small_float": [1.0, 2.0, 3.0, 4.0],
            "category_like": ["east", "west", "east", "west"],
        }
    )

    optimized = optimize_dataset(df)

    assert str(optimized["small_int"].dtype) in {"int8", "int16", "int32"}
    assert str(optimized["small_float"].dtype) == "float32"
    assert str(optimized["category_like"].dtype) == "category"


def test_analyze_apply_optimization_workflow():
    df = pd.DataFrame({"id": [1, 2, 3], "region": ["A", "A", "B"]})

    recommendations = analyze_optimization(df)
    optimized = apply_optimization(df, recommendations)

    assert "uint8" in recommendations
    assert "category" in recommendations
    assert str(optimized["id"].dtype) == "uint8"
    assert str(optimized["region"].dtype) == "category"


def test_profile_and_overview():
    df = pd.DataFrame({"x": [1, None, 3], "y": ["a", "b", None]})

    profile = profile_dataframe(df)
    overview = dataset_overview(df)

    assert profile["rows"] == 3
    assert profile["columns"] == 2
    assert profile["missing_values"]["x"] == 1
    assert set(overview.columns) == {"column", "dtype", "non_nulls", "nulls", "missing_pct", "unique_count", "memory_mb"}
    assert overview.loc["x", "missing_pct"] == 33.33
    assert overview.loc["y", "unique_count"] == 3


def test_simple_cleaning_compatibility_helpers():
    df = pd.DataFrame({"x": [1, None], "label": ["ok", None]})

    filled = fill_na(df, numeric_fill=-1, categorical_fill="missing")

    assert filled["x"].iloc[1] == -1
    assert filled["label"].iloc[1] == "missing"


def test_load_dataset_csv_defaults_to_pandas(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(path, index=False)

    df = load_dataset(path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 1)
