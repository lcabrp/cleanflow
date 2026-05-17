"""File loading utilities for tabular datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .logging_utils import log


def load_csv(path: str | Path, chunksize: int | None = None, logger: logging.Logger | None = None) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Load CSV with pandas.

    A chunksize returns an iterator, which keeps large-file workflows explicit.
    """
    log(logger, f"Loading CSV with pandas: {path}")
    if chunksize:
        return pd.read_csv(path, chunksize=chunksize)
    return pd.read_csv(path)


def load_parquet(path: str | Path, logger: logging.Logger | None = None) -> pd.DataFrame:
    """Load Parquet with pandas."""
    log(logger, f"Loading Parquet with pandas: {path}")
    return pd.read_parquet(path)


def load_csv_polars(path: str | Path, logger: logging.Logger | None = None) -> Any:
    """Load CSV with Polars when the optional dependency is installed."""
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("Polars support is optional. Install with: pip install 'cleanflow[polars]'") from exc

    log(logger, f"Loading CSV with Polars: {path}")
    return pl.read_csv(path)


def load_parquet_polars(path: str | Path, logger: logging.Logger | None = None) -> Any:
    """Load Parquet with Polars when the optional dependency is installed."""
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("Polars support is optional. Install with: pip install 'cleanflow[polars]'") from exc

    log(logger, f"Loading Parquet with Polars: {path}")
    return pl.read_parquet(path)


def load_dataset(path: str | Path, engine: str = "pandas", enable_logging: bool = False) -> Any:
    """Load a CSV or Parquet file with pandas or Polars.

    Pandas is the default because it is a core CleanFlow dependency. Polars stays
    an explicit opt-in backend for users who want speed and have the extra
    installed.
    """
    from .logging_utils import get_logger

    logger = get_logger(enabled=enable_logging)
    path = Path(path)
    suffix = path.suffix.lower()
    if engine not in {"pandas", "polars"}:
        raise ValueError("engine must be 'pandas' or 'polars'")

    if suffix == ".csv":
        return load_csv_polars(path, logger) if engine == "polars" else load_csv(path, logger=logger)
    if suffix == ".parquet":
        return load_parquet_polars(path, logger) if engine == "polars" else load_parquet(path, logger)
    raise ValueError(f"Unsupported file type: {suffix}")


def estimate_csv_memory(path: str | Path, sample_rows: int = 50_000) -> float:
    """Estimate full CSV memory usage in MB from a row sample."""
    path = Path(path)
    sample = pd.read_csv(path, nrows=sample_rows)
    if sample.empty:
        return 0.0
    sample_mem = sample.memory_usage(deep=True).sum()
    estimated_rows = int(path.stat().st_size / max(sample_mem / len(sample), 1))
    return (sample_mem / len(sample)) * estimated_rows / 1024**2

