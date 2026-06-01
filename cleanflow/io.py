"""File loading utilities for tabular datasets."""

from __future__ import annotations

import logging
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .logging_utils import log


def csv_read_kwargs_for_types(type_map: dict[str, list[str]] | None) -> dict[str, Any]:
    """Build pandas read_csv kwargs from a dtype mapping dictionary.

    Extracts datetime columns into 'parse_dates' and maps the remaining columns
    to their target pandas/numpy dtypes for read-time injection.
    """
    if not type_map:
        return {}
    
    dtype: dict[str, str] = {}
    parse_dates: list[str] = []
    
    for target_dtype, columns in type_map.items():
        if target_dtype == "datetime64[ns]":
            parse_dates.extend(columns)
        else:
            for col in columns:
                dtype[col] = target_dtype

    kwargs: dict[str, Any] = {}
    if dtype:
        kwargs["dtype"] = dtype
    if parse_dates:
        kwargs["parse_dates"] = parse_dates
    return kwargs


def _generate_cache_key(path: Path, type_map: dict[str, list[str]] | None) -> str:
    """Generate a unique cache key based on file metadata and type mapping."""
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "type_map": type_map,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def load_csv(
    path: str | Path,
    chunksize: int | None = None,
    type_map: dict[str, list[str]] | None = None,
    use_dtype_hints: bool = True,
    logger: logging.Logger | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Load CSV with pandas, optionally using dtype hints for speed and memory efficiency.

    A chunksize returns an iterator, which keeps large-file workflows explicit.
    If pyarrow is installed, engine='pyarrow' is utilized for high-performance parsing.
    """
    log(logger, f"Loading CSV with pandas: {path}")
    
    kwargs: dict[str, Any] = {}
    if use_dtype_hints and type_map:
        kwargs = csv_read_kwargs_for_types(type_map)

    # Check if pyarrow is installed for optimized loading
    pyarrow_available = False
    try:
        import pyarrow  # noqa: F401
        pyarrow_available = True
    except ImportError:
        pass

    if pyarrow_available and kwargs:
        kwargs["engine"] = "pyarrow"
        kwargs["dtype_backend"] = "pyarrow"
        log(logger, "PyArrow parser and dtype backend enabled for read_csv")

    if chunksize:
        return pd.read_csv(path, chunksize=chunksize, **kwargs)
        
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        if kwargs:
            log(logger, f"Dtype-hinted CSV read failed ({exc}); retrying without dtype hints")
            return pd.read_csv(path)
        raise


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


def load_dataset(
    path: str | Path,
    engine: str = "pandas",
    enable_logging: bool = False,
    type_map: dict[str, list[str]] | None = None,
    use_dtype_hints: bool = True,
    cache: bool = False,
    cache_dir: str | Path | None = None,
) -> Any:
    """Load a CSV or Parquet file with pandas or Polars, supporting caching and type hints.

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

    # Resolve caching destination
    cache_file = None
    if cache and suffix in {".csv", ".parquet"}:
        c_dir = Path(cache_dir) if cache_dir else path.parent / ".cleanflow_cache"
        cache_key = _generate_cache_key(path, type_map)
        cache_file = c_dir / f"{path.stem}.optimized.{cache_key}.parquet"
        
        if cache_file.exists():
            log(logger, f"Loading optimized dataset from cache: {cache_file}")
            if engine == "polars":
                return load_parquet_polars(cache_file, logger)
            return load_parquet(cache_file, logger)

    # Standard loading path
    if suffix == ".csv":
        if engine == "polars":
            # Polars downcasting is done natively via lazy schemas; type_map is pandas-specific
            df = load_csv_polars(path, logger)
        else:
            df = load_csv(path, type_map=type_map, use_dtype_hints=use_dtype_hints, logger=logger)
    elif suffix == ".parquet":
        if engine == "polars":
            df = load_parquet_polars(path, logger)
        else:
            df = load_parquet(path, logger)
            if use_dtype_hints and type_map:
                log(logger, "Applying dtype hints to Parquet DataFrame after load")
                for target_dtype, columns in type_map.items():
                    for col in columns:
                        if col in df.columns:
                            try:
                                if target_dtype == "datetime64[ns]":
                                    df[col] = pd.to_datetime(df[col])
                                else:
                                    df[col] = df[col].astype(target_dtype)
                            except Exception as exc:
                                log(logger, f"Could not cast {col} to {target_dtype}: {exc}")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # Save to cache if caching is requested and wasn't loaded from it
    if cache_file and not cache_file.exists():
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            log(logger, f"Caching optimized dataset to Parquet: {cache_file}")
            if engine == "polars":
                try:
                    from .optimization.backends.polars_backend import to_parquet
                    to_parquet(df, cache_file, logger=logger)
                except ImportError:
                    df.write_parquet(cache_file)
            else:
                df.to_parquet(cache_file, index=False)
        except Exception as exc:
            log(logger, f"Could not write cache file due to: {exc}")

    return df


def estimate_csv_memory(path: str | Path, sample_rows: int = 50_000) -> float:
    """Estimate full CSV memory usage in MB from a row sample."""
    path = Path(path)
    sample = pd.read_csv(path, nrows=sample_rows)
    if sample.empty:
        return 0.0
    sample_mem = sample.memory_usage(deep=True).sum()
    estimated_rows = int(path.stat().st_size / max(sample_mem / len(sample), 1))
    return (sample_mem / len(sample)) * estimated_rows / 1024**2
