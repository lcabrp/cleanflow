"""Public optimization API used by the top-level cleanflow package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..logging_utils import get_logger, log
from .analysis import analyze_optimization as _analyze_optimization
from .analysis import apply_optimization as _apply_optimization
from .backends import optimize_dask, optimize_pandas, optimize_polars
from .utils import detect_engine


def optimize_dataset(df: Any, enable_logging: bool = False, for_parquet: bool = False) -> Any:
    """Optimize a pandas, Polars, or Dask DataFrame-like object."""
    logger = get_logger(enabled=enable_logging)
    engine = detect_engine(df)

    if engine == "pandas":
        return optimize_pandas(df, logger, for_parquet=for_parquet)
    if engine == "polars":
        if optimize_polars is None:
            raise ModuleNotFoundError("Polars support is optional. Install with: pip install 'cleanflow[polars]'")
        return optimize_polars(df, logger, for_parquet=for_parquet)
    if engine == "dask":
        if optimize_dask is None:
            raise ModuleNotFoundError("Dask support is optional. Install with: pip install 'cleanflow[dask]'")
        return optimize_dask(df, logger)
    raise RuntimeError("Unknown dataframe backend")


def analyze_optimization(
    df: Any,
    cat_threshold: float = 0.5,
    cat_max_unique: int = 1000,
    downcast_float: bool = True,
    require_float_notnull: bool = True,
    for_parquet: bool = False,
    verbose: bool = False,
    enable_logging: bool = False,
) -> dict[str, list[str]]:
    """Return dtype optimization recommendations without applying them."""
    logger = get_logger(enabled=enable_logging)
    return _analyze_optimization(
        df,
        cat_threshold=cat_threshold,
        cat_max_unique=cat_max_unique,
        downcast_float=downcast_float,
        require_float_notnull=require_float_notnull,
        for_parquet=for_parquet,
        verbose=verbose,
        logger=logger,
    )


def apply_optimization(df: Any, type_map: dict[str, list[str]], enable_logging: bool = False) -> Any:
    """Apply dtype optimization recommendations."""
    return _apply_optimization(df, type_map, logger=get_logger(enabled=enable_logging))


def convert_to_parquet(path: str | Path, output_path: str | Path, engine: str = "duckdb", enable_logging: bool = False) -> str:
    """Convert CSV to Parquet.

    DuckDB is the default for out-of-core speed. Use
    convert_to_parquet_optimized(..., engine="pandas") when DuckDB is not
    installed and chunked pandas conversion is preferred.
    """
    logger = get_logger(enabled=enable_logging)
    if engine != "duckdb":
        raise ValueError("convert_to_parquet currently supports engine='duckdb'. Use convert_to_parquet_optimized for pandas/polars.")

    from .backends.duckdb_backend import csv_to_parquet

    return csv_to_parquet(path, output_path, logger=logger)


def convert_to_parquet_optimized(
    path: str | Path,
    output_path: str | Path,
    compression: str = "snappy",
    chunksize: int = 1_000_000,
    enable_logging: bool = False,
    engine: str = "pandas",
) -> str:
    """Convert CSV to Parquet with dtype optimization.

    Pandas is the default because it is always installed with CleanFlow. Polars
    remains available as a faster optional backend.
    """
    logger = get_logger(enabled=enable_logging)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if engine == "polars":
        try:
            from .backends.polars_backend import optimize_and_write_parquet
        except ModuleNotFoundError as exc:
            if getattr(exc, "name", "") == "polars":
                log(logger, "Polars is not installed; falling back to pandas conversion.")
                engine = "pandas"
            else:
                raise
        else:
            return optimize_and_write_parquet(path, output_path, compression=compression, logger=logger)

    if engine != "pandas":
        raise ValueError("engine must be 'pandas' or 'polars'")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("PyArrow is required for optimized Parquet conversion. Install with: pip install 'cleanflow[parquet]'") from exc

    writer = None
    codec = None if compression == "none" else compression
    try:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk = optimize_pandas(chunk, for_parquet=True)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression=codec)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return str(output_path)

