"""Polars optimization backend."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from ...logging_utils import log


def optimize(df: pl.DataFrame, logger: logging.Logger | None = None, for_parquet: bool = False) -> pl.DataFrame:
    """Reduce Polars memory use with native dtype shrinking."""
    start_mem = df.estimated_size("mb")
    out = df.select(pl.all().shrink_dtype())
    end_mem = out.estimated_size("mb")
    log(logger, f"Polars memory: {start_mem:.2f} -> {end_mem:.2f} MB")
    return out


def optimize_and_write_parquet(
    csv_path: str | Path,
    output_path: str | Path,
    compression: str = "zstd",
    logger: logging.Logger | None = None,
) -> str:
    """Stream CSV to optimized Parquet using Polars lazy execution."""
    compression_map = {"none": "uncompressed", "snappy": "snappy", "zstd": "zstd", "gzip": "gzip", "brotli": "brotli", "lz4": "lz4"}
    log(logger, f"Streaming CSV to Parquet with Polars: {csv_path} -> {output_path}")
    (
        pl.scan_csv(csv_path)
        .select(pl.all().shrink_dtype())
        .sink_parquet(output_path, compression=compression_map.get(compression, compression))
    )
    return str(output_path)


def to_parquet(df: pl.DataFrame, path: str | Path, compression: str = "zstd", logger: logging.Logger | None = None) -> str:
    """Write a Polars DataFrame to Parquet."""
    compression_map = {"none": "uncompressed", "snappy": "snappy", "zstd": "zstd", "gzip": "gzip", "brotli": "brotli", "lz4": "lz4"}
    log(logger, f"Writing Polars DataFrame to Parquet: {path}")
    df.write_parquet(path, compression=compression_map.get(compression, compression))
    return str(path)

