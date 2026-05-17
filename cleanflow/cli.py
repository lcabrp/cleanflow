"""CleanFlow command-line entry points."""

from __future__ import annotations

import argparse
from typing import Sequence

from .io import load_dataset
from .optimization import convert_to_parquet, convert_to_parquet_optimized, optimize_dataset
from .profiling import main as profile_main


def optimize_main(argv: Sequence[str] | None = None) -> int:
    """Optimize a dataset or convert CSV to Parquet from the CLI."""
    parser = argparse.ArgumentParser(description="CleanFlow dataset optimization tools.")
    parser.add_argument("path", help="Path to CSV or Parquet input.")
    parser.add_argument("--to-parquet", help="Output Parquet path.")
    parser.add_argument("--optimize", action="store_true", help="Optimize dtypes during Parquet conversion.")
    parser.add_argument("--engine", choices=["pandas", "polars"], default="pandas", help="DataFrame engine for loading/conversion.")
    parser.add_argument("--compression", choices=["snappy", "zstd", "gzip", "brotli", "lz4", "none"], default="snappy")
    parser.add_argument("--chunksize", type=int, default=1_000_000, help="Rows per chunk for pandas conversion.")
    parser.add_argument("--log", action="store_true", help="Enable progress logging.")
    args = parser.parse_args(argv)

    if args.to_parquet:
        if args.optimize:
            output = convert_to_parquet_optimized(
                args.path,
                args.to_parquet,
                compression=args.compression,
                chunksize=args.chunksize,
                engine=args.engine,
                enable_logging=args.log,
            )
        else:
            output = convert_to_parquet(args.path, args.to_parquet, enable_logging=args.log)
        print(f"Parquet file written to: {output}")
        return 0

    df = load_dataset(args.path, engine=args.engine, enable_logging=args.log)
    optimize_dataset(df, enable_logging=args.log)
    print("Dataset optimized successfully")
    return 0


def profile_main_entry(argv: Sequence[str] | None = None) -> int:
    """Console wrapper for profiling."""
    return profile_main(argv)


if __name__ == "__main__":
    raise SystemExit(optimize_main())

