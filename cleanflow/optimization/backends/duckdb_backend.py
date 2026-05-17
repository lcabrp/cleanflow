"""DuckDB helpers for out-of-core CSV and Parquet work."""

from __future__ import annotations

import logging
from pathlib import Path

from ...logging_utils import log


def _require_duckdb():
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("DuckDB support is optional. Install with: pip install 'cleanflow[duckdb]'") from exc
    return duckdb


def _escape_sql_literal(value: str | Path) -> str:
    return str(value).replace("'", "''")


def load_csv(path: str | Path, logger: logging.Logger | None = None):
    """Load CSV into a DuckDB relation without materializing everything in pandas."""
    duckdb = _require_duckdb()
    log(logger, f"Loading CSV via DuckDB: {path}")
    return duckdb.connect().sql(f"SELECT * FROM read_csv_auto('{_escape_sql_literal(path)}')")


def load_parquet(path: str | Path, logger: logging.Logger | None = None):
    """Load Parquet into a DuckDB relation."""
    duckdb = _require_duckdb()
    log(logger, f"Loading Parquet via DuckDB: {path}")
    return duckdb.connect().sql(f"SELECT * FROM '{_escape_sql_literal(path)}'")


def csv_to_parquet(path: str | Path, output_path: str | Path, logger: logging.Logger | None = None) -> str:
    """Convert CSV to Parquet with DuckDB's out-of-core engine."""
    duckdb = _require_duckdb()
    log(logger, f"Converting CSV -> Parquet via DuckDB: {path} -> {output_path}")
    con = duckdb.connect()
    con.execute(
        f"COPY (SELECT * FROM read_csv('{_escape_sql_literal(path)}', "
        "auto_detect=true, ignore_errors=true, null_padding=true)) "
        f"TO '{_escape_sql_literal(output_path)}' (FORMAT PARQUET)"
    )
    return str(output_path)

