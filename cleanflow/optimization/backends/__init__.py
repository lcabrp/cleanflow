"""Optimization backend imports.

Optional backends are imported lazily enough that installing CleanFlow without
Polars, Dask, or DuckDB still gives a working pandas cleaning package.
"""

from __future__ import annotations

from typing import Any, Callable

from .pandas_backend import optimize as optimize_pandas

optimize_polars: Callable[..., Any] | None
optimize_dask: Callable[..., Any] | None

try:
    from .polars_backend import optimize as optimize_polars
except ModuleNotFoundError as exc:
    if getattr(exc, "name", "") == "polars":
        optimize_polars = None
    else:
        raise

try:
    from .dask_backend import optimize as optimize_dask
except ModuleNotFoundError as exc:
    if getattr(exc, "name", "").startswith("dask"):
        optimize_dask = None
    else:
        raise

__all__ = ["optimize_pandas", "optimize_polars", "optimize_dask"]

