"""Dask optimization backend."""

from __future__ import annotations

import logging
from typing import Any

from ...logging_utils import log


def optimize(df: Any, logger: logging.Logger | None = None) -> Any:
    """Return Dask objects unchanged because Dask optimizes lazily at compute time."""
    log(logger, "Dask optimization deferred to lazy execution.")
    return df

