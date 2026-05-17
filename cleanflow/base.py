"""
Base classes for all CleanFlow transformers.

Every transformer follows the analyze → transform → report pattern:
  1. analyze() — inspect the data and recommend actions
  2. transform() — apply cleaning transformations
  3. get_report() — return a summary of what was done
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseTransformer(ABC):
    """Abstract base class for all cleaning transformers."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation to the DataFrame and return a cleaned copy."""
        pass

    def get_report(self) -> pd.DataFrame:
        """Return a DataFrame summarizing the actions taken.

        Subclasses should override this to provide transformer-specific reports.
        Returns an empty DataFrame by default.
        """
        return pd.DataFrame()


class FitTransformer(BaseTransformer):
    """Base class for transformers that learn from data before transforming.

    Use this when the transformer needs a fit step (e.g., learning medians
    before imputation, or learning category mappings).
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "FitTransformer":
        """Learn parameters from the DataFrame. Returns self for chaining."""
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method: fit then transform."""
        self.fit(df)
        return self.transform(df)