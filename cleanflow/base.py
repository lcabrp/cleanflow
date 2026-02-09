from abc import ABC, abstractmethod
import pandas as pd

class BaseTransformer(ABC):
    """Abstract Base Class for all cleaning transformers."""
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation to the dataframe."""
        pass

class FitTransformer(BaseTransformer):
    """Base class for transformers that need to learn from data (e.g., imputer)."""
    
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """Learn parameters from the dataframe."""
        pass