import pandas as pd
from sklearn.impute import SimpleImputer

class DataStandardizer:
    """Handles type conversion and standardization[cite: 7, 8]."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for column in df.columns:
            if df[column].dtype == 'object':
                try:
                    df[column] = pd.to_datetime(df[column])
                except (ValueError, TypeError):
                    try:
                        df[column] = pd.to_numeric(
                            df[column].astype(str).str.replace(r'[$,]', '', regex=True)
                        )
                    except:
                        continue
        return df

class MissingValueHandler:
    """Imputes missing data using median or most frequent values[cite: 10]."""
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        num_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['object']).columns
        
        if len(num_cols) > 0:
            df[num_cols] = self.num_imputer.fit_transform(df[num_cols])
        if len(cat_cols) > 0:
            df[cat_cols] = self.cat_imputer.fit_transform(df[cat_cols])
        return df