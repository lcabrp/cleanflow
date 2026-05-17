"""
Feature Engineering Module
--------------------------
Advanced transformers for feature engineering, including automated
numerical transformations, enhanced date extraction, and feature selection.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from .base import FitTransformer

class NumericalTransformer(FitTransformer):
    """
    Advanced numerical transformer that automatically selects the best
    transformation (Log, Sqrt, Box-Cox, Yeo-Johnson) to improve normality,
    followed by scaling.
    """
    def __init__(self, columns=None, auto_transform=True, scaling_method="standard"):
        super().__init__()
        self.columns = columns
        self.auto_transform = auto_transform
        self.scaling_method = scaling_method
        self.transformations_ = {}  # Stores best transform per column
        self.scalers_ = {}          # Stores scaler per column
        self.stats_ = {}            # Stores normality improvement stats

    def _evaluate_normality(self, data):
        """Evaluate normality using skewness and kurtosis."""
        skew = data.skew()
        kurt = data.kurtosis()
        # Simple score: lower is better (0 is perfect normal)
        score = abs(skew) + abs(kurt) / 3.0
        return score

    def fit(self, X, y=None):
        X = X.copy()
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in self.columns:
            if col not in X.columns:
                continue
                
            data = X[col].dropna()
            # Need minimum samples for normality check/transform, but maybe not for scaling
            # Box-Cox requires > 0, Yeo-Johnson supports negative
            # Skew test technically runs on small data but noisy. 
            # Let's say we need at least 3 samples to attempt transformation logic
            can_transform = len(data) >= 3

            # 1. auto-transformation selection
            if self.auto_transform and can_transform:
                best_method, params, transformed_data = self._find_best_transform(data)
                self.transformations_[col] = {'method': best_method, 'params': params}
            else:
                best_method = 'none'
                transformed_data = data
                self.transformations_[col] = {'method': 'none', 'params': {}}

            # 2. Fit Scaler
            if self.scaling_method:
                if self.scaling_method == "standard":
                    scaler = StandardScaler()
                elif self.scaling_method == "minmax":
                    scaler = MinMaxScaler()
                elif self.scaling_method == "robust":
                    scaler = RobustScaler()
                else:
                    scaler = None
                
                if scaler:
                    # reshapes for sklearn
                    if len(transformed_data) > 0:
                        scaler.fit(transformed_data.values.reshape(-1, 1))
                        self.scalers_[col] = scaler

        return self

    def _find_best_transform(self, data):
        """Find the transformation that minimizes normality score."""
        original_score = self._evaluate_normality(data)
        best_score = original_score
        best_method = 'none'
        best_params = {}
        best_data = data
        
        # Try Log
        try:
            min_val = data.min()
            shift = abs(min_val) + 1 if min_val <= 0 else 0
            log_data = np.log(data + shift)
            score = self._evaluate_normality(log_data)
            if score < best_score:
                best_score = score
                best_method = 'log'
                best_params = {'shift': shift}
                best_data = log_data
        except Exception:
            # Some transforms are only valid for certain distributions; keep
            # trying the remaining candidates instead of failing the pipeline.
            pass

        # Try Box-Cox (requires positive)
        try:
            min_val = data.min()
            shift = abs(min_val) + 1 if min_val <= 0 else 0
            # Only try if strictly positive after shift
            if (data + shift > 0).all():
                bc_data, lmbda = stats.boxcox(data + shift)
                bc_series = pd.Series(bc_data, index=data.index)
                score = self._evaluate_normality(bc_series)
                if score < best_score:
                    best_score = score
                    best_method = 'boxcox'
                    best_params = {'lambda': lmbda, 'shift': shift}
                    best_data = bc_series
        except Exception:
            # Box-Cox is intentionally opportunistic here because bad inputs
            # should fall through to Yeo-Johnson or the original data.
            pass

        # Try Yeo-Johnson (works on negatives)
        try:
            yj_data, lmbda = stats.yeojohnson(data)
            yj_series = pd.Series(yj_data, index=data.index)
            score = self._evaluate_normality(yj_series)
            if score < best_score:
                best_score = score
                best_method = 'yeojohnson'
                best_params = {'lambda': lmbda}
                best_data = yj_series
        except Exception:
            # Yeo-Johnson is a best-effort normalization candidate.
            pass
        
        return best_method, best_params, best_data

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col not in X.columns or col not in self.transformations_:
                continue
            
            # 1. Apply Transformation
            trans_info = self.transformations_[col]
            method = trans_info['method']
            params = trans_info['params']
            
            data = X[col]
            mask = data.notna()
            if not mask.any():
                continue
                
            vals = data[mask]
            
            if method == 'log':
                vals = np.log(vals + params.get('shift', 0))
            elif method == 'boxcox':
                vals = stats.boxcox(vals + params.get('shift', 0), lmbda=params['lambda'])
            elif method == 'yeojohnson':
                vals = stats.yeojohnson(vals, lmbda=params['lambda'])
            
            # Ensure column is float if we are putting floats in it
            if not np.issubdtype(X[col].dtype, np.floating):
                 X[col] = X[col].astype(float)

            # Update X with transformed values
            X.loc[mask, col] = vals
            
            # 2. Apply Scaling
            if col in self.scalers_:
                scaler = self.scalers_[col]
                scaled_vals = scaler.transform(X.loc[mask, col].values.reshape(-1, 1)).flatten()
                X.loc[mask, col] = scaled_vals
                
        return X

    def get_report(self) -> pd.DataFrame:
        report_data = []
        for col, info in self.transformations_.items():
            report_data.append({
                "column": col,
                "transformation": info['method'],
                "scaling": self.scaling_method if col in self.scalers_ else "none"
            })
        return pd.DataFrame(report_data)

# Alias for backward compatibility
FeatureScaler = NumericalTransformer


class DateFeatureExtractor(FitTransformer):
    """
    Extracts datetime components including cyclical features (sin/cos),
    seasonality, and time differences.
    """
    def __init__(self, features=None, drop_original=False):
        super().__init__()
        self.default_features = ["year", "month", "day", "weekday"]
        self.features = features
        self.drop_original = drop_original
        self.created_features_ = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        self.created_features_ = []
        
        date_cols = X.select_dtypes(include=["datetime", "datetime64[ns]"]).columns.tolist()
        if not date_cols:
            return X
            
        features_to_extract = self.features if self.features else self.default_features
        
        for col in date_cols:
            series = X[col]
            if series.isna().all():
                continue
                
            dt = series.dt
            
            # Basic Features
            if "year" in features_to_extract:
                X[f"{col}_year"] = dt.year
                self.created_features_.append({"source": col, "feature": f"{col}_year", "type": "basic"})
            if "month" in features_to_extract:
                X[f"{col}_month"] = dt.month
                self.created_features_.append({"source": col, "feature": f"{col}_month", "type": "basic"})
            if "day" in features_to_extract:
                X[f"{col}_day"] = dt.day
                self.created_features_.append({"source": col, "feature": f"{col}_day", "type": "basic"})
            if "weekday" in features_to_extract:
                X[f"{col}_weekday"] = dt.dayofweek
                self.created_features_.append({"source": col, "feature": f"{col}_weekday", "type": "basic"})
            if "quarter" in features_to_extract:
                X[f"{col}_quarter"] = dt.quarter
                self.created_features_.append({"source": col, "feature": f"{col}_quarter", "type": "basic"})

            # Flags
            if "is_weekend" in features_to_extract or "flags" in features_to_extract:
                feat = f"{col}_is_weekend"
                X[feat] = (dt.dayofweek >= 5).astype(int)
                self.created_features_.append({"source": col, "feature": feat, "type": "flag"})
            if "is_month_start" in features_to_extract:
                feat = f"{col}_is_month_start"
                X[feat] = dt.is_month_start.astype(int)
                self.created_features_.append({"source": col, "feature": feat, "type": "flag"})
            if "is_month_end" in features_to_extract:
                feat = f"{col}_is_month_end"
                X[feat] = dt.is_month_end.astype(int)
                self.created_features_.append({"source": col, "feature": feat, "type": "flag"})

            # Cyclical Features
            if "cyclical" in features_to_extract or any("sin" in f for f in features_to_extract):
                if "month_sin" in features_to_extract or "cyclical" in features_to_extract:
                    X[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.month / 12)
                    X[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.month / 12)
                    self.created_features_.append({"source": col, "feature": f"{col}_month_sin", "type": "cyclical"})
                    self.created_features_.append({"source": col, "feature": f"{col}_month_cos", "type": "cyclical"})
                if "weekday_sin" in features_to_extract or "cyclical" in features_to_extract:
                    X[f"{col}_weekday_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
                    X[f"{col}_weekday_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)
                    self.created_features_.append({"source": col, "feature": f"{col}_weekday_sin", "type": "cyclical"})
                    self.created_features_.append({"source": col, "feature": f"{col}_weekday_cos", "type": "cyclical"})

            # Season
            if "season" in features_to_extract:
                month = dt.month
                conditions = [
                    month.isin([12, 1, 2]),
                    month.isin([3, 4, 5]),
                    month.isin([6, 7, 8]),
                    month.isin([9, 10, 11])
                ]
                choices = ["Winter", "Spring", "Summer", "Fall"]
                X[f"{col}_season"] = np.select(conditions, choices, default=None)
                self.created_features_.append({"source": col, "feature": f"{col}_season", "type": "season"})

            if self.drop_original:
                X.drop(columns=[col], inplace=True)
                
        return X

    def get_report(self) -> pd.DataFrame:
        return pd.DataFrame(self.created_features_)

class FeatureSelector(FitTransformer):
    """
    Selects important features based on variance and correlation.
    """
    def __init__(self, variance_threshold=0.0, correlation_threshold=0.95):
        super().__init__()
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.drop_columns_ = [] # List of tuples (column, reason)

    def fit(self, X, y=None):
        self.drop_columns_ = []
        X_num = X.select_dtypes(include=[np.number])
        if X_num.empty:
            return self

        # 1. Variance Check
        variances = X_num.var()
        low_variance_cols = variances[variances <= self.variance_threshold].index.tolist()
        for col in low_variance_cols:
            self.drop_columns_.append({"column": col, "reason": "low_variance"})

        # 2. Correlation Check
        current_dropped = [item["column"] for item in self.drop_columns_]
        remaining_cols = [c for c in X_num.columns if c not in current_dropped]
        
        if len(remaining_cols) > 1:
            df_corr = X_num[remaining_cols].corr().abs()
            upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
            
            for column in upper.columns:
                if any(upper[column] > self.correlation_threshold):
                    # Check if already dropped (shouldn't be, but safe)
                    if column not in [x["column"] for x in self.drop_columns_]:
                        self.drop_columns_.append({"column": column, "reason": "high_correlation"})

        return self

    def transform(self, X):
        X = X.copy()
        if self.drop_columns_:
            cols_to_drop = [item["column"] for item in self.drop_columns_ if item["column"] in X.columns]
            X.drop(columns=cols_to_drop, inplace=True)
        return X

    def get_report(self) -> pd.DataFrame:
        return pd.DataFrame(self.drop_columns_)

class MissingIndicator(FitTransformer):
    """
    Creates binary indicators for missing values.
    """
    def __init__(self, suffix="_is_missing"):
        super().__init__()
        self.suffix = suffix
        self.missing_cols_ = []
        self.created_indicators_ = []

    def fit(self, X, y=None):
        self.missing_cols_ = [col for col in X.columns if X[col].isna().any()]
        return self

    def transform(self, X):
        X = X.copy()
        self.created_indicators_ = []
        for col in self.missing_cols_:
            if col in X.columns:
                indicator_col = f"{col}{self.suffix}"
                X[indicator_col] = X[col].isna().astype(int)
                self.created_indicators_.append({"column": col, "indicator": indicator_col})
        return X

    def get_report(self) -> pd.DataFrame:
        return pd.DataFrame(self.created_indicators_)
