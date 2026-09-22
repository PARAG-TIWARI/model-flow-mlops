"""Feature engineering transformations for customer churn modeling."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives domain-specific financial and customer relationship indicators."""

    def __init__(self, add_ratio_features: bool = True):
        self.add_ratio_features = add_ratio_features

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Enrich input dataframe with engineered business metrics."""
        df = X.copy()

        if self.add_ratio_features and "TotalCharges" in df.columns and "tenure" in df.columns:
            # Average monthly charge over customer lifespan vs currently billed monthly charge
            safe_tenure = np.maximum(df["tenure"].astype(float), 1.0)
            df["ChargesRatio"] = np.round(df["TotalCharges"].astype(float) / safe_tenure, 2)
            if "MonthlyCharges" in df.columns:
                df["ChargeDiff"] = np.round(df["MonthlyCharges"].astype(float) - df["ChargesRatio"], 2)

        # Flag customers with bundled security & support
        if "OnlineSecurity" in df.columns and "TechSupport" in df.columns:
            df["HasSecuritySupportBundle"] = (
                (df["OnlineSecurity"] == "Yes") & (df["TechSupport"] == "Yes")
            ).astype(int)

        # Flag streaming adopters
        if "StreamingTV" in df.columns and "StreamingMovies" in df.columns:
            df["HasFullStreamingBundle"] = (
                (df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes")
            ).astype(int)

        return df
