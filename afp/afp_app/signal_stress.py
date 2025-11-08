import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

class StressProbabilityModel:
    def __init__(self, threshold_drawdown: float = -0.05, threshold_vol_spike: float = 1.5):
        self.threshold_drawdown = threshold_drawdown
        self.threshold_vol_spike = threshold_vol_spike
        self.model = None
        self.scaler = StandardScaler()

    def label_stress(self, df: pd.DataFrame) -> pd.Series:
        y = pd.Series(index=df.index, data=0)
        if "market_return" in df.columns:
            rr = df["market_return"].rolling(21).sum()
            y[rr < self.threshold_drawdown] = 1
        if "vix_close" in df.columns:
            vma = df["vix_close"].rolling(20).mean()
            spike = df["vix_close"] / vma
            y[spike > self.threshold_vol_spike] = 1
        if "credit_spread_1m_change" in df.columns:
            q = df["credit_spread_1m_change"].quantile(0.9)
            y[df["credit_spread_1m_change"] > q] = 1
        return y

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        if "vix_close" in df.columns:
            X["vix_level"] = df["vix_close"]
            X["vix_ma_ratio"] = df["vix_close"] / df["vix_close"].rolling(20).mean()
            X["vix_percentile"] = df["vix_close"].rolling(252).rank(pct=True)
        if "rates_1m_change" in df.columns:
            X["rates_change"] = df["rates_1m_change"]
            X["rates_vol"] = df["rates_1m_change"].rolling(20).std()
        if "term_spread_10y2y" in df.columns:
            X["term_spread"] = df["term_spread_10y2y"]
            X["term_spread_chg"] = df["term_spread_10y2y"].diff(21)
        if "credit_spread_level" in df.columns:
            X["credit_level"] = df["credit_spread_level"]
            X["credit_change"] = df["credit_spread_1m_change"]
        if "market_return" in df.columns:
            X["momo_5d"] = df["market_return"].rolling(5).sum()
            X["momo_21d"] = df["market_return"].rolling(21).sum()
        return X.ffill().fillna(0)

    def fit(self, df: pd.DataFrame):
        y = self.label_stress(df)
        X = self._features(df)
        valid = X.notna().all(axis=1) & y.notna()
        X = X[valid]
        y = y[valid]
        if len(X) < 100:
            return None
        Xs = self.scaler.fit_transform(X)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        scores = cross_val_score(clf, Xs, y, cv=5, scoring="roc_auc")
        clf.fit(Xs, y)
        self.model = clf
        return float(scores.mean())

    def predict(self, df: pd.DataFrame) -> dict | None:
        if self.model is None:
            auc = self.fit(df)
            if self.model is None:
                return None
        X = self._features(df).iloc[-1:].fillna(0)
        Xs = self.scaler.transform(X)
        p = float(self.model.predict_proba(Xs)[0,1])
        regime = "HIGH RISK" if p > 0.7 else ("ELEVATED" if p > 0.3 else "NORMAL")
        return {
            "stress_probability": p,
            "regime": regime,
            "key_indicators": {
                "vix_level": float(X["vix_level"].values[0]) if "vix_level" in X else None,
                "credit_spread": float(X["credit_level"].values[0]) if "credit_level" in X else None,
                "market_momentum": float(X["momo_21d"].values[0]) if "momo_21d" in X else None
            }
        }
