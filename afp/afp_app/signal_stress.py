from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

class StressProbabilityModel:


    def __init__(
        self,
        threshold_drawdown: float = -0.05,
        threshold_vol_spike: float = 1.5,
    ):

        self.threshold_drawdown = threshold_drawdown
        self.threshold_vol_spike = threshold_vol_spike

        self.model: GradientBoostingClassifier | None = None
        self.scaler: StandardScaler = StandardScaler()

        self.cv_auc_mean: float | None = None
        self.stress_share: float | None = None

    def label_stress_periods(self, market_data: pd.DataFrame) -> pd.Series:
      
        idx = market_data.index
        stress_labels = pd.Series(index=idx, data=0, dtype=int)

        if "market_return" in market_data.columns:
            rolling_return = market_data["market_return"].rolling(window=21).sum()
            stress_labels[rolling_return < self.threshold_drawdown] = 1

        if "vix_close" in market_data.columns:
            vix = market_data["vix_close"]
            vix_ma = vix.rolling(window=20).mean()
            vix_spike = vix / vix_ma
            stress_labels[vix_spike > self.threshold_vol_spike] = 1

        if "credit_spread_1m_change" in market_data.columns:
            spread_chg = market_data["credit_spread_1m_change"]
            if spread_chg.notna().sum() > 0:
                q90 = spread_chg.quantile(0.9)
                stress_labels[spread_chg > q90] = 1

        return stress_labels

    def prepare_stress_features(self, data: pd.DataFrame) -> pd.DataFrame:
      
        feats = pd.DataFrame(index=data.index)

        if "vix_close" in data.columns:
            vix = data["vix_close"]
            feats["vix_level"] = vix
            vix_ma = vix.rolling(window=20).mean()
            feats["vix_ma_ratio"] = vix / vix_ma
            feats["vix_percentile"] = vix.rolling(window=252).rank(pct=True)

        if "rates_1m_change" in data.columns:
            rc = data["rates_1m_change"]
            feats["rates_change"] = rc
            feats["rates_volatility"] = rc.rolling(window=20).std()

        if "term_spread_10y2y" in data.columns:
            ts = data["term_spread_10y2y"]
            feats["term_spread"] = ts
            feats["term_spread_change"] = ts.diff(21)

        if "credit_spread_level" in data.columns:
            feats["credit_level"] = data["credit_spread_level"]
        if "credit_spread_1m_change" in data.columns:
            feats["credit_change"] = data["credit_spread_1m_change"]

        if "market_return" in data.columns:
            mr = data["market_return"]
            feats["market_momentum_5d"] = mr.rolling(window=5).sum()
            feats["market_momentum_21d"] = mr.rolling(window=21).sum()

        feats = feats.fillna(method="ffill").fillna(0.0)
        return feats

    def fit(self, data: pd.DataFrame):
     
        print("Training stress probability model...")

        stress_labels = self.label_stress_periods(data)

        X = self.prepare_stress_features(data)
        y = stress_labels

        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) < 100:
            print("Insufficient data for stress model training")
            self.model = None
            self.cv_auc_mean = None
            self.stress_share = None
            return None

        X_scaled = self.scaler.fit_transform(X)

        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42,
        )

        cv_scores = cross_val_score(gb, X_scaled, y, cv=5, scoring="roc_auc")

        gb.fit(X_scaled, y)

        self.model = gb
        self.cv_auc_mean = float(cv_scores.mean())
        self.stress_share = float(y.mean())

        print(f"Stress model trained. CV AUC: {self.cv_auc_mean:.3f}")
        print(
            f"Stress periods: {y.sum()} out of {len(y)} days "
            f"({self.stress_share*100:.1f}%)"
        )

        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": gb.feature_importances_}
        ).sort_values("importance", ascending=False)

        return feature_importance

    def predict(self, data: pd.DataFrame) -> dict | None:
    

        if self.model is None:
            self.fit(data)
        if self.model is None:
            return None

        feats = self.prepare_stress_features(data)
        if feats.empty:
            return None

        x = feats.iloc[-1].copy()
        X_scaled = self.scaler.transform(x.values.reshape(1, -1))
        prob = float(self.model.predict_proba(X_scaled)[0, 1])

        if prob > 0.7:
            regime = "HIGH RISK"
        elif prob > 0.3:
            regime = "ELEVATED"
        else:
            regime = "NORMAL"

        key_indicators = {}
        if "vix_level" in feats.columns:
            key_indicators["vix_level"] = float(feats["vix_level"].iloc[-1])
        if "credit_level" in feats.columns:
            key_indicators["credit_spread"] = float(
                feats["credit_level"].iloc[-1]
            )
        if "market_momentum_21d" in feats.columns:
            key_indicators["market_momentum_21d"] = float(
                feats["market_momentum_21d"].iloc[-1]
            )

        return {
            "stress_probability": prob,
            "regime": regime,
            "key_indicators": key_indicators,
        }

