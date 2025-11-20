# afp_app/signal_stress.py

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class StressProbabilityModel:
    """
    Predict probability of a market stress regime from macro and market features.

    Main methods:
        - fit(data): trains classifier and stores cv_auc_mean and stress_share
        - predict(data): returns current stress probability, regime, key indicators
    """

    def __init__(
        self,
        threshold_drawdown: float = -0.05,
        threshold_vol_spike: float = 1.5,
    ):
        # Thresholds for labeling stress periods
        self.threshold_drawdown = threshold_drawdown
        self.threshold_vol_spike = threshold_vol_spike

        # Model and scaler will be set in fit()
        self.model: GradientBoostingClassifier | None = None
        self.scaler: StandardScaler = StandardScaler()

        # Validation summary
        self.cv_auc_mean: float | None = None
        self.stress_share: float | None = None

    # ------------------------------------------------------------------
    # Label stress periods
    # ------------------------------------------------------------------
    def label_stress_periods(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Label each date as stress (1) or normal (0) based on:

          1. Rolling 21 day market drawdown below threshold_drawdown
          2. VIX spike relative to 20 day moving average
          3. Large positive change in credit_spread_1m_change (top 10 percent)

        Returns a Series with the same index as market_data.
        """
        idx = market_data.index
        stress_labels = pd.Series(index=idx, data=0, dtype=int)

        # Condition 1: market drawdown
        if "market_return" in market_data.columns:
            rolling_return = market_data["market_return"].rolling(window=21).sum()
            stress_labels[rolling_return < self.threshold_drawdown] = 1

        # Condition 2: volatility spike via VIX
        if "vix_close" in market_data.columns:
            vix = market_data["vix_close"]
            vix_ma = vix.rolling(window=20).mean()
            vix_spike = vix / vix_ma
            stress_labels[vix_spike > self.threshold_vol_spike] = 1

        # Condition 3: credit spread widening
        if "credit_spread_1m_change" in market_data.columns:
            spread_chg = market_data["credit_spread_1m_change"]
            if spread_chg.notna().sum() > 0:
                q90 = spread_chg.quantile(0.9)
                stress_labels[spread_chg > q90] = 1

        return stress_labels

    # ------------------------------------------------------------------
    # Build stress features
    # ------------------------------------------------------------------
    def prepare_stress_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature matrix for stress classification from the modeling frame.

        Features:
            - vix_level, vix_ma_ratio, vix_percentile
            - rates_change, rates_volatility
            - term_spread, term_spread_change
            - credit_level, credit_change
            - market_momentum_5d, market_momentum_21d
        """
        feats = pd.DataFrame(index=data.index)

        # VIX based features
        if "vix_close" in data.columns:
            vix = data["vix_close"]
            feats["vix_level"] = vix
            vix_ma = vix.rolling(window=20).mean()
            feats["vix_ma_ratio"] = vix / vix_ma
            feats["vix_percentile"] = vix.rolling(window=252).rank(pct=True)

        # Rate changes
        if "rates_1m_change" in data.columns:
            rc = data["rates_1m_change"]
            feats["rates_change"] = rc
            feats["rates_volatility"] = rc.rolling(window=20).std()

        # Term structure - use 10y minus 2y
        if "term_spread_10y2y" in data.columns:
            ts = data["term_spread_10y2y"]
            feats["term_spread"] = ts
            feats["term_spread_change"] = ts.diff(21)

        # Credit spreads
        if "credit_spread_level" in data.columns:
            feats["credit_level"] = data["credit_spread_level"]
        if "credit_spread_1m_change" in data.columns:
            feats["credit_change"] = data["credit_spread_1m_change"]

        # Market momentum
        if "market_return" in data.columns:
            mr = data["market_return"]
            feats["market_momentum_5d"] = mr.rolling(window=5).sum()
            feats["market_momentum_21d"] = mr.rolling(window=21).sum()

        # Forward fill and replace remaining NaNs with zero
        feats = feats.fillna(method="ffill").fillna(0.0)
        return feats

    # ------------------------------------------------------------------
    # Fit classifier and record quality metrics
    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame):
        """
        Train the stress probability model and store:

            - cv_auc_mean: cross validated AUC
            - stress_share: fraction of days labeled as stress

        Returns feature importance DataFrame, or None if not enough data.
        """
        print("Training stress probability model...")

        # Labels
        stress_labels = self.label_stress_periods(data)

        # Features
        X = self.prepare_stress_features(data)
        y = stress_labels

        # Valid rows (no NaNs)
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) < 100:
            print("Insufficient data for stress model training")
            self.model = None
            self.cv_auc_mean = None
            self.stress_share = None
            return None

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Classifier
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42,
        )

        # Cross validated AUC
        cv_scores = cross_val_score(gb, X_scaled, y, cv=5, scoring="roc_auc")

        # Final fit
        gb.fit(X_scaled, y)

        self.model = gb
        self.cv_auc_mean = float(cv_scores.mean())
        self.stress_share = float(y.mean())

        print(f"Stress model trained. CV AUC: {self.cv_auc_mean:.3f}")
        print(
            f"Stress periods: {y.sum()} out of {len(y)} days "
            f"({self.stress_share*100:.1f}%)"
        )

        # Feature importance
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": gb.feature_importances_}
        ).sort_values("importance", ascending=False)

        return feature_importance

    # ------------------------------------------------------------------
    # Predict stress probability for the latest date
    # ------------------------------------------------------------------
    def predict(self, data: pd.DataFrame) -> dict | None:
        """
        Predict stress probability for the latest date.

        Returns dict with:
            - stress_probability (0 to 1)
            - regime ("NORMAL", "ELEVATED", "HIGH RISK")
            - key_indicators: latest vix_level, credit_level, market_momentum_21d
        """
        # Ensure model is trained
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

        # Map to regime
        if prob > 0.7:
            regime = "HIGH RISK"
        elif prob > 0.3:
            regime = "ELEVATED"
        else:
            regime = "NORMAL"

        # Build key indicators from latest feature row if present
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



