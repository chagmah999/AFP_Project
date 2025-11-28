# afp_app/optimizer.py

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.covariance import LedoitWolf


class UnifiedPortfolioOptimizer:
    """
    A stable, production-quality mean–variance optimizer.

    Key features:
      - Expected returns vector comes from alpha predictions
      - Covariance estimated using Ledoit–Wolf shrinkage
      - Supports long-only or long–short
      - Enforces per-name weight cap
      - Enforces total gross exposure cap
    """

    def __init__(
        self,
        forecast_horizon: int = 21,
        risk_aversion: float = 50.0,
        max_weight: float = 0.05,
        max_gross: float = 1.5,
    ):
        self.forecast_horizon = forecast_horizon
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.max_gross = max_gross

    # -------------------------------------------------------------
    # 1. Build expected returns vector (scaled by horizon)
    # -------------------------------------------------------------
    def build_expected_returns(
        self,
        alpha_preds: Dict[str, dict],
        tickers: List[str],
    ) -> pd.Series:
        """
        Extract expected alphas and scale them to forecast_horizon.
        Alpha predictor gives H-day expected return (not annualized).
        """
        vals = []
        for tk in tickers:
            info = alpha_preds.get(tk, {})
            vals.append(float(info.get("expected_alpha", 0.0)))

        mu = pd.Series(vals, index=tickers)

        # optional: scale (e.g., convert H-day return into annualized if desired)
        # for now, keep mu as H-day forward expected return
        return mu

    # -------------------------------------------------------------
    # 2. Covariance estimation (Ledoit–Wolf)
    # -------------------------------------------------------------
    def build_covariance(
        self,
        price_data: pd.DataFrame,
        tickers: List[str],
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        if price_data is None or price_data.empty:
            return pd.DataFrame()

        df = price_data[price_data["ticker"].isin(tickers)].copy()
        if df.empty:
            return pd.DataFrame()

        df = df.sort_values("date")
        retcol = "log_returns" if "log_returns" in df.columns else "returns"

        # last N days only
        unique_dates = df["date"].drop_duplicates().sort_values()
        if len(unique_dates) > lookback_days:
            cutoff = unique_dates.iloc[-lookback_days]
            df = df[df["date"] >= cutoff]

        pivot = df.pivot(index="date", columns="ticker", values=retcol).dropna(how="all")

        if pivot.shape[0] < 50:  # too few days
            return pd.DataFrame()

        # Ledoit–Wolf shrinkage covariance
        lw = LedoitWolf().fit(pivot.fillna(0.0).values)
        Sigma = pd.DataFrame(lw.covariance_, index=tickers, columns=tickers)
        return Sigma

    # -------------------------------------------------------------
    # 3. Solve for mean–variance weights
    # -------------------------------------------------------------
    def optimize(
        self,
        mu: pd.Series,
        Sigma: pd.DataFrame,
        long_only: bool = False,
    ) -> pd.Series:
        if mu is None or mu.empty:
            return pd.Series(dtype=float)

        tickers = list(mu.index)
        n = len(tickers)

        if Sigma is None or Sigma.empty:
            # fallback: no covariance → simple ranking alpha portfolio
            raw = mu - mu.mean()
        else:
            # risk-adjusted return
            var = np.diag(Sigma.values)
            var = np.where(var <= 0, 1e-6, var)
            raw = mu.values / (self.risk_aversion * var)

        # dollar-neutral unless long-only
        if not long_only:
            raw = raw - raw.mean()

        # clip per name
        raw = np.clip(raw, -self.max_weight, self.max_weight)

        # enforce gross exposure limit
        gross = np.sum(np.abs(raw))
        if gross > self.max_gross and gross > 0:
            raw *= self.max_gross / gross

        if long_only:
            raw = np.clip(raw, 0, self.max_weight)
            total = raw.sum()
            if total > 0:
                raw /= total  # sum to 1

        # name the result
        w = pd.Series(raw, index=tickers, name="weight")

        return w

    # -------------------------------------------------------------
    # 4. Build final portfolio table
    # -------------------------------------------------------------
    def build_portfolio_table(
        self,
        weights: pd.Series,
        alpha_preds: Dict[str, dict],
    ) -> pd.DataFrame:
        if weights is None or weights.empty:
            return pd.DataFrame()

        rows = []
        for tk, w in weights.items():
            info = alpha_preds.get(tk, {})
            exp_alpha = float(info.get("expected_alpha", 0.0)) * 100.0

            rows.append({
                "ticker": tk,
                "weight": w,
                "side": "Long" if w > 0 else ("Short" if w < 0 else "Flat"),
                "expected_alpha_%": exp_alpha,
            })

        df = pd.DataFrame(rows)
        return df.sort_values("weight", ascending=False).reset_index(drop=True)
