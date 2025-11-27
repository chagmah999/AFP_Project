# afp_app/optimizer.py

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List


class UnifiedPortfolioOptimizer:
    """
    Unified portfolio optimizer that builds a single long short portfolio
    from per ticker expected alpha and a covariance estimate.

    It uses a simple mean variance style heuristic:
      - tilt positions in proportion to alpha / variance
      - roughly dollar neutral
      - enforce per name weight cap and gross exposure cap
    """

    def __init__(
        self,
        risk_aversion: float = 10.0,
        max_gross: float = 1.5,
        max_weight: float = 0.10,
    ):
        """
        Parameters
        ----------
        risk_aversion : float
            Lambda on variance in the heuristic objective.
        max_gross : float
            Soft cap on total gross exposure, sum_i |w_i|.
        max_weight : float
            Hard cap on |w_i| for any single name.
        """
        self.risk_aversion = risk_aversion
        self.max_gross = max_gross
        self.max_weight = max_weight

        # Backwards compatible alias in case other code uses this name
        self.max_weight_per_name = max_weight

    # ----------------------------------------------------------
    # 1. Expected returns from alpha predictions
    # ----------------------------------------------------------
    def build_expected_returns(
        self,
        alpha_preds: Dict[str, dict],
        tickers: List[str],
    ) -> pd.Series:
        """
        Build a vector of expected returns mu from the per ticker alpha predictions.

        alpha_preds[ticker]["expected_alpha"] is assumed to be in return units,
        not percent.
        """
        vals = []
        for tk in tickers:
            info = alpha_preds.get(tk, {})
            vals.append(float(info.get("expected_alpha", 0.0)))
        mu = pd.Series(vals, index=tickers, name="mu")
        return mu

    # ----------------------------------------------------------
    # 2. Covariance matrix from price data
    # ----------------------------------------------------------
    def build_covariance(
        self,
        price_data: pd.DataFrame,
        tickers: List[str],
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """
        Build a return covariance matrix Sigma for the given tickers using
        recent daily returns.

        Uses log_returns if available, else returns.
        """
        if price_data is None or price_data.empty:
            return pd.DataFrame()

        df = price_data[price_data["ticker"].isin(tickers)].copy()
        if df.empty:
            return pd.DataFrame()

        df = df.sort_values("date")

        ret_col = "log_returns" if "log_returns" in df.columns else "returns"

        # Restrict to last lookback_days by date
        unique_dates = df["date"].drop_duplicates().sort_values()
        if len(unique_dates) > lookback_days:
            cutoff = unique_dates.iloc[-lookback_days]
            df = df[df["date"] >= cutoff]

        pivot = df.pivot_table(index="date", columns="ticker", values=ret_col)
        pivot = pivot.dropna(how="all")
        if pivot.shape[0] < 2:
            return pd.DataFrame()

        Sigma = pivot.cov()

        # Ensure all tickers appear in index/columns in a fixed order
        Sigma = Sigma.reindex(index=tickers, columns=tickers)

        # Fill diagonal if missing with sample variance, others with zero
        for tk in Sigma.index:
            if np.isnan(Sigma.loc[tk, tk]):
                if tk in pivot.columns:
                    Sigma.loc[tk, tk] = pivot[tk].var()
                else:
                    Sigma.loc[tk, tk] = 0.0

        Sigma = Sigma.fillna(0.0)
        return Sigma

    # ----------------------------------------------------------
    # 3. Heuristic mean variance optimization
    # ----------------------------------------------------------
    def optimize(self, mu: pd.Series, Sigma: pd.DataFrame) -> pd.Series:
        """
        Produce a vector of weights w given expected returns mu and covariance Sigma.

        Heuristic:
          - Start with w_raw proportional to mu / (lambda * variance)
          - Center weights to be roughly dollar neutral
          - Clip to per name bound |w_i| <= max_weight
          - Scale to respect gross exposure cap
          - Finally renormalize net exposure to sum to 1 (if possible)
        """
        if mu is None or mu.empty:
            return pd.Series(dtype=float)

        tickers = list(mu.index)
        n = len(tickers)
        if n == 0:
            return pd.Series(dtype=float)

        if Sigma is None or Sigma.empty:
            # Fallback: only alpha, no risk. Use simple centered weights.
            raw = mu.values.astype(float)
            raw = raw - raw.mean()
        else:
            diag = np.diag(Sigma.values)
            diag = np.where(diag <= 0, 1e-6, diag)
            raw = mu.values / (self.risk_aversion * diag)

            # Center to roughly dollar neutral
            raw = raw - raw.mean()

        # Clip per name
        raw = np.clip(raw, -self.max_weight, self.max_weight)

        # If all zero, fallback to equal weight
        if np.allclose(raw, 0.0):
            w_equal = np.repeat(1.0 / n, n)
            return pd.Series(w_equal, index=tickers, name="weight")

        # Enforce gross exposure cap
        gross = np.sum(np.abs(raw))
        if gross > 0:
            scale = min(self.max_gross / gross, 1.0)
            raw = raw * scale

        # Renormalize net exposure to sum to 1 if not tiny
        net = raw.sum()
        if np.abs(net) > 1e-6:
            raw = raw / net

        w = pd.Series(raw, index=tickers, name="weight")
        return w

    # ----------------------------------------------------------
    # 4. Pretty table for display
    # ----------------------------------------------------------
    def build_portfolio_table(
        self,
        weights: pd.Series,
        alpha_preds: Dict[str, dict],
    ) -> pd.DataFrame:
        """
        Turn weights and alpha predictions into a nice display table.
        """
        if weights is None or weights.empty:
            return pd.DataFrame()

        rows = []
        for tk, w in weights.items():
            info = alpha_preds.get(tk, {})
            exp_alpha = float(info.get("expected_alpha", 0.0)) * 100.0
            rows.append(
                {
                    "ticker": tk,
                    "weight": w,
                    "side": "Long" if w > 0 else ("Short" if w < 0 else "Flat"),
                    "expected_alpha_%": exp_alpha,
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("weight", ascending=False).reset_index(drop=True)
        return df
