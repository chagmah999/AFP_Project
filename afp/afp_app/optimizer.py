# afp_app/optimizer.py

from __future__ import annotations

import numpy as np
import pandas as pd


class UnifiedPortfolioOptimizer:
    """
    Build a unified optimized portfolio over the stock universe using
    the per ticker alpha forecasts and historical covariance.

    This version intentionally avoids cvxpy and uses a simple
    Markowitz style maximum Sharpe solution with basic constraints.

    High level:
      1) Construct expected alpha vector μ from alpha_preds
      2) Estimate covariance matrix Σ from recent daily returns
      3) Compute tangency style weights w ∝ Σ^{-1} μ
      4) Apply simple constraints:
           - long_only: clamp to w_i >= 0, normalize to sum to 1
           - if not long_only: scale so that sum(|w_i|) <= max_gross
    """

    def __init__(
        self,
        lookback_days: int = 252,
        max_gross: float = 1.0,
        long_only: bool = True,
    ):
        """
        Parameters
        ----------
        lookback_days : int
            Number of calendar days of history to use for covariance.
        max_gross : float
            Maximum gross exposure (sum of absolute weights) if allowing
            long/short. For a long-only portfolio this is effectively 1.
        long_only : bool
            If True, enforce w_i >= 0 and sum(w) = 1.
        """
        self.lookback_days = lookback_days
        self.max_gross = max_gross
        self.long_only = long_only

    # ---------------------------------------------------------
    # Build inputs: expected alpha vector and covariance matrix
    # ---------------------------------------------------------
    def build_inputs(
        self,
        alpha_preds: dict,
        price_data: pd.DataFrame,
    ) -> tuple[pd.Series, pd.DataFrame | None]:
        """
        From:
          alpha_preds: dict[ticker -> alpha_pred_dict]
              where alpha_pred_dict["expected_alpha"] is the H day excess alpha

          price_data: daily price DataFrame with columns:
              ["date", "ticker", "returns", "adjClose", ...]

        Returns:
          exp_alpha: pd.Series indexed by ticker with expected alpha
          cov: pd.DataFrame covariance matrix of daily returns
        """
        if not alpha_preds:
            return pd.Series(dtype=float), None

        # Build expected alpha vector
        exp_alpha = pd.Series(
            {
                tk: v.get("expected_alpha", np.nan)
                for tk, v in alpha_preds.items()
            },
            dtype=float,
        )
        exp_alpha = exp_alpha.replace([np.inf, -np.inf], np.nan).dropna()
        if exp_alpha.empty:
            return exp_alpha, None

        tickers = exp_alpha.index.tolist()

        if price_data is None or price_data.empty:
            return exp_alpha, None

        px = price_data[price_data["ticker"].isin(tickers)].copy()
        if px.empty:
            return exp_alpha, None

        # Restrict to recent window for covariance estimation
        px = px.sort_values("date")
        if "date" in px.columns:
            last_date = px["date"].max()
            cutoff = last_date - pd.Timedelta(days=self.lookback_days)
            px = px[px["date"] >= cutoff]

        # Pivot to date x ticker returns matrix
        ret_pivot = px.pivot_table(
            index="date",
            columns="ticker",
            values="returns",
        )

        # Drop dates that are all NaN
        ret_pivot = ret_pivot.dropna(how="all")
        if ret_pivot.empty:
            return exp_alpha, None

        cov = ret_pivot.cov()

        # Align rows/cols to alpha tickers
        cov = cov.reindex(index=tickers, columns=tickers)

        return exp_alpha, cov

    # ---------------------------------------------------------
    # Core optimizer: Markowitz tangency style
    # ---------------------------------------------------------
    def optimize(
        self,
        exp_alpha: pd.Series,
        cov: pd.DataFrame | None,
        long_only: bool | None = None,
    ) -> pd.Series:
        """
        Compute portfolio weights.

        If covariance is unavailable or singular, falls back to equal weight.

        Constraints:
          - Sum of weights = 1 if long_only is True
          - If long_only True: w_i >= 0
          - If long_only False: scale so sum(|w_i|) <= max_gross
        """
        if long_only is None:
            long_only = self.long_only

        tickers = exp_alpha.index.tolist()
        n = len(tickers)

        if cov is None or cov.empty:
            # Fallback: equal weight
            return pd.Series(
                np.ones(n) / n,
                index=tickers,
                name="weight",
            )

        # Replace NaNs with zero to avoid numerical issues
        cov = cov.fillna(0.0)

        mu = exp_alpha.values
        try:
            inv_cov = np.linalg.pinv(cov.values)
        except Exception:
            # Fallback if inversion fails
            return pd.Series(
                np.ones(n) / n,
                index=tickers,
                name="weight",
            )

        # Unconstrained tangency weights (up to scale)
        raw_w = inv_cov @ mu

        if long_only:
            raw_w = np.maximum(raw_w, 0.0)

        # If everything got clamped to zero or is non-finite, fall back
        if not np.any(np.isfinite(raw_w)) or np.all(raw_w == 0):
            return pd.Series(
                np.ones(n) / n,
                index=tickers,
                name="weight",
            )

        if long_only:
            # Normalize to sum to 1
            w = raw_w / np.sum(raw_w)
        else:
            # Allow long/short but cap gross exposure
            gross = np.sum(np.abs(raw_w))
            if gross > 0:
                w = raw_w / gross
            else:
                w = raw_w

            if self.max_gross is not None and self.max_gross > 0:
                gross = np.sum(np.abs(w))
                if gross > self.max_gross:
                    w = w * (self.max_gross / gross)

        return pd.Series(w, index=tickers, name="weight")

    # ---------------------------------------------------------
    # High level helper: from alpha_preds + price_data
    # ---------------------------------------------------------
    def build_portfolio(
        self,
        alpha_preds: dict,
        price_data: pd.DataFrame,
        long_only: bool | None = None,
    ) -> dict | None:
        """
        Convenience wrapper:

        1) Build expected alpha vector and covariance
        2) Optimize to get weights
        3) Return a dict with weights and expected alpha per ticker

        Returns None if inputs are empty.
        """
        exp_alpha, cov = self.build_inputs(alpha_preds, price_data)
        if exp_alpha.empty:
            return None

        weights = self.optimize(exp_alpha, cov, long_only=long_only)

        # Align expected alpha to the same index as weights
        exp_alpha_aligned = exp_alpha.reindex(weights.index)

        return {
            "weights": weights.sort_values(ascending=False),
            "expected_alpha": exp_alpha_aligned,
        }
