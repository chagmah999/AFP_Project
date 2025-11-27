# afp_app/optimizer.py

import numpy as np
import pandas as pd
import cvxpy as cp

class UnifiedPortfolioOptimizer:
    """
    Build a single unified long/short portfolio using:
      - forecasted factor premia (dict of factor → premium)
      - stock-level factor exposures (from factor metrics)
    """

    def __init__(
        self,
        max_gross=1.0,
        max_weight=0.05,
        risk_aversion=1.0,
    ):
        self.max_gross = max_gross
        self.max_weight = max_weight
        self.risk_aversion = risk_aversion

    def build_expected_returns(self, metrics_df, forecasts):
        """
        Expected stock return = sum_k (forecast_k * exposure_i,k)
        """
        betas = {}

        for factor, fc in forecasts.items():
            prem = fc.get("ar1_forecast", None)
            if prem is None:
                prem = fc.get("ensemble_forecast", 0.0)

            col = {
                "VALUE": "value_score",
                "QUALITY": "quality_score",
                "MOMENTUM": "momentum_score",
                "LOW_VOL": "lowvol_score",
            }.get(factor)

            if col in metrics_df.columns:
                betas[factor] = metrics_df[col].fillna(0.0) * prem

        # Sum contributions across factors → expected return per stock
        er = pd.DataFrame(betas).sum(axis=1)
        return er.values.reshape(-1), betas

    def optimize(self, expected_returns, cov_matrix, tickers):
        """
        Max Sharpe portfolio under:
           1. dollar neutrality
           2. gross exposure ≤ max_gross
           3. |w_i| ≤ max_weight
        """

        n = len(tickers)
        w = cp.Variable(n)

        expected_port_return = expected_returns @ w
        port_var = cp.quad_form(w, cov_matrix)

        objective = cp.Maximize(expected_port_return - self.risk_aversion * port_var)

        constraints = [
            cp.sum(w) == 0,                               # dollar neutral
            cp.norm1(w) <= self.max_gross,                # leverage limit
            w <= self.max_weight,
            w >= -self.max_weight,
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)

        if w.value is None:
            return pd.DataFrame()

        return pd.DataFrame({
            "ticker": tickers,
            "weight": w.value,
            "expected_return": expected_returns * w.value
        }).sort_values("weight", ascending=False)
