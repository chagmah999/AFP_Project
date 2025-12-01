from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.covariance import LedoitWolf

class UnifiedPortfolioOptimizer:

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

    def build_expected_returns(
        self,
        alpha_preds: Dict[str, dict],
        tickers: List[str],
    ) -> pd.Series:
        
        vals = []
        for tk in tickers:
            info = alpha_preds.get(tk, {})
            vals.append(float(info.get("expected_alpha", 0.0)))

        mu = pd.Series(vals, index=tickers)

        return mu

    def build_covariance(
        self,
        price_data: pd.DataFrame,
        tickers: List[str],
        lookback_days: int = 252,
    ) -> tuple[pd.DataFrame, List[str]]:
      
        if price_data is None or price_data.empty:
            return pd.DataFrame(), []

        df = price_data[price_data["ticker"].isin(tickers)].copy()
        if df.empty:
            return pd.DataFrame(), []

        df = df.sort_values("date")
        retcol = "log_returns" if "log_returns" in df.columns else "returns"

        unique_dates = df["date"].drop_duplicates().sort_values()
        if len(unique_dates) > lookback_days:
            cutoff = unique_dates.iloc[-lookback_days]
            df = df[df["date"] >= cutoff]

        pivot = df.pivot(index="date", columns="ticker", values=retcol)

        valid = pivot.count()[pivot.count() >= 50].index.tolist()

        if len(valid) < 2:  

            return pd.DataFrame(), []

        pivot_valid = pivot[valid].fillna(0.0)

        lw = LedoitWolf().fit(pivot_valid.values)
        Sigma = pd.DataFrame(lw.covariance_, index=valid, columns=valid)

        return Sigma, valid

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
        if n == 0:
            return pd.Series(dtype=float)

        if Sigma is None or Sigma.empty:
            if long_only:
                raw = np.clip(mu.values, a_min=0.0, a_max=None)
                if raw.sum() <= 0:
                    return pd.Series(dtype=float)
                w = raw / raw.sum()
            else:
                scores = mu.values
                pos = np.clip(scores, a_min=0.0, a_max=None)
                neg = np.clip(-scores, a_min=0.0, a_max=None)

                w = np.zeros(n)
                if pos.sum() > 0:
                    w += pos / pos.sum()
                if neg.sum() > 0:
                    w -= neg / neg.sum()

                gross = np.sum(np.abs(w))
                if gross > 0:
                    scale = min(self.max_gross / gross, 1.0)
                    w *= scale

            w = np.clip(w, -self.max_weight, self.max_weight)
            return pd.Series(w, index=tickers, name="weight")

        Sigma = Sigma.loc[mu.index, mu.index]  

        diag = np.diag(Sigma.values)
        diag = np.where(diag <= 0, 1e-6, diag)

        if long_only:

            scores = mu.values / np.sqrt(diag)
            scores = np.clip(scores, a_min=0.0, a_max=None)
            if scores.sum() <= 0:
                return pd.Series(dtype=float)
            w = scores / scores.sum()
        else:

            scores = mu.values / np.sqrt(diag)

            pos = np.clip(scores, a_min=0.0, a_max=None)
            neg = np.clip(-scores, a_min=0.0, a_max=None)

            w = np.zeros(n)

            target_gross = min(self.max_gross, 1.0)
            target_long = target_gross / 2.0
            target_short = target_gross / 2.0

            if pos.sum() > 0:
                w_long = pos / pos.sum() * target_long
                w += w_long
            if neg.sum() > 0:
                w_short = neg / neg.sum() * target_short
                w -= w_short

        w = np.clip(w, -self.max_weight, self.max_weight)

        gross = np.sum(np.abs(w))
        if gross > self.max_gross and gross > 0:
            w *= self.max_gross / gross

        return pd.Series(w, index=tickers, name="weight")

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
