# afp_app/engine.py

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


class MarketMancerEngine:
    """
    MarketMancerEngine

    This class takes the model outputs produced by the pipeline
    and turns them into a simple, human-readable set of
    "recommendations" for factors and single-name stocks.

    Inputs:
      - factor_forecasts: dict keyed by factor name
          Each value is expected to contain at least:
            {
              "ensemble_forecast": float,   # H-day expected premium (in decimal form)
              ... (other fields ignored here)
            }

      - alpha_preds: dict keyed by ticker
          Each value is expected to contain at least:
            {
              "expected_alpha": float,      # H-day expected idiosyncratic return (decimal)
              "drivers": {...},             # may include fundamental_score, features, etc.
            }

      - stress: kept only for backwards compatibility with older versions
        of the app. It is ignored in this simplified engine.

    Output (via generate()):
      A dictionary with three sections:
        {
          "factors": {
              "top_overweight": [...],
              "top_underweight": [...],
          },
          "stocks": {
              "top_longs": [...],
              "top_shorts": [...],
          },
          "meta": {
              "description": "...",
          }
        }

      The exact structure is easy to inspect in the Streamlit app
      and is deliberately simple so it is easy to explain to sponsors.
    """

    def __init__(
        self,
        factor_forecasts: Dict[str, dict] | None,
        alpha_preds: Dict[str, dict] | None,
        stress: Any | None = None,
    ) -> None:
        # Store inputs, falling back to empty dicts if None
        self.factor_forecasts = factor_forecasts or {}
        self.alpha_preds = alpha_preds or {}
        # Stress is ignored by design, but kept in the signature so that
        # older code that passed a third argument still runs without error.
        self.stress = None

    # -------------------------------------------------------------
    # Helper: build factor summary DataFrame
    # -------------------------------------------------------------
    def _build_factor_df(self) -> pd.DataFrame:
        rows = []
        for name, info in self.factor_forecasts.items():
            if info is None:
                continue
            prem = info.get("ensemble_forecast", None)
            if prem is None:
                continue
            rows.append(
                {
                    "factor": name,
                    "expected_premium_dec": float(prem),
                    "expected_premium_pct": float(prem) * 100.0,
                }
            )

        if not rows:
            return pd.DataFrame(columns=["factor", "expected_premium_dec", "expected_premium_pct"])

        df = pd.DataFrame(rows)
        df = df.sort_values("expected_premium_dec", ascending=False).reset_index(drop=True)
        return df

    # -------------------------------------------------------------
    # Helper: build alpha summary DataFrame
    # -------------------------------------------------------------
    def _build_alpha_df(self) -> pd.DataFrame:
        rows = []
        for tk, info in self.alpha_preds.items():
            if info is None:
                continue
            alpha_dec = info.get("expected_alpha", None)
            if alpha_dec is None:
                continue

            drivers = info.get("drivers", {}) or {}
            rows.append(
                {
                    "ticker": tk,
                    "expected_alpha_dec": float(alpha_dec),
                    "expected_alpha_pct": float(alpha_dec) * 100.0,
                    "fundamental_score": drivers.get("fundamental_score", None),
                }
            )

        if not rows:
            return pd.DataFrame(columns=["ticker", "expected_alpha_dec", "expected_alpha_pct", "fundamental_score"])

        df = pd.DataFrame(rows)
        df = df.sort_values("expected_alpha_dec", ascending=False).reset_index(drop=True)
        return df

    # -------------------------------------------------------------
    # Main public method: generate recommendation object
    # -------------------------------------------------------------
    def generate(self) -> Dict[str, Any]:
        """
        Build a simple recommendation dictionary based on:

          - factor forecasts (which factors to overweight / underweight)
          - single-name alpha forecasts (which stocks to tilt long / short)

        There is deliberately no use of a stress regime or
        complex scenario logic in this engine. Everything is
        transparent and directly tied to the model outputs.
        """

        # ---------- Factor recommendations ----------
        df_factors = self._build_factor_df()
        factor_recs: Dict[str, Any] = {
            "top_overweight": [],
            "top_underweight": [],
        }

        if not df_factors.empty:
            # Top 3 positive premia: overweight
            top_over = df_factors.head(3)
            for _, row in top_over.iterrows():
                if row["expected_premium_dec"] <= 0:
                    continue
                factor_recs["top_overweight"].append(
                    {
                        "factor": row["factor"],
                        "expected_premium_pct": row["expected_premium_pct"],
                        "direction": "overweight",
                    }
                )

            # Bottom 3 premia (most negative): underweight
            bottom_under = df_factors.sort_values(
                "expected_premium_dec", ascending=True
            ).head(3)
            for _, row in bottom_under.iterrows():
                if row["expected_premium_dec"] >= 0:
                    continue
                factor_recs["top_underweight"].append(
                    {
                        "factor": row["factor"],
                        "expected_premium_pct": row["expected_premium_pct"],
                        "direction": "underweight",
                    }
                )

        # ---------- Stock recommendations ----------
        df_alpha = self._build_alpha_df()
        stock_recs: Dict[str, Any] = {
            "top_longs": [],
            "top_shorts": [],
        }

        if not df_alpha.empty:
            # Top 10 positive alphas: candidates for long tilt
            top_longs = df_alpha[df_alpha["expected_alpha_dec"] > 0].head(10)
            for _, row in top_longs.iterrows():
                stock_recs["top_longs"].append(
                    {
                        "ticker": row["ticker"],
                        "expected_alpha_pct": row["expected_alpha_pct"],
                        "fundamental_score": row.get("fundamental_score", None),
                    }
                )

            # Top 10 negative alphas: candidates for short / underweight tilt
            top_shorts = (
                df_alpha[df_alpha["expected_alpha_dec"] < 0]
                .sort_values("expected_alpha_dec", ascending=True)
                .head(10)
            )
            for _, row in top_shorts.iterrows():
                stock_recs["top_shorts"].append(
                    {
                        "ticker": row["ticker"],
                        "expected_alpha_pct": row["expected_alpha_pct"],
                        "fundamental_score": row.get("fundamental_score", None),
                    }
                )

        # ---------- Meta info ----------
        meta = {
            "description": (
                "Recommendations are based solely on factor premia forecasts "
                "and per-stock alpha forecasts. No stress regime or scenario "
                "logic is used in this engine."
            ),
            "num_factors": int(len(self.factor_forecasts)),
            "num_alpha_names": int(len(self.alpha_preds)),
        }

        return {
            "factors": factor_recs,
            "stocks": stock_recs,
            "meta": meta,
        }
