from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

MacroShock = dict  # keys: rates_bp, term_10y2y_bp, term_10y3m_bp, vix_pct, credit_bp


def apply_macro_shocks_to_row(row: pd.Series, shocks: MacroShock) -> pd.Series:
    """
    Return a copy of the feature row with macro shocks applied.
    Assumptions:
      - rates_level, term_spread_* are in percentage points (eg 4.25), so +25 bps = +0.25
      - credit_spread_level is in percentage points (eg 1.30), so +50 bps = +0.50
      - vix_close is in index units, so a +10% shock multiplies by 1.10
    """
    r = row.copy()

    # Rates level shift in bps
    if "rates_level" in r and shocks.get("rates_bp") is not None:
        r["rates_level"] = float(r["rates_level"]) + (shocks["rates_bp"] / 100.0)

    # Term spread shifts in bps
    if "term_spread_10y2y" in r and shocks.get("term_10y2y_bp") is not None:
        r["term_spread_10y2y"] = float(r["term_spread_10y2y"]) + (shocks["term_10y2y_bp"] / 100.0)
    if "term_spread_10y3m" in r and shocks.get("term_10y3m_bp") is not None:
        r["term_spread_10y3m"] = float(r["term_spread_10y3m"]) + (shocks["term_10y3m_bp"] / 100.0)

    # Credit spread level in bps
    if "credit_spread_level" in r and shocks.get("credit_bp") is not None:
        r["credit_spread_level"] = float(r["credit_spread_level"]) + (shocks["credit_bp"] / 100.0)

    # VIX percent change
    if "vix_close" in r and shocks.get("vix_pct") is not None:
        r["vix_close"] = float(r["vix_close"]) * (1.0 + shocks["vix_pct"] / 100.0)

    return r


def _expected_feature_list_from_forecaster(forecaster, factor: str) -> Optional[List[str]]:
    """
    Try to recover the model's expected feature order from stored feature importance.
    Falls back to None if not available.
    """
    fi = getattr(forecaster, "feature_importance", {})
    if isinstance(fi, dict) and factor in fi and not fi[factor].empty and "feature" in fi[factor].columns:
        return list(fi[factor]["feature"].tolist())
    return None


def scenario_factor_premia(
    forecaster,
    modeling: pd.DataFrame,
    factor: str,
    shocks: MacroShock | None = None,
):
    """
    Scenario forecast using the exact cached feature vector if available; otherwise
    falls back to rebuilding from `modeling`.
    """
    # Need trained model + scaler
    if not hasattr(forecaster, "models") or factor not in forecaster.models:
        return None
    if not hasattr(forecaster, "scalers") or factor not in forecaster.scalers:
        return None

    # Prefer the exact row/ordering used in the base forecast
    if getattr(forecaster, "latest_feature_row", None) and factor in forecaster.latest_feature_row:
        x = forecaster.latest_feature_row[factor].copy()
        feat_list = getattr(forecaster, "feature_order", {}).get(factor, list(x.index))
        x = x.reindex(feat_list).astype(float)
    else:
        # Fallback: rebuild from modeling using feature_importance order
        fi = getattr(forecaster, "feature_importance", {}).get(factor, None)
        if fi is None or fi.empty or "feature" not in fi.columns or modeling.empty:
            return None
        feat_list = list(fi["feature"].tolist())
        modeling = modeling.sort_values("date") if "date" in modeling.columns else modeling
        latest = modeling.iloc[-1]
        x = pd.Series({col: float(latest[col]) if col in latest.index else 0.0 for col in feat_list})
        x = x.reindex(feat_list).astype(float)

    # Apply shocks
    if shocks:
        x = apply_macro_shocks_to_row(x, shocks)

    # Predict
    scaler = forecaster.scalers[factor]
    X_scaled = scaler.transform(x.values.reshape(1, -1))
    preds = {name: float(mdl.predict(X_scaled)[0]) for name, mdl in forecaster.models[factor].items()}
    ensemble = float(np.mean(list(preds.values())))

    # Drivers from stored importance
    fi = getattr(forecaster, "feature_importance", {}).get(factor, None)
    top = []
    if fi is not None and not fi.empty:
        top = fi.sort_values("rf_importance", ascending=False).head(5)[["feature", "rf_importance"]].to_dict("records")

    return {
        "factor": factor,
        "ensemble_forecast": ensemble,
        "model_forecasts": preds,
        "top_drivers": top,
    }



def scenario_stress(
    stress_model,
    data: pd.DataFrame,
    shocks: MacroShock | None = None,
) -> Optional[Dict]:
    """
    Compute stress probability for a one-step what-if by:
      - taking last feature row
      - applying macro shocks
      - using trained scaler + model to predict_proba
    """
    if stress_model.model is None:
        stress_model.fit(data)

    feats = stress_model.prepare_stress_features(data)
    if feats.empty:
        return None

    x = feats.iloc[-1].copy()

    if shocks:
        if "vix_level" in x and shocks.get("vix_pct") is not None:
            x["vix_level"] = float(x["vix_level"]) * (1.0 + shocks["vix_pct"] / 100.0)
        if "credit_level" in x and shocks.get("credit_bp") is not None:
            x["credit_level"] = float(x["credit_level"]) + (shocks["credit_bp"] / 100.0)
        if "rates_change" in x and shocks.get("rates_bp") is not None:
            x["rates_change"] = float(x.get("rates_change", 0.0)) + (shocks["rates_bp"] / 100.0)
        if "term_spread" in x and shocks.get("term_10y2y_bp") is not None:
            x["term_spread"] = float(x["term_spread"]) + (shocks["term_10y2y_bp"] / 100.0)

    X_scaled = stress_model.scaler.transform(x.values.reshape(1, -1))
    prob = float(stress_model.model.predict_proba(X_scaled)[0, 1])

    regime = "NORMAL"
    if prob > 0.7:
        regime = "HIGH RISK"
    elif prob > 0.3:
        regime = "ELEVATED"

    return {"stress_probability": prob, "regime": regime}
