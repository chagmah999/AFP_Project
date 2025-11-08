import numpy as np
from datetime import datetime

class MarketMancerEngine:
    def __init__(self, factor_forecasts: dict, alpha_predictions: dict, stress_forecast: dict):
        self.factor_forecasts = factor_forecasts or {}
        self.alpha_predictions = alpha_predictions or {}
        self.stress_forecast = stress_forecast or {}

    def generate(self) -> dict:
        recs = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_regime": self.stress_forecast.get("regime", "UNKNOWN"),
            "factor_tilts": {},
            "stock_picks": [],
            "summary": ""
        }
        risk_adj = 1.0
        if self.stress_forecast:
            r = self.stress_forecast.get("regime")
            if r == "HIGH RISK":
                risk_adj = 0.3
            elif r == "ELEVATED":
                risk_adj = 0.7

        if self.factor_forecasts:
            ordered = sorted(self.factor_forecasts.items(), key=lambda kv: kv[1]["ensemble_forecast"], reverse=True)
            for f, fc in ordered:
                er = fc["ensemble_forecast"]
                size = float(np.clip(er * 10 * risk_adj, -1, 1))
                recs["factor_tilts"][f] = {
                    "expected_premium": er,
                    "position": "LONG" if size > 0 else "SHORT",
                    "size": abs(size),
                    "confidence": fc.get("confidence","Medium")
                }

        if self.alpha_predictions:
            ordered = sorted(self.alpha_predictions.items(), key=lambda kv: kv[1]["expected_alpha"], reverse=True)
            for tk, pred in ordered[:5]:
                recs["stock_picks"].append({
                    "ticker": tk,
                    "expected_alpha": float(pred["expected_alpha"]),
                    "position_size": float(np.clip(pred["expected_alpha"] * 20 * risk_adj, 0, 0.1))
                })

        parts = []
        if self.stress_forecast:
            parts.append(f"Market regime {self.stress_forecast['regime']} "
                         f"(stress probability {self.stress_forecast['stress_probability']*100:.1f}%)")
        if recs["factor_tilts"]:
            top = list(recs["factor_tilts"].keys())[0]
            parts.append(f"Favor {top} factor "
                         f"(expected premium {recs['factor_tilts'][top]['expected_premium']*100:.2f}%)")
        if recs["stock_picks"]:
            parts.append(f"Top pick {recs['stock_picks'][0]['ticker']} "
                         f"(expected alpha {recs['stock_picks'][0]['expected_alpha']*100:.2f}%)")
        recs["summary"] = ". ".join(parts)
        return recs
