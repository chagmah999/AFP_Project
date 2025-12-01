from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

class FactorPremiaForecaster:
  

    def __init__(self, lookback_window: int = 252, forecast_horizon: int = 21):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon

        self.models: dict[str, dict] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.feature_importance: dict[str, pd.DataFrame] = {}
        self.validation_summary: dict[str, dict] = {}

    def prepare_features_targets(
        self,
        data: pd.DataFrame,
        target_factor: str,
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        
        H = self.forecast_horizon
        df = data.copy()

        base_features = [
            "rates_level",
            "rates_1m_change",
            "term_spread_10y2y",
            "term_spread_10y3m",
            "vix_close",
            "vix_percentile",
            "credit_spread_level",
            "credit_spread_1m_change",
        ]

        feature_cols: list[str] = base_features.copy()

        if target_factor in df.columns:

            for lag in [1, 5, 21, 63]:
                col = f"{target_factor}_lag{lag}"
                df[col] = df[target_factor].shift(lag)
                feature_cols.append(col)

            ma21_col = f"{target_factor}_ma21"
            ma63_col = f"{target_factor}_ma63"
            df[ma21_col] = df[target_factor].rolling(21).mean()
            df[ma63_col] = df[target_factor].rolling(63).mean()
            feature_cols.extend([ma21_col, ma63_col])

            fwd_col = f"{target_factor}_forward"
            df[fwd_col] = df[target_factor].shift(-H).rolling(H).mean()
        else:

            fwd_col = f"{target_factor}_forward"
            df[fwd_col] = np.nan

        feature_cols = [c for c in feature_cols if c in df.columns]
        feature_cols = list(dict.fromkeys(feature_cols))

        valid = df[feature_cols + [fwd_col]].dropna()
        if valid.empty:
            return pd.DataFrame(), pd.Series(dtype=float), feature_cols

        X = valid[feature_cols]
        y = valid[fwd_col]

        return X, y, feature_cols

    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        factor_name: str,
    ) -> dict[str, object]:
     
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[factor_name] = scaler

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, y_train)

        lasso = Lasso(alpha=0.001)
        lasso.fit(X_scaled, y_train)

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )
        rf.fit(X_scaled, y_train)

        self.models[factor_name] = {
            "ridge": ridge,
            "lasso": lasso,
            "random_forest": rf,
        }

        self.feature_importance[factor_name] = pd.DataFrame(
            {
                "feature": X_train.columns,
                "ridge_coef": np.abs(ridge.coef_),
                "lasso_coef": np.abs(lasso.coef_),
                "rf_importance": rf.feature_importances_,
            }
        ).sort_values("rf_importance", ascending=False)

        return self.models[factor_name]

    def walk_forward_validation(
        self,
        data: pd.DataFrame,
        target_factor: str,
    ) -> pd.DataFrame | None:
     
        print(f"\nWalk-forward validation for {target_factor}...")

        X, y, features = self.prepare_features_targets(data, target_factor)
        if X.empty or y.empty:
            print(f"No data for factor {target_factor}")
            return None

        if len(X) < self.lookback_window + self.forecast_horizon:
            print("Insufficient data for walk-forward validation")
            return None

        y_lag_all = y.shift(1)

        n_splits = 5
        test_size = len(X) // (n_splits + 1)
        results = []

        for i in range(n_splits):
            train_end = (i + 1) * test_size
            test_end = min(train_end + test_size, len(X))

            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]

            y_train_lag = y_lag_all.iloc[:train_end]
            y_test_lag = y_lag_all.iloc[train_end:test_end]

            if len(X_train) < 50 or len(X_test) < 10:
                continue

            self.train_models(X_train, y_train, target_factor)

            X_test_scaled = self.scalers[target_factor].transform(X_test)
            predictions = {}
            for name, mdl in self.models[target_factor].items():
                predictions[name] = mdl.predict(X_test_scaled)

            ensemble_pred = np.mean(list(predictions.values()), axis=0)

            mask_train = y_train.notna() & y_train_lag.notna()
            y_curr = y_train[mask_train]
            y_lag = y_train_lag[mask_train]

            if len(y_curr) >= 20:

                b, a = np.polyfit(y_lag.values, y_curr.values, 1)
            else:

                a = float(y_train.mean())
                b = 0.0

            if y_train.dropna().empty:
                fallback_lag = 0.0
            else:
                fallback_lag = float(y_train.dropna().iloc[-1])

            y_test_lag_filled = y_test_lag.fillna(fallback_lag)
            ar1_pred = a + b * y_test_lag_filled.values

            fold = {
                "factor": target_factor,
                "fold": i,
                "n_test": len(y_test),
                "test_start": (
                    data.iloc[train_end]["date"]
                    if "date" in data.columns
                    else train_end
                ),
                "test_end": (
                    data.iloc[test_end - 1]["date"]
                    if "date" in data.columns
                    else test_end
                ),
            }

            for name, pred in predictions.items():
                fold[f"{name}_rmse"] = float(
                    np.sqrt(mean_squared_error(y_test, pred))
                )
                fold[f"{name}_mae"] = float(
                    mean_absolute_error(y_test, pred)
                )
                fold[f"{name}_hit"] = float(
                    np.mean(np.sign(pred) == np.sign(y_test))
                )

            fold["ensemble_rmse"] = float(
                np.sqrt(mean_squared_error(y_test, ensemble_pred))
            )
            fold["ensemble_mae"] = float(
                mean_absolute_error(y_test, ensemble_pred)
            )
            fold["ensemble_hit"] = float(
                np.mean(np.sign(ensemble_pred) == np.sign(y_test))
            )

            fold["ar1_rmse"] = float(
                np.sqrt(mean_squared_error(y_test, ar1_pred))
            )
            fold["ar1_mae"] = float(
                mean_absolute_error(y_test, ar1_pred)
            )
            fold["ar1_hit"] = float(
                np.mean(np.sign(ar1_pred) == np.sign(y_test))
            )

            results.append(fold)

        if not results:

            return None

        results_df = pd.DataFrame(results)

        summary = {
            "factor": target_factor,
            "ensemble_rmse": float(results_df["ensemble_rmse"].mean()),
            "ensemble_mae": float(results_df["ensemble_mae"].mean()),
            "ensemble_hit_rate": float(results_df["ensemble_hit"].mean()),
            "ar1_rmse": float(results_df["ar1_rmse"].mean()),
            "ar1_mae": float(results_df["ar1_mae"].mean()),
            "ar1_hit_rate": float(results_df["ar1_hit"].mean()),
        }
        self.validation_summary[target_factor] = summary

        print(f"\n{target_factor} Validation Results")
        print("=" * 50)
        print(f"Average Ensemble RMSE: {summary['ensemble_rmse']:.4f}")
        print(f"Average Ensemble MAE : {summary['ensemble_mae']:.4f}")
        print(f"Average Ensemble Hit : {summary['ensemble_hit_rate']:.2%}")
        print(f"Average AR(1) RMSE   : {summary['ar1_rmse']:.4f}")
        print(f"Average AR(1) MAE    : {summary['ar1_mae']:.4f}")
        print(f"Average AR(1) Hit    : {summary['ar1_hit_rate']:.2%}")

        return results_df

    def forecast_next(
        self,
        data: pd.DataFrame,
        target_factor: str,
    ) -> dict | None:
        
        X, y, features = self.prepare_features_targets(data, target_factor)
        if X.empty or y.empty:
            return None

        if len(X) < self.lookback_window:
            print(f"[{target_factor}] Not enough data to forecast")
            return None

        self.train_models(X, y, target_factor)

        X_latest = X.iloc[-1:].values
        scaler = self.scalers[target_factor]
        X_scaled = scaler.transform(X_latest)

        preds = {}
        for name, mdl in self.models[target_factor].items():
            preds[name] = float(mdl.predict(X_scaled)[0])

        ensemble = float(np.mean(list(preds.values())))

        y_lag_all = y.shift(1)

        mask = y.notna() & y_lag_all.notna()
        y_curr = y[mask]
        y_lag = y_lag_all[mask]

        if len(y_curr) >= 20:

            b, a = np.polyfit(y_lag.values, y_curr.values, 1)
        else:

            a = float(y.mean())
            b = 0.0

        if y.dropna().empty:
            last_y = 0.0
        else:
            last_y = float(y.dropna().iloc[-1])

        ar1_forecast = a + b * last_y

        drivers = self.feature_importance.get(target_factor, None)
        top = []
        if drivers is not None and not drivers.empty:
            top = (
                drivers.sort_values("rf_importance", ascending=False)
                .head(10)[["feature", "rf_importance"]]
                .to_dict("records")
            )

        coef_norm = float(
            np.linalg.norm(drivers["rf_importance"].values)
        ) if drivers is not None else 0.0
        if coef_norm > 0.5:
            conf = "High"
        elif coef_norm > 0.2:
            conf = "Medium"
        else:
            conf = "Low"

        forecast = {
            "factor": target_factor,
            "forecast_horizon_days": self.forecast_horizon,
            "ensemble_forecast": ensemble,
            "ar1_forecast": float(ar1_forecast),
            "model_forecasts": preds,
            "top_drivers": top,
            "forecast_date": (
                data["date"].iloc[-1] if "date" in data.columns else "latest"
            ),
            "confidence": conf,
        }

        return forecast
