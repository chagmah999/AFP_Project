import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class FactorPremiaForecaster:
    """
    Forecast expected factor premia using macro variables.
    Trains simple models and supports walk-forward validation + one-step forecast.
    """

    def __init__(self, lookback_window: int = 252, forecast_horizon: int = 21):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.models = {}             # factor -> {ridge, lasso, random_forest}
        self.scalers = {}            # factor -> StandardScaler
        self.feature_importance = {} # factor -> DataFrame(feature, ridge_coef, lasso_coef, rf_importance)

        # NEW: cache to ensure scenarios reuse the exact same feature vector & order
        self.latest_feature_row = {}  # factor -> pd.Series (unscaled, in training feature order)
        self.feature_order = {}       # factor -> list[str]

    def prepare_features_targets(self, data: pd.DataFrame, target_factor: str):
        """
        Build X (features) and y (forward target) for a given factor.
        """
        macro_features = [
            'rates_level', 'rates_1m_change', 'term_spread_10y2y', 'term_spread_10y3m',
            'vix_close', 'vix_percentile', 'credit_spread_level', 'credit_spread_1m_change'
        ]

        # Add lagged target returns as features if present
        for lag in [1, 5, 21, 63]:
            if target_factor in data.columns:
                data[f'{target_factor}_lag{lag}'] = data[target_factor].shift(lag)
                macro_features.append(f'{target_factor}_lag{lag}')

        # Keep only available cols
        features = [f for f in macro_features if f in data.columns]

        # Forward average over the horizon
        if target_factor in data.columns:
            data[f'{target_factor}_forward'] = (
                data[target_factor].shift(-self.forecast_horizon)
                                   .rolling(self.forecast_horizon).mean()
            )
        else:
            data[f'{target_factor}_forward'] = np.nan  # fallback; you can synthesize if needed

        valid = data[features + [f'{target_factor}_forward']].dropna()
        X = valid[features]
        y = valid[f'{target_factor}_forward']

        return X, y, features

    def train_models(self, X_train, y_train, factor_name: str):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        self.scalers[factor_name] = scaler

        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(Xs, y_train)

        lasso = Lasso(alpha=0.001, random_state=42)
        lasso.fit(Xs, y_train)

        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(Xs, y_train)

        self.models[factor_name] = {'ridge': ridge, 'lasso': lasso, 'random_forest': rf}
        self.feature_importance[factor_name] = pd.DataFrame({
            'feature': X_train.columns,
            'ridge_coef': np.abs(getattr(ridge, 'coef_', np.zeros(X_train.shape[1]))),
            'lasso_coef': np.abs(getattr(lasso, 'coef_', np.zeros(X_train.shape[1]))),
            'rf_importance': getattr(rf, 'feature_importances_', np.zeros(X_train.shape[1]))
        }).sort_values('rf_importance', ascending=False)
        return self.models[factor_name]

    def walk_forward_validation(self, data: pd.DataFrame, target_factor: str):
        """
        Perform walk-forward validation and store summary metrics:
        - average RMSE, MAE, hit rate across folds
        Also returns a DataFrame of fold-level results.
        """
        print(f"\nWalk-forward validation for {target_factor}...")

        X, y, features = self.prepare_features_targets(data, target_factor)

        if len(X) < self.lookback_window + self.forecast_horizon:
            print(f"Insufficient data for walk-forward validation")
            return None

        # Split points for walk-forward
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

            if len(X_train) < 50 or len(X_test) < 10:
                continue

            # Train models
            self.train_models(X_train, y_train, target_factor)

            # Make predictions
            X_test_scaled = self.scalers[target_factor].transform(X_test)

            predictions = {}
            for model_name, model in self.models[target_factor].items():
                pred = model.predict(X_test_scaled)
                predictions[model_name] = pred

            # Ensemble prediction (simple average)
            ensemble_pred = np.mean(list(predictions.values()), axis=0)

            # Fold level metrics
            fold_results = {
                "factor": target_factor,
                "fold": i,
                "n_test": len(y_test),
                "test_start": data.iloc[train_end]["date"] if "date" in data.columns else train_end,
                "test_end": data.iloc[test_end - 1]["date"] if "date" in data.columns else test_end,
            }

            # Metrics for each model (RMSE, MAE, hit rate)
            for model_name, pred in predictions.items():
                fold_results[f"{model_name}_rmse"] = float(np.sqrt(mean_squared_error(y_test, pred)))
                fold_results[f"{model_name}_mae"] = float(mean_absolute_error(y_test, pred))
                fold_results[f"{model_name}_hit"] = float(np.mean(np.sign(pred) == np.sign(y_test)))

            # Ensemble metrics
            fold_results["ensemble_rmse"] = float(np.sqrt(mean_squared_error(y_test, ensemble_pred)))
            fold_results["ensemble_mae"] = float(mean_absolute_error(y_test, ensemble_pred))
            fold_results["ensemble_hit"] = float(np.mean(np.sign(ensemble_pred) == np.sign(y_test)))

            results.append(fold_results)

        if not results:
            return None

        results_df = pd.DataFrame(results)

        # Summary statistics (store for later retrieval in the app)
        summary = {
            "factor": target_factor,
            "ensemble_rmse": results_df["ensemble_rmse"].mean(),
            "ensemble_mae": results_df["ensemble_mae"].mean(),
            "ensemble_hit_rate": results_df["ensemble_hit"].mean(),
        }

        # Keep a per factor summary dictionary on the object
        if not hasattr(self, "validation_summary"):
            self.validation_summary = {}
        self.validation_summary[target_factor] = summary

        print(f"\n{target_factor} Validation Results:")
        print("=" * 50)
        print(f"Average Ensemble RMSE: {summary['ensemble_rmse']:.4f}")
        print(f"Average Ensemble MAE: {summary['ensemble_mae']:.4f}")
        print(f"Average Ensemble Hit Rate: {summary['ensemble_hit_rate']:.2%}")

        return results_df


    def forecast_next(self, data: pd.DataFrame, target_factor: str):
        """
        Train on all available data and predict the next-period ER.
        Also caches the *exact* feature vector and ordering used, so scenarios with zero shocks will match.
        """
        X, y, feats = self.prepare_features_targets(data.copy(), target_factor)
        if len(X) < max(50, self.lookback_window // 2):
            return None

        # Train on full set
        self.train_models(X, y, target_factor)

        # Latest feature row (unscaled) in training order; cache it for scenarios
        x = X.iloc[-1].copy()
        self.feature_order[target_factor] = list(X.columns)
        self.latest_feature_row[target_factor] = x.copy()

        # Predict ensemble
        Xs = self.scalers[target_factor].transform(x.values.reshape(1, -1))
        preds = {name: float(mdl.predict(Xs)[0]) for name, mdl in self.models[target_factor].items()}
        ensemble = float(np.mean(list(preds.values())))

        # Drivers
        drivers = self.feature_importance.get(target_factor, pd.DataFrame())
        top = []
        if not drivers.empty:
            top = drivers.sort_values('rf_importance', ascending=False).head(5)[['feature', 'rf_importance']].to_dict('records')

        return {
            'factor': target_factor,
            'forecast_horizon_days': self.forecast_horizon,
            'ensemble_forecast': ensemble,
            'model_forecasts': preds,
            'top_drivers': top,
            'forecast_date': data['date'].iloc[-1] if 'date' in data.columns else 'latest',
            'confidence': 'Medium'
        }
