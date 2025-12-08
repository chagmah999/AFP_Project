import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AlphaPredictor:
    def __init__(self, factor_returns: pd.DataFrame, fundamentals: dict, price_data: pd.DataFrame,
                 horizon: int = 21, lookback: int = 252*2):
        self.factor_returns = factor_returns if factor_returns is not None else pd.DataFrame()
        self.fundamentals = fundamentals if fundamentals is not None else {
            "balance_sheet": pd.DataFrame(), "income_statement": pd.DataFrame(), "cash_flow": pd.DataFrame()
        }
        self.price_data = price_data if price_data is not None else pd.DataFrame()
        self.horizon = horizon
        self.lookback = lookback
        self.models = {}
        self.scalers = {}
        self._features_used = {}
        
        # Pre-compute all technical features ONCE (vectorized) - this is the key optimization
        self._technical_cache = {}
        self._fundamental_cache = {}
        self._precompute_all_features()

    def _precompute_all_features(self):
        """Pre-compute ALL features for ALL tickers using vectorized operations."""
        
        # === TECHNICAL FEATURES (vectorized per ticker) ===
        if not self.price_data.empty and "ticker" in self.price_data.columns:
            for ticker in self.price_data["ticker"].unique():
                px = self.price_data[self.price_data["ticker"] == ticker].copy()
                if px.empty or "date" not in px.columns:
                    continue
                    
                px = px.sort_values("date").set_index("date")
                
                # Ensure we have returns
                if "returns" not in px.columns:
                    if "adjClose" in px.columns:
                        px["returns"] = np.log(px["adjClose"]).diff()
                    elif "close" in px.columns:
                        px["returns"] = np.log(px["close"]).diff()
                    else:
                        continue
                
                if "returns" not in px.columns:
                    continue
                
                # Compute all technical features vectorized (ONCE per ticker)
                tech_df = pd.DataFrame(index=px.index)
                for w in [5, 21, 63]:
                    tech_df[f"mom_{w}d"] = px["returns"].rolling(w, min_periods=w).sum()
                    tech_df[f"vol_{w}d"] = px["returns"].rolling(w, min_periods=w).std()
                
                self._technical_cache[ticker] = tech_df
        
        # === FUNDAMENTAL FEATURES (pre-process per ticker) ===
        bs = self.fundamentals.get("balance_sheet", pd.DataFrame())
        inc = self.fundamentals.get("income_statement", pd.DataFrame())
        cf = self.fundamentals.get("cash_flow", pd.DataFrame())
        
        # Get all unique tickers from fundamentals
        all_tickers = set()
        for df in [bs, inc, cf]:
            if not df.empty and "ticker" in df.columns:
                all_tickers.update(df["ticker"].unique())
        
        for ticker in all_tickers:
            # Get latest fundamental data for this ticker
            fund_data = {}
            
            # Balance sheet
            if not bs.empty and "ticker" in bs.columns and "date" in bs.columns:
                tk_bs = bs[bs["ticker"] == ticker].copy()
                if not tk_bs.empty:
                    tk_bs["date"] = pd.to_datetime(tk_bs["date"], errors="coerce")
                    tk_bs = tk_bs.dropna(subset=["date"]).sort_values("date")
                    if not tk_bs.empty:
                        latest = tk_bs.iloc[-1]
                        for col in ["totalStockholdersEquity", "totalAssets", "totalDebt"]:
                            if col in latest.index and pd.notna(latest[col]):
                                fund_data[col] = float(latest[col])
            
            # Income statement
            if not inc.empty and "ticker" in inc.columns and "date" in inc.columns:
                tk_inc = inc[inc["ticker"] == ticker].copy()
                if not tk_inc.empty:
                    tk_inc["date"] = pd.to_datetime(tk_inc["date"], errors="coerce")
                    tk_inc = tk_inc.dropna(subset=["date"]).sort_values("date")
                    if not tk_inc.empty:
                        latest = tk_inc.iloc[-1]
                        for col in ["netIncome", "revenue", "grossProfit"]:
                            if col in latest.index and pd.notna(latest[col]):
                                fund_data[col] = float(latest[col])
            
            # Cash flow
            if not cf.empty and "ticker" in cf.columns and "date" in cf.columns:
                tk_cf = cf[cf["ticker"] == ticker].copy()
                if not tk_cf.empty:
                    tk_cf["date"] = pd.to_datetime(tk_cf["date"], errors="coerce")
                    tk_cf = tk_cf.dropna(subset=["date"]).sort_values("date")
                    if not tk_cf.empty:
                        latest = tk_cf.iloc[-1]
                        if "freeCashFlow" in latest.index and pd.notna(latest["freeCashFlow"]):
                            fund_data["freeCashFlow"] = float(latest["freeCashFlow"])
            
            if fund_data:
                self._fundamental_cache[ticker] = fund_data

    def _compute_fundamental_ratios(self, fund_data: dict) -> dict:
        """Compute fundamental ratios from raw fundamental data."""
        feats = {}
        
        total_equity = fund_data.get("totalStockholdersEquity")
        total_assets = fund_data.get("totalAssets")
        total_debt = fund_data.get("totalDebt")
        net_income = fund_data.get("netIncome")
        revenue = fund_data.get("revenue")
        gross_profit = fund_data.get("grossProfit")
        fcf = fund_data.get("freeCashFlow")
        
        # Compute ratios safely
        if net_income is not None and total_equity is not None and total_equity != 0:
            feats["roe"] = net_income / total_equity
        else:
            feats["roe"] = np.nan
            
        if net_income is not None and total_assets is not None and total_assets != 0:
            feats["roa"] = net_income / total_assets
        else:
            feats["roa"] = np.nan
            
        if gross_profit is not None and revenue is not None and revenue != 0:
            feats["gross_margin"] = gross_profit / revenue
        else:
            feats["gross_margin"] = np.nan
            
        if total_debt is not None and total_equity is not None and total_equity != 0:
            feats["debt_to_equity"] = total_debt / total_equity
        else:
            feats["debt_to_equity"] = np.nan
            
        if fcf is not None and revenue is not None and revenue != 0:
            feats["fcf_margin"] = fcf / revenue
        else:
            feats["fcf_margin"] = np.nan
            
        return feats

    def _fundamental_score(self, feats: dict) -> int:
        """Compute a simple fundamental score."""
        score = 0
        if "roe" in feats and pd.notna(feats["roe"]):
            score += 1 if feats["roe"] > 0.15 else -1
        if "gross_margin" in feats and pd.notna(feats["gross_margin"]):
            score += 1 if feats["gross_margin"] > 0.30 else -1
        if "debt_to_equity" in feats and pd.notna(feats["debt_to_equity"]):
            score += 1 if feats["debt_to_equity"] < 1.0 else -1
        if "fcf_margin" in feats and pd.notna(feats["fcf_margin"]):
            score += 1 if feats["fcf_margin"] > 0 else -1
        return int(score)

    def train_ticker(self, ticker: str) -> bool:
        """Train a model for a single ticker using pre-computed features."""
        
        # Check if we have price data for this ticker
        if self.price_data.empty:
            return False
            
        px = self.price_data[self.price_data["ticker"] == ticker].copy()
        if px.empty or len(px) < 60:
            return False
            
        px = px.sort_values("date")
        
        # Ensure we have returns
        if "returns" not in px.columns:
            if "adjClose" in px.columns:
                px["returns"] = np.log(px["adjClose"]).diff()
            elif "close" in px.columns:
                px["returns"] = np.log(px["close"]).diff()
            else:
                return False
        
        # Compute forward returns (target variable)
        px["fwd_ret"] = px["returns"].rolling(self.horizon).sum().shift(-self.horizon)
        
        # Limit to lookback period
        end_date = px["date"].max()
        start_date = end_date - pd.Timedelta(days=int(self.lookback * 1.5))
        px = px[(px["date"] >= start_date) & (px["date"] <= end_date)].copy()
        
        if len(px) < 60:
            return False
        
        # Get pre-computed technical features
        tech_feats = self._technical_cache.get(ticker)
        if tech_feats is None or tech_feats.empty:
            return False
        
        # Get pre-computed fundamental ratios
        fund_data = self._fundamental_cache.get(ticker, {})
        fund_ratios = self._compute_fundamental_ratios(fund_data)
        
        # Build feature matrix - USE VECTORIZED MERGE, NOT LOOP
        px_indexed = px.set_index("date")
        
        # Align technical features with price data
        X = tech_feats.reindex(px_indexed.index)
        
        # Add fundamental features (constant across all dates for simplicity)
        for feat_name, feat_val in fund_ratios.items():
            X[feat_name] = feat_val
        
        # Target
        y = px_indexed["fwd_ret"]
        
        # Combine and drop NaN
        combined = pd.concat([y.rename("target"), X], axis=1).dropna()
        
        if len(combined) < 50:
            return False
        
        y_train = combined["target"].values
        X_train = combined.drop(columns=["target"])
        
        # Remove columns with no variance or all NaN
        valid_cols = []
        for col in X_train.columns:
            col_data = X_train[col]
            if col_data.notna().sum() > len(col_data) * 0.5:  # At least 50% non-NaN
                if col_data.std() > 1e-10:  # Has variance
                    valid_cols.append(col)
        
        if len(valid_cols) < 2:
            return False
        
        X_train = X_train[valid_cols].fillna(X_train[valid_cols].median())
        
        # Scale and fit model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        try:
            # Use fast settings: fewer alphas, fewer CV folds
            model = LassoCV(cv=3, random_state=42, n_alphas=10, max_iter=2000)
            model.fit(X_scaled, y_train)
        except Exception:
            try:
                model = RidgeCV(cv=3)
                model.fit(X_scaled, y_train)
            except Exception:
                return False
        
        self.models[ticker] = model
        self.scalers[ticker] = scaler
        self._features_used[ticker] = list(valid_cols)
        return True

    def predict_alpha(self, ticker: str, horizon: int | None = None) -> dict | None:
        """Predict alpha for a ticker."""
        if horizon is None:
            horizon = self.horizon
        
        # Get fundamental ratios for scoring
        fund_data = self._fundamental_cache.get(ticker, {})
        fund_ratios = self._compute_fundamental_ratios(fund_data)
        score = self._fundamental_score(fund_ratios)
        
        # Try to train if we don't have a model
        if ticker not in self.models:
            if not self.train_ticker(ticker):
                # Return fallback prediction based on fundamentals only
                exp = 0.002 * score
                return {
                    "ticker": ticker,
                    "expected_alpha": float(exp),
                    "horizon_days": horizon,
                    "confidence": "Low",
                    "drivers": {
                        "fundamental_score": int(score),
                        "key_metrics": fund_ratios,
                        "top_features": []
                    }
                }
        
        # Double-check we have a model
        if ticker not in self.models:
            exp = 0.002 * score
            return {
                "ticker": ticker,
                "expected_alpha": float(exp),
                "horizon_days": horizon,
                "confidence": "Low",
                "drivers": {
                    "fundamental_score": int(score),
                    "key_metrics": fund_ratios,
                    "top_features": []
                }
            }
        
        # Get latest features for prediction
        cols = self._features_used.get(ticker, [])
        if not cols:
            exp = 0.002 * score
            return {
                "ticker": ticker,
                "expected_alpha": float(exp),
                "horizon_days": horizon,
                "confidence": "Low",
                "drivers": {
                    "fundamental_score": int(score),
                    "key_metrics": fund_ratios,
                    "top_features": []
                }
            }
        
        # Build feature vector for latest date
        tech_feats = self._technical_cache.get(ticker)
        if tech_feats is not None and not tech_feats.empty:
            latest_tech = tech_feats.iloc[-1].to_dict()
        else:
            latest_tech = {}
        
        # Combine with fundamentals
        all_feats = {**latest_tech, **fund_ratios}
        
        # Create feature vector in correct order
        x = pd.DataFrame([all_feats]).reindex(columns=cols).fillna(0)
        
        try:
            scaler = self.scalers[ticker]
            X_scaled = scaler.transform(x.values)
            model = self.models[ticker]
            pred = float(model.predict(X_scaled)[0])
        except Exception:
            pred = 0.002 * score
        
        # Get feature importances
        coefs = getattr(self.models.get(ticker), "coef_", np.zeros(len(cols)))
        imp = pd.DataFrame({
            "feature": cols,
            "coef": coefs,
            "abs": np.abs(coefs)
        }).sort_values("abs", ascending=False)
        top = imp.head(5)[["feature", "coef"]].to_dict("records")
        
        coef_norm = float(np.linalg.norm(coefs)) if len(coefs) > 0 else 0.0
        conf = "High" if coef_norm > 0.5 else ("Medium" if coef_norm > 0.2 else "Low")
        
        return {
            "ticker": ticker,
            "expected_alpha": pred,
            "horizon_days": horizon,
            "confidence": conf,
            "drivers": {
                "fundamental_score": int(score),
                "key_metrics": fund_ratios,
                "top_features": top
            }
        }
