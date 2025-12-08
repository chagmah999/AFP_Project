import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler


class AlphaPredictor:
    def __init__(
        self,
        factor_returns: pd.DataFrame,
        fundamentals: dict,
        price_data: pd.DataFrame,
        horizon: int = 21,
        lookback: int = 252 * 2,
    ):
        self.factor_returns = factor_returns if factor_returns is not None else pd.DataFrame()
        self.fundamentals = (
            fundamentals
            if fundamentals is not None
            else {
                "balance_sheet": pd.DataFrame(),
                "income_statement": pd.DataFrame(),
                "cash_flow": pd.DataFrame(),
            }
        )
        self.price_data = price_data if price_data is not None else pd.DataFrame()
        self.horizon = horizon
        self.lookback = lookback
        self.models: dict[str, object] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self._features_used: dict[str, list[str]] = {}

        # Pre-compute fundamental data lookup for efficiency with large universes
        self._fundamental_cache: dict[str, pd.core.groupby.generic.DataFrameGroupBy | None] = {}
        self._precompute_fundamentals()

    def _precompute_fundamentals(self) -> None:
        """Pre-compute fundamental data indexed by ticker for faster lookups."""
        for key in ["balance_sheet", "income_statement", "cash_flow"]:
            df = self.fundamentals.get(key, pd.DataFrame())
            if not df.empty and "ticker" in df.columns and "date" in df.columns:
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date")
                self._fundamental_cache[key] = df.groupby("ticker")
            else:
                self._fundamental_cache[key] = None

    @staticmethod
    def _last_before(df: pd.DataFrame, date_col: str, date_val: pd.Timestamp) -> pd.DataFrame:
        if df.empty or date_col not in df.columns:
            return pd.DataFrame()
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        mask = df[date_col].notna() & (df[date_col] <= date_val)
        df = df[mask]
        return df.iloc[[-1]] if not df.empty else pd.DataFrame()

    def _get_fundamental_before(
        self,
        key: str,
        ticker: str,
        asof: pd.Timestamp,
    ) -> pd.DataFrame:
        """Efficiently get the last fundamental record before a given date."""
        cache = self._fundamental_cache.get(key)
        if cache is None:
            return pd.DataFrame()

        try:
            group = cache.get_group(ticker)
            mask = group["date"] <= asof
            filtered = group[mask]
            if filtered.empty:
                return pd.DataFrame()
            return filtered.iloc[[-1]]
        except KeyError:
            return pd.DataFrame()

    def _build_fundamental_row(self, ticker: str, asof: pd.Timestamp) -> dict:
        bsl = self._get_fundamental_before("balance_sheet", ticker, asof)
        incl = self._get_fundamental_before("income_statement", ticker, asof)
        cfl = self._get_fundamental_before("cash_flow", ticker, asof)

        def gv(d: pd.DataFrame, col: str) -> float:
            return (
                float(d[col].values[0])
                if (not d.empty and col in d.columns and pd.notna(d[col].values[0]))
                else np.nan
            )

        total_equity = gv(bsl, "totalStockholdersEquity")
        total_assets = gv(bsl, "totalAssets")
        total_debt = gv(bsl, "totalDebt")
        net_income = gv(incl, "netIncome")
        revenue = gv(incl, "revenue")
        gross_profit = gv(incl, "grossProfit")
        fcf = gv(cfl, "freeCashFlow")

        feats: dict[str, float] = {}

        feats["roe"] = (
            net_income / total_equity
            if (pd.notna(net_income) and pd.notna(total_equity) and total_equity)
            else np.nan
        )
        feats["roa"] = (
            net_income / total_assets
            if (pd.notna(net_income) and pd.notna(total_assets) and total_assets)
            else np.nan
        )
        feats["gross_margin"] = (
            gross_profit / revenue
            if (pd.notna(gross_profit) and pd.notna(revenue) and revenue)
            else np.nan
        )
        feats["debt_to_equity"] = (
            total_debt / total_equity
            if (pd.notna(total_debt) and pd.notna(total_equity) and total_equity)
            else np.nan
        )
        feats["fcf_margin"] = (
            fcf / revenue
            if (pd.notna(fcf) and pd.notna(revenue) and revenue)
            else np.nan
        )

        return feats

    def _build_technical_row(self, ticker: str, asof: pd.Timestamp) -> dict:
        feats: dict[str, float] = {}

        if (
            self.price_data.empty
            or "date" not in self.price_data.columns
            or "ticker" not in self.price_data.columns
        ):
            return feats

        px = self.price_data[self.price_data["ticker"] == ticker].copy()
        if px.empty:
            return feats

        px = px.sort_values("date").set_index("date")

        if "returns" not in px.columns and "close" in px.columns:
            px["returns"] = np.log(px["close"]).diff()

        if "returns" not in px.columns:
            return feats

        start = asof - pd.Timedelta(days=int(self.lookback * 1.5))
        px = px.loc[(px.index >= start) & (px.index <= asof)]

        for w in [5, 21, 63]:
            feats[f"mom_{w}d"] = (
                px["returns"].rolling(w).sum().iloc[-1] if len(px) >= w else np.nan
            )
            feats[f"vol_{w}d"] = (
                px["returns"].rolling(w).std().iloc[-1] if len(px) >= w else np.nan
            )

        return feats

    def _fundamental_score(self, feats: dict) -> int:
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
        if self.price_data.empty:
            return False

        px = (
            self.price_data[self.price_data["ticker"] == ticker]
            .copy()
            .sort_values("date")
        )
        if px.empty:
            return False

        if "returns" not in px.columns and "close" in px.columns:
            px["returns"] = np.log(px["close"]).diff()

        if "returns" not in px.columns:
            return False

        px["fwd_ret"] = px["returns"].rolling(self.horizon).sum().shift(-self.horizon)
        end = px["date"].max()

        # CHANGE 1: More adaptive lookback - use available data more flexibly
        # Use at least 2x the lookback or all available data, whichever is smaller
        min_start = end - pd.Timedelta(days=int(self.lookback * 2))
        px = px[(px["date"] >= min_start) & (px["date"] <= end)].copy()

        if len(px) < 60:  # Absolute minimum
            return False

        y = px.set_index("date")["fwd_ret"]

        # CHANGE 2: Build features more efficiently using vectorized operations where possible
        # Sample dates to reduce computation for very long histories
        dates_to_use = y.index
        if len(dates_to_use) > 500:
            # Sample every N-th date to keep ~500 training points
            step = max(len(dates_to_use) // 500, 1)
            dates_to_use = dates_to_use[::step]

        rows: list[pd.Series] = []
        for dt in dates_to_use:
            f = self._build_fundamental_row(ticker, dt)
            t = self._build_technical_row(ticker, dt)
            rows.append(pd.Series({**f, **t}, name=dt))

        X = pd.DataFrame(rows).sort_index()
        y = y.reindex(X.index)
        df = pd.concat([y, X], axis=1).dropna()

        # CHANGE 3: Relaxed minimum requirements for training
        # Reduced from 120 rows and 5 columns to 50 rows and 3 columns
        min_rows = max(50, self.horizon * 2)  # At least 50 or 2x horizon
        if df.shape[0] < min_rows or df.shape[1] < 3:
            return False

        y_tr = df["fwd_ret"].values
        X_tr = df.drop(columns=["fwd_ret"])

        # Drop columns that are all NaN or have too many NaNs
        valid_cols = X_tr.columns[X_tr.notna().sum() > len(X_tr) * 0.5]  # ≥ 50% non-NaN
        if len(valid_cols) < 2:
            return False

        X_tr = X_tr[valid_cols].fillna(X_tr[valid_cols].median())

        # Check for zero variance columns
        var = X_tr.var()
        valid_cols = var[var > 1e-10].index
        if len(valid_cols) < 2:
            return False

        X_tr = X_tr[valid_cols]

        sc = StandardScaler()
        Xs = sc.fit_transform(X_tr)

        # CHANGE 4: Use faster model fitting for large universes
        # Reduced n_alphas and cv folds for speed
        model = None
        try:
            model = LassoCV(cv=3, random_state=42, n_alphas=20, max_iter=5000)
            model.fit(Xs, y_tr)
        except Exception:
            # Fallback to Ridge if Lasso fails
            try:
                model = RidgeCV(cv=3)
                model.fit(Xs, y_tr)
            except Exception:
                return False

        if model is None:
            return False

        self.models[ticker] = model
        self.scalers[ticker] = sc
        self._features_used[ticker] = list(X_tr.columns)

        return True

    def predict_alpha(self, ticker: str, horizon: int | None = None) -> dict | None:
        if horizon is None:
            horizon = self.horizon

        today = pd.Timestamp.today().normalize()

        # CHANGE 5: Always try to return a prediction, even if model training fails
        if ticker not in self.models:
            if not self.train_ticker(ticker):
                # Return fallback prediction based on fundamentals
                feats = self._build_fundamental_row(ticker, today)
                score = self._fundamental_score(feats)
                exp = 0.002 * score  # Simple heuristic: 0.2% per point

                return {
                    "ticker": ticker,
                    "expected_alpha": float(exp),
                    "horizon_days": horizon,
                    "confidence": "Low",
                    "drivers": {
                        "fundamental_score": int(score),
                        "key_metrics": feats,
                        "top_features": [],
                    },
                }

        # If we still do not have a model after training attempt, use fallback
        if ticker not in self.models:
            feats = self._build_fundamental_row(ticker, today)
            score = self._fundamental_score(feats)
            exp = 0.002 * score

            return {
                "ticker": ticker,
                "expected_alpha": float(exp),
                "horizon_days": horizon,
                "confidence": "Low",
                "drivers": {
                    "fundamental_score": int(score),
                    "key_metrics": feats,
                    "top_features": [],
                },
            }

        feats_f = self._build_fundamental_row(ticker, today)
        feats_t = self._build_technical_row(ticker, today)
        feats = {**feats_f, **feats_t}
        score = self._fundamental_score(feats)
        cols = self._features_used.get(ticker, [])

        if not cols:
            # No features available, use fallback
            exp = 0.002 * score

            return {
                "ticker": ticker,
                "expected_alpha": float(exp),
                "horizon_days": horizon,
                "confidence": "Low",
                "drivers": {
                    "fundamental_score": int(score),
                    "key_metrics": feats_f,
                    "top_features": [],
                },
            }

        x = (
            pd.DataFrame([feats], index=[today])
            .reindex(columns=cols)
            .ffill()
            .bfill()
            .fillna(0.0)
        )

        sc = self.scalers[ticker]
        model = self.models[ticker]

        try:
            Xs = sc.transform(x.values)
            pred = float(model.predict(Xs)[0])
        except Exception:
            # If prediction fails, use fallback
            pred = 0.002 * score

        coefs = getattr(model, "coef_", np.zeros(len(cols)))
        imp = (
            pd.DataFrame({"feature": cols, "coef": coefs, "abs": np.abs(coefs)})
            .sort_values("abs", ascending=False)
        )
        top = imp.head(5)[["feature", "coef"]].to_dict("records")

        coef_norm = float(np.linalg.norm(coefs)) if len(coefs) > 0 else 0.0
        if coef_norm > 0.5:
            conf = "High"
        elif coef_norm > 0.2:
            conf = "Medium"
        else:
            conf = "Low"

        return {
            "ticker": ticker,
            "expected_alpha": pred,
            "horizon_days": horizon,
            "confidence": conf,
            "drivers": {
                "fundamental_score": int(score),
                "key_metrics": feats_f,
                "top_features": top,
            },
        }
