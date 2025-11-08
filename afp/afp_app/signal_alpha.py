import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

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

    @staticmethod
    def _last_before(df, date_col, date_val):
        if df.empty:
            return pd.DataFrame()
        df = df[df[date_col] <= date_val]
        return df.iloc[[-1]] if not df.empty else pd.DataFrame()

    def _build_fundamental_row(self, ticker: str, asof: pd.Timestamp) -> dict:
        bs = self.fundamentals.get("balance_sheet", pd.DataFrame())
        inc = self.fundamentals.get("income_statement", pd.DataFrame())
        cf  = self.fundamentals.get("cash_flow", pd.DataFrame())
        bsl = self._last_before(bs[bs.get("ticker", pd.Series(dtype=str)) == ticker].sort_values("date"), "date", asof) if not bs.empty else pd.DataFrame()
        incl = self._last_before(inc[inc.get("ticker", pd.Series(dtype=str)) == ticker].sort_values("date"), "date", asof) if not inc.empty else pd.DataFrame()
        cfl = self._last_before(cf[cf.get("ticker", pd.Series(dtype=str)) == ticker].sort_values("date"), "date", asof) if not cf.empty else pd.DataFrame()

        def gv(d, col):
            return float(d[col].values[0]) if (not d.empty and col in d.columns and pd.notna(d[col].values[0])) else np.nan

        total_equity = gv(bsl, "totalStockholdersEquity")
        total_assets = gv(bsl, "totalAssets")
        total_debt   = gv(bsl, "totalDebt")
        net_income   = gv(incl, "netIncome")
        revenue      = gv(incl, "revenue")
        gross_profit = gv(incl, "grossProfit")
        fcf          = gv(cfl, "freeCashFlow")

        feats = {}
        feats["roe"] = (net_income/total_equity) if (pd.notna(net_income) and pd.notna(total_equity) and total_equity) else np.nan
        feats["roa"] = (net_income/total_assets) if (pd.notna(net_income) and pd.notna(total_assets) and total_assets) else np.nan
        feats["gross_margin"] = (gross_profit/revenue) if (pd.notna(gross_profit) and pd.notna(revenue) and revenue) else np.nan
        feats["debt_to_equity"] = (total_debt/total_equity) if (pd.notna(total_debt) and pd.notna(total_equity) and total_equity) else np.nan
        feats["fcf_margin"] = (fcf/revenue) if (pd.notna(fcf) and pd.notna(revenue) and revenue) else np.nan
        return feats

    def _build_technical_row(self, ticker: str, asof: pd.Timestamp) -> dict:
        feats = {}
        if self.price_data.empty or "date" not in self.price_data.columns or "ticker" not in self.price_data.columns:
            return feats
        px = self.price_data[self.price_data["ticker"] == ticker].copy()
        if px.empty:
            return feats
        px = px.sort_values("date").set_index("date")
        if "returns" not in px.columns and "close" in px.columns:
            px["returns"] = np.log(px["close"]).diff()
        if "returns" not in px.columns:
            return feats
        start = asof - pd.Timedelta(days=int(self.lookback*1.5))
        px = px.loc[(px.index >= start) & (px.index <= asof)]
        for w in [5,21,63]:
            feats[f"mom_{w}d"] = px["returns"].rolling(w).sum().iloc[-1] if len(px) >= w else np.nan
            feats[f"vol_{w}d"] = px["returns"].rolling(w).std().iloc[-1] if len(px) >= w else np.nan
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
        px = self.price_data[self.price_data["ticker"] == ticker].copy().sort_values("date")
        if px.empty:
            return False
        if "returns" not in px.columns and "close" in px.columns:
            px["returns"] = np.log(px["close"]).diff()
        px["fwd_21d"] = px["returns"].rolling(self.horizon).sum().shift(-self.horizon)
        end = px["date"].max()
        start = end - pd.Timedelta(days=int(self.lookback*1.5))
        px = px[(px["date"] >= start) & (px["date"] <= end)].copy()
        y = px.set_index("date")["fwd_21d"]
        rows = []
        for dt in y.index:
            f = self._build_fundamental_row(ticker, dt)
            t = self._build_technical_row(ticker, dt)
            rows.append(pd.Series({**f, **t}, name=dt))
        X = pd.DataFrame(rows).sort_index()
        df = pd.concat([y, X], axis=1).dropna()
        if df.shape[0] < 120 or df.shape[1] < 5:
            return False
        y_tr = df["fwd_21d"].values
        X_tr = df.drop(columns=["fwd_21d"])
        sc = StandardScaler()
        Xs = sc.fit_transform(X_tr)
        model = LassoCV(cv=5, random_state=42, n_alphas=50, max_iter=20000)
        model.fit(Xs, y_tr)
        self.models[ticker] = model
        self.scalers[ticker] = sc
        self._features_used[ticker] = list(X_tr.columns)
        return True

    def predict_alpha(self, ticker: str, horizon: int | None = None) -> dict | None:
        if horizon is None:
            horizon = self.horizon
        today = pd.Timestamp.today().normalize()
        if ticker not in self.models:
            if not self.train_ticker(ticker):
                feats = self._build_fundamental_row(ticker, today)
                score = self._fundamental_score(feats)
                exp = 0.002 * score
                return {
                    "ticker": ticker,
                    "expected_alpha": float(exp),
                    "horizon_days": horizon,
                    "confidence": "Low",
                    "drivers": {"fundamental_score": int(score), "key_metrics": feats, "top_features": []}
                }
        feats_f = self._build_fundamental_row(ticker, today)
        feats_t = self._build_technical_row(ticker, today)
        feats = {**feats_f, **feats_t}
        score = self._fundamental_score(feats)
        cols = self._features_used.get(ticker, [])
        x = pd.DataFrame([feats], index=[today]).reindex(columns=cols).ffill().bfill().fillna(0)
        sc = self.scalers[ticker]
        Xs = sc.transform(x.values)
        model = self.models[ticker]
        pred = float(model.predict(Xs)[0])
        coefs = getattr(model, "coef_", np.zeros(len(cols)))
        imp = pd.DataFrame({"feature": cols, "coef": coefs, "abs": np.abs(coefs)}).sort_values("abs", ascending=False)
        top = imp.head(5)[["feature","coef"]].to_dict("records")
        conf = "High" if float(np.linalg.norm(coefs)) > 0.5 else ("Medium" if float(np.linalg.norm(coefs)) > 0.2 else "Low")
        return {
            "ticker": ticker,
            "expected_alpha": pred,
            "horizon_days": horizon,
            "confidence": conf,
            "drivers": {"fundamental_score": int(score), "key_metrics": feats_f, "top_features": top}
        }
