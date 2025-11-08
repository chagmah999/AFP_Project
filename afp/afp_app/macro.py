import time
import requests
import pandas as pd
from datetime import datetime
from .config import FMP_API_KEY, FMP_STABLE_BASE, FMP_V4_BASE


class MacroDataFetcher:
    def __init__(self, api_key: str = FMP_API_KEY,
                 stable_base: str = FMP_STABLE_BASE,
                 v4_base: str = FMP_V4_BASE):
        self.api_key = api_key
        self.stable_base = stable_base
        self.v4_base = v4_base

    def _get(self, url: str, params: dict) -> dict | list:
        params = {**params, "apikey": self.api_key}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _to_dt(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

    # -------------------- Treasury -------------------- #
    def fetch_treasury_rates(self, from_date: str = "2022-01-01", to_date: str | None = None) -> pd.DataFrame:
        """
        Fetch U.S. Treasury rates from FMP stable endpoint with robust numeric handling.
        Computes term spreads and a simple average 'rates_level'.
        """
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")

        url = f"{self.stable_base}/treasury-rates"
        js = self._get(url, {"from": from_date, "to": to_date})
        if not js:
            return pd.DataFrame()

        df = pd.DataFrame(js)
        if "date" not in df.columns:
            return pd.DataFrame()

        # Normalize date and sort
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # Column picker to tolerate naming variants seen in FMP payloads
        def pick(frame: pd.DataFrame, candidates: list[str]) -> str | None:
            for c in candidates:
                if c in frame.columns:
                    return c
            return None

        c_3m = pick(df, ["month3", "3M"])
        c_2y = pick(df, ["year2", "2Y"])
        c_5y = pick(df, ["year5", "5Y"])
        c_10y = pick(df, ["year10", "10Y"])

        # Coerce available tenor columns to numeric
        num_cols = [c for c in [c_3m, c_2y, c_5y, c_10y] if c is not None]
        if num_cols:
            df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

        # Spreads
        if c_10y and c_2y:
            df["term_spread_10y2y"] = df[c_10y] - df[c_2y]
        if c_10y and c_3m:
            df["term_spread_10y3m"] = df[c_10y] - df[c_3m]

        # Rates level features
        levels = [c for c in [c_3m, c_2y, c_5y, c_10y] if c is not None]
        if levels:
            # Use DataFrame mean over numeric columns, not pd.to_numeric on a 2-D object
            df["rates_level"] = df[levels].astype(float).mean(axis=1)
            df["rates_1m_change"] = df["rates_level"].diff(21)

        return df

    # -------------------- VIX -------------------- #
    def fetch_vix(self, from_date: str = "2022-01-01") -> pd.DataFrame:
        url = f"{self.stable_base}/historical-price-eod/full"
        js = self._get(url, {"symbol": "^VIX"})
        if isinstance(js, dict) and "historical" in js:
            df = pd.DataFrame(js["historical"])
        else:
            df = pd.DataFrame(js)

        if df.empty or "date" not in df or "close" not in df:
            return pd.DataFrame()

        df["date"] = self._to_dt(df["date"])
        df = df[df["date"] >= pd.Timestamp(from_date)].copy()
        df = df.rename(columns={"close": "vix_close"})
        df["vix_ma20"] = df["vix_close"].rolling(20, min_periods=1).mean()
        df["vix_percentile"] = df["vix_close"].rolling(252, min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        return df[["date", "vix_close", "vix_ma20", "vix_percentile"]].sort_values("date").reset_index(drop=True)

    # -------------------- Credit proxies -------------------- #
    def fetch_credit_spreads(self, from_date: str = "2022-01-01") -> pd.DataFrame:
        frames = []
        for sym in ["HYG", "LQD", "TLT"]:
            url = f"{self.stable_base}/historical-price-eod/full"
            js = self._get(url, {"symbol": sym})
            if isinstance(js, dict) and "historical" in js:
                df = pd.DataFrame(js["historical"])
            else:
                df = pd.DataFrame(js)

            if df.empty or "date" not in df or "close" not in df:
                continue

            df["date"] = self._to_dt(df["date"])
            df = df[df["date"] >= pd.Timestamp(from_date)].copy()
            df["ticker"] = sym
            frames.append(df[["date", "ticker", "close"]])
            time.sleep(0.05)

        if not frames:
            return pd.DataFrame()

        wide = pd.concat(frames).pivot(index="date", columns="ticker", values="close").sort_index()

        if {"TLT", "HYG"}.issubset(wide.columns):
            wide["hy_spread"] = (wide["TLT"] / wide["HYG"] - 1.0) * 100.0
        if {"TLT", "LQD"}.issubset(wide.columns):
            wide["ig_spread"] = (wide["TLT"] / wide["LQD"] - 1.0) * 100.0

        if {"hy_spread", "ig_spread"}.issubset(wide.columns):
            wide["credit_spread_level"] = wide[["hy_spread", "ig_spread"]].mean(axis=1)
            wide["credit_spread_1m_change"] = wide["credit_spread_level"].diff(21)

        return wide.reset_index()
