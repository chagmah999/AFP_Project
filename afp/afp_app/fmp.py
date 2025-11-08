import time
import numpy as np
import requests
import pandas as pd
from .config import FMP_API_KEY, FMP_BASE

class FMPDataFetcher:
    def __init__(self, api_key: str = FMP_API_KEY):
        self.api_key = api_key

    def _request(self, url: str, params: dict | None = None) -> dict | list:
        if params is None:
            params = {}
        params["apikey"] = self.api_key
        for i in range(3):
            try:
                r = requests.get(url, params=params, timeout=30)
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                if i == 2:
                    raise
                time.sleep(2 ** i)

    def fetch_balance_sheet(self, ticker: str, period: str = "quarter", limit: int = 20) -> pd.DataFrame:
        url = f"{FMP_BASE}/balance-sheet-statement/{ticker}"
        js = self._request(url, {"period": period, "limit": limit})
        if not js:
            return pd.DataFrame()
        df = pd.DataFrame(js)
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = ticker
        return df.sort_values("date")

    def fetch_income_statement(self, ticker: str, period: str = "quarter", limit: int = 20) -> pd.DataFrame:
        url = f"{FMP_BASE}/income-statement/{ticker}"
        js = self._request(url, {"period": period, "limit": limit})
        if not js:
            return pd.DataFrame()
        df = pd.DataFrame(js)
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = ticker
        return df.sort_values("date")

    def fetch_cash_flow(self, ticker: str, period: str = "quarter", limit: int = 20) -> pd.DataFrame:
        url = f"{FMP_BASE}/cash-flow-statement/{ticker}"
        js = self._request(url, {"period": period, "limit": limit})
        if not js:
            return pd.DataFrame()
        df = pd.DataFrame(js)
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = ticker
        return df.sort_values("date")

    def fetch_historical_prices(self, ticker: str, from_date: str, to_date: str | None = None) -> pd.DataFrame:
        url = f"{FMP_BASE}/historical-price-full/{ticker}"
        params = {"from": from_date}
        if to_date:
            params["to"] = to_date
        js = self._request(url, params)
        if not js or "historical" not in js:
            return pd.DataFrame()
        df = pd.DataFrame(js["historical"])
        if "date" not in df:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = ticker
        df = df.sort_values("date")
        if "adjClose" in df:
            df["returns"] = df["adjClose"].pct_change()
            ratio = df["adjClose"] / df["adjClose"].shift(1)
            df["log_returns"] = np.where(ratio > 0, np.log(ratio), np.nan)
        return df
