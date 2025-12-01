import requests
import pandas as pd
import time

FMP_BASE = "https://financialmodelingprep.com/api/v3"

class FMPDataFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _get(self, endpoint: str, params=None):
        if params is None:
            params = {}
        params["apikey"] = self.api_key

        for attempt in range(3):
            r = requests.get(f"{FMP_BASE}/{endpoint}", params=params)
            if r.status_code == 200:
                try:
                    data = r.json()
                    return data
                except Exception:
                    time.sleep(0.5)
            time.sleep(0.5)
        return None

    def get_balance_sheet(self, ticker: str):
        return self._get(f"balance-sheet-statement/{ticker}")

    def get_income_statement(self, ticker: str):
        return self._get(f"income-statement/{ticker}")

    def get_cash_flow(self, ticker: str):
        return self._get(f"cash-flow-statement/{ticker}")

    def get_profile(self, ticker: str):
        data = self._get(f"profile/{ticker}")
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        return {
            "ticker": ticker,
            "sector": data[0].get("sector"),
            "industry": data[0].get("industry"),
        }

    def get_price_history(self, ticker: str, start_date: str, end_date: str | None = None):
        params = {"from": start_date}
        if end_date:
            params["to"] = end_date

        data = self._get(f"historical-price-full/{ticker}", params=params)

        if not data or "historical" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["historical"])
        if df.empty:
            return df

        if "adjClose" not in df.columns:
            if "close" in df.columns:
                df = df.rename(columns={"close": "adjClose"})
            else:
                return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        return df[["date", "adjClose"]]

