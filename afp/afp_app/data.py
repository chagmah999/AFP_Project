import time
import pandas as pd
import numpy as np           
from datetime import datetime
from .fmp import FMPDataFetcher


def collect_fundamental_data(tickers: list[str], start_date: str, fetcher: FMPDataFetcher) -> dict[str, pd.DataFrame]:
    bs_all, inc_all, cf_all = [], [], []
    for i, t in enumerate(tickers, 1):
        try:
            bs = fetcher.fetch_balance_sheet(t, period="quarter", limit=20)
            inc = fetcher.fetch_income_statement(t, period="quarter", limit=20)
            cf  = fetcher.fetch_cash_flow(t, period="quarter", limit=20)
            if not bs.empty:
                bs_all.append(bs[bs["date"] >= start_date])
            if not inc.empty:
                inc_all.append(inc[inc["date"] >= start_date])
            if not cf.empty:
                cf_all.append(cf[cf["date"] >= start_date])
        except Exception as e:
            print(f"[warn] fundamentals {t}: {e}")
        if i % 5 == 0:
            time.sleep(0.1)
    return {
        "balance_sheet": pd.concat(bs_all, ignore_index=True) if bs_all else pd.DataFrame(),
        "income_statement": pd.concat(inc_all, ignore_index=True) if inc_all else pd.DataFrame(),
        "cash_flow": pd.concat(cf_all, ignore_index=True) if cf_all else pd.DataFrame(),
    }

def collect_price_data(tickers: list[str], start_date: str, end_date: str | None, fetcher: FMPDataFetcher) -> pd.DataFrame:
    frames = []
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    for i, t in enumerate(tickers, 1):
        try:
            px = fetcher.fetch_historical_prices(t, from_date=start_date, to_date=end_date)
            if not px.empty:
                # choose adjusted first, else close
                price_col = "adjClose" if "adjClose" in px.columns else ("close" if "close" in px.columns else None)
                if price_col:
                    s = pd.to_numeric(px[price_col], errors="coerce")
                    px["returns"] = s.pct_change()
                    # guard against non-positive ratios before log
                    ratio = s.div(s.shift(1))
                    ratio = ratio.clip(lower=1e-12)  # avoid log of <= 0
                    px["log_returns"] = np.log(ratio)
                frames.append(px)
        except Exception as e:
            print(f"[warn] prices {t}: {e}")
        if i % 5 == 0:
            time.sleep(0.1)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    return out

