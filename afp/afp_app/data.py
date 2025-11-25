import time
import pandas as pd
import numpy as np
from datetime import datetime
from .fmp import FMPDataFetcher


def collect_fundamental_data(tickers, start_date, fetcher):
    bs_rows = []
    inc_rows = []
    cf_rows = []
    profile_rows = []

    for tk in tickers:
        # Fetch fundamentals
        bs = fetcher.get_balance_sheet(tk)
        inc = fetcher.get_income_statement(tk)
        cf = fetcher.get_cash_flow(tk)

        # Fetch metadata (sector, industry)
        prof = fetcher.get_profile(tk)
        if prof:
            profile_rows.append(prof)

        if isinstance(bs, list):
            for row in bs:
                row["ticker"] = tk
                bs_rows.append(row)
        if isinstance(inc, list):
            for row in inc:
                row["ticker"] = tk
                inc_rows.append(row)
        if isinstance(cf, list):
            for row in cf:
                row["ticker"] = tk
                cf_rows.append(row)

    # Convert to DataFrames
    bs_df = pd.DataFrame(bs_rows)
    inc_df = pd.DataFrame(inc_rows)
    cf_df = pd.DataFrame(cf_rows)
    profile_df = pd.DataFrame(profile_rows)

    # Merge sector/industry into each fundamental table
    if not profile_df.empty:
        bs_df = bs_df.merge(profile_df, on="ticker", how="left")
        inc_df = inc_df.merge(profile_df, on="ticker", how="left")
        cf_df = cf_df.merge(profile_df, on="ticker", how="left")

        # Ensure date columns are proper datetimes
    for df in (bs_df, inc_df, cf_df):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")


    return {
        "balance_sheet": bs_df,
        "income_statement": inc_df,
        "cash_flow": cf_df,
    }


def collect_price_data(
    tickers: list[str],
    start_date: str,
    end_date: str | None,
    fetcher: FMPDataFetcher,
) -> pd.DataFrame:
    """
    Collect daily price data for all tickers using FMPDataFetcher.get_price_history.

    Returns a DataFrame with at least:
        date, ticker, adjClose, returns, log_returns
    """
    frames: list[pd.DataFrame] = []

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    for i, t in enumerate(tickers, 1):
        try:
            # Use the new method defined in fmp.py
            px = fetcher.get_price_history(t, start_date, end_date)

            if not isinstance(px, pd.DataFrame) or px.empty:
                continue

            # Ensure required columns are present
            if "adjClose" not in px.columns:
                # If get_price_history was changed to return a different column, handle gracefully
                if "close" in px.columns:
                    px = px.rename(columns={"close": "adjClose"})
                else:
                    continue

            # Attach ticker
            px = px.copy()
            px["ticker"] = t

            # Ensure date is datetime and sorted
            if "date" in px.columns:
                px["date"] = pd.to_datetime(px["date"])
                px = px.sort_values("date")

            # Compute returns and log_returns
            s = pd.to_numeric(px["adjClose"], errors="coerce")
            px["returns"] = s.pct_change()

            # Guard against non positive ratios before log
            ratio = s.div(s.shift(1))
            ratio = ratio.clip(lower=1e-12)
            px["log_returns"] = np.log(ratio)

            frames.append(px)

        except Exception as e:
            print(f"[warn] prices {t}: {e}")

        # Small pause to be gentle with the API
        if i % 5 == 0:
            time.sleep(0.1)

    if not frames:
        return pd.DataFrame()

    out = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    return out
