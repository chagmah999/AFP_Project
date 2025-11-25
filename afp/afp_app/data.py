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

    return {
        "balance_sheet": bs_df,
        "income_statement": inc_df,
        "cash_flow": cf_df,
    }



def collect_price_data(tickers, start_date, end_date, fetcher):
    """
    Pull daily price history for each ticker and build a single DataFrame with:
      - date
      - ticker
      - adjClose
      - returns
    """
    frames = []

    for tk in tickers:
        try:
            df = fetcher.fetch_price_history(
                symbol=tk,
                start_date=start_date,
                end_date=end_date,
            )

            if df is None or df.empty:
                print(f"[warn] no prices for {tk}")
                continue

            # Ensure expected columns and clean
            if "date" not in df.columns:
                print(f"[warn] prices {tk}: no 'date' column in response")
                continue

            # FMP usually gives 'adjClose'; if only 'close' exists, fall back to that
            if "adjClose" not in df.columns and "close" in df.columns:
                df = df.rename(columns={"close": "adjClose"})

            if "adjClose" not in df.columns:
                print(f"[warn] prices {tk}: no 'adjClose' or 'close' column")
                continue

            df = df[["date", "adjClose"]].copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            # Compute simple daily returns
            df["returns"] = df["adjClose"].pct_change()

            df["ticker"] = tk

            frames.append(df)

        except Exception as e:
            print(f"[warn] prices {tk}: {e}")
            continue

    if not frames:
        # Return an empty DataFrame with the right columns so app.py will see empty
        return pd.DataFrame(columns=["date", "ticker", "adjClose", "returns"])

    prices = pd.concat(frames, ignore_index=True)

    # Final sanity check: drop rows with NaN prices
    prices = prices.dropna(subset=["adjClose"])

    return prices


