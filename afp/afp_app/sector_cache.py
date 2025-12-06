from __future__ import annotations

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import FMP_API_KEY, DEFAULT_START_DATE
from .fmp import FMPDataFetcher
from .universe import fetch_sp500_from_fmp, ALL_SP500

CACHE_DIR = os.getenv("SP500_CACHE_DIR", "./cache")

FUNDAMENTALS_CACHE_FILENAME = "sp500_fundamentals.parquet"
PRICES_CACHE_FILENAME = "sp500_prices.parquet"
CACHE_METADATA_FILENAME = "sp500_cache_metadata.json"


FUNDAMENTALS_REFRESH_DAYS = 30  
PRICES_REFRESH_DAYS = 7         

MIN_PRICE_HISTORY_DAYS = 60

API_CALL_DELAY = 0.1  
BATCH_SIZE = 10       
BATCH_DELAY = 1.0     

def _get_cache_dir() -> Path:
    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _get_fundamentals_cache_path() -> Path:
    return _get_cache_dir() / FUNDAMENTALS_CACHE_FILENAME


def _get_prices_cache_path() -> Path:
    return _get_cache_dir() / PRICES_CACHE_FILENAME


def _get_metadata_file_path() -> Path:
    return _get_cache_dir() / CACHE_METADATA_FILENAME

def _load_cache_metadata() -> dict:
    """Load cache metadata (last update times, ticker counts, etc.)."""
    meta_path = _get_metadata_file_path()
    if not meta_path.exists():
        return {}
    
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_cache_metadata(metadata: dict) -> None:
    meta_path = _get_metadata_file_path()
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def _update_metadata(cache_type: str, ticker_count: int, extra_info: dict = None) -> None:
    metadata = _load_cache_metadata()
    
    metadata[f"{cache_type}_last_updated"] = datetime.now().isoformat()
    metadata[f"{cache_type}_ticker_count"] = ticker_count
    
    if extra_info:
        for key, value in extra_info.items():
            metadata[f"{cache_type}_{key}"] = value
    
    _save_cache_metadata(metadata)
   
def _is_fundamentals_stale() -> bool:
    cache_path = _get_fundamentals_cache_path()
    
    if not cache_path.exists():
        print("[sector_cache] Fundamentals cache does not exist")
        return True
    
    metadata = _load_cache_metadata()
    last_updated_str = metadata.get("fundamentals_last_updated")
    
    if not last_updated_str:
        print("[sector_cache] Fundamentals cache metadata missing")
        return True
    
    try:
        last_updated = datetime.fromisoformat(last_updated_str)
    except (ValueError, TypeError):
        print("[sector_cache] Invalid fundamentals timestamp in metadata")
        return True
    
    age_days = (datetime.now() - last_updated).days
    is_stale = age_days >= FUNDAMENTALS_REFRESH_DAYS
    
    if is_stale:
        print(f"[sector_cache] Fundamentals cache is stale ({age_days} days old, limit is {FUNDAMENTALS_REFRESH_DAYS})")
    else:
        print(f"[sector_cache] Fundamentals cache is fresh ({age_days} days old)")
    
    return is_stale


def _is_prices_stale() -> bool:
    cache_path = _get_prices_cache_path()
    
    if not cache_path.exists():
        print("[sector_cache] Prices cache does not exist")
        return True
    
    metadata = _load_cache_metadata()
    last_updated_str = metadata.get("prices_last_updated")
    
    if not last_updated_str:
        print("[sector_cache] Prices cache metadata missing")
        return True
    
    try:
        last_updated = datetime.fromisoformat(last_updated_str)
    except (ValueError, TypeError):
        print("[sector_cache] Invalid prices timestamp in metadata")
        return True
    
    age_days = (datetime.now() - last_updated).days
    is_stale = age_days >= PRICES_REFRESH_DAYS
    
    if is_stale:
        print(f"[sector_cache] Prices cache is stale ({age_days} days old, limit is {PRICES_REFRESH_DAYS})")
    else:
        print(f"[sector_cache] Prices cache is fresh ({age_days} days old)")
    
    return is_stale


def _load_fundamentals_cache() -> Optional[pd.DataFrame]:
    cache_path = _get_fundamentals_cache_path()
    
    if not cache_path.exists():
        return None
    
    try:
        df = pd.read_parquet(cache_path)
        print(f"[sector_cache] Loaded fundamentals for {len(df)} stocks from cache")
        return df
    except Exception as e:
        print(f"[sector_cache] Error loading fundamentals cache: {e}")
        return None


def _load_prices_cache() -> Optional[pd.DataFrame]:
    cache_path = _get_prices_cache_path()
    
    if not cache_path.exists():
        return None
    
    try:
        df = pd.read_parquet(cache_path)
        print(f"[sector_cache] Loaded prices for {df['ticker'].nunique()} stocks from cache")
        return df
    except Exception as e:
        print(f"[sector_cache] Error loading prices cache: {e}")
        return None

def _save_fundamentals_cache(df: pd.DataFrame, sectors: list = None) -> bool:
    cache_path = _get_fundamentals_cache_path()
    
    try:
        df.to_parquet(cache_path, index=False)
        _update_metadata(
            "fundamentals",
            len(df),
            {"sectors": sectors or []}
        )
        print(f"[sector_cache] Saved fundamentals for {len(df)} stocks to cache")
        return True
    except Exception as e:
        print(f"[sector_cache] Error saving fundamentals cache: {e}")
        return False


def _save_prices_cache(df: pd.DataFrame) -> bool:
    cache_path = _get_prices_cache_path()
    
    try:
        df.to_parquet(cache_path, index=False)
        ticker_count = df["ticker"].nunique() if "ticker" in df.columns else 0
        _update_metadata("prices", ticker_count)
        print(f"[sector_cache] Saved prices for {ticker_count} stocks to cache")
        return True
    except Exception as e:
        print(f"[sector_cache] Error saving prices cache: {e}")
        return False

def _collect_sp500_fundamentals(
    tickers: list[str],
    fetcher: FMPDataFetcher,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> pd.DataFrame:

    bs_rows = []
    inc_rows = []
    cf_rows = []
    profile_data = {}
    
    total = len(tickers)
    
    for i, tk in enumerate(tickers):
        try:
            prof = fetcher.get_profile(tk)
            if prof:
                profile_data[tk] = {
                    "sector": prof.get("sector"),
                    "industry": prof.get("industry"),
                }
            
            bs = fetcher.get_balance_sheet(tk)
            inc = fetcher.get_income_statement(tk)
            cf = fetcher.get_cash_flow(tk)
            
            if isinstance(bs, list) and bs:
                row = bs[0].copy()  
                row["ticker"] = tk
                bs_rows.append(row)
            
            if isinstance(inc, list) and inc:
                row = inc[0].copy()  
                row["ticker"] = tk
                inc_rows.append(row)
            
            if isinstance(cf, list) and cf:
                row = cf[0].copy()  
                row["ticker"] = tk
                cf_rows.append(row)
            
            if progress_callback:
                progress_callback(i + 1, total, f"Fundamentals: {tk}")
            
        except Exception as e:
            print(f"[sector_cache] Error fetching fundamentals for {tk}: {e}")
        
        time.sleep(API_CALL_DELAY)
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(BATCH_DELAY)
    
    bs_df = pd.DataFrame(bs_rows) if bs_rows else pd.DataFrame()
    inc_df = pd.DataFrame(inc_rows) if inc_rows else pd.DataFrame()
    cf_df = pd.DataFrame(cf_rows) if cf_rows else pd.DataFrame()
    
    if bs_df.empty:
        return pd.DataFrame()
    
    result = bs_df.copy()
    
    if not inc_df.empty:
        inc_cols = ["ticker", "revenue", "netIncome", "grossProfit", "operatingIncome", 
                    "eps", "ebitda", "weightedAverageShsOut", "weightedAverageShsOutDil"]
        inc_cols = [c for c in inc_cols if c in inc_df.columns]
        result = result.merge(inc_df[inc_cols], on="ticker", how="left")
    
    if not cf_df.empty:
        cf_cols = ["ticker", "freeCashFlow", "operatingCashFlow"]
        cf_cols = [c for c in cf_cols if c in cf_df.columns]
        result = result.merge(cf_df[cf_cols], on="ticker", how="left")
    
    result["sector"] = result["ticker"].map(lambda tk: profile_data.get(tk, {}).get("sector"))
    result["industry"] = result["ticker"].map(lambda tk: profile_data.get(tk, {}).get("industry"))
    
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
    
    return result

def _collect_sp500_prices(
    tickers: list[str],
    fetcher: FMPDataFetcher,
    start_date: str,
    end_date: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> pd.DataFrame:

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    frames = []
    total = len(tickers)
    
    for i, tk in enumerate(tickers):
        try:
            px = fetcher.get_price_history(tk, start_date, end_date)
            
            if px is None or px.empty:
                continue
            
            if "adjClose" not in px.columns:
                if "close" in px.columns:
                    px = px.rename(columns={"close": "adjClose"})
                else:
                    continue
            
            px = px.copy()
            px["ticker"] = tk
            
            if "date" in px.columns:
                px["date"] = pd.to_datetime(px["date"])
                px = px.sort_values("date")
            
            s = pd.to_numeric(px["adjClose"], errors="coerce")
            px["returns"] = s.pct_change()
            
            ratio = s.div(s.shift(1)).clip(lower=1e-12)
            px["log_returns"] = np.log(ratio)
            
            keep_cols = ["date", "ticker", "adjClose", "returns", "log_returns"]
            keep_cols = [c for c in keep_cols if c in px.columns]
            frames.append(px[keep_cols])
            
            if progress_callback:
                progress_callback(i + 1, total, f"Prices: {tk}")
            
        except Exception as e:
            print(f"[sector_cache] Error fetching prices for {tk}: {e}")
        
        time.sleep(API_CALL_DELAY)
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(BATCH_DELAY)
    
    if not frames:
        return pd.DataFrame()
    
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)

def _safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


def _compute_factor_scores(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame
) -> pd.DataFrame:
    if fundamentals.empty:
        return pd.DataFrame()
    
    metrics = fundamentals.copy()
    
    share_candidates = ["outstandingShares", "weightedAverageShsOut", "weightedAverageShsOutDil"]
    metrics["shares_out"] = np.nan
    for col in share_candidates:
        if col in metrics.columns:
            metrics["shares_out"] = metrics["shares_out"].fillna(
                pd.to_numeric(metrics[col], errors="coerce")
            )
    
    if not prices.empty and "ticker" in prices.columns:
        last_price = (
            prices.sort_values("date")
            .groupby("ticker")["adjClose"]
            .last()
        )
        metrics["price_last"] = metrics["ticker"].map(last_price)
    else:
        metrics["price_last"] = np.nan
    
    metrics["book_equity"] = pd.to_numeric(
        metrics.get("totalStockholdersEquity"), errors="coerce"
    )
    
    metrics["market_cap"] = metrics["shares_out"] * metrics["price_last"]
    
    metrics["bp_ratio"] = _safe_div(metrics["book_equity"], metrics["market_cap"])
    metrics["ep_ratio"] = _safe_div(
        pd.to_numeric(metrics.get("netIncome"), errors="coerce"),
        metrics["market_cap"]
    )
    metrics["fcfp_ratio"] = _safe_div(
        pd.to_numeric(metrics.get("freeCashFlow"), errors="coerce") if "freeCashFlow" in metrics.columns else np.nan,
        metrics["market_cap"]
    )
    
    metrics["roe"] = _safe_div(
        pd.to_numeric(metrics.get("netIncome"), errors="coerce"),
        pd.to_numeric(metrics.get("totalStockholdersEquity"), errors="coerce")
    )
    metrics["roa"] = _safe_div(
        pd.to_numeric(metrics.get("netIncome"), errors="coerce"),
        pd.to_numeric(metrics.get("totalAssets"), errors="coerce")
    )
    metrics["gross_margin"] = _safe_div(
        pd.to_numeric(metrics.get("grossProfit"), errors="coerce"),
        pd.to_numeric(metrics.get("revenue"), errors="coerce")
    )
    metrics["fcf_margin"] = _safe_div(
        pd.to_numeric(metrics.get("freeCashFlow"), errors="coerce") if "freeCashFlow" in metrics.columns else np.nan,
        pd.to_numeric(metrics.get("revenue"), errors="coerce")
    )
    metrics["debt_to_equity"] = _safe_div(
        pd.to_numeric(metrics.get("totalDebt"), errors="coerce"),
        pd.to_numeric(metrics.get("totalStockholdersEquity"), errors="coerce")
    )
    
    if not prices.empty:
        px_pivot = prices.pivot_table(index="date", columns="ticker", values="adjClose")
        
        if not px_pivot.empty:
            mom_60d = px_pivot.pct_change(60)
            if not mom_60d.empty:
                last_mom = mom_60d.iloc[-1]
                metrics["momentum_60d"] = metrics["ticker"].map(last_mom.to_dict())
        
        vol_60d = (
            prices
            .sort_values(["ticker", "date"])
            .groupby("ticker")["returns"]
            .apply(lambda x: x.rolling(60, min_periods=30).std().iloc[-1] if len(x) >= 30 else np.nan)
        )
        metrics["volatility_60d"] = metrics["ticker"].map(vol_60d.to_dict())
    
    metrics = metrics.replace([np.inf, -np.inf], np.nan)
    
    output_cols = [
        "ticker", "sector", "industry",
       
        "bp_ratio", "ep_ratio", "fcfp_ratio",
        
        "roe", "roa", "gross_margin", "fcf_margin", "debt_to_equity",
       
        "momentum_60d",
        
        "volatility_60d",
       
        "market_cap", "price_last"
    ]
    
    output_cols = [c for c in output_cols if c in metrics.columns]
    
    return metrics[output_cols].copy()

def get_sp500_sector_scores(
    fetcher: Optional[FMPDataFetcher] = None,
    force_refresh: bool = False,
    force_refresh_fundamentals: bool = False,
    force_refresh_prices: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> pd.DataFrame:

    refresh_fundamentals = force_refresh or force_refresh_fundamentals or _is_fundamentals_stale()
    refresh_prices = force_refresh or force_refresh_prices or _is_prices_stale()
    
    fundamentals_df = _load_fundamentals_cache() if not refresh_fundamentals else None
    prices_df = _load_prices_cache() if not refresh_prices else None
    
    need_fetcher = refresh_fundamentals or refresh_prices
    
    if need_fetcher:
        if fetcher is None:
            if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY":
                print("[sector_cache] ERROR: No valid FMP API key configured")
                if fundamentals_df is None:
                    fundamentals_df = _load_fundamentals_cache()
                if prices_df is None:
                    prices_df = _load_prices_cache()
                
                if fundamentals_df is None or prices_df is None:
                    return pd.DataFrame()
            else:
                fetcher = FMPDataFetcher(FMP_API_KEY)
        
        tickers = fetch_sp500_from_fmp()
        if not tickers:
            print("[sector_cache] Using fallback S&P 500 list")
            tickers = ALL_SP500.copy()
        
        print(f"[sector_cache] Processing {len(tickers)} S&P 500 stocks...")
        
        total_steps = 0
        if refresh_fundamentals:
            total_steps += len(tickers)
        if refresh_prices:
            total_steps += len(tickers)
        
        current_step = 0
        
        if refresh_fundamentals and fetcher:
            print("[sector_cache] Refreshing fundamentals cache (monthly)...")
            
            def fundamentals_progress(current, total, message):
                nonlocal current_step
                if progress_callback:
                    progress_callback(current, total_steps, message)
            
            fundamentals_df = _collect_sp500_fundamentals(
                tickers, fetcher, progress_callback=fundamentals_progress
            )
            
            if not fundamentals_df.empty:
                sectors = fundamentals_df["sector"].dropna().unique().tolist()
                _save_fundamentals_cache(fundamentals_df, sectors)
            
            current_step = len(tickers)
        
        if refresh_prices and fetcher:
            print(f"[sector_cache] Refreshing prices cache ({PRICES_REFRESH_DAYS}-day cycle)...")
            
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            def prices_progress(current, total, message):
                if progress_callback:
                    progress_callback(current_step + current, total_steps, message)
            
            prices_df = _collect_sp500_prices(
                tickers, fetcher, start_date, end_date, progress_callback=prices_progress
            )
            
            if not prices_df.empty:
                _save_prices_cache(prices_df)
    
    if fundamentals_df is None or fundamentals_df.empty:
        print("[sector_cache] No fundamentals data available")
        return pd.DataFrame()
    
    if prices_df is None or prices_df.empty:
        print("[sector_cache] No prices data available, computing without momentum/volatility")
        prices_df = pd.DataFrame()
    
    factor_scores = _compute_factor_scores(fundamentals_df, prices_df)
    
    print(f"[sector_cache] Computed factor scores for {len(factor_scores)} stocks")
    return factor_scores


def get_cache_status() -> dict:
    metadata = _load_cache_metadata()
    
    fundamentals_path = _get_fundamentals_cache_path()
    prices_path = _get_prices_cache_path()
    
    status = {
        "cache_exists": fundamentals_path.exists() and prices_path.exists(),
        "cache_dir": str(_get_cache_dir()),
        
        "fundamentals_exists": fundamentals_path.exists(),
        "fundamentals_last_updated": metadata.get("fundamentals_last_updated"),
        "fundamentals_ticker_count": metadata.get("fundamentals_ticker_count", 0),
        "fundamentals_sectors": metadata.get("fundamentals_sectors", []),
        "fundamentals_is_stale": _is_fundamentals_stale(),
        "fundamentals_refresh_days": FUNDAMENTALS_REFRESH_DAYS,
        
        "prices_exists": prices_path.exists(),
        "prices_last_updated": metadata.get("prices_last_updated"),
        "prices_ticker_count": metadata.get("prices_ticker_count", 0),
        "prices_is_stale": _is_prices_stale(),
        "prices_refresh_days": PRICES_REFRESH_DAYS,
        
        "is_stale": _is_fundamentals_stale() or _is_prices_stale(),
        "last_updated": metadata.get("prices_last_updated") or metadata.get("fundamentals_last_updated"),
        "ticker_count": metadata.get("fundamentals_ticker_count", 0),
        "sectors": metadata.get("fundamentals_sectors", []),
    }
    
    return status


def clear_cache() -> bool:
    try:
        fundamentals_path = _get_fundamentals_cache_path()
        prices_path = _get_prices_cache_path()
        meta_path = _get_metadata_file_path()
        
        if fundamentals_path.exists():
            fundamentals_path.unlink()
        if prices_path.exists():
            prices_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        
        print("[sector_cache] All caches cleared")
        return True
    except Exception as e:
        print(f"[sector_cache] Error clearing cache: {e}")
        return False


def clear_fundamentals_cache() -> bool:
    try:
        path = _get_fundamentals_cache_path()
        if path.exists():
            path.unlink()
        
        metadata = _load_cache_metadata()
        for key in list(metadata.keys()):
            if key.startswith("fundamentals_"):
                del metadata[key]
        _save_cache_metadata(metadata)
        
        print("[sector_cache] Fundamentals cache cleared")
        return True
    except Exception as e:
        print(f"[sector_cache] Error clearing fundamentals cache: {e}")
        return False


def clear_prices_cache() -> bool:
    try:
        path = _get_prices_cache_path()
        if path.exists():
            path.unlink()
        
        metadata = _load_cache_metadata()
        for key in list(metadata.keys()):
            if key.startswith("prices_"):
                del metadata[key]
        _save_cache_metadata(metadata)
        
        print("[sector_cache] Prices cache cleared")
        return True
    except Exception as e:
        print(f"[sector_cache] Error clearing prices cache: {e}")
        return False

def _effective_rank_percentile(value: float, peer_values: np.ndarray, ascending: bool = True) -> float:
    if pd.isna(value) or len(peer_values) == 0:
        return np.nan
    
    peer_values = peer_values[~np.isnan(peer_values)]
    
    if len(peer_values) == 0:
        return np.nan
    
    all_values = np.append(peer_values, value)
    n = len(all_values)
    
    if n == 1:
        return 0.5 
    
    if ascending:
        ranks = rankdata(all_values, method='average')
        target_rank = ranks[-1]  
        percentile = (target_rank - 1) / (n - 1)
        return 1.0 - percentile  
    else:
        ranks = rankdata(all_values, method='average')
        target_rank = ranks[-1]
        percentile = (target_rank - 1) / (n - 1)
        return percentile

def compute_sector_percentiles(
    stock_metrics: pd.DataFrame,
    sp500_reference: pd.DataFrame,
    factor_columns: Optional[list[str]] = None
) -> pd.DataFrame:

    if stock_metrics.empty or sp500_reference.empty:
        return stock_metrics.copy()
    
    if "ticker" not in stock_metrics.columns or "sector" not in stock_metrics.columns:
        print("[sector_cache] WARNING: stock_metrics missing 'ticker' or 'sector' column")
        return stock_metrics.copy()
    
    if "sector" not in sp500_reference.columns:
        print("[sector_cache] WARNING: sp500_reference missing 'sector' column")
        return stock_metrics.copy()
    
    if factor_columns is None:
        factor_columns = [
            "bp_ratio", "ep_ratio", "fcfp_ratio",
            "roe", "roa", "gross_margin", "fcf_margin", "debt_to_equity",
            "momentum_60d",
            "volatility_60d",
        ]
    
    available_factors = [c for c in factor_columns 
                         if c in stock_metrics.columns and c in sp500_reference.columns]
    
    if not available_factors:
        print("[sector_cache] WARNING: No common factor columns found")
        return stock_metrics.copy()
    
    ascending_factors = {"debt_to_equity", "volatility_60d"}
    
    result = stock_metrics.copy()
    
    for _, row in stock_metrics.iterrows():
        ticker = row["ticker"]
        sector = row.get("sector")
        
        if pd.isna(sector) or sector is None:
            sector_peers = sp500_reference
        else:
            sector_peers = sp500_reference[sp500_reference["sector"] == sector]
        
        if sector_peers.empty:
            sector_peers = sp500_reference
        
        for factor in available_factors:
            stock_value = row.get(factor)
            
            if pd.isna(stock_value):
                continue
            
            peer_values = sector_peers[factor].dropna().values
            
            if len(peer_values) == 0:
                continue
            
            is_ascending = factor in ascending_factors
            percentile = _effective_rank_percentile(stock_value, peer_values, ascending=is_ascending)
            
            result.loc[result["ticker"] == ticker, f"{factor}_score"] = percentile
    
    return result


def get_sector_percentile_for_ticker(
    ticker: str,
    sector: str,
    raw_metrics: dict,
    sp500_reference: pd.DataFrame
) -> dict:

    if sp500_reference.empty:
        return {}
    
    if pd.isna(sector) or sector is None:
        sector_peers = sp500_reference
    else:
        sector_peers = sp500_reference[sp500_reference["sector"] == sector]
        if sector_peers.empty:
            sector_peers = sp500_reference
    
    ascending_factors = {"debt_to_equity", "volatility_60d"}
    
    percentiles = {}
    
    for factor, value in raw_metrics.items():
        if pd.isna(value) or factor not in sp500_reference.columns:
            continue
        
        peer_values = sector_peers[factor].dropna().values
        
        if len(peer_values) == 0:
            continue
        
        is_ascending = factor in ascending_factors
        percentile = _effective_rank_percentile(value, peer_values, ascending=is_ascending)
        
        if not np.isnan(percentile):
            percentiles[factor] = float(percentile)
    
    return percentiles
