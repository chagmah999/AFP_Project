"""
sector_cache.py

Daily caching system for S&P 500 factor scores.
Computes and stores factor metrics for all S&P 500 constituents once per day,
enabling sector-relative percentile calculations for any subset of stocks.
"""

from __future__ import annotations

import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import FMP_API_KEY, DEFAULT_START_DATE
from .fmp import FMPDataFetcher
from .universe import fetch_sp500_from_fmp, ALL_SP500


# =============================================================================
# Configuration
# =============================================================================

CACHE_DIR = os.getenv("SP500_CACHE_DIR", "./cache")
CACHE_FILENAME = "sp500_factor_scores.parquet"
CACHE_METADATA_FILENAME = "sp500_cache_metadata.json"

# Cache is considered stale after this hour (in local time)
# e.g., 6 means refresh if it's after 6 AM and cache is from yesterday
CACHE_REFRESH_HOUR = 6

# Minimum number of trading days of price history required
MIN_PRICE_HISTORY_DAYS = 60

# Rate limiting for API calls
API_CALL_DELAY = 0.1  # seconds between API calls
BATCH_SIZE = 10  # Number of tickers to process before a longer pause
BATCH_DELAY = 1.0  # seconds to pause after each batch


# =============================================================================
# Cache Management
# =============================================================================

def _get_cache_dir() -> Path:
    """Get or create the cache directory."""
    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _get_cache_file_path() -> Path:
    """Get the path to the main cache file."""
    return _get_cache_dir() / CACHE_FILENAME


def _get_metadata_file_path() -> Path:
    """Get the path to the cache metadata file."""
    return _get_cache_dir() / CACHE_METADATA_FILENAME


def _load_cache_metadata() -> dict:
    """Load cache metadata (last update time, ticker count, etc.)."""
    meta_path = _get_metadata_file_path()
    if not meta_path.exists():
        return {}
    
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_cache_metadata(metadata: dict) -> None:
    """Save cache metadata."""
    meta_path = _get_metadata_file_path()
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def _is_cache_stale() -> bool:
    """
    Check if the cache needs to be refreshed.
    
    Cache is considered stale if:
    1. Cache file doesn't exist
    2. Cache was created before today's refresh hour
    3. Cache metadata is missing or corrupted
    """
    cache_path = _get_cache_file_path()
    
    if not cache_path.exists():
        print("[sector_cache] Cache file does not exist")
        return True
    
    metadata = _load_cache_metadata()
    if not metadata or "last_updated" not in metadata:
        print("[sector_cache] Cache metadata missing or corrupted")
        return True
    
    try:
        last_updated = datetime.fromisoformat(metadata["last_updated"])
    except (ValueError, TypeError):
        print("[sector_cache] Invalid last_updated timestamp in metadata")
        return True
    
    now = datetime.now()
    today_refresh_time = now.replace(
        hour=CACHE_REFRESH_HOUR, minute=0, second=0, microsecond=0
    )
    
    # If it's past refresh hour today and cache is from before that time
    if now >= today_refresh_time and last_updated < today_refresh_time:
        print(f"[sector_cache] Cache is stale (last updated: {last_updated})")
        return True
    
    # If cache is from a previous day entirely
    if last_updated.date() < now.date() and now.hour >= CACHE_REFRESH_HOUR:
        print(f"[sector_cache] Cache is from a previous day (last updated: {last_updated})")
        return True
    
    print(f"[sector_cache] Cache is fresh (last updated: {last_updated})")
    return False


def _load_cache() -> Optional[pd.DataFrame]:
    """Load the cached S&P 500 factor scores."""
    cache_path = _get_cache_file_path()
    
    if not cache_path.exists():
        return None
    
    try:
        df = pd.read_parquet(cache_path)
        print(f"[sector_cache] Loaded {len(df)} stocks from cache")
        return df
    except Exception as e:
        print(f"[sector_cache] Error loading cache: {e}")
        return None


def _save_cache(df: pd.DataFrame) -> bool:
    """Save the S&P 500 factor scores to cache."""
    cache_path = _get_cache_file_path()
    
    try:
        df.to_parquet(cache_path, index=False)
        
        # Save metadata
        metadata = {
            "last_updated": datetime.now().isoformat(),
            "ticker_count": len(df),
            "sectors": df["sector"].dropna().unique().tolist() if "sector" in df.columns else [],
            "columns": df.columns.tolist(),
        }
        _save_cache_metadata(metadata)
        
        print(f"[sector_cache] Saved {len(df)} stocks to cache")
        return True
    except Exception as e:
        print(f"[sector_cache] Error saving cache: {e}")
        return False


# =============================================================================
# Data Collection for Full S&P 500
# =============================================================================

def _collect_sp500_fundamentals(
    tickers: list[str],
    fetcher: FMPDataFetcher,
    progress_callback: Optional[callable] = None
) -> dict:
    """
    Collect fundamental data for all S&P 500 stocks.
    
    Args:
        tickers: List of ticker symbols
        fetcher: FMPDataFetcher instance
        progress_callback: Optional callback(current, total, ticker) for progress updates
    
    Returns:
        Dictionary with 'balance_sheet', 'income_statement', 'cash_flow' DataFrames
    """
    bs_rows = []
    inc_rows = []
    cf_rows = []
    profile_rows = []
    
    total = len(tickers)
    
    for i, tk in enumerate(tickers):
        try:
            # Fetch profile for sector/industry
            prof = fetcher.get_profile(tk)
            if prof:
                profile_rows.append(prof)
            
            # Fetch fundamentals
            bs = fetcher.get_balance_sheet(tk)
            inc = fetcher.get_income_statement(tk)
            cf = fetcher.get_cash_flow(tk)
            
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
            
            if progress_callback:
                progress_callback(i + 1, total, tk)
            
        except Exception as e:
            print(f"[sector_cache] Error fetching {tk}: {e}")
        
        # Rate limiting
        time.sleep(API_CALL_DELAY)
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(BATCH_DELAY)
    
    # Build DataFrames
    bs_df = pd.DataFrame(bs_rows)
    inc_df = pd.DataFrame(inc_rows)
    cf_df = pd.DataFrame(cf_rows)
    profile_df = pd.DataFrame(profile_rows)
    
    # Merge profile (sector/industry) into fundamentals
    if not profile_df.empty:
        for df in [bs_df, inc_df, cf_df]:
            if not df.empty and "ticker" in df.columns:
                for col in ["sector", "industry"]:
                    if col in profile_df.columns and col not in df.columns:
                        mapping = profile_df.set_index("ticker")[col].to_dict()
                        df[col] = df["ticker"].map(mapping)
    
    # Convert date columns
    for df in [bs_df, inc_df, cf_df]:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    return {
        "balance_sheet": bs_df,
        "income_statement": inc_df,
        "cash_flow": cf_df,
        "profiles": profile_df,
    }


def _collect_sp500_prices(
    tickers: list[str],
    fetcher: FMPDataFetcher,
    start_date: str,
    end_date: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> pd.DataFrame:
    """
    Collect price data for all S&P 500 stocks.
    
    Args:
        tickers: List of ticker symbols
        fetcher: FMPDataFetcher instance
        start_date: Start date for price history
        end_date: End date (defaults to today)
        progress_callback: Optional callback(current, total, ticker) for progress updates
    
    Returns:
        DataFrame with date, ticker, adjClose, returns, log_returns
    """
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
            
            # Calculate returns
            s = pd.to_numeric(px["adjClose"], errors="coerce")
            px["returns"] = s.pct_change()
            
            ratio = s.div(s.shift(1)).clip(lower=1e-12)
            px["log_returns"] = np.log(ratio)
            
            frames.append(px)
            
            if progress_callback:
                progress_callback(i + 1, total, tk)
            
        except Exception as e:
            print(f"[sector_cache] Error fetching prices for {tk}: {e}")
        
        # Rate limiting
        time.sleep(API_CALL_DELAY)
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(BATCH_DELAY)
    
    if not frames:
        return pd.DataFrame()
    
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


# =============================================================================
# Factor Score Computation (Raw Values)
# =============================================================================

def _safe_div(num, den):
    """Safe division handling zeros and infinities."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


def _compute_raw_factor_metrics(
    fundamentals: dict,
    price_data: pd.DataFrame,
    profiles: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute raw factor metrics WITHOUT percentile ranking.
    
    This computes the underlying ratios and values that will later be
    ranked against sector peers.
    
    Returns DataFrame with columns:
        ticker, sector, industry, date,
        bp_ratio, ep_ratio, fcfp_ratio (VALUE)
        roe, roa, gross_margin, fcf_margin, debt_to_equity (QUALITY)
        momentum_60d (MOMENTUM)
        volatility_60d (LOW_VOL)
    """
    bs = fundamentals.get("balance_sheet", pd.DataFrame())
    inc = fundamentals.get("income_statement", pd.DataFrame())
    cf = fundamentals.get("cash_flow", pd.DataFrame())
    
    if bs.empty or inc.empty:
        return pd.DataFrame()
    
    # Select relevant columns from balance sheet
    bs_cols = ["ticker", "date", "totalStockholdersEquity", "totalAssets", 
               "totalLiabilities", "totalDebt", "cashAndCashEquivalents"]
    
    if "outstandingShares" in bs.columns:
        bs_cols.append("outstandingShares")
    
    for meta_col in ["sector", "industry"]:
        if meta_col in bs.columns:
            bs_cols.append(meta_col)
    
    bs_cols = [c for c in bs_cols if c in bs.columns]
    bs_use = bs[bs_cols].copy()
    
    # Select relevant columns from income statement
    inc_cols = ["ticker", "date", "revenue", "netIncome", "grossProfit", 
                "operatingIncome", "eps", "ebitda", "weightedAverageShsOut", 
                "weightedAverageShsOutDil"]
    inc_cols = [c for c in inc_cols if c in inc.columns]
    inc_use = inc[inc_cols].copy()
    
    # Merge balance sheet and income statement
    metrics = pd.merge(bs_use, inc_use, on=["ticker", "date"], how="inner")
    
    # Get shares outstanding
    share_candidates = ["outstandingShares", "weightedAverageShsOut", "weightedAverageShsOutDil"]
    metrics["shares_out"] = np.nan
    for col in share_candidates:
        if col in metrics.columns:
            metrics["shares_out"] = metrics["shares_out"].fillna(
                pd.to_numeric(metrics[col], errors="coerce")
            )
    
    # Merge cash flow data
    if not cf.empty:
        cf_cols = ["ticker", "date", "freeCashFlow", "operatingCashFlow"]
        cf_cols = [c for c in cf_cols if c in cf.columns]
        if cf_cols:
            cf_use = cf[cf_cols].copy()
            metrics = pd.merge(metrics, cf_use, on=["ticker", "date"], how="left")
    
    # Add sector/industry from profiles if not already present
    if not profiles.empty and "ticker" in profiles.columns:
        for col in ["sector", "industry"]:
            if col in profiles.columns and col not in metrics.columns:
                mapping = profiles.set_index("ticker")[col].to_dict()
                metrics[col] = metrics["ticker"].map(mapping)
    
    # Get latest data per ticker
    metrics["date"] = pd.to_datetime(metrics["date"], errors="coerce")
    metrics = metrics.sort_values("date").groupby("ticker").last().reset_index()
    
    # Book equity
    metrics["book_equity"] = pd.to_numeric(metrics.get("totalStockholdersEquity"), errors="coerce")
    
    # Get last price for each ticker
    if not price_data.empty and "ticker" in price_data.columns:
        last_price = (
            price_data.sort_values("date")
            .groupby("ticker")["adjClose"]
            .last()
        )
        metrics["price_last"] = metrics["ticker"].map(last_price)
    else:
        metrics["price_last"] = np.nan
    
    # Market cap
    metrics["market_cap"] = metrics["shares_out"] * metrics["price_last"]
    
    # =========================================================================
    # VALUE FACTORS (raw ratios)
    # =========================================================================
    metrics["bp_ratio"] = _safe_div(metrics["book_equity"], metrics["market_cap"])
    metrics["ep_ratio"] = _safe_div(
        pd.to_numeric(metrics.get("netIncome"), errors="coerce"),
        metrics["market_cap"]
    )
    metrics["fcfp_ratio"] = _safe_div(
        pd.to_numeric(metrics.get("freeCashFlow"), errors="coerce"),
        metrics["market_cap"]
    )
    
    # =========================================================================
    # QUALITY FACTORS (raw ratios)
    # =========================================================================
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
        pd.to_numeric(metrics.get("freeCashFlow"), errors="coerce"),
        pd.to_numeric(metrics.get("revenue"), errors="coerce")
    )
    metrics["debt_to_equity"] = _safe_div(
        pd.to_numeric(metrics.get("totalDebt"), errors="coerce"),
        pd.to_numeric(metrics.get("totalStockholdersEquity"), errors="coerce")
    )
    
    # =========================================================================
    # MOMENTUM & VOLATILITY (from price data)
    # =========================================================================
    if not price_data.empty:
        # Momentum: 60-day price change
        px_pivot = price_data.pivot_table(index="date", columns="ticker", values="adjClose")
        
        if not px_pivot.empty:
            mom_60d = px_pivot.pct_change(60)
            if not mom_60d.empty:
                last_mom = mom_60d.iloc[-1]
                metrics["momentum_60d"] = metrics["ticker"].map(last_mom.to_dict())
        
        # Volatility: 60-day rolling std of returns
        vol_60d = (
            price_data
            .sort_values(["ticker", "date"])
            .groupby("ticker")["returns"]
            .apply(lambda x: x.rolling(60, min_periods=30).std().iloc[-1] if len(x) >= 30 else np.nan)
        )
        metrics["volatility_60d"] = metrics["ticker"].map(vol_60d.to_dict())
    
    # Clean up infinities
    metrics = metrics.replace([np.inf, -np.inf], np.nan)
    
    # Select output columns
    output_cols = [
        "ticker", "sector", "industry", "date",
        # Value
        "bp_ratio", "ep_ratio", "fcfp_ratio",
        # Quality
        "roe", "roa", "gross_margin", "fcf_margin", "debt_to_equity",
        # Momentum
        "momentum_60d",
        # Low Vol
        "volatility_60d",
        # Additional useful data
        "market_cap", "price_last"
    ]
    
    output_cols = [c for c in output_cols if c in metrics.columns]
    
    return metrics[output_cols].copy()


# =============================================================================
# Main Public Interface
# =============================================================================

def get_sp500_sector_scores(
    fetcher: Optional[FMPDataFetcher] = None,
    force_refresh: bool = False,
    progress_callback: Optional[callable] = None
) -> pd.DataFrame:
    """
    Get S&P 500 factor scores, using cache if available and fresh.
    
    Args:
        fetcher: FMPDataFetcher instance (created if not provided)
        force_refresh: If True, ignore cache and recompute
        progress_callback: Optional callback(current, total, ticker) for progress updates
    
    Returns:
        DataFrame with raw factor metrics for all S&P 500 stocks, including:
        - ticker, sector, industry
        - Value: bp_ratio, ep_ratio, fcfp_ratio
        - Quality: roe, roa, gross_margin, fcf_margin, debt_to_equity
        - Momentum: momentum_60d
        - Low Vol: volatility_60d
    """
    # Check if cache is fresh
    if not force_refresh and not _is_cache_stale():
        cached = _load_cache()
        if cached is not None and not cached.empty:
            return cached
    
    # Need to refresh cache
    print("[sector_cache] Refreshing S&P 500 factor scores cache...")
    
    if fetcher is None:
        if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY":
            print("[sector_cache] ERROR: No valid FMP API key configured")
            # Return empty DataFrame or try to load stale cache
            stale_cache = _load_cache()
            if stale_cache is not None:
                print("[sector_cache] Using stale cache due to missing API key")
                return stale_cache
            return pd.DataFrame()
        
        fetcher = FMPDataFetcher(FMP_API_KEY)
    
    # Get S&P 500 constituents
    tickers = fetch_sp500_from_fmp()
    if not tickers:
        print("[sector_cache] Using fallback S&P 500 list")
        tickers = ALL_SP500.copy()
    
    print(f"[sector_cache] Processing {len(tickers)} S&P 500 stocks...")
    
    # Calculate date range for price history
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Collect fundamentals
    print("[sector_cache] Fetching fundamentals...")
    fundamentals = _collect_sp500_fundamentals(
        tickers, fetcher, 
        progress_callback=lambda c, t, tk: progress_callback(c, t * 2, f"Fundamentals: {tk}") if progress_callback else None
    )
    
    # Collect prices
    print("[sector_cache] Fetching prices...")
    price_data = _collect_sp500_prices(
        tickers, fetcher, start_date, end_date,
        progress_callback=lambda c, t, tk: progress_callback(t + c, t * 2, f"Prices: {tk}") if progress_callback else None
    )
    
    # Compute raw factor metrics
    print("[sector_cache] Computing factor metrics...")
    profiles = fundamentals.get("profiles", pd.DataFrame())
    factor_scores = _compute_raw_factor_metrics(fundamentals, price_data, profiles)
    
    if factor_scores.empty:
        print("[sector_cache] WARNING: No factor scores computed")
        return pd.DataFrame()
    
    # Save to cache
    _save_cache(factor_scores)
    
    print(f"[sector_cache] Cache refreshed with {len(factor_scores)} stocks")
    return factor_scores


def compute_sector_percentiles(
    stock_metrics: pd.DataFrame,
    sp500_reference: pd.DataFrame,
    factor_columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Compute sector-relative percentiles for stocks against S&P 500 reference.
    
    For each stock, calculates where it ranks (0-1 percentile) among all 
    S&P 500 stocks in the same sector for each factor.
    
    Args:
        stock_metrics: DataFrame with raw factor values for stocks to score
                       Must have 'ticker' and 'sector' columns
        sp500_reference: Full S&P 500 reference DataFrame from get_sp500_sector_scores()
        factor_columns: List of factor columns to compute percentiles for
                        If None, uses default factor columns
    
    Returns:
        DataFrame with same tickers as input, but factor columns replaced with
        sector-relative percentile scores (0-1)
    """
    if stock_metrics.empty or sp500_reference.empty:
        return stock_metrics.copy()
    
    if "ticker" not in stock_metrics.columns or "sector" not in stock_metrics.columns:
        print("[sector_cache] WARNING: stock_metrics missing 'ticker' or 'sector' column")
        return stock_metrics.copy()
    
    if "sector" not in sp500_reference.columns:
        print("[sector_cache] WARNING: sp500_reference missing 'sector' column")
        return stock_metrics.copy()
    
    # Default factor columns
    if factor_columns is None:
        factor_columns = [
            # Value (higher is better)
            "bp_ratio", "ep_ratio", "fcfp_ratio",
            # Quality (higher is better, except debt_to_equity)
            "roe", "roa", "gross_margin", "fcf_margin", "debt_to_equity",
            # Momentum (higher is better)
            "momentum_60d",
            # Low Vol (lower is better)
            "volatility_60d",
        ]
    
    # Filter to columns that exist in both DataFrames
    available_factors = [c for c in factor_columns 
                         if c in stock_metrics.columns and c in sp500_reference.columns]
    
    if not available_factors:
        print("[sector_cache] WARNING: No common factor columns found")
        return stock_metrics.copy()
    
    # Define which factors should be ranked ascending (lower is better)
    ascending_factors = {"debt_to_equity", "volatility_60d"}
    
    result = stock_metrics.copy()
    
    for _, row in stock_metrics.iterrows():
        ticker = row["ticker"]
        sector = row.get("sector")
        
        if pd.isna(sector) or sector is None:
            # No sector - compare against entire S&P 500
            sector_peers = sp500_reference
        else:
            # Get all S&P 500 stocks in same sector
            sector_peers = sp500_reference[sp500_reference["sector"] == sector]
        
        if sector_peers.empty:
            # Fallback to entire S&P 500 if sector has no peers
            sector_peers = sp500_reference
        
        # Compute percentile for each factor
        for factor in available_factors:
            stock_value = row.get(factor)
            
            if pd.isna(stock_value):
                continue
            
            peer_values = sector_peers[factor].dropna()
            
            if peer_values.empty:
                continue
            
            # Compute percentile rank
            if factor in ascending_factors:
                # Lower is better - percentile of stocks with HIGHER values
                percentile = (peer_values > stock_value).sum() / len(peer_values)
            else:
                # Higher is better - percentile of stocks with LOWER values
                percentile = (peer_values < stock_value).sum() / len(peer_values)
            
            # Update result
            result.loc[result["ticker"] == ticker, f"{factor}_score"] = percentile
    
    return result


def get_sector_percentile_for_ticker(
    ticker: str,
    sector: str,
    raw_metrics: dict,
    sp500_reference: pd.DataFrame
) -> dict:
    """
    Convenience function to get sector percentiles for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        sector: Sector classification
        raw_metrics: Dictionary of raw factor values for the stock
        sp500_reference: Full S&P 500 reference DataFrame
    
    Returns:
        Dictionary mapping factor names to percentile scores (0-1)
    """
    if sp500_reference.empty:
        return {}
    
    # Get sector peers
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
        
        peer_values = sector_peers[factor].dropna()
        
        if peer_values.empty:
            continue
        
        if factor in ascending_factors:
            percentile = (peer_values > value).sum() / len(peer_values)
        else:
            percentile = (peer_values < value).sum() / len(peer_values)
        
        percentiles[factor] = float(percentile)
    
    return percentiles


def get_cache_status() -> dict:
    """
    Get information about the current cache status.
    
    Returns:
        Dictionary with cache status information
    """
    cache_path = _get_cache_file_path()
    metadata = _load_cache_metadata()
    
    status = {
        "cache_exists": cache_path.exists(),
        "cache_path": str(cache_path),
        "is_stale": _is_cache_stale(),
        "last_updated": metadata.get("last_updated"),
        "ticker_count": metadata.get("ticker_count", 0),
        "sectors": metadata.get("sectors", []),
    }
    
    return status


def clear_cache() -> bool:
    """
    Clear the cache files.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cache_path = _get_cache_file_path()
        meta_path = _get_metadata_file_path()
        
        if cache_path.exists():
            cache_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        
        print("[sector_cache] Cache cleared")
        return True
    except Exception as e:
        print(f"[sector_cache] Error clearing cache: {e}")
        return False
