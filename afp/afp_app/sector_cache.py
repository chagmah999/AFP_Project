"""
sector_cache.py

Hybrid caching system for S&P 500 factor scores.

This module maintains two separate caches:
1. Fundamentals cache (profiles, balance sheet, income statement, cash flow)
   - Refreshed MONTHLY (these change quarterly at most)
2. Prices cache (price history for momentum/volatility)
   - Refreshed WEEKLY (or daily if API limits allow)

This hybrid approach minimizes API calls while keeping time-sensitive
data (momentum, volatility) reasonably fresh.
"""

from __future__ import annotations

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd

from .config import FMP_API_KEY, DEFAULT_START_DATE
from .fmp import FMPDataFetcher
from .universe import fetch_sp500_from_fmp, ALL_SP500


# =============================================================================
# Configuration
# =============================================================================

CACHE_DIR = os.getenv("SP500_CACHE_DIR", "./cache")

# Cache filenames
FUNDAMENTALS_CACHE_FILENAME = "sp500_fundamentals.parquet"
PRICES_CACHE_FILENAME = "sp500_prices.parquet"
CACHE_METADATA_FILENAME = "sp500_cache_metadata.json"

# Refresh frequencies (in days)
# Change PRICES_REFRESH_DAYS to 1 for daily refresh if API limits allow
FUNDAMENTALS_REFRESH_DAYS = 30  # Monthly
PRICES_REFRESH_DAYS = 7         # Weekly (change to 1 for daily)

# Minimum data requirements
MIN_PRICE_HISTORY_DAYS = 60

# Rate limiting for API calls
API_CALL_DELAY = 0.1  # seconds between API calls
BATCH_SIZE = 10       # Number of tickers to process before a longer pause
BATCH_DELAY = 1.0     # seconds to pause after each batch


# =============================================================================
# Cache Path Helpers
# =============================================================================

def _get_cache_dir() -> Path:
    """Get or create the cache directory."""
    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _get_fundamentals_cache_path() -> Path:
    """Get the path to the fundamentals cache file."""
    return _get_cache_dir() / FUNDAMENTALS_CACHE_FILENAME


def _get_prices_cache_path() -> Path:
    """Get the path to the prices cache file."""
    return _get_cache_dir() / PRICES_CACHE_FILENAME


def _get_metadata_file_path() -> Path:
    """Get the path to the cache metadata file."""
    return _get_cache_dir() / CACHE_METADATA_FILENAME


# =============================================================================
# Metadata Management
# =============================================================================

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
    """Save cache metadata."""
    meta_path = _get_metadata_file_path()
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def _update_metadata(cache_type: str, ticker_count: int, extra_info: dict = None) -> None:
    """Update metadata for a specific cache type."""
    metadata = _load_cache_metadata()
    
    metadata[f"{cache_type}_last_updated"] = datetime.now().isoformat()
    metadata[f"{cache_type}_ticker_count"] = ticker_count
    
    if extra_info:
        for key, value in extra_info.items():
            metadata[f"{cache_type}_{key}"] = value
    
    _save_cache_metadata(metadata)


# =============================================================================
# Staleness Checks
# =============================================================================

def _is_fundamentals_stale() -> bool:
    """Check if the fundamentals cache needs to be refreshed."""
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
    """Check if the prices cache needs to be refreshed."""
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


# =============================================================================
# Cache Loading
# =============================================================================

def _load_fundamentals_cache() -> Optional[pd.DataFrame]:
    """Load the cached fundamentals data."""
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
    """Load the cached prices data."""
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


# =============================================================================
# Cache Saving
# =============================================================================

def _save_fundamentals_cache(df: pd.DataFrame, sectors: list = None) -> bool:
    """Save the fundamentals data to cache."""
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
    """Save the prices data to cache."""
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


# =============================================================================
# Data Collection: Fundamentals
# =============================================================================

def _collect_sp500_fundamentals(
    tickers: list[str],
    fetcher: FMPDataFetcher,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> pd.DataFrame:
    """
    Collect fundamental data (profiles, BS, IS, CF) for all S&P 500 stocks.
    
    Returns a DataFrame with one row per ticker containing:
    - ticker, sector, industry
    - Key balance sheet items
    - Key income statement items
    - Key cash flow items
    """
    bs_rows = []
    inc_rows = []
    cf_rows = []
    profile_data = {}
    
    total = len(tickers)
    
    for i, tk in enumerate(tickers):
        try:
            # Fetch profile for sector/industry
            prof = fetcher.get_profile(tk)
            if prof:
                profile_data[tk] = {
                    "sector": prof.get("sector"),
                    "industry": prof.get("industry"),
                }
            
            # Fetch fundamentals
            bs = fetcher.get_balance_sheet(tk)
            inc = fetcher.get_income_statement(tk)
            cf = fetcher.get_cash_flow(tk)
            
            if isinstance(bs, list) and bs:
                row = bs[0].copy()  # Most recent
                row["ticker"] = tk
                bs_rows.append(row)
            
            if isinstance(inc, list) and inc:
                row = inc[0].copy()  # Most recent
                row["ticker"] = tk
                inc_rows.append(row)
            
            if isinstance(cf, list) and cf:
                row = cf[0].copy()  # Most recent
                row["ticker"] = tk
                cf_rows.append(row)
            
            if progress_callback:
                progress_callback(i + 1, total, f"Fundamentals: {tk}")
            
        except Exception as e:
            print(f"[sector_cache] Error fetching fundamentals for {tk}: {e}")
        
        # Rate limiting
        time.sleep(API_CALL_DELAY)
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(BATCH_DELAY)
    
    # Build DataFrames
    bs_df = pd.DataFrame(bs_rows) if bs_rows else pd.DataFrame()
    inc_df = pd.DataFrame(inc_rows) if inc_rows else pd.DataFrame()
    cf_df = pd.DataFrame(cf_rows) if cf_rows else pd.DataFrame()
    
    # Merge all fundamentals
    if bs_df.empty:
        return pd.DataFrame()
    
    result = bs_df.copy()
    
    if not inc_df.empty:
        # Select key income statement columns
        inc_cols = ["ticker", "revenue", "netIncome", "grossProfit", "operatingIncome", 
                    "eps", "ebitda", "weightedAverageShsOut", "weightedAverageShsOutDil"]
        inc_cols = [c for c in inc_cols if c in inc_df.columns]
        result = result.merge(inc_df[inc_cols], on="ticker", how="left")
    
    if not cf_df.empty:
        # Select key cash flow columns
        cf_cols = ["ticker", "freeCashFlow", "operatingCashFlow"]
        cf_cols = [c for c in cf_cols if c in cf_df.columns]
        result = result.merge(cf_df[cf_cols], on="ticker", how="left")
    
    # Add sector/industry from profiles
    result["sector"] = result["ticker"].map(lambda tk: profile_data.get(tk, {}).get("sector"))
    result["industry"] = result["ticker"].map(lambda tk: profile_data.get(tk, {}).get("industry"))
    
    # Convert date column
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
    
    return result


# =============================================================================
# Data Collection: Prices
# =============================================================================

def _collect_sp500_prices(
    tickers: list[str],
    fetcher: FMPDataFetcher,
    start_date: str,
    end_date: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> pd.DataFrame:
    """
    Collect price data for all S&P 500 stocks.
    
    Returns DataFrame with: date, ticker, adjClose, returns, log_returns
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
            
            # Keep only necessary columns
            keep_cols = ["date", "ticker", "adjClose", "returns", "log_returns"]
            keep_cols = [c for c in keep_cols if c in px.columns]
            frames.append(px[keep_cols])
            
            if progress_callback:
                progress_callback(i + 1, total, f"Prices: {tk}")
            
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
# Factor Score Computation
# =============================================================================

def _safe_div(num, den):
    """Safe division handling zeros and infinities."""
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
    """
    Compute raw factor metrics from fundamentals and prices.
    
    Returns DataFrame with columns:
        ticker, sector, industry,
        bp_ratio, ep_ratio, fcfp_ratio (VALUE)
        roe, roa, gross_margin, fcf_margin, debt_to_equity (QUALITY)
        momentum_60d (MOMENTUM)
        volatility_60d (LOW_VOL)
    """
    if fundamentals.empty:
        return pd.DataFrame()
    
    metrics = fundamentals.copy()
    
    # Get shares outstanding
    share_candidates = ["outstandingShares", "weightedAverageShsOut", "weightedAverageShsOutDil"]
    metrics["shares_out"] = np.nan
    for col in share_candidates:
        if col in metrics.columns:
            metrics["shares_out"] = metrics["shares_out"].fillna(
                pd.to_numeric(metrics[col], errors="coerce")
            )
    
    # Get last price for each ticker
    if not prices.empty and "ticker" in prices.columns:
        last_price = (
            prices.sort_values("date")
            .groupby("ticker")["adjClose"]
            .last()
        )
        metrics["price_last"] = metrics["ticker"].map(last_price)
    else:
        metrics["price_last"] = np.nan
    
    # Book equity
    metrics["book_equity"] = pd.to_numeric(
        metrics.get("totalStockholdersEquity"), errors="coerce"
    )
    
    # Market cap
    metrics["market_cap"] = metrics["shares_out"] * metrics["price_last"]
    
    # VALUE FACTORS
    metrics["bp_ratio"] = _safe_div(metrics["book_equity"], metrics["market_cap"])
    metrics["ep_ratio"] = _safe_div(
        pd.to_numeric(metrics.get("netIncome"), errors="coerce"),
        metrics["market_cap"]
    )
    metrics["fcfp_ratio"] = _safe_div(
        pd.to_numeric(metrics.get("freeCashFlow"), errors="coerce") if "freeCashFlow" in metrics.columns else np.nan,
        metrics["market_cap"]
    )
    
    # QUALITY FACTORS
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
    
    # MOMENTUM & VOLATILITY (from prices)
    if not prices.empty:
        # Momentum: 60-day price change
        px_pivot = prices.pivot_table(index="date", columns="ticker", values="adjClose")
        
        if not px_pivot.empty:
            mom_60d = px_pivot.pct_change(60)
            if not mom_60d.empty:
                last_mom = mom_60d.iloc[-1]
                metrics["momentum_60d"] = metrics["ticker"].map(last_mom.to_dict())
        
        # Volatility: 60-day rolling std of returns
        vol_60d = (
            prices
            .sort_values(["ticker", "date"])
            .groupby("ticker")["returns"]
            .apply(lambda x: x.rolling(60, min_periods=30).std().iloc[-1] if len(x) >= 30 else np.nan)
        )
        metrics["volatility_60d"] = metrics["ticker"].map(vol_60d.to_dict())
    
    # Clean up infinities
    metrics = metrics.replace([np.inf, -np.inf], np.nan)
    
    # Select output columns
    output_cols = [
        "ticker", "sector", "industry",
        # Value
        "bp_ratio", "ep_ratio", "fcfp_ratio",
        # Quality
        "roe", "roa", "gross_margin", "fcf_margin", "debt_to_equity",
        # Momentum
        "momentum_60d",
        # Low Vol
        "volatility_60d",
        # Additional
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
    force_refresh_fundamentals: bool = False,
    force_refresh_prices: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> pd.DataFrame:
    """
    Get S&P 500 factor scores using hybrid caching.
    
    Fundamentals are refreshed monthly, prices are refreshed weekly (or daily).
    
    Args:
        fetcher: FMPDataFetcher instance (created if not provided)
        force_refresh: If True, force refresh of both caches
        force_refresh_fundamentals: If True, force refresh fundamentals only
        force_refresh_prices: If True, force refresh prices only
        progress_callback: Optional callback(current, total, message) for progress updates
    
    Returns:
        DataFrame with factor scores for all S&P 500 stocks
    """
    # Determine what needs refreshing
    refresh_fundamentals = force_refresh or force_refresh_fundamentals or _is_fundamentals_stale()
    refresh_prices = force_refresh or force_refresh_prices or _is_prices_stale()
    
    # Load existing caches
    fundamentals_df = _load_fundamentals_cache() if not refresh_fundamentals else None
    prices_df = _load_prices_cache() if not refresh_prices else None
    
    # Check if we need to fetch anything
    need_fetcher = refresh_fundamentals or refresh_prices
    
    if need_fetcher:
        if fetcher is None:
            if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY":
                print("[sector_cache] ERROR: No valid FMP API key configured")
                # Try to use stale caches if available
                if fundamentals_df is None:
                    fundamentals_df = _load_fundamentals_cache()
                if prices_df is None:
                    prices_df = _load_prices_cache()
                
                if fundamentals_df is None or prices_df is None:
                    return pd.DataFrame()
            else:
                fetcher = FMPDataFetcher(FMP_API_KEY)
        
        # Get S&P 500 constituents
        tickers = fetch_sp500_from_fmp()
        if not tickers:
            print("[sector_cache] Using fallback S&P 500 list")
            tickers = ALL_SP500.copy()
        
        print(f"[sector_cache] Processing {len(tickers)} S&P 500 stocks...")
        
        # Calculate total steps for progress
        total_steps = 0
        if refresh_fundamentals:
            total_steps += len(tickers)
        if refresh_prices:
            total_steps += len(tickers)
        
        current_step = 0
        
        # Refresh fundamentals if needed
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
        
        # Refresh prices if needed
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
    
    # Compute factor scores from cached data
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
    """
    Get detailed information about the current cache status.
    
    Returns:
        Dictionary with cache status information for both fundamentals and prices
    """
    metadata = _load_cache_metadata()
    
    fundamentals_path = _get_fundamentals_cache_path()
    prices_path = _get_prices_cache_path()
    
    status = {
        # Overall
        "cache_exists": fundamentals_path.exists() and prices_path.exists(),
        "cache_dir": str(_get_cache_dir()),
        
        # Fundamentals
        "fundamentals_exists": fundamentals_path.exists(),
        "fundamentals_last_updated": metadata.get("fundamentals_last_updated"),
        "fundamentals_ticker_count": metadata.get("fundamentals_ticker_count", 0),
        "fundamentals_sectors": metadata.get("fundamentals_sectors", []),
        "fundamentals_is_stale": _is_fundamentals_stale(),
        "fundamentals_refresh_days": FUNDAMENTALS_REFRESH_DAYS,
        
        # Prices
        "prices_exists": prices_path.exists(),
        "prices_last_updated": metadata.get("prices_last_updated"),
        "prices_ticker_count": metadata.get("prices_ticker_count", 0),
        "prices_is_stale": _is_prices_stale(),
        "prices_refresh_days": PRICES_REFRESH_DAYS,
        
        # Combined staleness (for backward compatibility)
        "is_stale": _is_fundamentals_stale() or _is_prices_stale(),
        "last_updated": metadata.get("prices_last_updated") or metadata.get("fundamentals_last_updated"),
        "ticker_count": metadata.get("fundamentals_ticker_count", 0),
        "sectors": metadata.get("fundamentals_sectors", []),
    }
    
    return status


def clear_cache() -> bool:
    """
    Clear all cache files.
    
    Returns:
        True if successful, False otherwise
    """
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
    """Clear only the fundamentals cache."""
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
    """Clear only the prices cache."""
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


# =============================================================================
# Sector Percentile Computation (unchanged from original)
# =============================================================================

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
