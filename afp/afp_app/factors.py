"""
factors.py

Factor metric computation and portfolio construction.

This module provides:
1. Raw factor metric computation (ratios, returns, volatility)
2. Percentile ranking (universe-only or sector-relative via S&P 500 reference)
3. Factor portfolio construction (long-short portfolios)
4. Factor return calculation
"""

import numpy as np
import pandas as pd
from typing import Optional


def _safe_div(num, den):
    """Safe division handling zeros and infinities."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


# =============================================================================
# RAW FACTOR METRICS (No Percentile Ranking)
# =============================================================================

def calculate_raw_factor_metrics(
    fundamentals: dict,
    price_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute raw factor metrics WITHOUT percentile ranking.
    
    This function calculates the underlying ratios and values that can later
    be ranked against sector peers or the full universe.
    
    Args:
        fundamentals: Dictionary with 'balance_sheet', 'income_statement', 'cash_flow' DataFrames
        price_data: DataFrame with columns: date, ticker, adjClose, returns
    
    Returns:
        DataFrame with columns:
            ticker, sector, industry, date,
            bp_ratio, ep_ratio, fcfp_ratio (VALUE)
            roe, roa, gross_margin, fcf_margin, debt_to_equity (QUALITY)
            momentum_60d (MOMENTUM)
            volatility_60d (LOW_VOL)
            market_cap, price_last (additional)
    """
    bs = fundamentals.get("balance_sheet", pd.DataFrame())
    inc = fundamentals.get("income_statement", pd.DataFrame())
    cf = fundamentals.get("cash_flow", pd.DataFrame())

    if bs.empty or inc.empty:
        return pd.DataFrame()

    # Select relevant columns from balance sheet
    bs_cols = [
        "ticker", "date",
        "totalStockholdersEquity",
        "totalAssets",
        "totalLiabilities",
        "totalDebt",
        "cashAndCashEquivalents",
    ]

    if "outstandingShares" in bs.columns:
        bs_cols.append("outstandingShares")

    for meta_col in ["sector", "industry"]:
        if meta_col in bs.columns:
            bs_cols.append(meta_col)

    bs_cols = [c for c in bs_cols if c in bs.columns]
    bs_use = bs[bs_cols].copy()

    # Select relevant columns from income statement
    inc_cols = [
        "ticker", "date",
        "revenue",
        "netIncome",
        "grossProfit",
        "operatingIncome",
        "eps",
        "ebitda",
        "weightedAverageShsOut",
        "weightedAverageShsOutDil",
    ]
    inc_cols = [c for c in inc_cols if c in inc.columns]
    inc_use = inc[inc_cols].copy()

    # Merge balance sheet and income statement
    metrics = pd.merge(
        bs_use,
        inc_use,
        on=["ticker", "date"],
        how="inner",
    )

    # Get shares outstanding from multiple possible columns
    share_candidates = [
        "outstandingShares",
        "weightedAverageShsOut",
        "weightedAverageShsOutDil",
    ]
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
            metrics = pd.merge(
                metrics,
                cf_use,
                on=["ticker", "date"],
                how="left",
            )

    # Book equity
    metrics["book_equity"] = pd.to_numeric(
        metrics["totalStockholdersEquity"], errors="coerce"
    )

    # Get last price for each ticker
    if not price_data.empty:
        last_price = (
            price_data.sort_values("date")
            .groupby("ticker")["adjClose"]
            .last()
        )
        metrics["price_last"] = metrics["ticker"].map(last_price)
    else:
        metrics["price_last"] = np.nan

    # Market cap
    metrics["market_cap"] = _safe_div(
        metrics["shares_out"] * metrics["price_last"],
        1.0,
    )

    # =========================================================================
    # VALUE FACTORS (raw ratios)
    # =========================================================================
    metrics["bp_ratio"] = _safe_div(
        metrics["book_equity"],
        metrics["market_cap"],
    )
    metrics["ep_ratio"] = _safe_div(
        pd.to_numeric(metrics.get("netIncome"), errors="coerce"),
        metrics["market_cap"],
    )
    metrics["fcfp_ratio"] = _safe_div(
        pd.to_numeric(metrics.get("freeCashFlow"), errors="coerce") if "freeCashFlow" in metrics.columns else np.nan,
        metrics["market_cap"],
    )

    # =========================================================================
    # QUALITY FACTORS (raw ratios)
    # =========================================================================
    metrics["roe"] = _safe_div(
        pd.to_numeric(metrics.get("netIncome"), errors="coerce"),
        pd.to_numeric(metrics.get("totalStockholdersEquity"), errors="coerce"),
    )
    metrics["roa"] = _safe_div(
        pd.to_numeric(metrics.get("netIncome"), errors="coerce"),
        pd.to_numeric(metrics.get("totalAssets"), errors="coerce"),
    )
    metrics["gross_margin"] = _safe_div(
        pd.to_numeric(metrics.get("grossProfit"), errors="coerce"),
        pd.to_numeric(metrics.get("revenue"), errors="coerce"),
    )
    metrics["fcf_margin"] = _safe_div(
        pd.to_numeric(metrics.get("freeCashFlow"), errors="coerce") if "freeCashFlow" in metrics.columns else np.nan,
        pd.to_numeric(metrics.get("revenue"), errors="coerce"),
    )
    metrics["debt_to_equity"] = _safe_div(
        pd.to_numeric(metrics.get("totalDebt"), errors="coerce"),
        pd.to_numeric(metrics.get("totalStockholdersEquity"), errors="coerce"),
    )

    # =========================================================================
    # MOMENTUM & VOLATILITY (from price data)
    # =========================================================================
    if not price_data.empty:
        px_pivot = price_data.pivot_table(
            index="date",
            columns="ticker",
            values="adjClose",
        )

        # Momentum: 60-day price change
        mom_60d = px_pivot.pct_change(60)
        if not mom_60d.empty:
            last_mom = mom_60d.iloc[-1]
            for tk in last_mom.index:
                metrics.loc[metrics["ticker"] == tk, "momentum_60d"] = last_mom[tk]

        # Volatility: 60-day rolling std of returns
        vol_60d = (
            price_data
            .sort_values(["ticker", "date"])
            .groupby("ticker")["returns"]
            .apply(
                lambda x: x.rolling(60, min_periods=30).std().iloc[-1]
                if len(x) >= 30 else np.nan
            )
        )
        for tk in vol_60d.index:
            metrics.loc[metrics["ticker"] == tk, "volatility_60d"] = vol_60d[tk]

    # Clean up infinities
    metrics = metrics.replace([np.inf, -np.inf], np.nan)

    return metrics


# =============================================================================
# PERCENTILE RANKING FUNCTIONS
# =============================================================================

def _rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Compute percentile rank (0-1) for a series."""
    return series.rank(method="average", pct=True, ascending=ascending)


def _zscore_grouped(series: pd.Series) -> pd.Series:
    """Compute z-score within a group."""
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(index=series.index, data=np.nan)
    return (series - mu) / sigma


def compute_universe_percentiles(
    metrics: pd.DataFrame,
    group_by_sector: bool = True
) -> pd.DataFrame:
    """
    Compute percentile scores relative to the provided universe only.
    
    This is the ORIGINAL behavior - ranking stocks only against others
    in the same DataFrame.
    
    Args:
        metrics: DataFrame from calculate_raw_factor_metrics()
        group_by_sector: If True, rank within sector; if False, rank across all stocks
    
    Returns:
        DataFrame with added score columns: value_score, quality_score, 
        momentum_score, lowvol_score
    """
    if metrics.empty:
        return metrics
    
    metrics = metrics.copy()
    
    # Determine grouping
    if group_by_sector and "sector" in metrics.columns:
        group_cols = ["sector"]
    elif "industry" in metrics.columns:
        group_cols = ["industry"]
    else:
        group_cols = []

    def _group_rank(df: pd.DataFrame, col_name: str, ascending: bool = True) -> pd.Series:
        if col_name not in df.columns:
            return pd.Series(index=df.index, data=np.nan)
        if group_cols:
            return (
                df.groupby(group_cols)[col_name]
                .transform(lambda s: _rank_pct(s, ascending=ascending))
            )
        else:
            return _rank_pct(df[col_name], ascending=ascending)

    # =========================================================================
    # VALUE SCORE (composite of bp, ep, fcfp z-scores)
    # =========================================================================
    value_components = []

    if "bp_ratio" in metrics.columns:
        if group_cols:
            metrics["z_bp"] = (
                metrics.groupby(group_cols)["bp_ratio"]
                .transform(_zscore_grouped)
            )
        else:
            metrics["z_bp"] = _zscore_grouped(metrics["bp_ratio"])
        value_components.append("z_bp")

    if "ep_ratio" in metrics.columns:
        if group_cols:
            metrics["z_ep"] = (
                metrics.groupby(group_cols)["ep_ratio"]
                .transform(_zscore_grouped)
            )
        else:
            metrics["z_ep"] = _zscore_grouped(metrics["ep_ratio"])
        value_components.append("z_ep")

    if "fcfp_ratio" in metrics.columns:
        if group_cols:
            metrics["z_fcfp"] = (
                metrics.groupby(group_cols)["fcfp_ratio"]
                .transform(_zscore_grouped)
            )
        else:
            metrics["z_fcfp"] = _zscore_grouped(metrics["fcfp_ratio"])
        value_components.append("z_fcfp")

    if value_components:
        metrics["value_raw"] = metrics[value_components].mean(axis=1, skipna=True)

        if group_cols:
            metrics["value_score"] = (
                metrics.groupby(group_cols)["value_raw"]
                .transform(lambda s: s.rank(method="average", pct=True, ascending=False))
            )
        else:
            metrics["value_score"] = _rank_pct(metrics["value_raw"], ascending=False)

    # =========================================================================
    # QUALITY SCORE (composite of roe, roa, margins, inverse leverage)
    # =========================================================================
    quality_components = []

    if "roe" in metrics.columns:
        metrics["q_roe"] = _group_rank(metrics, "roe", ascending=True)
        quality_components.append("q_roe")

    if "roa" in metrics.columns:
        metrics["q_roa"] = _group_rank(metrics, "roa", ascending=True)
        quality_components.append("q_roa")

    if "gross_margin" in metrics.columns:
        metrics["q_gm"] = _group_rank(metrics, "gross_margin", ascending=True)
        quality_components.append("q_gm")

    if "fcf_margin" in metrics.columns:
        metrics["q_fcfm"] = _group_rank(metrics, "fcf_margin", ascending=True)
        quality_components.append("q_fcfm")

    if "debt_to_equity" in metrics.columns:
        # Lower debt is better
        lev_rank = _group_rank(metrics, "debt_to_equity", ascending=True)
        metrics["q_levinv"] = 1.0 - lev_rank
        quality_components.append("q_levinv")

    if quality_components:
        metrics["quality_score"] = metrics[quality_components].mean(axis=1)

    # =========================================================================
    # LOW VOLATILITY SCORE (lower vol = higher score)
    # =========================================================================
    if "volatility_60d" in metrics.columns:
        vol_rank = _group_rank(metrics, "volatility_60d", ascending=True)
        metrics["lowvol_score"] = 1.0 - vol_rank

    # =========================================================================
    # MOMENTUM SCORE (higher momentum = higher score)
    # =========================================================================
    if "momentum_60d" in metrics.columns:
        metrics["momentum_score"] = _group_rank(metrics, "momentum_60d", ascending=True)

    return metrics


def compute_sector_relative_percentiles(
    universe_metrics: pd.DataFrame,
    sp500_reference: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute percentile scores relative to ALL S&P 500 stocks in the same sector.
    
    This is the NEW behavior - ranking each stock in the user's universe 
    against all ~500 S&P 500 stocks with the same sector.
    
    Args:
        universe_metrics: DataFrame from calculate_raw_factor_metrics() for user's universe
        sp500_reference: Full S&P 500 reference DataFrame from sector_cache.get_sp500_sector_scores()
    
    Returns:
        DataFrame with added score columns: value_score, quality_score, 
        momentum_score, lowvol_score (all relative to S&P 500 sector peers)
    """
    if universe_metrics.empty:
        return universe_metrics
    
    if sp500_reference.empty:
        print("[factors] WARNING: No S&P 500 reference provided, falling back to universe-only ranking")
        return compute_universe_percentiles(universe_metrics, group_by_sector=True)
    
    if "sector" not in universe_metrics.columns or "sector" not in sp500_reference.columns:
        print("[factors] WARNING: Missing sector column, falling back to universe-only ranking")
        return compute_universe_percentiles(universe_metrics, group_by_sector=False)
    
    result = universe_metrics.copy()
    
    # Define raw metric columns for each factor
    value_metrics = ["bp_ratio", "ep_ratio", "fcfp_ratio"]
    quality_metrics = ["roe", "roa", "gross_margin", "fcf_margin", "debt_to_equity"]
    momentum_metrics = ["momentum_60d"]
    lowvol_metrics = ["volatility_60d"]
    
    # Define which metrics are "lower is better"
    ascending_metrics = {"debt_to_equity", "volatility_60d"}
    
    # Initialize score columns
    result["value_score"] = np.nan
    result["quality_score"] = np.nan
    result["momentum_score"] = np.nan
    result["lowvol_score"] = np.nan
    
    # Process each stock in the universe
    for idx, row in universe_metrics.iterrows():
        ticker = row["ticker"]
        sector = row.get("sector")
        
        # Get sector peers from S&P 500 reference
        if pd.isna(sector) or sector is None or sector == "":
            # No sector - compare against entire S&P 500
            sector_peers = sp500_reference
        else:
            sector_peers = sp500_reference[sp500_reference["sector"] == sector]
            if sector_peers.empty:
                # Fallback to entire S&P 500 if no peers found
                sector_peers = sp500_reference
        
        # ---------------------------------------------------------------------
        # VALUE SCORE
        # ---------------------------------------------------------------------
        value_percentiles = []
        for metric in value_metrics:
            if metric not in row or pd.isna(row[metric]):
                continue
            if metric not in sector_peers.columns:
                continue
            
            peer_values = sector_peers[metric].dropna()
            if peer_values.empty:
                continue
            
            stock_value = row[metric]
            # Higher is better for value metrics
            percentile = (peer_values < stock_value).sum() / len(peer_values)
            value_percentiles.append(percentile)
        
        if value_percentiles:
            result.loc[idx, "value_score"] = np.mean(value_percentiles)
        
        # ---------------------------------------------------------------------
        # QUALITY SCORE
        # ---------------------------------------------------------------------
        quality_percentiles = []
        for metric in quality_metrics:
            if metric not in row or pd.isna(row[metric]):
                continue
            if metric not in sector_peers.columns:
                continue
            
            peer_values = sector_peers[metric].dropna()
            if peer_values.empty:
                continue
            
            stock_value = row[metric]
            
            if metric in ascending_metrics:
                # Lower is better (debt_to_equity)
                percentile = (peer_values > stock_value).sum() / len(peer_values)
            else:
                # Higher is better
                percentile = (peer_values < stock_value).sum() / len(peer_values)
            
            quality_percentiles.append(percentile)
        
        if quality_percentiles:
            result.loc[idx, "quality_score"] = np.mean(quality_percentiles)
        
        # ---------------------------------------------------------------------
        # MOMENTUM SCORE
        # ---------------------------------------------------------------------
        for metric in momentum_metrics:
            if metric not in row or pd.isna(row[metric]):
                continue
            if metric not in sector_peers.columns:
                continue
            
            peer_values = sector_peers[metric].dropna()
            if peer_values.empty:
                continue
            
            stock_value = row[metric]
            # Higher is better for momentum
            percentile = (peer_values < stock_value).sum() / len(peer_values)
            result.loc[idx, "momentum_score"] = percentile
        
        # ---------------------------------------------------------------------
        # LOW VOLATILITY SCORE
        # ---------------------------------------------------------------------
        for metric in lowvol_metrics:
            if metric not in row or pd.isna(row[metric]):
                continue
            if metric not in sector_peers.columns:
                continue
            
            peer_values = sector_peers[metric].dropna()
            if peer_values.empty:
                continue
            
            stock_value = row[metric]
            # Lower is better for volatility
            percentile = (peer_values > stock_value).sum() / len(peer_values)
            result.loc[idx, "lowvol_score"] = percentile
    
    return result


# =============================================================================
# MAIN ENTRY POINT (Backward Compatible)
# =============================================================================

def calculate_factor_metrics(
    fundamentals: dict,
    price_data: pd.DataFrame,
    sp500_reference: Optional[pd.DataFrame] = None,
    use_sector_relative: bool = True
) -> pd.DataFrame:
    """
    Calculate factor metrics with percentile scores.
    
    This is the main entry point that maintains backward compatibility.
    
    Args:
        fundamentals: Dictionary with 'balance_sheet', 'income_statement', 'cash_flow' DataFrames
        price_data: DataFrame with columns: date, ticker, adjClose, returns
        sp500_reference: Optional S&P 500 reference DataFrame for sector-relative scoring.
                         If provided and use_sector_relative=True, percentiles are computed
                         relative to all S&P 500 stocks in the same sector.
        use_sector_relative: If True and sp500_reference is provided, use sector-relative
                             percentiles. If False, use universe-only percentiles.
    
    Returns:
        DataFrame with all factor metrics and score columns
    """
    # Step 1: Compute raw metrics
    metrics = calculate_raw_factor_metrics(fundamentals, price_data)
    
    if metrics.empty:
        return metrics
    
    # Step 2: Compute percentile scores
    if use_sector_relative and sp500_reference is not None and not sp500_reference.empty:
        # NEW BEHAVIOR: Rank against S&P 500 sector peers
        metrics = compute_sector_relative_percentiles(metrics, sp500_reference)
    else:
        # ORIGINAL BEHAVIOR: Rank within universe only
        metrics = compute_universe_percentiles(metrics, group_by_sector=True)
    
    return metrics


# =============================================================================
# FACTOR PORTFOLIO CONSTRUCTION
# =============================================================================

class FactorPortfolioConstructor:
    """
    Construct long-short factor portfolios from 0-1 factor scores and compute factor returns.
    """

    def __init__(self, metrics_df: pd.DataFrame, price_df: pd.DataFrame):
        self.metrics = metrics_df
        self.prices = price_df
        self.portfolios: dict[str, pd.DataFrame] = {}

    def construct_factor_portfolio(
        self,
        factor_name: str,
        metric_column: str,
        ascending: bool,
        percentile: float = 0.3,
    ) -> pd.DataFrame:
        """
        Construct a long-short portfolio for a single factor.
        
        Args:
            factor_name: Name of the factor (e.g., "VALUE")
            metric_column: Column name containing the factor score
            ascending: If True, low values are "good" (go long); 
                       if False, high values are "good" (go long)
            percentile: Percentile threshold for long/short selection
        
        Returns:
            DataFrame with columns: factor, ticker, position, weight
        """
        if self.metrics.empty or metric_column not in self.metrics.columns:
            return pd.DataFrame()

        # Get latest metrics per ticker
        latest = (
            self.metrics.sort_values("date")
            .groupby("ticker")
            .last()
        )

        valid = latest[metric_column].dropna()
        if len(valid) < 3:
            return pd.DataFrame()

        low = valid.quantile(percentile)
        high = valid.quantile(1 - percentile)

        if ascending:
            # Low values are good - go long on low, short on high
            long_tk = valid[valid <= low].index.tolist()
            short_tk = valid[valid >= high].index.tolist()
        else:
            # High values are good - go long on high, short on low
            long_tk = valid[valid >= high].index.tolist()
            short_tk = valid[valid <= low].index.tolist()

        if not long_tk or not short_tk:
            return pd.DataFrame()

        w_long = [1.0 / len(long_tk)] * len(long_tk)
        w_short = [-1.0 / len(short_tk)] * len(short_tk)

        port = pd.DataFrame({
            "factor": factor_name,
            "ticker": long_tk + short_tk,
            "position": ["long"] * len(long_tk) + ["short"] * len(short_tk),
            "weight": w_long + w_short,
        })
        return port

    def construct_all(self) -> dict[str, pd.DataFrame]:
        """
        Construct portfolios for all standard factors.
        
        Returns:
            Dictionary mapping factor names to portfolio DataFrames
        """
        ports: dict[str, pd.DataFrame] = {}

        if "value_score" in self.metrics.columns:
            ports["VALUE"] = self.construct_factor_portfolio(
                "VALUE", "value_score", ascending=False
            )
        else:
            ports["VALUE"] = pd.DataFrame()

        if "quality_score" in self.metrics.columns:
            ports["QUALITY"] = self.construct_factor_portfolio(
                "QUALITY", "quality_score", ascending=False
            )
        else:
            ports["QUALITY"] = pd.DataFrame()

        if "momentum_score" in self.metrics.columns:
            ports["MOMENTUM"] = self.construct_factor_portfolio(
                "MOMENTUM", "momentum_score", ascending=False
            )
        else:
            ports["MOMENTUM"] = pd.DataFrame()

        if "lowvol_score" in self.metrics.columns:
            ports["LOW_VOL"] = self.construct_factor_portfolio(
                "LOW_VOL", "lowvol_score", ascending=False
            )
        else:
            ports["LOW_VOL"] = pd.DataFrame()

        self.portfolios = ports
        return ports

    def calculate_factor_returns(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Calculate daily returns for each factor portfolio.
        
        Args:
            start_date: Start date for return calculation
            end_date: End date for return calculation
        
        Returns:
            DataFrame with columns: date, factor, return
        """
        rets = []

        for fname, port in self.portfolios.items():
            if port is None or port.empty:
                continue

            px = self.prices[
                (self.prices["ticker"].isin(port["ticker"])) &
                (self.prices["date"] >= start_date) &
                (self.prices["date"] <= end_date)
            ]
            if px.empty:
                continue

            for dt, day in px.groupby("date"):
                wr = 0.0
                for _, row in port.iterrows():
                    tr = day.loc[day["ticker"] == row["ticker"], "returns"]
                    if not tr.empty:
                        wr += row["weight"] * tr.values[0]
                rets.append({"date": dt, "factor": fname, "return": wr})

        return pd.DataFrame(rets)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_factor_score_columns() -> list[str]:
    """Return list of standard factor score column names."""
    return ["value_score", "quality_score", "momentum_score", "lowvol_score"]


def get_raw_metric_columns() -> dict[str, list[str]]:
    """Return mapping of factor names to their raw metric columns."""
    return {
        "VALUE": ["bp_ratio", "ep_ratio", "fcfp_ratio"],
        "QUALITY": ["roe", "roa", "gross_margin", "fcf_margin", "debt_to_equity"],
        "MOMENTUM": ["momentum_60d"],
        "LOW_VOL": ["volatility_60d"],
    }


def summarize_factor_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table of factor scores for display.
    
    Args:
        metrics: DataFrame with factor score columns
    
    Returns:
        DataFrame with ticker, sector, and factor scores formatted for display
    """
    if metrics.empty:
        return pd.DataFrame()
    
    display_cols = ["ticker"]
    
    if "sector" in metrics.columns:
        display_cols.append("sector")
    
    score_cols = get_factor_score_columns()
    for col in score_cols:
        if col in metrics.columns:
            display_cols.append(col)
    
    result = metrics[display_cols].copy()
    
    # Rename columns for display
    rename_map = {
        "value_score": "Value",
        "quality_score": "Quality",
        "momentum_score": "Momentum",
        "lowvol_score": "Low Vol",
    }
    result = result.rename(columns=rename_map)
    
    return result.sort_values("ticker").reset_index(drop=True)
