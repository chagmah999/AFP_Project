import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import rankdata

def _safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out

def _rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
    valid_mask = series.notna()
    if valid_mask.sum() < 2:
        return pd.Series(index=series.index, data=np.nan)

    result = pd.Series(index=series.index, data=np.nan)
    valid_values = series[valid_mask].values

    if ascending:

        ranks = rankdata(valid_values, method='average')
    else:

        ranks = rankdata(-valid_values, method='average')

    n = len(valid_values)
    if n > 1:
        percentiles = (ranks - 1) / (n - 1)
    else:
        percentiles = np.array([0.5])  

    result.loc[valid_mask] = percentiles
    return result

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

def _zscore_grouped(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(index=series.index, data=np.nan)
    return (series - mu) / sigma

def calculate_raw_factor_metrics(
    fundamentals: dict,
    price_data: pd.DataFrame
) -> pd.DataFrame:
    
    bs = fundamentals.get("balance_sheet", pd.DataFrame())
    inc = fundamentals.get("income_statement", pd.DataFrame())
    cf = fundamentals.get("cash_flow", pd.DataFrame())

    if bs.empty or inc.empty:
        return pd.DataFrame()

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

    metrics = pd.merge(
        bs_use,
        inc_use,
        on=["ticker", "date"],
        how="inner",
    )

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

    metrics["book_equity"] = pd.to_numeric(
        metrics["totalStockholdersEquity"], errors="coerce"
    )

    if not price_data.empty:
        last_price = (
            price_data.sort_values("date")
            .groupby("ticker")["adjClose"]
            .last()
        )
        metrics["price_last"] = metrics["ticker"].map(last_price)
    else:
        metrics["price_last"] = np.nan

    metrics["market_cap"] = _safe_div(
        metrics["shares_out"] * metrics["price_last"],
        1.0,
    )

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

    if not price_data.empty:
        px_pivot = price_data.pivot_table(
            index="date",
            columns="ticker",
            values="adjClose",
        )

        mom_60d = px_pivot.pct_change(60)
        if not mom_60d.empty:
            last_mom = mom_60d.iloc[-1]
            for tk in last_mom.index:
                metrics.loc[metrics["ticker"] == tk, "momentum_60d"] = last_mom[tk]

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

    metrics = metrics.replace([np.inf, -np.inf], np.nan)

    return metrics

def compute_universe_percentiles(
    metrics: pd.DataFrame,
    group_by_sector: bool = True
) -> pd.DataFrame:
    if metrics.empty:
        return metrics

    metrics = metrics.copy()

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
                .transform(lambda s: _rank_pct(s, ascending=False))
            )
        else:
            metrics["value_score"] = _rank_pct(metrics["value_raw"], ascending=False)

    quality_components = []

    if "roe" in metrics.columns:
        metrics["q_roe"] = _group_rank(metrics, "roe", ascending=False)  

        quality_components.append("q_roe")

    if "roa" in metrics.columns:
        metrics["q_roa"] = _group_rank(metrics, "roa", ascending=False)  

        quality_components.append("q_roa")

    if "gross_margin" in metrics.columns:
        metrics["q_gm"] = _group_rank(metrics, "gross_margin", ascending=False)  

        quality_components.append("q_gm")

    if "fcf_margin" in metrics.columns:
        metrics["q_fcfm"] = _group_rank(metrics, "fcf_margin", ascending=False)  

        quality_components.append("q_fcfm")

    if "debt_to_equity" in metrics.columns:

        metrics["q_levinv"] = _group_rank(metrics, "debt_to_equity", ascending=True)
        quality_components.append("q_levinv")

    if quality_components:
        metrics["quality_score"] = metrics[quality_components].mean(axis=1)

    if "volatility_60d" in metrics.columns:
        metrics["lowvol_score"] = _group_rank(metrics, "volatility_60d", ascending=True)

    if "momentum_60d" in metrics.columns:
        metrics["momentum_score"] = _group_rank(metrics, "momentum_60d", ascending=False)

    return metrics

def compute_sector_relative_percentiles(
    universe_metrics: pd.DataFrame,
    sp500_reference: pd.DataFrame
) -> pd.DataFrame:
    if universe_metrics.empty:
        return universe_metrics

    if sp500_reference.empty:
        print("[factors] WARNING: No S&P 500 reference provided, falling back to universe-only ranking")
        return compute_universe_percentiles(universe_metrics, group_by_sector=True)

    if "sector" not in universe_metrics.columns or "sector" not in sp500_reference.columns:
        print("[factors] WARNING: Missing sector column, falling back to universe-only ranking")
        return compute_universe_percentiles(universe_metrics, group_by_sector=False)

    result = universe_metrics.copy()

    value_metrics = ["bp_ratio", "ep_ratio", "fcfp_ratio"]
    quality_metrics = ["roe", "roa", "gross_margin", "fcf_margin", "debt_to_equity"]
    momentum_metrics = ["momentum_60d"]
    lowvol_metrics = ["volatility_60d"]

    ascending_metrics = {"debt_to_equity", "volatility_60d"}

    result["value_score"] = np.nan
    result["quality_score"] = np.nan
    result["momentum_score"] = np.nan
    result["lowvol_score"] = np.nan

    for idx, row in universe_metrics.iterrows():
        ticker = row["ticker"]
        sector = row.get("sector")

        if pd.isna(sector) or sector is None or sector == "":

            sector_peers = sp500_reference
        else:
            sector_peers = sp500_reference[sp500_reference["sector"] == sector]
            if sector_peers.empty:

                sector_peers = sp500_reference

        value_percentiles = []
        for metric in value_metrics:
            if metric not in row or pd.isna(row[metric]):
                continue
            if metric not in sector_peers.columns:
                continue

            peer_values = sector_peers[metric].dropna().values
            if len(peer_values) == 0:
                continue

            stock_value = row[metric]

            percentile = _effective_rank_percentile(stock_value, peer_values, ascending=False)
            if not np.isnan(percentile):
                value_percentiles.append(percentile)

        if value_percentiles:
            result.loc[idx, "value_score"] = np.mean(value_percentiles)

        quality_percentiles = []
        for metric in quality_metrics:
            if metric not in row or pd.isna(row[metric]):
                continue
            if metric not in sector_peers.columns:
                continue

            peer_values = sector_peers[metric].dropna().values
            if len(peer_values) == 0:
                continue

            stock_value = row[metric]

            is_ascending = metric in ascending_metrics
            percentile = _effective_rank_percentile(stock_value, peer_values, ascending=is_ascending)
            if not np.isnan(percentile):
                quality_percentiles.append(percentile)

        if quality_percentiles:
            result.loc[idx, "quality_score"] = np.mean(quality_percentiles)

        for metric in momentum_metrics:
            if metric not in row or pd.isna(row[metric]):
                continue
            if metric not in sector_peers.columns:
                continue

            peer_values = sector_peers[metric].dropna().values
            if len(peer_values) == 0:
                continue

            stock_value = row[metric]

            percentile = _effective_rank_percentile(stock_value, peer_values, ascending=False)
            result.loc[idx, "momentum_score"] = percentile

        for metric in lowvol_metrics:
            if metric not in row or pd.isna(row[metric]):
                continue
            if metric not in sector_peers.columns:
                continue

            peer_values = sector_peers[metric].dropna().values
            if len(peer_values) == 0:
                continue

            stock_value = row[metric]

            percentile = _effective_rank_percentile(stock_value, peer_values, ascending=True)
            result.loc[idx, "lowvol_score"] = percentile

    return result

def calculate_factor_metrics(
    fundamentals: dict,
    price_data: pd.DataFrame,
    sp500_reference: Optional[pd.DataFrame] = None,
    use_sector_relative: bool = True
) -> pd.DataFrame:

    metrics = calculate_raw_factor_metrics(fundamentals, price_data)

    if metrics.empty:
        return metrics

    if use_sector_relative and sp500_reference is not None and not sp500_reference.empty:

        metrics = compute_sector_relative_percentiles(metrics, sp500_reference)
    else:

        metrics = compute_universe_percentiles(metrics, group_by_sector=True)

    return metrics

class FactorPortfolioConstructor:
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

        if self.metrics.empty or metric_column not in self.metrics.columns:
            return pd.DataFrame()

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

            long_tk = valid[valid <= low].index.tolist()
            short_tk = valid[valid >= high].index.tolist()
        else:

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

def get_factor_score_columns() -> list[str]:
    return ["value_score", "quality_score", "momentum_score", "lowvol_score"]

def get_raw_metric_columns() -> dict[str, list[str]]:
    return {
        "VALUE": ["bp_ratio", "ep_ratio", "fcfp_ratio"],
        "QUALITY": ["roe", "roa", "gross_margin", "fcf_margin", "debt_to_equity"],
        "MOMENTUM": ["momentum_60d"],
        "LOW_VOL": ["volatility_60d"],
    }

def summarize_factor_scores(metrics: pd.DataFrame) -> pd.DataFrame:

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

    rename_map = {
        "value_score": "Value",
        "quality_score": "Quality",
        "momentum_score": "Momentum",
        "lowvol_score": "Low Vol",
    }
    result = result.rename(columns=rename_map)

    return result.sort_values("ticker").reset_index(drop=True)
