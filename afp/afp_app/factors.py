import numpy as np
import pandas as pd


def _safe_div(num, den):
    """Safe division that returns NaN if denominator is zero or NaN."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


def calculate_factor_metrics(
    fundamentals: dict,
    price_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Build factor inputs and industry-neutral 0-1 scores for VALUE, QUALITY,
    MOMENTUM, and LOW_VOL.

    - Uses FMP fundamentals (balance sheet, income statement, cash flow)
    - Carries through 'sector' / 'industry' if present
    - Computes raw ratios
    - Computes 60d momentum and 60d realized volatility
    - Normalizes within sector (or industry) using percentile ranks
    """
    bs = fundamentals.get("balance_sheet", pd.DataFrame())
    inc = fundamentals.get("income_statement", pd.DataFrame())
    cf = fundamentals.get("cash_flow", pd.DataFrame())

    if bs.empty or inc.empty:
        return pd.DataFrame()

    # ---------------- Merge fundamentals ----------------
        # Columns to keep from balance sheet, including optional sector metadata if present
    bs_cols = [
        "ticker", "date",
        "totalStockholdersEquity",
        "totalAssets",
        "totalLiabilities",
        "totalDebt",
        "cashAndCashEquivalents",
        # common share count if available
        "commonStockSharesOutstanding",
    ]
    for meta_col in ["sector", "industry"]:
        if meta_col in bs.columns:
            bs_cols.append(meta_col)

    bs_use = bs[bs_cols].copy()

    # Income statement
    inc_cols = [
        "ticker", "date",
        "revenue",
        "netIncome",
        "grossProfit",
        "operatingIncome",
        "eps",
        "ebitda",
        # some FMP endpoints put share counts here instead
        "weightedAverageShsOut",
        "weightedAverageShsOutDil",
    ]
    inc_use = inc[inc_cols].copy()


    metrics = pd.merge(
        bs_use,
        inc_use,
        on=["ticker", "date"],
        how="inner",
    )

    if not cf.empty:
        cf_cols = ["ticker", "date", "freeCashFlow", "operatingCashFlow"]
        cf_use = cf[cf_cols].copy()
        metrics = pd.merge(
            metrics,
            cf_use,
            on=["ticker", "date"],
            how="left",
        )

    # ---------------- Fundamental ratios ----------------
    metrics["book_equity"] = metrics["totalStockholdersEquity"]

    # Simplified value proxy (consistent with earlier codebase)
    metrics["earnings_yield"] = _safe_div(
        metrics["netIncome"],
        metrics["totalAssets"],
    )

    metrics["roe"] = _safe_div(
        metrics["netIncome"],
        metrics["totalStockholdersEquity"],
    )
    metrics["roa"] = _safe_div(
        metrics["netIncome"],
        metrics["totalAssets"],
    )
    metrics["gross_margin"] = _safe_div(
        metrics["grossProfit"],
        metrics["revenue"],
    )
    metrics["fcf_margin"] = _safe_div(
        metrics.get("freeCashFlow", np.nan),
        metrics["revenue"],
    )
    metrics["debt_to_equity"] = _safe_div(
        metrics["totalDebt"],
        metrics["totalStockholdersEquity"],
    )

    # ---------------- Momentum and volatility from price data ----------------
    if price_data.empty:
        metrics = metrics.replace([np.inf, -np.inf], np.nan)
        return metrics

    px_pivot = price_data.pivot_table(
        index="date",
        columns="ticker",
        values="adjClose",
    )

    # 60-day momentum: simple percentage change over 60 trading days
    mom_60d = px_pivot.pct_change(60)
    if not mom_60d.empty:
        last_mom = mom_60d.iloc[-1]
        for tk in last_mom.index:
            metrics.loc[metrics["ticker"] == tk, "momentum_60d"] = last_mom[tk]

    # 60-day realized volatility from daily returns
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

    # ---------------- Cross-sectional 0-1 scores (industry-neutral) ----------------
    # Grouping level: sector if available, else industry, else whole universe
    if "sector" in metrics.columns:
        group_cols = ["sector"]
    elif "industry" in metrics.columns:
        group_cols = ["industry"]
    else:
        group_cols = []  # all names together

    def _rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
        return series.rank(method="average", pct=True, ascending=ascending)

    # Helper to apply groupwise rank or global rank if no group_cols
    def _group_rank(col_name: str, ascending: bool = True) -> pd.Series:
        if group_cols:
            return (
                metrics.groupby(group_cols)[col_name]
                .transform(lambda s: _rank_pct(s, ascending=ascending))
            )
        else:
            return _rank_pct(metrics[col_name], ascending=ascending)

    # VALUE: high earnings_yield = cheap
    if "earnings_yield" in metrics.columns:
        metrics["value_score"] = _group_rank("earnings_yield", ascending=True)

    # QUALITY: average of ROE, ROA, margins, inverse leverage
    quality_components = []

    if "roe" in metrics.columns:
        metrics["q_roe"] = _group_rank("roe", ascending=True)
        quality_components.append("q_roe")

    if "roa" in metrics.columns:
        metrics["q_roa"] = _group_rank("roa", ascending=True)
        quality_components.append("q_roa")

    if "gross_margin" in metrics.columns:
        metrics["q_gm"] = _group_rank("gross_margin", ascending=True)
        quality_components.append("q_gm")

    if "fcf_margin" in metrics.columns:
        metrics["q_fcfm"] = _group_rank("fcf_margin", ascending=True)
        quality_components.append("q_fcfm")

    if "debt_to_equity" in metrics.columns:
        # Lower leverage is better, so 1 - percentile of debt_to_equity
        if group_cols:
            lev_rank = (
                metrics.groupby(group_cols)["debt_to_equity"]
                .transform(lambda s: _rank_pct(s, ascending=True))
            )
        else:
            lev_rank = _rank_pct(metrics["debt_to_equity"], ascending=True)
        metrics["q_levinv"] = 1.0 - lev_rank
        quality_components.append("q_levinv")

    if quality_components:
        metrics["quality_score"] = metrics[quality_components].mean(axis=1)

    # LOW VOL: low volatility should have high score
    if "volatility_60d" in metrics.columns:
        if group_cols:
            vol_rank = (
                metrics.groupby(group_cols)["volatility_60d"]
                .transform(lambda s: _rank_pct(s, ascending=True))
            )
        else:
            vol_rank = _rank_pct(metrics["volatility_60d"], ascending=True)
        metrics["lowvol_score"] = 1.0 - vol_rank

    # MOMENTUM: percentile of 60d momentum within group
    if "momentum_60d" in metrics.columns:
        metrics["momentum_score"] = _group_rank("momentum_60d", ascending=True)

    metrics = metrics.replace([np.inf, -np.inf], np.nan)
    return metrics


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
        Build a single factor portfolio from a given score column.

        - metric_column is expected to be a 0-1 score where higher is "more of" the factor.
        - We take top (1 - percentile) as the long leg and bottom percentile as the short leg.
        - ascending flag keeps the old API but is interpreted as:
            ascending = False: long = high scores, short = low scores (typical)
            ascending = True:  long = low scores,  short = high scores
        """
        if self.metrics.empty or metric_column not in self.metrics.columns:
            return pd.DataFrame()

        # Use the latest metrics for each ticker
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
            # For a metric where "low" means more of the factor
            long_tk = valid[valid <= low].index.tolist()
            short_tk = valid[valid >= high].index.tolist()
        else:
            # For a metric where "high" means more of the factor
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
        Construct portfolios for all four factors using the 0-1 scores.
        """
        ports: dict[str, pd.DataFrame] = {}

        # Value: high value_score
        if "value_score" in self.metrics.columns:
            ports["VALUE"] = self.construct_factor_portfolio(
                "VALUE", "value_score", ascending=False
            )
        else:
            ports["VALUE"] = pd.DataFrame()

        # Quality: high quality_score
        if "quality_score" in self.metrics.columns:
            ports["QUALITY"] = self.construct_factor_portfolio(
                "QUALITY", "quality_score", ascending=False
            )
        else:
            ports["QUALITY"] = pd.DataFrame()

        # Momentum: high momentum_score
        if "momentum_score" in self.metrics.columns:
            ports["MOMENTUM"] = self.construct_factor_portfolio(
                "MOMENTUM", "momentum_score", ascending=False
            )
        else:
            ports["MOMENTUM"] = pd.DataFrame()

        # Low volatility: high lowvol_score means lower risk
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
        Compute daily long-short factor returns from the constructed portfolios.
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
