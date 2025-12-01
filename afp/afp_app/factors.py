import numpy as np
import pandas as pd

def _safe_div(num, den):
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
        metrics["netIncome"],
        metrics["market_cap"],
    )
    metrics["fcfp_ratio"] = _safe_div(
        metrics.get("freeCashFlow", np.nan),
        metrics["market_cap"],
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

    if price_data.empty:
        metrics = metrics.replace([np.inf, -np.inf], np.nan)
        return metrics

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

    if "sector" in metrics.columns:
        group_cols = ["sector"]
    elif "industry" in metrics.columns:
        group_cols = ["industry"]
    else:
        group_cols = []  

    def _rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
        return series.rank(method="average", pct=True, ascending=ascending)

    def _group_rank(col_name: str, ascending: bool = True) -> pd.Series:
        if group_cols:
            return (
                metrics.groupby(group_cols)[col_name]
                .transform(lambda s: _rank_pct(s, ascending=ascending))
            )
        else:
            return _rank_pct(metrics[col_name], ascending=ascending)

    def _zscore_grouped(series: pd.Series) -> pd.Series:
        mu = series.mean()
        sigma = series.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(index=series.index, data=np.nan)
        return (series - mu) / sigma

    value_components = []

    if "bp_ratio" in metrics.columns:
        metrics["z_bp"] = (
            metrics.groupby(group_cols)["bp_ratio"]
            .transform(_zscore_grouped)
        )
        value_components.append("z_bp")

    if "ep_ratio" in metrics.columns:
        metrics["z_ep"] = (
            metrics.groupby(group_cols)["ep_ratio"]
            .transform(_zscore_grouped)
        )
        value_components.append("z_ep")

    if "fcfp_ratio" in metrics.columns:
        metrics["z_fcfp"] = (
            metrics.groupby(group_cols)["fcfp_ratio"]
            .transform(_zscore_grouped)
        )
        value_components.append("z_fcfp")

    if value_components:
        metrics["value_raw"] = metrics[value_components].mean(axis=1, skipna=True)

        if group_cols:
            metrics["value_score"] = (
                metrics.groupby(group_cols)["value_raw"]
                .transform(lambda s: s.rank(method="average", pct=True, ascending=False))
            )
        else:
            metrics["value_score"] = _rank_pct(
                metrics["value_raw"], ascending=False
            )

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

    if "volatility_60d" in metrics.columns:
        if group_cols:
            vol_rank = (
                metrics.groupby(group_cols)["volatility_60d"]
                .transform(lambda s: _rank_pct(s, ascending=True))
            )
        else:
            vol_rank = _rank_pct(metrics["volatility_60d"], ascending=True)
        metrics["lowvol_score"] = 1.0 - vol_rank

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
