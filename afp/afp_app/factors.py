import numpy as np
import pandas as pd


def _pick_industry_column(df: pd.DataFrame) -> str | None:
    """
    Try to find an industry/sector-type column in a DataFrame.
    Returns the column name if found, else None.
    """
    candidates = [
        "industry",
        "sector",
        "gicsSector",
        "gics_sector",
        "gicsIndustry",
        "gics_industry",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def calculate_factor_metrics(fundamentals: dict, price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Build cross-sectional factor inputs (Value, Quality, Momentum, Low Vol)
    with industry-normalized scores.

    - Value: composite of cheapness proxies, ranked within industry
    - Quality: composite of profitability + margins + low leverage, ranked within industry
    - Momentum: 60d price momentum (kept simple, can later extend)
    - Low Vol: 60d realized volatility, inverted and ranked within industry

    Returns one row per (ticker, accounting date) with factor-relevant columns.
    """
    bs = fundamentals.get("balance_sheet", pd.DataFrame())
    inc = fundamentals.get("income_statement", pd.DataFrame())
    cf = fundamentals.get("cash_flow", pd.DataFrame())

    if bs.empty or inc.empty:
        return pd.DataFrame()

    # ----- merge fundamentals -----
    metrics = pd.merge(
        bs[
            [
                "ticker",
                "date",
                "totalStockholdersEquity",
                "totalAssets",
                "totalLiabilities",
                "totalDebt",
                "cashAndCashEquivalents",
            ]
        ],
        inc[
            [
                "ticker",
                "date",
                "revenue",
                "netIncome",
                "grossProfit",
                "operatingIncome",
                "eps",
                "ebitda",
            ]
        ],
        on=["ticker", "date"],
        how="inner",
    )

    if not cf.empty:
        metrics = pd.merge(
            metrics,
            cf[["ticker", "date", "freeCashFlow", "operatingCashFlow"]],
            on=["ticker", "date"],
            how="left",
        )

    # ----- attach industry / sector if available -----
    # We assume any industry/sector-like column lives in price_data, then map by ticker.
    industry_col = None
    if not price_data.empty:
        ind_col = _pick_industry_column(price_data)
        if ind_col is not None:
            ticker_ind = (
                price_data[["ticker", ind_col]]
                .dropna()
                .drop_duplicates(subset="ticker")
            )
            metrics = metrics.merge(ticker_ind, on="ticker", how="left")
            industry_col = ind_col

    # If still no industry information, fall back to a single dummy group.
    if industry_col is None:
        industry_col = "industry_group"
        metrics[industry_col] = "ALL"

    # ----- basic accounting ratios -----
    metrics["book_equity"] = metrics["totalStockholdersEquity"]

    # Value-style cheapness proxies (we do not have true market cap here, so use accounting proxies)
    # You can later swap these for book/price, earnings/price, FCF/price once market cap is available.
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics["earnings_yield"] = metrics["netIncome"] / metrics["totalAssets"]
        metrics["bp_proxy"] = metrics["book_equity"] / metrics["totalAssets"]
        metrics["fcf_proxy"] = metrics["freeCashFlow"] / metrics["totalAssets"]

    # Quality-style ratios
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics["roe"] = metrics["netIncome"] / metrics["totalStockholdersEquity"]
        metrics["roa"] = metrics["netIncome"] / metrics["totalAssets"]
        metrics["gross_margin"] = metrics["grossProfit"] / metrics["revenue"]
        metrics["fcf_margin"] = metrics["freeCashFlow"] / metrics["revenue"]
        metrics["debt_to_equity"] = metrics["totalDebt"] / metrics["totalStockholdersEquity"]

    # ----- price-based metrics for momentum and volatility -----
    # Use adjusted close for momentum, and returns for volatility
    if "adjClose" in price_data.columns:
        price_pivot = price_data.pivot_table(index="date", columns="ticker", values="adjClose")
    else:
        price_pivot = price_data.pivot_table(index="date", columns="ticker", values="close")

    # 60-day momentum: simple % change over last 60 trading days
    mom_60 = price_pivot.pct_change(60).iloc[-1] if not price_pivot.empty else pd.Series(dtype=float)
    for tk in mom_60.index:
        metrics.loc[metrics["ticker"] == tk, "momentum_60d"] = mom_60[tk]

    # 60-day volatility: rolling std of daily returns
    if "returns" in price_data.columns:
        vol_60 = price_data.groupby("ticker")["returns"].apply(
            lambda x: x.rolling(60, min_periods=30).std().iloc[-1] if len(x) > 30 else np.nan
        )
        for tk in vol_60.index:
            metrics.loc[metrics["ticker"] == tk, "volatility_60d"] = vol_60[tk]
    else:
        metrics["volatility_60d"] = np.nan

    # ----- industry-normalized factor scores -----
    # Helper: within-industry percentile rank in [0,1]
    def pct_rank_by_industry(series: pd.Series, ind: pd.Series, ascending: bool = True) -> pd.Series:
        df = pd.DataFrame({"val": series, "ind": ind})
        return df.groupby("ind")["val"].transform(
            lambda x: x.rank(pct=True, ascending=ascending)
        )

    # 1) VALUE: composite of bp_proxy, earnings_yield, fcf_proxy
    value_components = ["bp_proxy", "earnings_yield", "fcf_proxy"]
    for col in value_components:
        if col not in metrics.columns:
            metrics[col] = np.nan

    for col in value_components:
        metrics[f"value_{col}_pct"] = pct_rank_by_industry(
            metrics[col], metrics[industry_col], ascending=True
        )
    metrics["value_score"] = metrics[[f"value_{c}_pct" for c in value_components]].mean(axis=1)

    # 2) QUALITY: composite of roe, roa, gross_margin, fcf_margin, inverted leverage
    quality_components = ["roe", "roa", "gross_margin", "fcf_margin", "inv_leverage"]
    metrics["inv_leverage"] = -metrics["debt_to_equity"]  # lower leverage = better, so invert

    for col in quality_components:
        if col not in metrics.columns:
            metrics[col] = np.nan

    for col in quality_components:
        metrics[f"quality_{col}_pct"] = pct_rank_by_industry(
            metrics[col], metrics[industry_col], ascending=True
        )
    metrics["quality_score"] = metrics[[f"quality_{c}_pct" for c in quality_components]].mean(axis=1)

    # 3) LOW VOL: percentile of inverted volatility (high score = low vol)
    # If vol is missing, leave score as NaN for that ticker.
    if "volatility_60d" not in metrics.columns:
        metrics["volatility_60d"] = np.nan

    # Within industry, lower vol should get higher score, so ascending=False on vol, or ascending=True on -vol
    metrics["lowvol_score"] = pct_rank_by_industry(
        -metrics["volatility_60d"], metrics[industry_col], ascending=True
    )

    # Clean infinities
    metrics = metrics.replace([np.inf, -np.inf], np.nan)

    return metrics


class FactorPortfolioConstructor:
    """
    Build long-short factor portfolios from factor scores.
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
        Build a simple long-short portfolio:
          - Use the latest metrics per ticker
          - Rank tickers by `metric_column`
          - Long top percentile, short bottom percentile

        `ascending=False` means high metric -> high rank -> long.
        """
        if self.metrics.empty or metric_column not in self.metrics.columns:
            return pd.DataFrame()

        latest = self.metrics.sort_values("date").groupby("ticker").last()
        valid = latest[metric_column].dropna()
        if len(valid) < 3:
            return pd.DataFrame()

        low = valid.quantile(percentile)
        high = valid.quantile(1 - percentile)

        if ascending:
            # Lower metric = "better" (e.g., raw volatility)
            long_tk = valid[valid <= low].index.tolist()
            short_tk = valid[valid >= high].index.tolist()
        else:
            # Higher metric = "better" (e.g., value_score, quality_score, momentum)
            long_tk = valid[valid >= high].index.tolist()
            short_tk = valid[valid <= low].index.tolist()

        w_long = [1 / len(long_tk)] * len(long_tk) if long_tk else []
        w_short = [-1 / len(short_tk)] * len(short_tk) if short_tk else []

        return pd.DataFrame(
            {
                "factor": factor_name,
                "ticker": long_tk + short_tk,
                "position": ["long"] * len(long_tk) + ["short"] * len(short_tk),
                "weight": w_long + w_short,
            }
        )

    def construct_all(self) -> dict[str, pd.DataFrame]:
        """
        Construct portfolios for all four factors using the new, industry-normalized scores.
        """
        # VALUE: high value_score
        self.portfolios["VALUE"] = self.construct_factor_portfolio(
            "VALUE", "value_score", ascending=False
        )

        # QUALITY: high quality_score
        self.portfolios["QUALITY"] = self.construct_factor_portfolio(
            "QUALITY", "quality_score", ascending=False
        )

        # MOMENTUM: high 60d momentum (still cross-sectional; can later normalize by industry)
        if "momentum_60d" in self.metrics.columns:
            self.portfolios["MOMENTUM"] = self.construct_factor_portfolio(
                "MOMENTUM", "momentum_60d", ascending=False
            )

        # LOW_VOL: high lowvol_score (which already inverts volatility)
        if "lowvol_score" in self.metrics.columns:
            self.portfolios["LOW_VOL"] = self.construct_factor_portfolio(
                "LOW_VOL", "lowvol_score", ascending=False
            )

        return self.portfolios

    def calculate_factor_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Compute daily long-short factor returns for each factor:
          fp_{f,t} = sum_i w_i * r_{i,t}
        where weights are +1/N for long names and -1/N for short names.
        """
        rets = []
        if self.prices.empty:
            return pd.DataFrame()

        for fname, port in self.portfolios.items():
            if port is None or port.empty:
                continue

            px = self.prices[
                (self.prices["ticker"].isin(port["ticker"]))
                & (self.prices["date"] >= start_date)
                & (self.prices["date"] <= end_date)
            ]
            if px.empty:
                continue

            for dt, day in px.groupby("date"):
                wr = 0.0
                for _, row in port.iterrows():
                    tr = day[day["ticker"] == row["ticker"]]["returns"]
                    if not tr.empty:
                        wr += row["weight"] * tr.values[0]
                rets.append({"date": dt, "factor": fname, "return": wr})

        return pd.DataFrame(rets)
