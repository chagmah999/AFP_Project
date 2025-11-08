import numpy as np
import pandas as pd

def calculate_factor_metrics(fundamentals: dict, price_data: pd.DataFrame) -> pd.DataFrame:
    bs = fundamentals.get("balance_sheet", pd.DataFrame())
    inc = fundamentals.get("income_statement", pd.DataFrame())
    cf  = fundamentals.get("cash_flow", pd.DataFrame())
    if bs.empty or inc.empty:
        return pd.DataFrame()

    metrics = pd.merge(
        bs[["ticker","date","totalStockholdersEquity","totalAssets","totalLiabilities","totalDebt","cashAndCashEquivalents"]],
        inc[["ticker","date","revenue","netIncome","grossProfit","operatingIncome","eps","ebitda"]],
        on=["ticker","date"],
        how="inner"
    )
    if not cf.empty:
        metrics = pd.merge(
            metrics,
            cf[["ticker","date","freeCashFlow","operatingCashFlow"]],
            on=["ticker","date"], how="left"
        )

    metrics["book_equity"] = metrics["totalStockholdersEquity"]
    metrics["earnings_yield"] = metrics["netIncome"] / metrics["totalAssets"]  # same simplified proxy as NB

    metrics["roe"] = metrics["netIncome"] / metrics["totalStockholdersEquity"]
    metrics["roa"] = metrics["netIncome"] / metrics["totalAssets"]
    metrics["gross_margin"] = metrics["grossProfit"] / metrics["revenue"]
    metrics["debt_to_equity"] = metrics["totalDebt"] / metrics["totalStockholdersEquity"]

    price_pivot = price_data.pivot_table(index="date", columns="ticker", values="adjClose")
    for w in [20, 60, 120, 250]:
        mom = price_pivot.pct_change(w)
        last = mom.iloc[-1]
        for tk in last.index:
            metrics.loc[metrics["ticker"] == tk, f"momentum_{w}d"] = last[tk]
    vol = price_data.groupby("ticker")["returns"].apply(
        lambda x: x.rolling(60, min_periods=30).std().iloc[-1] if len(x) > 30 else np.nan
    )
    for tk in vol.index:
        metrics.loc[metrics["ticker"] == tk, "volatility_60d"] = vol[tk]

    metrics = metrics.replace([np.inf, -np.inf], np.nan)
    return metrics

class FactorPortfolioConstructor:
    def __init__(self, metrics_df: pd.DataFrame, price_df: pd.DataFrame):
        self.metrics = metrics_df
        self.prices = price_df
        self.portfolios = {}

    def construct_factor_portfolio(self, factor_name: str, metric_column: str, ascending: bool, percentile: float = 0.3) -> pd.DataFrame:
        latest = self.metrics.sort_values("date").groupby("ticker").last()
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
        w_long = [1/len(long_tk)] * len(long_tk) if long_tk else []
        w_short = [-1/len(short_tk)] * len(short_tk) if short_tk else []
        return pd.DataFrame({
            "factor": factor_name,
            "ticker": long_tk + short_tk,
            "position": ["long"]*len(long_tk) + ["short"]*len(short_tk),
            "weight": w_long + w_short
        })

    def construct_all(self) -> dict[str, pd.DataFrame]:
        self.portfolios["VALUE"] = self.construct_factor_portfolio("VALUE", "earnings_yield", ascending=False)
        self.portfolios["QUALITY"] = self.construct_factor_portfolio("QUALITY", "roe", ascending=False)
        if "momentum_60d" in self.metrics.columns:
            self.portfolios["MOMENTUM"] = self.construct_factor_portfolio("MOMENTUM", "momentum_60d", ascending=False)
        if "volatility_60d" in self.metrics.columns:
            self.portfolios["LOW_VOL"] = self.construct_factor_portfolio("LOW_VOL", "volatility_60d", ascending=True)
        return self.portfolios

    def calculate_factor_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        rets = []
        for fname, port in self.portfolios.items():
            if port.empty:
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
                    tr = day[day["ticker"] == row["ticker"]]["returns"]
                    if not tr.empty:
                        wr += row["weight"] * tr.values[0]
                rets.append({"date": dt, "factor": fname, "return": wr})
        return pd.DataFrame(rets)
