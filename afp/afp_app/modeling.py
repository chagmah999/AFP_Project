import pandas as pd

def prepare_modeling_data(factor_returns: pd.DataFrame, macro: dict, price_data: pd.DataFrame) -> pd.DataFrame:
    if factor_returns is None or factor_returns.empty:
        modeling = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=1000, freq="D")})
    else:
        modeling = factor_returns.pivot_table(index="date", columns="factor", values="return").reset_index()

    if "treasury" in macro and not macro["treasury"].empty:
        keep = [c for c in ["date","month3","year2","year5","year10","term_spread_10y2y","term_spread_10y3m","rates_level","rates_1m_change"]
                if c in macro["treasury"].columns]
        modeling = modeling.merge(macro["treasury"][keep], on="date", how="left")

    if "vix" in macro and not macro["vix"].empty:
        keep = [c for c in ["date","vix_close","vix_ma20","vix_percentile"] if c in macro["vix"].columns]
        modeling = modeling.merge(macro["vix"][keep], on="date", how="left")

    if "credit" in macro and not macro["credit"].empty:
        keep = [c for c in ["date","credit_spread_level","credit_spread_1m_change","hy_spread","ig_spread"]
                if c in macro["credit"].columns]
        modeling = modeling.merge(macro["credit"][keep], on="date", how="left")

    spy = price_data[price_data["ticker"] == "SPY"] if "ticker" in price_data.columns else pd.DataFrame()
    if not spy.empty:
        modeling = modeling.merge(spy[["date","returns"]].rename(columns={"returns":"market_return"}), on="date", how="left")

    modeling = modeling.sort_values("date").ffill().bfill()
    return modeling
