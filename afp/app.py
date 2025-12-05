import os
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime


from afp_app.config import (
    FMP_API_KEY,
    DEFAULT_START_DATE,
    DEFAULT_UNIVERSE_SIZE,
    LOOKBACK_DAYS,
)

from afp_app.universe import get_universe
from afp_app.fmp import FMPDataFetcher
from afp_app.data import collect_fundamental_data, collect_price_data
from afp_app.factors import calculate_factor_metrics, FactorPortfolioConstructor
from afp_app.macro import MacroDataFetcher
from afp_app.modeling import prepare_modeling_data
from afp_app.signal_factor_premia import FactorPremiaForecaster
from afp_app.signal_alpha import AlphaPredictor
from afp_app.engine import MarketMancerEngine
from afp_app.optimizer import UnifiedPortfolioOptimizer
def get_sp500_tickers_from_fmp(api_key: str) -> list[str]:
    """
    Fetch the full current S&P 500 constituent list directly from FMP.

    Returns a list of ticker symbols.
    """
    url = f"https://financialmodelingprep.com/stable/sp500_constituent?apikey={api_key}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    if "symbol" not in df.columns:
        raise ValueError("sp500_constituent payload does not contain a 'symbol' column")

    tickers = (
        df["symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    return sorted(tickers)


def load_or_build_sp500_metrics(
    fetcher,
    api_key: str,
    start_date: str,
    cache_dir: str = "tempdata",
) -> pd.DataFrame:
    """
    Load full S&P 500 factor metrics from a daily cache if available.
    Otherwise, fetch S&P 500 constituents, collect fundamentals and prices
    for the full set, compute factor metrics, and cache them to disk.

    The cache key is today's date (UTC) so runs within the same day reuse
    the same metrics_full.
    """
    today_str = datetime.utcnow().date().isoformat()
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"sp500_factor_metrics_{today_str}.parquet")

    # Try to load from cache
    if os.path.exists(cache_path):
        try:
            metrics_full = pd.read_parquet(cache_path)
            return metrics_full
        except Exception:
            # If reading fails, fall through and rebuild
            pass

    # If cache not present or unreadable, rebuild it
    sp500_tickers = get_sp500_tickers_from_fmp(api_key)

    # Full S&P 500 fundamentals and prices
    fundamentals_full = collect_fundamental_data(sp500_tickers, start_date, fetcher)
    prices_full = collect_price_data(sp500_tickers, start_date, None, fetcher)

    from afp_app.factors import calculate_factor_metrics  # local import to avoid cycles

    metrics_full = calculate_factor_metrics(fundamentals_full, prices_full)

    # Persist to disk for the rest of the day
    try:
        metrics_full.to_parquet(cache_path)
    except Exception:
        # If writing fails, just continue without cache for this run
        pass

    return metrics_full


def compute_factor_performance(factor_returns_hist: pd.DataFrame):
    """
    Compute performance statistics and cumulative return paths
    for each factor.

    The input can be either:
      1) Long format with columns ['date','factor','return', ...], or
      2) Wide format with a date index (or 'date' column) and one
         numeric column per factor (plus optional 'rf_daily').

    Expected columns in long format:
      - 'date': calendar date of the factor return
      - 'factor': factor name (e.g. 'VALUE', 'QUALITY', ...)
      - 'return': daily factor return in decimals
      - optional 'rf_daily': daily risk free rate in decimals
        (if present, it can either be repeated per factor row or
         stored in a separate row; it will be aligned by date)

    Returns
    -------
    perf_summary : DataFrame
        One row per factor with columns:
        ['factor', 'total_return', 'ann_return', 'ann_vol',
         'sharpe', 'max_drawdown']
    cum_paths : DataFrame
        Cumulative simple return paths (decimal) for each factor,
        indexed by date, one column per factor.
    """
    raw = factor_returns_hist.copy()
    if raw.empty:
        empty_perf = pd.DataFrame(
            columns=[
                "factor",
                "total_return",
                "ann_return",
                "ann_vol",
                "sharpe",
                "max_drawdown",
            ]
        )
        empty_cum = pd.DataFrame()
        return empty_perf, empty_cum

    # ------------------------------------------------------------------
    # Case 1: long format ['date','factor','return', ...]
    # ------------------------------------------------------------------
    if {"date", "factor", "return"}.issubset(raw.columns):
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[~df["date"].isna()]

        # Pull out risk free if present
        if "rf_daily" in df.columns:
            # Take one rf value per date (mean across rows if repeated)
            rf_daily = (
                df[["date", "rf_daily"]]
                .dropna()
                .groupby("date")["rf_daily"]
                .mean()
                .sort_index()
            )
        else:
            rf_daily = None

        # Pivot factor returns to wide: one column per factor
        wide = (
            df.pivot_table(
                index="date",
                columns="factor",
                values="return",
                aggfunc="mean",
            )
            .sort_index()
        )

        # Merge risk free back in as a separate column if available
        if rf_daily is not None:
            wide["rf_daily"] = rf_daily.reindex(wide.index).ffill()

        wide.index = pd.DatetimeIndex(wide.index)
        wide = wide[~wide.index.isna()]

    # ------------------------------------------------------------------
    # Case 2: already wide format
    # ------------------------------------------------------------------
    else:
        df = raw.copy()

        def _ensure_datetime_index(d: pd.DataFrame) -> pd.DataFrame:
            d = d.copy()
            if "date" in d.columns:
                d["date"] = pd.to_datetime(d["date"], errors="coerce")
                d = d[~d["date"].isna()]
                d = d.sort_values("date").set_index("date")
            else:
                if not isinstance(d.index, pd.DatetimeIndex):
                    d.index = pd.to_datetime(d.index, errors="coerce")
                d = d.sort_index()
                d = d[~d.index.isna()]
            if not d.index.is_unique:
                d = d.groupby(d.index).last()
            return d

        wide = _ensure_datetime_index(df)

    if wide.empty:
        empty_perf = pd.DataFrame(
            columns=[
                "factor",
                "total_return",
                "ann_return",
                "ann_vol",
                "sharpe",
                "max_drawdown",
            ]
        )
        empty_cum = pd.DataFrame()
        return empty_perf, empty_cum

    # Identify risk free and factor columns
    if "rf_daily" in wide.columns:
        rf_daily = wide["rf_daily"].astype(float)
        candidate_cols = [c for c in wide.columns if c != "rf_daily"]
    else:
        rf_daily = None
        candidate_cols = list(wide.columns)

    factor_cols = []
    for col in candidate_cols:
        if col.lower() in ["rf", "rf_daily", "r_3m", "r_1m"]:
            continue
        if pd.api.types.is_numeric_dtype(wide[col]):
            factor_cols.append(col)

    if not factor_cols:
        empty_perf = pd.DataFrame(
            columns=[
                "factor",
                "total_return",
                "ann_return",
                "ann_vol",
                "sharpe",
                "max_drawdown",
            ]
        )
        empty_cum = pd.DataFrame()
        return empty_perf, empty_cum

    ann_factor = 252.0
    perf_rows = []
    cum_paths = pd.DataFrame(index=wide.index)

    for fac in factor_cols:
        r = wide[fac].dropna()
        if r.empty:
            continue

        # Align rf_daily to this factor's dates if present
        if isinstance(rf_daily, pd.Series):
            rf_used = rf_daily.reindex(r.index).ffill().fillna(0.0)
        else:
            rf_used = 0.0

        n_days = len(r)

        # Cumulative simple return path
        cum = (1.0 + r).cumprod() - 1.0
        cum_paths[fac] = cum

        # Total and annualized return
        total_return = (1.0 + r).prod() - 1.0
        if n_days > 0:
            ann_return = (1.0 + total_return) ** (ann_factor / n_days) - 1.0
        else:
            ann_return = np.nan

        # Annualized volatility
        daily_vol = r.std(ddof=1)
        ann_vol = daily_vol * np.sqrt(ann_factor)

        # Sharpe ratio (excess return vs rf_daily if available)
        if isinstance(rf_used, pd.Series):
            excess = r - rf_used
            excess_mean = excess.mean()
            excess_std = r.std(ddof=1)
        else:
            excess_mean = r.mean()
            excess_std = r.std(ddof=1)

        if excess_std > 0:
            sharpe = excess_mean / excess_std * np.sqrt(ann_factor)
        else:
            sharpe = np.nan

        # Max drawdown on gross return path
        gross_path = (1.0 + r).cumprod()
        running_max = gross_path.cummax()
        drawdown = gross_path / running_max - 1.0
        max_dd = drawdown.min()

        perf_rows.append(
            {
                "factor": fac,
                "total_return": total_return,
                "ann_return": ann_return,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
            }
        )

    perf_summary = pd.DataFrame(perf_rows).sort_values("factor").reset_index(drop=True)
    return perf_summary, cum_paths


st.set_page_config(page_title="AFP Forecasting Tool", layout="wide")

st.title("AFP Forecasting Tool")
st.caption("Factor premia forecasts, stock-level alpha, and unified portfolio optimization")

for key, default in [
    ("base_forecasts", None),
    ("base_factor_eval", None),
    ("base_alpha", None),
    ("modeling_frame", None),
    ("forecaster_obj", None),
    ("universe_tickers", None),
    ("factor_portfolio_sizes", None),
    ("sample_factor_scores", None),
    ("optimized_portfolio", None),
    ("portfolio_horizon", None),
    ("factor_returns", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.subheader("Configuration")

    api_key = st.text_input(
        "FMP API Key",
        value=FMP_API_KEY or "",
        type="password",
        help="Your FMP API key",
    )

    start_date = st.text_input(
        "Start date (YYYY-MM-DD)",
        value=DEFAULT_START_DATE,
        help="Earliest date for analysis",
    )

    st.markdown("### Universe")
    universe_size = st.slider(
        "Universe size",
        min_value=10,
        max_value=509,
        value=DEFAULT_UNIVERSE_SIZE,
        step=5,
    )
    randomize = st.checkbox(
        "Randomize universe selection",
        value=True,
    )
    seed = st.number_input(
        "Random seed",
        value=42,
        step=1,
    )

    st.markdown("### Forecasting")
    forecast_horizon = st.slider(
        "Forecast horizon (days)",
        min_value=5,
        max_value=63,
        value=21,
        step=1,
        help="Length of the forward window for factor premia and alpha",
    )

    top_k_drivers = st.radio(
        "Number of top drivers to show",
        options=[3, 5],
        index=0,
    )

    run_btn = st.button("Run pipeline")

status = st.empty()

if run_btn:
    t0 = time.time()

    if not api_key:
        st.error("Please enter a valid FMP API key.")
        st.stop()

    st.session_state["portfolio_horizon"] = int(forecast_horizon)

    status.info("Selecting universe...")
    tickers = get_universe(
        universe_size,
        randomize=randomize,
        seed=int(seed),
    )

    st.session_state["universe_tickers"] = tickers

    status.info("Fetching fundamentals and prices for the selected universe...")

    fetcher = FMPDataFetcher(api_key=api_key)

    # Fundamentals and prices only for the chosen universe U
    fundamentals_uni = collect_fundamental_data(tickers, start_date, fetcher)
    prices_uni = collect_price_data(tickers, start_date, None, fetcher)

    if prices_uni is None or not isinstance(prices_uni, pd.DataFrame) or prices_uni.empty:
        st.error("No price data returned for the selected universe. Check API key, tickers, or date range.")
        st.stop()

    st.success(
        f"Collected {len(prices_uni)} price rows for the selected universe. "
        f"Date range: {prices_uni['date'].min()} to {prices_uni['date'].max()}"
    )
    status.info("Computing factor scores and factor returns (using S&P 500 baseline)...")

    # Full S&P 500 factor metrics (computed once per day and cached)
    metrics_full = load_or_build_sp500_metrics(
        fetcher=fetcher,
        api_key=api_key,
        start_date=start_date,
    )

    # Restrict these metrics to the selected universe for portfolios, etc.
    if metrics_full is None or not isinstance(metrics_full, pd.DataFrame) or metrics_full.empty:
        metrics = pd.DataFrame()
    else:
        metrics = metrics_full[metrics_full["ticker"].isin(tickers)].copy()

    factor_returns = pd.DataFrame()

    if metrics.empty:
        st.warning("No factor metrics available for the selected universe. Check fundamentals coverage.")
        st.session_state["factor_portfolio_sizes"] = None
        st.session_state["sample_factor_scores"] = None
    else:
        # Factor portfolios and returns are built on the selected universe U
        ctor = FactorPortfolioConstructor(metrics, prices_uni)
        portfolios = ctor.construct_all()

        port_sizes = {
            k: (0 if v is None or v.empty else len(v))
            for k, v in portfolios.items()
        }

        factor_returns = ctor.calculate_factor_returns(
            start_date,
            prices_uni["date"].max().strftime("%Y-%m-%d"),
        )

        # === Display table: scores for U, computed vs full S&P 500 peers ===
        latest_full = (
            metrics_full.sort_values("date")
            .groupby("ticker")
            .last()
            .reset_index()
        )

        latest_universe = latest_full[latest_full["ticker"].isin(tickers)].copy()

        score_cols = [
            c
            for c in [
                "value_score",
                "quality_score",
                "momentum_score",
                "lowvol_score",
            ]
            if c in latest_universe.columns
        ]

        sample = None
        if score_cols:
            display_cols = ["ticker"] + score_cols

            if "sector" in latest_full.columns:
                sector_counts_full = latest_full.groupby("sector")["ticker"].nunique()
                latest_universe["sector"] = latest_universe["sector"]
                latest_universe["sector_peer_count"] = latest_universe["sector"].map(sector_counts_full)
                display_cols += ["sector", "sector_peer_count"]

            if "industry" in latest_universe.columns:
                display_cols.append("industry")

            sample = latest_universe[display_cols].sort_values("ticker")

        st.session_state["factor_portfolio_sizes"] = port_sizes
        st.session_state["sample_factor_scores"] = sample

    # Store factor return history for backtesting display
    if isinstance(factor_returns, pd.DataFrame) and not factor_returns.empty:
        st.session_state["factor_returns"] = factor_returns.copy()
    else:
        st.session_state["factor_returns"] = None



        
    status.info("Fetching macro data...")
    m = MacroDataFetcher(api_key=api_key)
    macro = {
        "treasury": m.fetch_treasury_rates(from_date=start_date),
        "vix": m.fetch_vix(from_date=start_date),
        "credit": m.fetch_credit_spreads(from_date=start_date),
    }

    status.info("Preparing modeling frame...")
    modeling = prepare_modeling_data(factor_returns, macro, prices)
    st.session_state["modeling_frame"] = modeling

    status.info("Forecasting factor premia...")
    forecaster = FactorPremiaForecaster(
        lookback_window=LOOKBACK_DAYS,
        forecast_horizon=forecast_horizon,
    )
    st.session_state["forecaster_obj"] = forecaster

    factors = ["VALUE", "QUALITY", "MOMENTUM", "LOW_VOL"]
    forecasts: dict[str, dict] = {}
    factor_eval: dict[str, dict] = {}

    for f in factors:
        val = forecaster.walk_forward_validation(modeling, f)
        if val is not None:
            factor_eval[f] = forecaster.validation_summary.get(f, {})

        fc = forecaster.forecast_next(modeling, f)
        if fc:
            fc["top_drivers"] = (fc.get("top_drivers") or [])[:top_k_drivers]
            forecasts[f] = fc

    st.session_state["base_forecasts"] = forecasts
    st.session_state["base_factor_eval"] = factor_eval

    status.info("Predicting per-ticker alpha...")
    alpha_model = AlphaPredictor(
        factor_returns,
        fundamentals,
        prices,
        horizon=forecast_horizon,
        lookback=252 * 2,
    )

    alpha_preds: dict[str, dict] = {}

    for tk in tickers:
        res = alpha_model.predict_alpha(tk, horizon=forecast_horizon)
        if res:
            if "drivers" in res:
                top = res["drivers"].get("top_features", [])
                res["drivers"]["top_features"] = top[:top_k_drivers]
            alpha_preds[tk] = res

    st.session_state["base_alpha"] = alpha_preds

    status.info("Constructing optimized unified portfolio...")

    try:
        opt_tickers = [tk for tk in tickers if tk in alpha_preds]

        if opt_tickers:
            optimizer = UnifiedPortfolioOptimizer(
                risk_aversion=10.0,
                max_gross=1.5,
                max_weight=0.10,
            )

            mu = optimizer.build_expected_returns(
                alpha_preds=alpha_preds,
                tickers=opt_tickers,
            )

            Sigma, valid_tickers = optimizer.build_covariance(
                price_data=prices,
                tickers=opt_tickers,
                lookback_days=252,
            )

            if Sigma is None or Sigma.empty or len(valid_tickers) < 2:
                st.session_state["optimized_portfolio"] = None
            else:
                common = [tk for tk in valid_tickers if tk in mu.index]
                if len(common) < 2:
                    st.session_state["optimized_portfolio"] = None
                else:
                    mu_use = mu.loc[common]
                    Sigma_use = Sigma.loc[common, common]

                    weights = optimizer.optimize(mu=mu_use, Sigma=Sigma_use)

                    port_table = optimizer.build_portfolio_table(
                        weights=weights,
                        alpha_preds=alpha_preds,
                    )
                    st.session_state["optimized_portfolio"] = port_table
        else:
            st.session_state["optimized_portfolio"] = None

    except Exception as e:
        st.session_state["optimized_portfolio"] = None
        st.warning(f"Error constructing optimized portfolio: {e}")

    t1 = time.time()
    st.success(f"Pipeline completed in {t1 - t0:.1f} seconds.")

forecasts = st.session_state.get("base_forecasts")
alpha_preds = st.session_state.get("base_alpha")
factor_eval = st.session_state.get("base_factor_eval")

if not forecasts and not alpha_preds:
    st.info("Run the pipeline from the sidebar to generate forecasts.")
else:
    # =========================================================
    # 0. Historical factor performance (backtest)
    # =========================================================
    factor_returns_hist = st.session_state.get("factor_returns")

    if isinstance(factor_returns_hist, pd.DataFrame) and not factor_returns_hist.empty:
        perf_summary, cum_paths = compute_factor_performance(factor_returns_hist)

        if not perf_summary.empty:
            st.subheader("Historical factor performance")

            st.caption(
                "This section summarizes how each long short factor portfolio "
                "has performed over the full sample used in the model. "
                "For each factor, we show total and annualized returns, "
                "annualized volatility, a Sharpe ratio computed relative to the "
                "available daily risk free series when present (otherwise zero), "
                "and the worst peak to trough drawdown over the period."
            )

            perf_display = perf_summary.copy()
            perf_display["Total return %"] = perf_display["total_return"] * 100.0
            perf_display["Annualized return %"] = perf_summary["ann_return"] * 100.0
            perf_display["Annualized volatility %"] = (
                perf_summary["ann_vol"] * 100.0
            )
            perf_display["Sharpe (rf adj)"] = perf_summary["sharpe"]
            perf_display["Max drawdown %"] = perf_summary["max_drawdown"] * 100.0

            st.dataframe(
                perf_display[
                    [
                        "factor",
                        "Total return %",
                        "Annualized return %",
                        "Annualized volatility %",
                        "Sharpe (rf adj)",
                        "Max drawdown %",
                    ]
                ]
                .rename(columns={"factor": "Factor"})
                .style.format(
                    {
                        "Total return %": "{:.2f}",
                        "Annualized return %": "{:.2f}",
                        "Annualized volatility %": "{:.2f}",
                        "Sharpe (rf adj)": "{:.2f}",
                        "Max drawdown %": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )

            if not cum_paths.empty:
                st.caption(
                    "Cumulative growth of one unit invested in each factor "
                    "long short portfolio over time."
                )
                st.line_chart(cum_paths)
    else:
        st.info(
            "No historical factor returns are available yet. "
            "Run the pipeline to compute factor portfolios and their return history."
        )

    # =========================================================
    # 1. Factor premia forecasts (first, core view)
    # =========================================================

    st.subheader("Factor premia forecasts")

    if forecasts:
        st.caption(
            "These forecasts estimate each factor’s expected return (in percent) over the next *H* days, "
            "where *H* is the forecast horizon set by the user. The primary forecast uses a simple AR(1) "
            "model on each factor’s own history, while macro features are used for diagnostics and driver "
            "analysis. Higher values indicate a stronger expected tailwind for that factor (candidates to "
            "overweight), while negative values indicate expected headwinds (candidates to underweight)."
        )

        summary_rows = []
        drivers_rows = []

        for f, v in forecasts.items():
            ar1_fc = v.get("ar1_forecast", v.get("ensemble_forecast", 0.0))

            summary_rows.append(
                {
                    "Factor": f,
                    "Expected Premium % (AR(1))": ar1_fc * 100.0,
                }
            )

            for d in (v.get("top_drivers") or []):
                drivers_rows.append(
                    {
                        "Factor": f,
                        "Driver": d.get("feature"),
                        "RF Importance": d.get("rf_importance"),
                    }
                )

        df_summary = pd.DataFrame(summary_rows).sort_values(
            "Expected Premium % (AR(1))", ascending=False
        )
        st.dataframe(
            df_summary.style.format(
                {"Expected Premium % (AR(1))": "{:.2f}"}
            ),
            use_container_width=True,
        )

        with st.expander(
            "Show machine learning ensemble factor forecasts (Ridge, Lasso, Random Forest)"
        ):
            ensemble_rows = []
            for f, v in forecasts.items():
                ensemble_rows.append(
                    {
                        "Factor": f,
                        "Ensemble Premium %": v.get("ensemble_forecast", np.nan)
                        * 100.0,
                        "AR(1) Premium %": v.get("ar1_forecast", np.nan) * 100.0,
                    }
                )

            df_ensemble = pd.DataFrame(ensemble_rows).sort_values(
                "Ensemble Premium %", ascending=False
            )
            st.dataframe(
                df_ensemble.style.format(
                    {
                        "Ensemble Premium %": "{:.2f}",
                        "AR(1) Premium %": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )

        if drivers_rows:
            st.subheader("Top drivers per factor")
            st.caption(
                "The table below shows which macro features the model found most predictive "
                "when forecasting each factor’s expected return over the next *H* days. "
                "These macro features include rate levels, "
                "term spreads, credit spreads, and volatility measures. More technically, "
                "'most predictive' means these features produced the largest reductions in forecast "
                "error in the random forest model, with RF importance measuring the average improvement "
                "in fit when that feature is used across the trees."
            )

            df_drivers = pd.DataFrame(drivers_rows)
            st.dataframe(
                df_drivers.style.format({"RF Importance": "{:.3f}"}),
                use_container_width=True,
            )

        if factor_eval:
            st.markdown("### Factor signal validation (walk-forward)")
            st.caption(
                "This section evaluates how well the forecasting models would have performed historically "
                "using walk-forward validation. The data is split into several consecutive train-test windows: "
                "for each window, the models are trained on past data and then tested only on the period that "
                "comes immediately after it, mimicking real-time forecasting. We report accuracy for two models: "
                "a simple AR(1) baseline and a three-model machine-learning ensemble (Ridge, Lasso, Random Forest). "
                "For each, we show the direction hit rate (how often the model correctly predicted the sign of the "
                "factor’s forward return), as well as RMSE and MAE to measure numerical forecast error. "
            )

            eval_rows = []
            for f, s in factor_eval.items():
                eval_rows.append(
                    {
                        "Factor": f,
                        "Ensemble Hit Rate": s.get("ensemble_hit_rate", np.nan),
                        "Ensemble RMSE": s.get("ensemble_rmse", np.nan),
                        "Ensemble MAE": s.get("ensemble_mae", np.nan),
                        "AR(1) Hit Rate": s.get("ar1_hit_rate", np.nan),
                        "AR(1) RMSE": s.get("ar1_rmse", np.nan),
                        "AR(1) MAE": s.get("ar1_mae", np.nan),
                    }
                )

            df_eval = pd.DataFrame(eval_rows)

            st.markdown("**AR(1) baseline performance**")
            df_ar1 = df_eval[
                ["Factor", "AR(1) Hit Rate", "AR(1) RMSE", "AR(1) MAE"]
            ]
            st.dataframe(
                df_ar1.style.format(
                    {
                        "AR(1) Hit Rate": "{:.2%}",
                        "AR(1) RMSE": "{:.4f}",
                        "AR(1) MAE": "{:.4f}",
                    }
                ),
                use_container_width=True,
            )

            with st.expander(
                "Show machine learning ensemble validation metrics (Ridge, Lasso, Random Forest)"
            ):
                st.dataframe(
                    df_eval.style.format(
                        {
                            "Ensemble Hit Rate": "{:.2%}",
                            "Ensemble RMSE": "{:.4f}",
                            "Ensemble MAE": "{:.4f}",
                            "AR(1) Hit Rate": "{:.2%}",
                            "AR(1) RMSE": "{:.4f}",
                            "AR(1) MAE": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                )
    else:
        st.info("No factor forecasts available.")

    with st.expander("Show universe and stock-level factor scores (details)"):
        uni = st.session_state.get("universe_tickers")
        port_sizes = st.session_state.get("factor_portfolio_sizes")
        sample_scores = st.session_state.get("sample_factor_scores")

        if uni:
            st.markdown(f"**Universe size**: {len(uni)}")
            st.dataframe(
                pd.DataFrame({"ticker": uni}),
                use_container_width=True,
            )

        if port_sizes:
            st.markdown("**Factor portfolios (size of long plus short)**")
            st.json(port_sizes)

        if isinstance(sample_scores, pd.DataFrame) and not sample_scores.empty:
            st.markdown("**Stock-level, sector-adjusted factor scores (0 to 1)**")
            st.dataframe(sample_scores, use_container_width=True)

    st.subheader("Optimized unified portfolio")

    optimized_portfolio = st.session_state.get("optimized_portfolio")

    if isinstance(optimized_portfolio, pd.DataFrame) and not optimized_portfolio.empty:
        st.caption(
            "This section shows a unified long/short portfolio built from the model’s *H*-day stock-level alpha forecasts. "
            "The optimizer chooses weights that maximize expected *H*-day portfolio alpha relative to portfolio risk, "
            "where risk is measured using a Ledoit Wolf shrinkage estimate of the recent return covariance matrix. "
            "Stocks with higher expected alpha and more favorable risk characteristics receive higher positive weights "
            "(long positions), while stocks with negative expected alpha receive negative weights (short positions). "
            "For stability, the portfolio enforces practical constraints: no individual position may exceed the per-stock "
            "weight cap (currently 10 percent), and total gross exposure is limited (currently at 1.5 times the portfolio’s "
            "capital). The table reports each stock’s portfolio weight, side, and its own expected *H*-day alpha in percent."
        )
        st.dataframe(
            optimized_portfolio.style.format(
                {
                    "weight": "{:.3f}",
                    "expected_alpha_%": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        try:
            portfolio_alpha_decimal = (
                optimized_portfolio["weight"]
                * (optimized_portfolio["expected_alpha_%"] / 100.0)
            ).sum()

            portfolio_alpha_pct = portfolio_alpha_decimal * 100.0

            horizon_used = st.session_state.get("portfolio_horizon")
            if horizon_used is None:
                horizon_used = int(forecast_horizon)

            st.markdown(
                f"**Portfolio expected {horizon_used}-day alpha:** "
                f"{portfolio_alpha_pct:.2f}%"
            )
        except Exception:
            pass
    else:
        st.info(
            "No unified optimized portfolio is available. "
            "This can happen if there are too few tickers with both "
            "alpha predictions and sufficient return history."
        )

    st.subheader("Alpha predictions (top 10)")
    st.caption(
        "This section lists the 10 stocks with the highest expected *H*-day alpha from a Lasso regression that links "
        "standardized stock characteristics (valuation, quality, momentum, size, etc.) to their future *H*-day returns. "
        "For each name, the expected alpha is the model’s forecast of its *H*-day excess return (in percent) based on "
        "those historical relationships, and the fundamental score summarizes how attractive its fundamentals look on those "
        "same dimensions."
    )

    if alpha_preds:
        df_alpha = pd.DataFrame(
            [
                {
                    "ticker": tk,
                    "expected_alpha_%": v["expected_alpha"] * 100.0,
                    "fundamental_score": v["drivers"].get("fundamental_score", None),
                    "top_features": v["drivers"].get("top_features", []),
                }
                for tk, v in alpha_preds.items()
            ]
        ).sort_values("expected_alpha_%", ascending=False)

        show_top = df_alpha.head(10)[
            ["ticker", "expected_alpha_%", "fundamental_score"]
        ]
        st.dataframe(
            show_top.style.format({"expected_alpha_%": "{:.2f}"}),
            use_container_width=True,
        )

        try:
            n = len(df_alpha)
            if n >= 30:
                k = max(int(n * 0.10), 3)
                top_mean = df_alpha.head(k)["expected_alpha_%"].mean()
                bottom_mean = df_alpha.tail(k)["expected_alpha_%"].mean()
                spread = top_mean - bottom_mean

                st.markdown(
                    f"**Alpha signal summary**: "
                    f"Top decile mean **{top_mean:.2f}%**, "
                    f"Bottom decile **{bottom_mean:.2f}%**, "
                    f"Spread **{spread:.2f}%**."
                )
        except Exception:
            pass

        st.markdown("**Top drivers for each of the top 10 stocks**")
        st.caption(
            "For each stock, the model predicts its expected *H*-day alpha using a Lasso regression trained on "
            "historical data. All input features are standardized before estimation, so each coefficient measures "
            "how a one standard deviation increase in that feature changes the stock’s predicted *H*-day alpha, "
            "holding other features fixed. A positive coefficient means higher values of the feature are associated "
            "with higher predicted alpha, and a negative coefficient means the opposite."
        )

        for _, row in df_alpha.head(10).iterrows():
            tk = row["ticker"]
            alpha_val = row["expected_alpha_%"]
            feats = row["top_features"]

            with st.expander(f"{tk} - {alpha_val:.2f}%"):
                if feats:
                    df_feats = pd.DataFrame(feats)
                    if "coef" in df_feats.columns:
                        df_feats.rename(columns={"coef": "Coefficient"}, inplace=True)
                    st.dataframe(df_feats, use_container_width=True)
                else:
                    st.write("No feature importances available for this ticker.")
    else:
        st.info("No alpha predictions available.")
