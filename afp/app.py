import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

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

from afp_app.sector_cache import (
    get_sp500_sector_scores,
    get_cache_status,
    clear_cache as clear_sp500_cache,
)

def compute_factor_performance(factor_returns_hist: pd.DataFrame):
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

    if {"date", "factor", "return"}.issubset(raw.columns):
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[~df["date"].isna()]

        if "rf_daily" in df.columns:
            rf_daily = (
                df[["date", "rf_daily"]]
                .dropna()
                .groupby("date")["rf_daily"]
                .mean()
                .sort_index()
            )
        else:
            rf_daily = None

        wide = (
            df.pivot_table(
                index="date",
                columns="factor",
                values="return",
                aggfunc="mean",
            )
            .sort_index()
        )

        if rf_daily is not None:
            wide["rf_daily"] = rf_daily.reindex(wide.index).ffill()

        wide.index = pd.DatetimeIndex(wide.index)
        wide = wide[~wide.index.isna()]

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

        if isinstance(rf_daily, pd.Series):
            rf_used = rf_daily.reindex(r.index).ffill().fillna(0.0)
        else:
            rf_used = 0.0

        n_days = len(r)

        cum = (1.0 + r).cumprod() - 1.0
        cum_paths[fac] = cum

        total_return = (1.0 + r).prod() - 1.0
        if n_days > 0:
            ann_return = (1.0 + total_return) ** (ann_factor / n_days) - 1.0
        else:
            ann_return = np.nan

        daily_vol = r.std(ddof=1)
        ann_vol = daily_vol * np.sqrt(ann_factor)

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


def load_sp500_reference_cache(
    api_key: str, 
    force_refresh: bool = False,
    force_refresh_prices: bool = False,
    show_progress: bool = True
) -> pd.DataFrame:

    try:
        if api_key and api_key != "YOUR_FMP_API_KEY":
            fetcher = FMPDataFetcher(api_key=api_key)
        else:
            fetcher = None
        
        cache_status = get_cache_status()
        needs_refresh = (
            force_refresh or 
            force_refresh_prices or
            cache_status.get("is_stale", True) or 
            not cache_status.get("cache_exists", False)
        )
        
        if needs_refresh and show_progress:
            if force_refresh:
                refresh_msg = "fundamentals (monthly) and prices (daily)"
            elif force_refresh_prices:
                refresh_msg = "prices (daily)"
            elif cache_status.get("fundamentals_is_stale") and cache_status.get("prices_is_stale"):
                refresh_msg = "fundamentals (monthly) and prices (daily)"
            elif cache_status.get("fundamentals_is_stale"):
                refresh_msg = "fundamentals (monthly)"
            elif cache_status.get("prices_is_stale"):
                refresh_msg = "prices (daily)"
            else:
                refresh_msg = "cache"
            
            progress_container = st.container()
            with progress_container:
                st.info(f"🔄 Refreshing S&P 500 {refresh_msg}...")
                progress_bar = st.progress(0, text="Initializing...")
                status_text = st.empty()
                
                def progress_callback(current: int, total: int, message: str):
                    pct = min(current / max(total, 1), 1.0)
                    progress_bar.progress(pct, text=f"{message} ({current}/{total})")
                    status_text.caption(f"Processing: {message}")
                
                sp500_ref = get_sp500_sector_scores(
                    fetcher=fetcher,
                    force_refresh=force_refresh,
                    force_refresh_prices=force_refresh_prices,
                    progress_callback=progress_callback,
                )
                
                progress_bar.progress(1.0, text="Complete!")
                time.sleep(0.5)
                progress_container.empty()
        else:
            sp500_ref = get_sp500_sector_scores(
                fetcher=fetcher,
                force_refresh=force_refresh,
                force_refresh_prices=force_refresh_prices,
            )
        
        return sp500_ref
    
    except Exception as e:
        st.warning(f"Could not load S&P 500 reference cache: {e}")
        return pd.DataFrame()


st.set_page_config(page_title="MarketMancer", layout="wide")

st.title("MarketMancer")
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
    ("sp500_reference", None), 
    ("sp500_cache_status", None),  
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

    st.markdown("### S&P 500 Sector Benchmark")
    
    cache_status = get_cache_status()
    st.session_state["sp500_cache_status"] = cache_status
    
    if cache_status.get("fundamentals_exists"):
        fund_updated = cache_status.get("fundamentals_last_updated", "Unknown")
        fund_stale = cache_status.get("fundamentals_is_stale", True)
        fund_icon = "✅" if not fund_stale else "⚠️"
        fund_date = fund_updated[:10] if fund_updated else "Unknown"
        st.caption(f"{fund_icon} Fundamentals: {fund_date} (monthly)")
    else:
        st.caption("⚠️ Fundamentals: Not cached")
    
    if cache_status.get("prices_exists"):
        prices_updated = cache_status.get("prices_last_updated", "Unknown")
        prices_stale = cache_status.get("prices_is_stale", True)
        prices_icon = "✅" if not prices_stale else "⚠️"
        prices_date = prices_updated[:10] if prices_updated else "Unknown"
        st.caption(f"{prices_icon} Prices: {prices_date} (daily)")
    else:
        st.caption("⚠️ Prices: Not cached")
    
    ticker_count = cache_status.get("fundamentals_ticker_count", 0)
    if ticker_count > 0:
        st.caption(f"📊 {ticker_count} S&P 500 stocks cached")
    
    use_sector_relative = st.checkbox(
        "Use S&P 500 sector-relative scoring",
        value=True,
        help="If enabled, factor scores are percentiles relative to all S&P 500 stocks "
             "in the same sector. If disabled, scores are relative to the selected universe only."
    )
    
    st.caption("Refresh cache:")
    col1, col2, col3 = st.columns(3)
    with col1:
        refresh_all_btn = st.button("All", help="Refresh both fundamentals and prices")
    with col2:
        refresh_prices_btn = st.button("Prices", help="Refresh prices only (daily)")
    with col3:
        clear_cache_btn = st.button("Clear", help="Delete all cached data")
    
    if clear_cache_btn:
        if clear_sp500_cache():
            st.success("Cache cleared!")
            st.session_state["sp500_reference"] = None
            st.rerun()
        else:
            st.error("Failed to clear cache.")
    
    if refresh_all_btn:
        if not api_key:
            st.error("Please enter an API key first.")
        else:
            sp500_ref = load_sp500_reference_cache(api_key, force_refresh=True, show_progress=True)
            if not sp500_ref.empty:
                st.session_state["sp500_reference"] = sp500_ref
                st.success(f"Cache refreshed with {len(sp500_ref)} stocks!")
                st.rerun()
            else:
                st.error("Failed to refresh cache.")
    
    if refresh_prices_btn:
        if not api_key:
            st.error("Please enter an API key first.")
        else:
            sp500_ref = load_sp500_reference_cache(
                api_key, 
                force_refresh=False, 
                force_refresh_prices=True,
                show_progress=True
            )
            if not sp500_ref.empty:
                st.session_state["sp500_reference"] = sp500_ref
                st.success(f"Prices cache refreshed!")
                st.rerun()
            else:
                st.error("Failed to refresh prices cache.")

    st.markdown("---")
    run_btn = st.button("Run pipeline", type="primary")

status = st.empty()

if run_btn:
    t0 = time.time()

    if not api_key:
        st.error("Please enter a valid FMP API key.")
        st.stop()

    st.session_state["portfolio_horizon"] = int(forecast_horizon)


    sp500_reference = pd.DataFrame()
    
    if use_sector_relative:
        status.info("Loading S&P 500 sector benchmark cache...")
        
        cached_ref = st.session_state.get("sp500_reference")
        cache_status = get_cache_status()
        
        if cached_ref is not None and not cached_ref.empty and not cache_status.get("is_stale", True):
            sp500_reference = cached_ref
            status.success(f"Using cached S&P 500 benchmark ({len(sp500_reference)} stocks)")
        else:
            sp500_reference = load_sp500_reference_cache(
                api_key, 
                force_refresh=False, 
                show_progress=True
            )
            
            if not sp500_reference.empty:
                st.session_state["sp500_reference"] = sp500_reference
                status.success(f"S&P 500 benchmark loaded ({len(sp500_reference)} stocks)")
            else:
                status.warning(
                    "Could not load S&P 500 benchmark. "
                    "Factor scores will be relative to the selected universe only."
                )

    status.info("Selecting universe...")
    tickers = get_universe(
        universe_size,
        randomize=randomize,
        seed=int(seed),
    )

    st.session_state["universe_tickers"] = tickers

    status.info("Fetching fundamentals and prices...")
    fetcher = FMPDataFetcher(api_key=api_key)
    fundamentals = collect_fundamental_data(tickers, start_date, fetcher)
    prices = collect_price_data(tickers, start_date, None, fetcher)

    if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
        st.error("No price data returned. Check API key, tickers, or date range.")
        st.stop()

    st.success(
        f"Collected {len(prices)} price rows. "
        f"Date range: {prices['date'].min()} to {prices['date'].max()}"
    )

    status.info("Computing factor scores and factor returns...")
    
    if use_sector_relative and not sp500_reference.empty:
        metrics = calculate_factor_metrics(
            fundamentals, 
            prices,
            sp500_reference=sp500_reference,
            use_sector_relative=True
        )
    else:
        metrics = calculate_factor_metrics(fundamentals, prices)

    factor_returns = pd.DataFrame()
    if metrics.empty:
        st.warning("No factor metrics available. Check fundamentals coverage.")
        st.session_state["factor_portfolio_sizes"] = None
        st.session_state["sample_factor_scores"] = None
    else:
        ctor = FactorPortfolioConstructor(metrics, prices)
        portfolios = ctor.construct_all()

        port_sizes = {
            k: (0 if v is None or v.empty else len(v))
            for k, v in portfolios.items()
        }

        factor_returns = ctor.calculate_factor_returns(
            start_date,
            prices["date"].max().strftime("%Y-%m-%d"),
        )

        latest = (
            metrics.sort_values("date")
            .groupby("ticker")
            .last()
            .reset_index()
        )

        score_cols = [
            c
            for c in [
                "value_score",
                "quality_score",
                "momentum_score",
                "lowvol_score",
            ]
            if c in latest.columns
        ]

        display_cols = ["ticker"]
        if "sector" in latest.columns:
            display_cols.append("sector")
        display_cols.extend(score_cols)

        if score_cols:
            sample = latest[display_cols].sort_values("ticker")
        else:
            sample = None

        st.session_state["factor_portfolio_sizes"] = port_sizes
        st.session_state["sample_factor_scores"] = sample

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


    status.info("Predicting per-ticker alpha (this may take a few minutes for large universes)...")
    
    alpha_model = AlphaPredictor(
        factor_returns,
        fundamentals,
        prices,
        horizon=forecast_horizon,
        lookback=252 * 2,
    )
    
    status.info("Alpha model initialized, generating predictions...")

    alpha_preds: dict[str, dict] = {}
    
    total_tickers = len(tickers)
    progress_bar = st.progress(0, text="Predicting alpha...")
    
    for i, tk in enumerate(tickers):
        # Update progress more frequently
        if i % 10 == 0 or i == total_tickers - 1:
            pct = (i + 1) / total_tickers
            progress_bar.progress(pct, text=f"Predicting alpha... ({i + 1}/{total_tickers} tickers)")
        
        res = alpha_model.predict_alpha(tk, horizon=forecast_horizon)
        if res:
            if "drivers" in res:
                top = res["drivers"].get("top_features", [])
                res["drivers"]["top_features"] = top[:top_k_drivers]
            alpha_preds[tk] = res
    
    progress_bar.progress(1.0, text="Alpha predictions complete!")
    st.session_state["base_alpha"] = alpha_preds
    
    if not alpha_preds:
        st.warning(f"No alpha predictions could be generated for any of the {len(tickers)} tickers. "
                   "This may indicate insufficient price or fundamental data.")
    else:
        st.success(f"Generated alpha predictions for {len(alpha_preds)}/{len(tickers)} tickers")

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
                # Try optimization without covariance matrix (alpha-only)
                st.info("Covariance matrix unavailable - using alpha-only optimization")
                weights = optimizer.optimize(mu=mu, Sigma=pd.DataFrame())
                
                if not weights.empty:
                    port_table = optimizer.build_portfolio_table(
                        weights=weights,
                        alpha_preds=alpha_preds,
                    )
                    st.session_state["optimized_portfolio"] = port_table
                else:
                    st.session_state["optimized_portfolio"] = None
            else:
                common = [tk for tk in valid_tickers if tk in mu.index]
                if len(common) < 2:
                    st.info("Insufficient common tickers - using alpha-only optimization")
                    weights = optimizer.optimize(mu=mu, Sigma=pd.DataFrame())
                    
                    if not weights.empty:
                        port_table = optimizer.build_portfolio_table(
                            weights=weights,
                            alpha_preds=alpha_preds,
                        )
                        st.session_state["optimized_portfolio"] = port_table
                    else:
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
            st.warning("No tickers with alpha predictions available for portfolio optimization.")

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

    factor_returns_hist = st.session_state.get("factor_returns")

    if isinstance(factor_returns_hist, pd.DataFrame) and not factor_returns_hist.empty:
        perf_summary, cum_paths = compute_factor_performance(factor_returns_hist)

        if not perf_summary.empty:
            st.subheader("Historical factor performance")
            st.caption("How have these factor portfolios actually behaved over time?")

            with st.expander("Details"):
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
            
                wealth_paths = (1.0 + cum_paths).clip(lower=1e-6)
            
                df_wealth = wealth_paths.reset_index()
                date_col = df_wealth.columns[0]
            
                df_wealth = df_wealth.melt(
                    id_vars=date_col,
                    var_name="Factor",
                    value_name="Wealth",
                ).rename(columns={date_col: "date"})
            
                chart = (
                    alt.Chart(df_wealth)
                    .mark_line()
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y(
                            "Wealth:Q",
                            title="Cumulative wealth (log scale)",
                            scale=alt.Scale(type="log"),
                        ),
                        color=alt.Color("Factor:N", title="Factor"),
                        tooltip=[
                            alt.Tooltip("date:T", title="Date"),
                            alt.Tooltip("Factor:N", title="Factor"),
                            alt.Tooltip("Wealth:Q", title="Wealth", format=".2f"),
                        ],
                    )
                    .properties(height=300)
                )
            
                st.altair_chart(chart, use_container_width=True)

    else:
        st.info(
            "No historical factor returns are available yet. "
            "Run the pipeline to compute factor portfolios and their return history."
        )

    st.subheader("Factor premia forecasts")
    st.caption("Which factor styles look poised to outperform next?")

    if forecasts:
        with st.expander("Details"):
            st.caption(
                "These forecasts estimate each factor's expected return (in percent) over the next *H* days, "
                "where *H* is the forecast horizon set by the user. The primary forecast uses a simple AR(1) "
                "model on each factor's own history, while macro features are used for diagnostics and driver "
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
            st.caption("What macro forces are actually moving these factors?")

            with st.expander("Details"):
                st.caption(
                    "The table below shows which macro features the model found most predictive "
                    "when forecasting each factor's expected return over the next *H* days. "
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
            st.caption("Can I trust these factor forecasts out of sample?")

            with st.expander("Details"):
                st.caption(
                    "This section evaluates how well the forecasting models would have performed historically "
                    "using walk-forward validation. The data is split into several consecutive train-test windows: "
                    "for each window, the models are trained on past data and then tested only on the period that "
                    "comes immediately after it, mimicking real-time forecasting. We report accuracy for two models: "
                    "a simple AR(1) baseline and a three-model machine-learning ensemble (Ridge, Lasso, Random Forest). "
                    "For each, we show the direction hit rate (how often the model correctly predicted the sign of the "
                    "factor's forward return), as well as RMSE and MAE to measure numerical forecast error. "
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
        sp500_ref = st.session_state.get("sp500_reference")
        cache_status = st.session_state.get("sp500_cache_status", {})

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
            if sp500_ref is not None and not sp500_ref.empty:
                sp500_count = len(sp500_ref)
                sectors_in_ref = sp500_ref["sector"].nunique() if "sector" in sp500_ref.columns else 0
                
                st.markdown("**Stock-level factor scores (0 to 1) — S&P 500 Sector-Relative**")
                st.caption(
                    f"Each score represents the stock's percentile rank compared to **all {sp500_count} S&P 500 stocks** "
                    f"in the same sector (across {sectors_in_ref} sectors). A score of 0.80 means the stock ranks "
                    "better than 80% of its S&P 500 sector peers on that factor."
                )
            else:
                st.markdown("**Stock-level factor scores (0 to 1) — Universe-Relative**")
                st.caption(
                    "Each score represents the stock's percentile rank compared to other stocks "
                    "in the selected universe (sector-adjusted where possible). A score of 0.80 means "
                    "the stock ranks better than 80% of universe peers on that factor."
                )
            
            score_display = sample_scores.copy()
            format_dict = {}
            for col in score_display.columns:
                if col.endswith("_score"):
                    format_dict[col] = "{:.2f}"
            
            st.dataframe(
                score_display.style.format(format_dict),
                use_container_width=True,
            )
            
            if cache_status:
                with st.expander("S&P 500 Benchmark Cache Info"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Fundamentals Cache**")
                        st.write(f"- Exists: {cache_status.get('fundamentals_exists', False)}")
                        st.write(f"- Last updated: {cache_status.get('fundamentals_last_updated', 'Never')[:10] if cache_status.get('fundamentals_last_updated') else 'Never'}")
                        st.write(f"- Stocks: {cache_status.get('fundamentals_ticker_count', 0)}")
                        st.write(f"- Refresh cycle: {cache_status.get('fundamentals_refresh_days', 30)} days")
                        st.write(f"- Stale: {cache_status.get('fundamentals_is_stale', True)}")
                    with col2:
                        st.markdown("**Prices Cache**")
                        st.write(f"- Exists: {cache_status.get('prices_exists', False)}")
                        st.write(f"- Last updated: {cache_status.get('prices_last_updated', 'Never')[:10] if cache_status.get('prices_last_updated') else 'Never'}")
                        st.write(f"- Stocks: {cache_status.get('prices_ticker_count', 0)}")
                        st.write(f"- Refresh cycle: {cache_status.get('prices_refresh_days', 7)} days")
                        st.write(f"- Stale: {cache_status.get('prices_is_stale', True)}")
                    
                    st.markdown("**Sectors in cache:**")
                    sectors = cache_status.get("fundamentals_sectors", [])
                    if sectors:
                        st.write(", ".join(sorted(sectors)))

    st.subheader("Optimized unified portfolio")
    st.caption("Given all signals together, what long/short portfolio should I actually hold?")

    optimized_portfolio = st.session_state.get("optimized_portfolio")

    if isinstance(optimized_portfolio, pd.DataFrame) and not optimized_portfolio.empty:
        with st.expander("Details"):
            st.caption(
                "This section shows a unified long/short portfolio built from the model's *H*-day stock-level alpha forecasts. "
                "The optimizer chooses weights that maximize expected *H*-day portfolio alpha relative to portfolio risk, "
                "where risk is measured using a Ledoit Wolf shrinkage estimate of the recent return covariance matrix. "
                "Stocks with higher expected alpha and more favorable risk characteristics receive higher positive weights "
                "(long positions), while stocks with negative expected alpha receive negative weights (short positions). "
                "For stability, the portfolio enforces practical constraints: no individual position may exceed the per-stock "
                "weight cap (currently 10 percent), and total gross exposure is limited (currently at 1.5 times the portfolio's "
                "capital). The table reports each stock's portfolio weight, side, and its own expected *H*-day alpha in percent."
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
    st.caption("Which individual stocks show the strongest forward alpha right now?")

    with st.expander("Details"):
        st.caption(
            "This section lists the 10 stocks with the highest expected *H*-day alpha from a Lasso regression that links "
            "standardized stock characteristics (valuation, quality, momentum, size, etc.) to their future *H*-day returns. "
            "For each name, the expected alpha is the model's forecast of its *H*-day excess return (in percent) based on "
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
        st.caption("Why is the model bullish or bearish on this stock?")

        with st.expander("Details"):
            st.caption(
                "For each stock, the model predicts its expected *H*-day alpha using a Lasso regression trained on "
                "historical data. All input features are standardized before estimation, so each coefficient measures "
                "how a one standard deviation increase in that feature changes the stock's predicted *H*-day alpha, "
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
