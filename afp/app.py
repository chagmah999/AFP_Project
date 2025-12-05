import os
import time
import numpy as np
import pandas as pd
import streamlit as st

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

import numpy as np
import pandas as pd

def compute_factor_performance(
    factor_returns: pd.DataFrame,
    macro_frame: pd.DataFrame | None = None,
    rf_col: str = "R_3M",
) -> pd.DataFrame:
    """
    Compute historical performance statistics for each factor portfolio.

    Inputs
    ------
    factor_returns : DataFrame
        Long-format factor returns with columns:
            - 'date'   : pandas.Timestamp or datetime-like
            - 'factor' : factor name (e.g. 'VALUE', 'QUALITY', 'MOMENTUM', 'LOW_VOL')
            - 'return' : daily factor return in decimal form (0.01 = 1%)

    macro_frame : DataFrame or None
        Optional macro data with at least:
            - 'date'
            - rf_col (for example 'R_3M'), which is a daily risk-free rate in percent
              (for example 4.5 means 4.5 percent per year)

        If provided and rf_col is present, Sharpe ratios are computed using
        *excess returns* over this risk-free rate. If not, the risk-free rate
        is treated as zero.

    rf_col : str
        Column name in macro_frame that contains the annualized risk-free rate
        in percent (for example 3-month Treasury yield).

    Output
    ------
    perf : DataFrame
        One row per factor with columns:
            - factor
            - start_date
            - end_date
            - n_days
            - total_return_pct
            - ann_return_pct
            - ann_excess_return_pct
            - ann_vol_pct
            - sharpe
            - max_drawdown_pct
    """

    if factor_returns is None or factor_returns.empty:
        return pd.DataFrame()

    # Ensure date is datetime
    factor_returns = factor_returns.copy()
    factor_returns["date"] = pd.to_datetime(factor_returns["date"])

    # Prepare risk-free series if available
    use_rf = False
    rf_df = None
    if macro_frame is not None and rf_col in macro_frame.columns:
        rf_df = macro_frame[["date", rf_col]].copy()
        rf_df["date"] = pd.to_datetime(rf_df["date"])
        # Convert from percent to daily decimal rate
        # Example: R_3M = 4.5 -> annual 0.045 -> daily approx 0.045 / 252
        rf_df["rf_daily"] = (rf_df[rf_col] / 100.0) / 252.0
        use_rf = True

    rows = []
    for f_name, df_f in factor_returns.groupby("factor"):
        df_f = df_f.sort_values("date").copy()
        if df_f.empty:
            continue

        # Merge risk-free if we have it
        if use_rf:
            tmp = df_f.merge(
                rf_df[["date", "rf_daily"]],
                on="date",
                how="left",
            )
            # If some rf_daily are missing (for example holidays), fill with the median daily rf
            median_rf = tmp["rf_daily"].median()
            tmp["rf_daily"] = tmp["rf_daily"].fillna(median_rf if pd.notna(median_rf) else 0.0)
            r = tmp["return"].astype(float)
            rf_daily = tmp["rf_daily"].astype(float)
        else:
            r = df_f["return"].astype(float)
            # Zero risk-free if not provided
            rf_daily = pd.Series(0.0, index=r.index)

        if r.empty:
            continue

        # Basic counts and dates
        n_days = len(r)
        start_date = df_f["date"].iloc[0]
        end_date = df_f["date"].iloc[-1]

        # Cumulative total return (compound)
        wealth = (1.0 + r).cumprod()
        total_return = wealth.iloc[-1] - 1.0

        # Annualized total return based on trading days
        if n_days > 0:
            ann_return = (1.0 + total_return) ** (252.0 / n_days) - 1.0
        else:
            ann_return = np.nan

        # Annualized volatility (using population std)
        daily_vol = r.std(ddof=0)
        ann_vol = daily_vol * np.sqrt(252.0) if pd.notna(daily_vol) else np.nan

        # Excess returns and Sharpe ratio
        # Daily excess = r_t - rf_t
        excess = r - rf_daily
        # Annualized excess return as mean(excess) * 252
        ex_ann_return = excess.mean() * 252.0

        if ann_vol > 0 and pd.notna(ann_vol):
            sharpe = ex_ann_return / ann_vol
        else:
            sharpe = np.nan

        # Max drawdown: based on wealth series
        running_max = wealth.cummax()
        drawdown = wealth / running_max - 1.0
        max_drawdown = drawdown.min() if not drawdown.empty else np.nan

        rows.append(
            {
                "factor": f_name,
                "start_date": start_date,
                "end_date": end_date,
                "n_days": n_days,
                "total_return_pct": total_return * 100.0,
                "ann_return_pct": ann_return * 100.0 if pd.notna(ann_return) else np.nan,
                "ann_excess_return_pct": ex_ann_return * 100.0,
                "ann_vol_pct": ann_vol * 100.0 if pd.notna(ann_vol) else np.nan,
                "sharpe": sharpe,
                "max_drawdown_pct": max_drawdown * 100.0 if pd.notna(max_drawdown) else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame()

    perf = pd.DataFrame(rows)
    return perf




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

        if score_cols:
            sample = latest[["ticker"] + score_cols].sort_values("ticker")

        else:
            sample = None

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
                "This section summarizes how each long–short factor portfolio "
                "has performed over the full sample used in the model. "
                "For each factor, we show total and annualized returns, "
                "annualized volatility, a simple Sharpe ratio using zero risk free, "
                "and the worst peak–to–trough drawdown over the period."
            )

            st.dataframe(
                perf_summary.style.format(
                    {
                        "Total return %": "{:.2f}",
                        "Annualized return %": "{:.2f}",
                        "Annualized volatility %": "{:.2f}",
                        "Sharpe (rf = 0)": "{:.2f}",
                        "Max drawdown %": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )

            # Optional cumulative return chart
            if not cum_paths.empty:
                st.caption(
                    "Cumulative growth of one unit invested in each factor "
                    "long–short portfolio over time."
                )
                cum_display = cum_paths.set_index("date")
                st.line_chart(cum_display)
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
                        "Ensemble Premium %": v.get("ensemble_forecast", np.nan) * 100.0,
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
            st.markdown("**Stock-level, sector-adjusted factor scores (0-1 scale)**")
            st.dataframe(sample_scores, use_container_width=True)

    st.subheader("Optimized unified portfolio")

    optimized_portfolio = st.session_state.get("optimized_portfolio")

    if isinstance(optimized_portfolio, pd.DataFrame) and not optimized_portfolio.empty:
        st.caption(
            "This section shows a unified long/short portfolio built from the model’s *H*-day stock-level alpha forecasts. "
            "The optimizer chooses weights that maximize expected *H*-day portfolio alpha relative to portfolio risk, "
            "where risk is measured using a Ledoit–Wolf shrinkage estimate of the recent return covariance matrix. "
            "Stocks with higher expected alpha and more favorable risk characteristics receive higher positive weights "
            "(long positions), while stocks with negative expected alpha receive negative weights (short positions). "
            "For stability, the portfolio enforces practical constraints: no individual position may exceed the per-stock "
            "weight cap (currently 10%), and total gross exposure is limited (currently at 1.5x the portfolio’s "
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
