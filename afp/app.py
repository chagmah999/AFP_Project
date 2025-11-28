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


st.set_page_config(page_title="AFP Forecasting Tool", layout="wide")

st.title("AFP Forecasting Tool")
st.caption("Factor premia forecasts, stock-level alpha, and unified portfolio optimization")

# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
for key, default in [
    ("base_forecasts", None),
    ("base_factor_eval", None),
    ("base_alpha", None),
    ("modeling_frame", None),
    ("forecaster_obj", None),
    # store universe and factor score info for later display
    ("universe_tickers", None),
    ("factor_portfolio_sizes", None),
    ("sample_factor_scores", None),
    # store optimized unified portfolio table
    ("optimized_portfolio", None),
    # store the forecast horizon used when pipeline was last run
    ("portfolio_horizon", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# -------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------------
if run_btn:
    t0 = time.time()

    if not api_key:
        st.error("Please enter a valid FMP API key.")
        st.stop()

    # Remember the horizon used for this run
    st.session_state["portfolio_horizon"] = int(forecast_horizon)

    # ------------------ Universe ------------------
    status.info("Selecting universe...")
    tickers = get_universe(
        universe_size,
        randomize=randomize,
        seed=int(seed),
    )

    # Store universe for later display instead of showing it first
    st.session_state["universe_tickers"] = tickers

    # ------------------ Data Collection ------------------
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

    # ------------------ Factor Metrics & Portfolios ------------------
    status.info("Computing factor scores and factor returns...")
    metrics = calculate_factor_metrics(fundamentals, prices)

    factor_returns = pd.DataFrame()
    if metrics.empty:
        st.warning("No factor metrics available. Check fundamentals coverage.")
        # Also clear any old stored portfolio info
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

        # Build latest per ticker scores for later display
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
            sample = (
                latest[["ticker"] + score_cols]
                .sort_values("ticker")
                .head(30)
            )
        else:
            sample = None

        # Store details in session state instead of showing them first
        st.session_state["factor_portfolio_sizes"] = port_sizes
        st.session_state["sample_factor_scores"] = sample

    # ------------------ Macro Data & Modeling Frame ------------------
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

    # ------------------ Factor Premia Forecasting (core signal) ------------------
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
        # Walk-forward validation (stores ensemble and AR(1) metrics)
        val = forecaster.walk_forward_validation(modeling, f)
        if val is not None:
            factor_eval[f] = forecaster.validation_summary.get(f, {})

        # Forward forecast for next H days
        fc = forecaster.forecast_next(modeling, f)
        if fc:
            fc["top_drivers"] = (fc.get("top_drivers") or [])[:top_k_drivers]
            forecasts[f] = fc

    st.session_state["base_forecasts"] = forecasts
    st.session_state["base_factor_eval"] = factor_eval

    # ------------------ Alpha Predictions ------------------
    status.info("Predicting per-ticker alpha...")
    alpha_model = AlphaPredictor(
        factor_returns,
        fundamentals,
        prices,
        horizon=forecast_horizon,
        lookback=252 * 2,
    )

    alpha_preds: dict[str, dict] = {}
    cap = min(100, len(tickers))

    for tk in tickers[:cap]:
        res = alpha_model.predict_alpha(tk, horizon=forecast_horizon)
        if res:
            if "drivers" in res:
                top = res["drivers"].get("top_features", [])
                res["drivers"]["top_features"] = top[:top_k_drivers]
            alpha_preds[tk] = res

    st.session_state["base_alpha"] = alpha_preds

    # ------------------ Unified Optimized Portfolio (compute only, store in session) ------------------
    status.info("Constructing optimized unified portfolio...")

    try:
        # Use only tickers for which we have alpha predictions
        opt_tickers = [tk for tk in tickers if tk in alpha_preds]

        if opt_tickers:
            optimizer = UnifiedPortfolioOptimizer(
                risk_aversion=10.0,
                max_gross=1.5,
                max_weight=0.10,
            )

            # 1. Expected returns vector from alpha predictions (H-day expected alphas)
            mu = optimizer.build_expected_returns(
                alpha_preds=alpha_preds,
                tickers=opt_tickers,
            )

            # 2. Covariance matrix and valid ticker subset
            Sigma, valid_tickers = optimizer.build_covariance(
                price_data=prices,
                tickers=opt_tickers,
                lookback_days=252,
            )

            if Sigma is None or Sigma.empty or len(valid_tickers) < 2:
                st.session_state["optimized_portfolio"] = None
            else:
                # Keep only tickers that have both alpha and sufficient history
                common = [tk for tk in valid_tickers if tk in mu.index]
                if len(common) < 2:
                    st.session_state["optimized_portfolio"] = None
                else:
                    mu_use = mu.loc[common]
                    Sigma_use = Sigma.loc[common, common]

                    # 3. Optimize weights
                    weights = optimizer.optimize(mu=mu_use, Sigma=Sigma_use)

                    # 4. Pretty table for display
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


# -------------------------------------------------------------
# Display Section (Persists after running pipeline)
# -------------------------------------------------------------
forecasts = st.session_state.get("base_forecasts")
alpha_preds = st.session_state.get("base_alpha")
factor_eval = st.session_state.get("base_factor_eval")

if not forecasts and not alpha_preds:
    st.info("Run the pipeline from the sidebar to generate forecasts.")
else:
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

        # 1a. Main view: AR(1) expected premium per factor
        summary_rows = []
        drivers_rows = []

        for f, v in forecasts.items():
            # Prefer AR(1); if missing for any reason, fall back to ensemble
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

        # 1b. Optional ensemble view in an expander
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

        # 1c. Top drivers per factor
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

        # 1d. Validation summary: AR(1) vs ensemble
        if factor_eval:
            st.markdown("### Factor signal validation (walk-forward)")

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

            # AR(1) baseline table (primary)
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

            # Ensemble vs AR(1) comparison in an expander
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

    # =========================================================
    # 1.b Universe and factor score details (after premia)
    # =========================================================
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
            st.markdown("**Sample stock-level factor scores (0 to 1)**")
            st.dataframe(sample_scores, use_container_width=True)

    # ------------------ Optimized unified portfolio ------------------
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

        # --- Portfolio-level expected H-day alpha with dynamic H ---
        try:
            portfolio_alpha_decimal = (
                optimized_portfolio["weight"]
                * (optimized_portfolio["expected_alpha_%"] / 100.0)
            ).sum()

            portfolio_alpha_pct = portfolio_alpha_decimal * 100.0

            # Use the horizon from the last pipeline run if available
            horizon_used = st.session_state.get("portfolio_horizon")
            if horizon_used is None:
                # Fallback to current slider value if something went wrong
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

    # =========================================================
    # 2. Alpha predictions
    # =========================================================
    st.subheader("Alpha predictions (top 10)")

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

        # Top 10 table
        show_top = df_alpha.head(10)[
            ["ticker", "expected_alpha_%", "fundamental_score"]
        ]
        st.dataframe(
            show_top.style.format({"expected_alpha_%": "{:.2f}"}),
            use_container_width=True,
        )

        # Alpha signal strength
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

        # Driver details for each of the top 10 stocks
        st.markdown("Top drivers for each of the top 10 stocks")
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
