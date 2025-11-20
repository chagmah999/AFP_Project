import os
import json
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
from afp_app.signal_stress import StressProbabilityModel
from afp_app.engine import MarketMancerEngine
from afp_app.scenario import scenario_factor_premia, scenario_stress


st.set_page_config(page_title="AFP Forecasting Tool", layout="wide")

st.title("AFP Forecasting Tool")
st.caption("Factor premia forecasts, per-ticker alpha, and market stress regime")

# -------------------------------------------------------------------
# Session state for reusing results (for sensitivity analysis)
# -------------------------------------------------------------------
if "base_forecasts" not in st.session_state:
    st.session_state["base_forecasts"] = None
if "base_alpha" not in st.session_state:
    st.session_state["base_alpha"] = None
if "base_stress" not in st.session_state:
    st.session_state["base_stress"] = None
if "base_factor_eval" not in st.session_state:
    st.session_state["base_factor_eval"] = None
if "modeling_frame" not in st.session_state:
    st.session_state["modeling_frame"] = None
if "forecaster_obj" not in st.session_state:
    st.session_state["forecaster_obj"] = None
if "stress_model_obj" not in st.session_state:
    st.session_state["stress_model_obj"] = None

# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
with st.sidebar:
    st.subheader("Configuration")

    api_key = st.text_input(
        "FMP API Key",
        value=FMP_API_KEY or "",
        type="password",
        help="Financial Modeling Prep API key",
    )

    start_date = st.text_input(
        "Start date (YYYY-MM-DD)",
        value=DEFAULT_START_DATE,
        help="Earliest date to pull data for backtests and model training",
    )

    st.markdown("**Universe**")
    universe_size = st.slider(
        "Universe size",
        min_value=10,
        max_value=509,
        value=DEFAULT_UNIVERSE_SIZE,
        step=5,
        help="Number of S&P 500 stocks to include",
    )
    randomize = st.checkbox(
        "Randomize universe selection",
        value=True,
        help="If unchecked, uses the first N tickers alphabetically",
    )
    seed = st.number_input(
        "Random seed",
        min_value=0,
        value=42,
        step=1,
        help="Seed for random universe selection",
    )

    st.markdown("**Forecasting**")
    forecast_horizon = st.slider(
        "Forecast horizon (days)",
        min_value=5,
        max_value=63,
        value=21,
        step=1,
        help="Length H of the forward window for factor premiums and alpha",
    )
    top_k_drivers = st.radio(
        "Top drivers to show (per factor/stock)",
        options=[3, 5],
        index=0,
        help="Number of top drivers by importance to display",
    )

    run_btn = st.button("Run pipeline")

status = st.empty()

# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
forecasts = st.session_state["base_forecasts"]
alpha_preds = st.session_state["base_alpha"]
stress_fc = st.session_state["base_stress"]
factor_eval = st.session_state["base_factor_eval"]
modeling = st.session_state["modeling_frame"]
forecaster = st.session_state["forecaster_obj"]
stress = st.session_state["stress_model_obj"]

if run_btn:
    t0 = time.time()

    if not api_key or api_key == "YOUR_FMP_API_KEY":
        st.error("Please set a valid FMP API key.")
        st.stop()

    # ---------------- Universe ----------------
    status.info("Building universe...")
    tickers = get_universe(
        universe_size,
        randomize=randomize,
        seed=int(seed),
    )
    st.write(f"Universe of {len(tickers)} tickers:")
    st.dataframe(pd.DataFrame({"ticker": tickers}), use_container_width=True)

    # ---------------- Data collection ----------------
    status.info("Fetching fundamentals and prices...")
    fetcher = FMPDataFetcher(api_key=api_key)
    fundamentals = collect_fundamental_data(tickers, start_date, fetcher)
    prices = collect_price_data(tickers, start_date, None, fetcher)

    if prices.empty:
        st.error("No price data returned. Check API key, tickers, or date range.")
        st.stop()

    st.success(
        f"Collected {len(prices)} price rows. "
        f"Date range: {prices['date'].min()} to {prices['date'].max()}"
    )

    # ---------------- Factors: metrics and portfolios ----------------
    status.info("Computing factor metrics...")
    metrics = calculate_factor_metrics(fundamentals, prices)

    if metrics.empty:
        st.warning("No factor metrics computed. Check fundamental coverage.")
        portfolios = {}
        factor_returns = pd.DataFrame()
    else:
        ctor = FactorPortfolioConstructor(metrics, prices)
        portfolios = ctor.construct_all()
        st.write("Portfolios built:")
        st.json(
            {
                k: 0 if v is None or v.empty else len(v)
                for k, v in portfolios.items()
            }
        )
        factor_returns = ctor.calculate_factor_returns(
            start_date, prices["date"].max().strftime("%Y-%m-%d")
        )

    # ---------------- Macro features and modeling frame ----------------
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

    # ---------------- Signal 1: factor premia ----------------
    status.info("Forecasting factor premia...")
    forecaster = FactorPremiaForecaster(
        lookback_window=LOOKBACK_DAYS,
        forecast_horizon=forecast_horizon,
    )
    st.session_state["forecaster_obj"] = forecaster

    factors = ["VALUE", "QUALITY", "MOMENTUM", "LOW_VOL"]
    forecasts = {}
    factor_eval = {}

    for f in factors:
        # Walk forward validation (gives direction hit rate and error metrics)
        res = forecaster.walk_forward_validation(modeling, f)
        if res is not None:
            factor_eval[f] = forecaster.validation_summary.get(f, {})

        # Forecast next H day factor premium
        fc = forecaster.forecast_next(modeling, f)
        if fc:
            fc["top_drivers"] = (fc.get("top_drivers") or [])[:top_k_drivers]
            forecasts[f] = fc

    st.session_state["base_forecasts"] = forecasts
    st.session_state["base_factor_eval"] = factor_eval

    # ---------------- Signal 2: alpha (per ticker) ----------------
    status.info("Predicting per-ticker alpha...")
    alpha = AlphaPredictor(
        factor_returns,
        fundamentals,
        prices,
        horizon=forecast_horizon,
        lookback=252 * 2,
    )
    alpha_preds = {}
    cap = min(100, len(tickers))

    for tk in tickers[:cap]:
        p = alpha.predict_alpha(tk, horizon=forecast_horizon)
        if p:
            # Keep only top_k_drivers for display
            if "drivers" in p and "top_features" in p["drivers"]:
                p["drivers"]["top_features"] = (
                    p["drivers"].get("top_features") or []
                )[:top_k_drivers]
            alpha_preds[tk] = p

    st.session_state["base_alpha"] = alpha_preds

    # ---------------- Signal 3: stress regime ----------------
    status.info("Estimating market stress probability...")
    stress = StressProbabilityModel()
    feature_importance_stress = stress.fit(modeling)
    stress_fc = stress.predict(modeling)

    st.session_state["stress_model_obj"] = stress
    st.session_state["base_stress"] = stress_fc

    # ---------------- Integrate recommendations ----------------
    status.info("Integrating recommendations...")
    engine = MarketMancerEngine(forecasts, alpha_preds, stress_fc or {})
    recs = engine.generate()

    # ---------------- Display sections ----------------
    # Stress
    st.subheader("Stress regime")

    if stress_fc:
        line = (
            f"Regime: **{stress_fc['regime']}**  |  "
            f"Stress probability: **{stress_fc['stress_probability']*100:.1f}%**"
        )
        auc = getattr(stress, "cv_auc_mean", None)
        share = getattr(stress, "stress_share", None)
        if auc is not None:
            line += f"  |  Model AUC: **{auc:.3f}**"
        if share is not None:
            line += (
                f"  |  Historical stress frequency: "
                f"**{share*100:.1f}%**"
            )
        st.write(line)

        key_ind = stress_fc.get("key_indicators", {})
        if key_ind:
            st.markdown("Key indicators at latest date:")
            rows = [{"Indicator": k, "Value": v} for k, v in key_ind.items()]
            if rows:
                df_ind = pd.DataFrame(rows)
                st.dataframe(
                    df_ind.style.format({"Value": "{:.3f}"}),
                    use_container_width=True,
                )
    else:
        st.info("No stress regime forecast available.")

    # Factor premia
    st.subheader("Factor premia forecasts")

    if forecasts:

        # 1. Summary table
        summary_rows = []
        drivers_rows = []
        for f, v in forecasts.items():
            summary_rows.append(
                {
                    "Factor": f,
                    "Expected Premium %": v["ensemble_forecast"] * 100.0,
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
            "Expected Premium %", ascending=False
        )
        st.dataframe(
            df_summary.style.format(
                {"Expected Premium %": "{:.2f}"}
            ),
            use_container_width=True,
        )

        # 2. Top drivers table
        if drivers_rows:
            st.markdown(f"Top **{top_k_drivers}** drivers per factor")
            df_drivers = pd.DataFrame(drivers_rows)
            st.dataframe(
                df_drivers.style.format(
                    {"RF Importance": "{:.3f}"}
                ),
                use_container_width=True,
            )

        # 3. Validation summary (hit rate and errors)
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
                    }
                )

            df_eval = pd.DataFrame(eval_rows)
            st.dataframe(
                df_eval.style.format(
                    {
                        "Ensemble Hit Rate": "{:.2%}",
                        "Ensemble RMSE": "{:.4f}",
                        "Ensemble MAE": "{:.4f}",
                    }
                ),
                use_container_width=True,
            )

    else:
        st.info("No factor forecasts available.")

    # Alpha
    st.subheader("Alpha predictions (top 10)")

    if alpha_preds:
        df_alpha = pd.DataFrame(
            [
                {
                    "ticker": tk,
                    "expected_alpha_%": v["expected_alpha"] * 100.0,
                    "fundamental_score": v["drivers"].get(
                        "fundamental_score", None
                    ),
                    "top_features": v["drivers"].get("top_features", []),
                }
                for tk, v in alpha_preds.items()
            ]
        ).sort_values("expected_alpha_%", ascending=False)

        # 1. Top 10 table
        show_top = df_alpha.head(10)[
            ["ticker", "expected_alpha_%", "fundamental_score"]
        ]
        st.dataframe(
            show_top.style.format({"expected_alpha_%": "{:.2f}"}),
            use_container_width=True,
        )

        # 2. Alpha signal strength (top vs bottom)
        try:
            n = len(df_alpha)
            if n >= 30:
                k = max(int(n * 0.10), 3)
                top_mean = df_alpha.head(k)["expected_alpha_%"].mean()
                bottom_mean = df_alpha.tail(k)["expected_alpha_%"].mean()
                spread = top_mean - bottom_mean

                st.markdown(
                    f"**Alpha signal summary**: top decile mean expected alpha "
                    f"**{top_mean:.2f}%**, bottom decile **{bottom_mean:.2f}%**, "
                    f"spread **{spread:.2f}%**."
                )
        except Exception:
            pass

        # 3. Driver tables for each top 10 stock
        st.markdown(
            f"Top **{top_k_drivers}** drivers for each of the top 10 stocks"
        )
        for _, row in df_alpha.head(10).iterrows():
            ticker = row["ticker"]
            alpha_val = row["expected_alpha_%"]
            feats = row["top_features"]

            with st.expander(f"{ticker} - {alpha_val:.2f}%"):
                if feats:
                    df_feats = pd.DataFrame(feats)
                    if "coef" in df_feats.columns:
                        df_feats = df_feats.rename(
                            columns={"coef": "Coefficient"}
                        )
                    st.dataframe(df_feats, use_container_width=True)
                else:
                    st.write("No feature importances available for this ticker.")
    else:
        st.info("No alpha predictions available.")

    # Integrated recommendations
    st.subheader("Integrated recommendations")
    st.json(recs)

    t1 = time.time()
    st.success(f"Done in {t1 - t0:.1f} seconds.")

# -------------------------------------------------------------------
# Sensitivity analysis section (uses stored base results)
# -------------------------------------------------------------------
forecasts = st.session_state.get("base_forecasts")
alpha_preds = st.session_state.get("base_alpha")
stress_fc = st.session_state.get("base_stress")
modeling = st.session_state.get("modeling_frame")
forecaster = st.session_state.get("forecaster_obj")
stress = st.session_state.get("stress_model_obj")

if forecasts and modeling is not None and forecaster and stress:
    st.markdown("---")
    st.subheader("Sensitivity analysis")

    st.caption(
        "Apply macro shocks to the latest feature vector and see how "
        "factor premium forecasts and stress probability would change."
    )

    SENS_KEYS = ["scn_rates", "scn_102y", "scn_103m", "scn_credit", "scn_vix"]

    def _reset_sensitivity():
        # Remove keys so sliders reinitialize to defaults on next run
        for k in SENS_KEYS:
            if k in st.session_state:
                del st.session_state[k]

    col_reset, col_run = st.columns([1, 1])
    with col_reset:
        if st.button("Reset shocks to 0"):
            _reset_sensitivity()
            st.experimental_rerun()
    with col_run:
        run_scen_btn = st.button("Run sensitivity scenario")

    colL, colR = st.columns([1, 1])
    with colL:
        shock_rates = st.slider(
            "Rates level shock (bps)",
            min_value=-300,
            max_value=300,
            value=0,
            step=5,
            key="scn_rates",
        )
        shock_term_10y2y = st.slider(
            "10y-2y term spread shock (bps)",
            min_value=-200,
            max_value=200,
            value=0,
            step=5,
            key="scn_102y",
        )
        shock_term_10y3m = st.slider(
            "10y-3m term spread shock (bps)",
            min_value=-300,
            max_value=300,
            value=0,
            step=5,
            key="scn_103m",
        )
    with colR:
        shock_credit = st.slider(
            "Credit spread level shock (bps)",
            min_value=-300,
            max_value=300,
            value=0,
            step=5,
            key="scn_credit",
        )
        shock_vix = st.slider(
            "VIX level shock (%)",
            min_value=-50,
            max_value=200,
            value=0,
            step=5,
            key="scn_vix",
        )

    if run_scen_btn:
        shocks = {
            "rates_bp": shock_rates,
            "term_10y2y_bp": shock_term_10y2y,
            "term_10y3m_bp": shock_term_10y3m,
            "credit_bp": shock_credit,
            "vix_pct": shock_vix,
        }

        # Factor scenarios
        scen_rows = []
        for f, base_fc in forecasts.items():
            scen_fc = scenario_factor_premia(
                forecaster,
                modeling,
                f,
                shocks=shocks,
            )
            if scen_fc is None:
                continue

            base_val = base_fc["ensemble_forecast"] * 100.0
            scen_val = scen_fc["ensemble_forecast"] * 100.0
            delta_bp = (scen_val - base_val) * 100.0

            scen_rows.append(
                {
                    "Factor": f,
                    "Base premium %": base_val,
                    "Scenario premium %": scen_val,
                    "Delta (bp)": delta_bp,
                }
            )

        if scen_rows:
            st.markdown("**Factor premium scenario results**")
            df_scen = pd.DataFrame(scen_rows).sort_values(
                "Factor"
            )
            st.dataframe(
                df_scen.style.format(
                    {
                        "Base premium %": "{:.2f}",
                        "Scenario premium %": "{:.2f}",
                        "Delta (bp)": "{:.1f}",
                    }
                ),
                use_container_width=True,
            )

        # Stress scenario
        scen_stress = scenario_stress(stress, modeling, shocks=shocks)
        if scen_stress and stress_fc:
            base_prob = stress_fc["stress_probability"] * 100.0
            scen_prob = scen_stress["stress_probability"] * 100.0
            delta_prob = scen_prob - base_prob

            st.markdown("**Stress probability scenario result**")
            st.write(
                f"Base regime **{stress_fc['regime']}** "
                f"({base_prob:.1f}%)  ->  "
                f"Scenario regime **{scen_stress['regime']}** "
                f"({scen_prob:.1f}%), "
                f"change **{delta_prob:.1f} percentage points**."
            )
else:
    st.markdown("---")
    st.info(
        "Run the pipeline first to enable sensitivity analysis "
        "based on the latest forecasts."
    )
