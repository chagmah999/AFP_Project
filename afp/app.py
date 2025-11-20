import os
import json
import time
import pandas as pd
import streamlit as st

from afp_app.config import (
    FMP_API_KEY, DEFAULT_START_DATE, DEFAULT_UNIVERSE_SIZE,
    LOOKBACK_DAYS
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
from afp_app.sensitivity import compute_latest_betas, factor_delta_map, reproject_alpha_for_top10

st.set_page_config(page_title="AFP Forecasting Tool", layout="wide")

# ---------------- Session state bootstrapping ---------------- #
for k, v in {
    "pipeline_ready": False,
    "pipeline_time": None,
    "tickers": None,
    "fundamentals": None,
    "prices": None,
    "metrics": None,
    "portfolios": None,
    "factor_returns": None,
    "macro": None,
    "modeling": None,
    "forecaster": None,
    "forecasts": None,
    "alpha_model": None,
    "alpha_preds": None,
    "stress_model": None,
    "stress_fc": None,
    "recs": None,
    "top10_tickers": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("AFP Forecasting Tool")
st.caption("Factor premia forecasts, per-ticker alpha, and market stress regime")

# ---------------- Sidebar controls ---------------- #
with st.sidebar:
    st.subheader("Configuration")
    api_key = st.text_input("FMP API Key", value=FMP_API_KEY or "", type="password")
    start_date = st.text_input("Start date (YYYY-MM-DD)", value=DEFAULT_START_DATE)

    st.markdown("**Universe**")
    universe_size = st.slider("Universe size", 10, 509, DEFAULT_UNIVERSE_SIZE, step=5)
    randomize = st.checkbox("Randomize universe selection", value=True)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.markdown("**Forecasting**")
    forecast_horizon = st.slider("Forecast horizon (days)", 5, 63, 21, step=1)
    top_k_drivers = st.radio("Top drivers to show (per factor/stock)", [3, 5], index=0)

    run_btn = st.button("Run pipeline", type="primary")

status = st.empty()
log = st.container()

# ---------------- Heavy pipeline, called only on button ---------------- #
def run_pipeline():
    t0 = time.time()
    if not api_key or api_key == "YOUR_FMP_API_KEY":
        st.error("Please set a valid FMP API key.")
        st.stop()

    # Universe
    status.info("Building universe...")
    tickers = get_universe(universe_size, randomize=randomize, seed=int(seed))

    # Data collection
    status.info("Fetching fundamentals and prices...")
    fetcher = FMPDataFetcher(api_key=api_key)
    fundamentals = collect_fundamental_data(tickers, start_date, fetcher)
    prices = collect_price_data(tickers, start_date, None, fetcher)

    if prices.empty:
        st.error("No price data returned. Check API key, tickers, or date range.")
        st.stop()

    # Factor metrics & portfolios
    status.info("Computing factor metrics...")
    metrics = calculate_factor_metrics(fundamentals, prices)
    ctor = FactorPortfolioConstructor(metrics, prices)
    portfolios = ctor.construct_all()
    factor_returns = ctor.calculate_factor_returns(start_date, prices["date"].max().strftime("%Y-%m-%d"))

    # Macro features & modeling frame
    status.info("Fetching macro data...")
    m = MacroDataFetcher(api_key=api_key)
    macro = {
        "treasury": m.fetch_treasury_rates(from_date=start_date),
        "vix": m.fetch_vix(from_date=start_date),
        "credit": m.fetch_credit_spreads(from_date=start_date),
    }

    status.info("Preparing modeling frame...")
    modeling = prepare_modeling_data(factor_returns, macro, prices)

    # Signal 1: Factor premia
    status.info("Forecasting factor premia...")
    forecaster = FactorPremiaForecaster(lookback_window=LOOKBACK_DAYS, forecast_horizon=forecast_horizon)
    factors = ["VALUE", "QUALITY", "MOMENTUM", "LOW_VOL"]
    forecasts = {}
    for f in factors:
        _ = forecaster.walk_forward_validation(modeling, f)  
        fc = forecaster.forecast_next(modeling, f)
        if fc:
            fc["top_drivers"] = (fc.get("top_drivers") or [])[:top_k_drivers]
            forecasts[f] = fc

    # Signal 2: Alpha
    status.info("Predicting per-ticker alpha...")
    alpha_model = AlphaPredictor(factor_returns, fundamentals, prices, horizon=forecast_horizon, lookback=252*2)
    alpha_preds = {}
    cap = min(100, len(tickers))  # UI cap
    for tk in tickers[:cap]:
        p = alpha_model.predict_alpha(tk, horizon=forecast_horizon)
        if p:
            p["drivers"]["top_features"] = (p["drivers"].get("top_features") or [])[:top_k_drivers]
            alpha_preds[tk] = p

    # Signal 3: Stress
    status.info("Estimating market stress probability...")
    stress_model = StressProbabilityModel()
    _ = stress_model.fit(modeling)
    stress_fc = stress_model.predict(modeling)

    # Integrate
    status.info("Integrating recommendations...")
    engine = MarketMancerEngine(forecasts, alpha_preds, stress_fc or {})
    recs = engine.generate()

    # Build top10 list for later sensitivity use
    if alpha_preds:
        df_alpha = pd.DataFrame([{
            "ticker": tk,
            "expected_alpha_%": v["expected_alpha"] * 100.0,
            "fundamental_score": v["drivers"]["fundamental_score"],
            "top_features": v["drivers"].get("top_features", [])
        } for tk, v in alpha_preds.items()]).sort_values("expected_alpha_%", ascending=False)
        top10_tickers = df_alpha.head(10)["ticker"].tolist()
    else:
        df_alpha = pd.DataFrame()
        top10_tickers = []

    t1 = time.time()

    # Persist everything
    st.session_state.pipeline_ready = True
    st.session_state.pipeline_time = round(t1 - t0, 1)
    st.session_state.tickers = tickers
    st.session_state.fundamentals = fundamentals
    st.session_state.prices = prices
    st.session_state.metrics = metrics
    st.session_state.portfolios = portfolios
    st.session_state.factor_returns = factor_returns
    st.session_state.macro = macro
    st.session_state.modeling = modeling
    st.session_state.forecaster = forecaster
    st.session_state.forecasts = forecasts
    st.session_state.alpha_model = alpha_model
    st.session_state.alpha_preds = alpha_preds
    st.session_state.stress_model = stress_model
    st.session_state.stress_fc = stress_fc
    st.session_state.recs = recs
    st.session_state.top10_tickers = top10_tickers
    # also store df_alpha for display reuse
    st.session_state.df_alpha = df_alpha


# ---------------- Trigger pipeline when button is pressed ---------------- #
if run_btn:
    run_pipeline()

# ---------------- Always show last pipeline results, if any ---------------- #
if st.session_state.pipeline_ready:
    # Universe
    st.subheader("Universe")
    tickers = st.session_state.tickers
    st.write(f"Universe of {len(tickers)} tickers:")
    st.dataframe(pd.DataFrame({"ticker": tickers}), use_container_width=True)

    # Data status
    prices = st.session_state.prices
    st.success(f"Collected {len(prices)} price rows. Date range: {prices['date'].min()} to {prices['date'].max()}")

    # Portfolios built
    st.subheader("Portfolios built")
    portfolios = st.session_state.portfolios or {}
    st.json({k: 0 if v is None or (hasattr(v, 'empty') and v.empty) else len(v) for k, v in portfolios.items()})

    # Factor premia forecasts
    st.subheader("Factor premia forecasts")
    forecasts = st.session_state.forecasts or {}
    if forecasts:
        summary_rows, drivers_rows = [], []
        for f, v in forecasts.items():
            summary_rows.append({"Factor": f, "Expected Premium %": v["ensemble_forecast"] * 100.0})
            for d in (v.get("top_drivers") or []):
                drivers_rows.append({"Factor": f, "Driver": d.get("feature"), "RF Importance": d.get("rf_importance")})
        df_summary = pd.DataFrame(summary_rows).sort_values("Expected Premium %", ascending=False)
        st.dataframe(df_summary.style.format({"Expected Premium %": "{:.2f}"}), use_container_width=True)
        if drivers_rows:
            st.markdown("Top drivers per factor")
            df_drivers = pd.DataFrame(drivers_rows)
            st.dataframe(df_drivers.style.format({"RF Importance": "{:.3f}"}), use_container_width=True)
    else:
        st.info("No factor forecasts available.")

    # Alpha predictions
    st.subheader("Alpha predictions (top 10)")
    alpha_preds = st.session_state.alpha_preds or {}
    if alpha_preds and hasattr(st.session_state, "df_alpha") and not st.session_state.df_alpha.empty:
        df_alpha = st.session_state.df_alpha
        show_top = df_alpha.head(10)[["ticker", "expected_alpha_%", "fundamental_score"]]
        st.dataframe(show_top.style.format({"expected_alpha_%": "{:.2f}"}), use_container_width=True)

        st.markdown("Top drivers for each of the top 10 stocks")
        for _, row in df_alpha.head(10).iterrows():
            with st.expander(f"{row['ticker']} — {row['expected_alpha_%']:.2f}%"):
                feats = row["top_features"]
                if feats:
                    df_feats = pd.DataFrame(feats)
                    if "coef" in df_feats.columns:
                        df_feats = df_feats.rename(columns={"coef": "Coefficient"})
                    st.dataframe(df_feats, use_container_width=True)
                else:
                    st.write("No feature importances available for this ticker.")
    else:
        st.info("No alpha predictions available.")

    # Stress regime
    st.subheader("Stress regime")
    stress_fc = st.session_state.stress_fc
    if stress_fc:
        st.write(f"Regime: **{stress_fc['regime']}**  |  Stress probability: **{stress_fc['stress_probability']*100:.1f}%**")

    # Integrated recommendations
    st.subheader("Integrated recommendations")
    st.json(st.session_state.recs or {})

    st.success(f"Finished pipeline in {st.session_state.pipeline_time} seconds.")

    # ---------------- Sensitivity ---------------- #
    st.markdown("---")
    st.header("Sensitivity analysis")
    st.caption("Adjust macro inputs and see impact on factor premia and on the top-10 alphas. No retraining or refetching.")

    # ---- Sensitivity reset ----
    if "sens_version" not in st.session_state:
        st.session_state.sens_version = 0

    def _reset_sensitivity():
        st.session_state.sens_version += 1

    st.button("Reset sensitivity to defaults", on_click=_reset_sensitivity)

    def skey(name: str) -> str:
        # Widget keys depend on current version
        return f"{name}_v{st.session_state.sens_version}"

    colL, colR = st.columns([1, 1])
    with colL:
        shock_rates      = st.slider("Rates level shock (bps)", -300, 300, 0, step=5, key=skey("scn_rates"))
        shock_term_10y2y = st.slider("10y-2y term spread shock (bps)", -200, 200, 0, step=5, key=skey("scn_102y"))
        shock_term_10y3m = st.slider("10y-3m term spread shock (bps)", -300, 300, 0, step=5, key=skey("scn_103m"))
    with colR:
        shock_credit     = st.slider("Credit spread level shock (bps)", -300, 300, 0, step=5, key=skey("scn_credit"))
        shock_vix        = st.slider("VIX level shock (%)", -50, 200, 0, step=5, key=skey("scn_vix"))



    # Compute scenarios from saved objects on every interaction, without re-running pipeline
    shocks = {
        "rates_bp":        shock_rates,
        "term_10y2y_bp":   shock_term_10y2y,
        "term_10y3m_bp":   shock_term_10y3m,
        "credit_bp":       shock_credit,
        "vix_pct":         shock_vix,
    }

    # Scenario factor forecasts
    forecaster = st.session_state.forecaster
    modeling = st.session_state.modeling
    base_forecasts = st.session_state.forecasts
    scen_fc = {}
    for f in ["VALUE", "QUALITY", "MOMENTUM", "LOW_VOL"]:
        scen = scenario_factor_premia(forecaster, modeling, f, shocks=shocks)
        if scen:
            scen_fc[f] = scen

    if scen_fc:
        factor_rows = []
        for f in ["VALUE", "QUALITY", "MOMENTUM", "LOW_VOL"]:
            base_er = base_forecasts.get(f, {}).get("ensemble_forecast", None)
            scen_er = scen_fc.get(f, {}).get("ensemble_forecast", None)
            if scen_er is None:
                continue
            factor_rows.append({
                "Factor": f,
                "Base ER %": None if base_er is None else base_er * 100.0,
                "Scenario ER %": scen_er * 100.0,
                "Delta (bp)": None if base_er is None else (scen_er - base_er) * 10000.0
            })
        if factor_rows:
            df_factors_scen = pd.DataFrame(factor_rows).sort_values("Scenario ER %", ascending=False)
            st.dataframe(
                df_factors_scen.style.format({"Base ER %": "{:.2f}", "Scenario ER %": "{:.2f}", "Delta (bp)": "{:.1f}"}),
                use_container_width=True
            )

    # Reproject top-10 alphas using factor deltas and betasx
    prices = st.session_state.prices
    factor_returns = st.session_state.factor_returns
    tickers = st.session_state.tickers
    top10_tickers = st.session_state.top10_tickers or []
    alpha_preds = st.session_state.alpha_preds or {}

    if top10_tickers:
        betas_wide = compute_latest_betas(prices, factor_returns, tickers, window=252)
        dER = factor_delta_map(base_forecasts, scen_fc)
        if not betas_wide.empty and dER:
            df_alpha_scen = reproject_alpha_for_top10(top10_tickers, alpha_preds, betas_wide, dER)
            st.markdown("Top-10 alpha under scenario")
            st.dataframe(
                df_alpha_scen.style.format({"base_alpha_%": "{:.2f}", "scenario_alpha_%": "{:.2f}", "delta_bp": "{:.1f}"}),
                use_container_width=True
            )
        else:
            st.info("Could not compute scenario alpha. Make sure factor deltas and betas are available.")
    else:
        st.info("No top-10 list available to project alpha.")

    st.caption("Alpha sensitivity = base alpha + sum(beta_f × Δfactor_ER_f). All other variables held constant.")

else:
    st.info("Set options in the sidebar and click Run pipeline.")
