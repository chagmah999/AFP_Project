from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List

def compute_latest_betas(
    prices: pd.DataFrame,
    factor_returns: pd.DataFrame,
    tickers: List[str],
    window: int = 252
) -> pd.DataFrame:

    if prices.empty or factor_returns.empty:
        return pd.DataFrame()

    px = prices[['date','ticker','returns']].dropna().copy()
    px['date'] = pd.to_datetime(px['date'])
    last_date = px['date'].max()
    start = last_date - pd.Timedelta(days=window * 2)  

    px = px[(px['date'] >= start) & (px['date'] <= last_date)]

    fac_wide = (
        factor_returns
        .pivot_table(index='date', columns='factor', values='return')
        .sort_index()
        .dropna(how='all')
    )

    fac_wide = fac_wide[(fac_wide.index >= start) & (fac_wide.index <= last_date)]
    wanted_factors = [c for c in ["VALUE","QUALITY","MOMENTUM","LOW_VOL"] if c in fac_wide.columns]
    if not wanted_factors:
        return pd.DataFrame()

    rows = []
    for tk in tickers:
        s = px[px['ticker'] == tk][['date','returns']].set_index('date').sort_index()
        if s.empty:
            continue
        df = s.join(fac_wide[wanted_factors], how='inner').dropna()
        if len(df) < max(60, int(window * 0.5)):  

            continue

        y = df['returns'].values.reshape(-1, 1)
        X = df[wanted_factors].values
        XtX = X.T @ X
        try:
            betas = np.linalg.solve(XtX, X.T @ y).flatten()
        except np.linalg.LinAlgError:

            betas = np.linalg.pinv(XtX) @ X.T @ y
            betas = betas.flatten()
        row = {'ticker': tk}
        for f, b in zip(wanted_factors, betas):
            row[f] = float(b)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index('ticker')

    for f in ["VALUE","QUALITY","MOMENTUM","LOW_VOL"]:
        if f not in out.columns:
            out[f] = np.nan
    return out[wanted_factors]

def factor_delta_map(
    base_fc: Dict[str, dict],
    scen_fc: Dict[str, dict]
) -> Dict[str, float]:

    out: Dict[str, float] = {}
    for f in ["VALUE","QUALITY","MOMENTUM","LOW_VOL"]:
        base_er = base_fc.get(f, {}).get("ensemble_forecast", None)
        scen_er = scen_fc.get(f, {}).get("ensemble_forecast", None)
        if base_er is None or scen_er is None:
            continue
        out[f] = float(scen_er - base_er)  

    return out

def reproject_alpha_for_top10(
    top10: List[str],
    alpha_preds: Dict[str, dict],
    betas_wide: pd.DataFrame,
    dER: Dict[str, float]
) -> pd.DataFrame:

    rows = []
    for tk in top10:
        base_alpha = alpha_preds.get(tk, {}).get("expected_alpha", None)
        if base_alpha is None:
            continue

        adj = 0.0
        if tk in betas_wide.index:
            for f, delta in dER.items():
                beta_f = betas_wide.at[tk, f] if f in betas_wide.columns else np.nan
                if pd.notna(beta_f):
                    adj += float(beta_f) * float(delta)
        scen_alpha = base_alpha + adj
        rows.append({
            "ticker": tk,
            "base_alpha_%": base_alpha * 100.0,
            "scenario_alpha_%": scen_alpha * 100.0,
            "delta_bp": (scen_alpha - base_alpha) * 10000.0
        })
    return pd.DataFrame(rows).sort_values("scenario_alpha_%", ascending=False)

