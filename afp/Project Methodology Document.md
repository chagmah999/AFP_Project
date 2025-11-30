**AFP Forecasting Tool**

**Full Methodology and Implementation Details (Current Version)**

This document describes the current version of the AFP forecasting tool in enough detail that every visible feature in the app can be explained mathematically and intuitively.

The overall story the tool now tells is:

1.  Build a stock universe and compute academic style factor scores for each stock
2.  Turn those scores into long short factor portfolios and daily factor premia
3.  Use macro data plus lagged factor behavior to forecast factor premia, with AR(1) as the main forecast
4.  Use fundamentals and technicals to estimate stock level H day alpha
5.  Combine those alphas with a Ledoit Wolf covariance matrix to build a constrained long/short “unified” portfolio
6.  Surface a small set of interpretable tables: factor forecasts, macro drivers, validation, universe and factor scores, unified portfolio, and top alpha names


----------

**1. Time Index, Horizon, and Universe**

**1.1 Time index and forecast horizon**

The user sets:

-   Start date $S$
-   Forecast horizon $H$ trading days (e.g. $H = 21$ )

If $S$ is not a trading day, the system uses the first trading day on or after $S$.

Define a sequence of trading days

$$  
t = 1, 2, \dots, T  
$$

where $t = 1$ is the first trading day on or after $S$, and $t = T$ is the last day for which we have prices and macro data.

All returns, factor premia, macro variables, and model targets are indexed by this time index $t$.

----------

**1.2 Universe construction**

We start from a dynamic large cap universe based on the current S&P 500 constituents. The tool queries FMP’s S&P 500 constituent endpoint in real time to obtain the live list of tickers. If that API call fails for any reason, it falls back to a hardcoded backup list of S&P 500 names stored in the code, so the universe selection remains stable and the tool continues to run.

The user chooses a universe size $U$ between 10 and 509.

-   If **Randomize universe selection** is **unchecked**:  
	   - We sort the full S&P 500 universe alphabetically by ticker symbol and take the first $U$ tickers from that sorted list.
-   If **Randomize universe selection** is **checked**:

	-   We use the user provided integer seed.
	-   We randomly sample $U$ tickers from the full list using that seed.
	-   This means the same seed produces the same random universe, which makes the tool run reproducibly.

The final ordered list of universe tickers is stored and later displayed (inside an expander) for transparency.

----------

**2. Data for Each Stock**

For each stock $i \in \{1,\dots,U\}$ and trading day $t$:

**2.1 Prices and returns**

We pull adjusted close prices $P_{i,t}$ from FMP via FMPDataFetcher.

Daily log return:

$$  
r_{i,t} = \ln P_{i,t} - \ln P_{i,t-1}  
$$

Log returns are preferred because they are approximately additive over time and work cleanly with rolling windows.

----------

**2.2 Fundamentals (point in time)**

For each stock and each date $t$, we pull the latest available financials as of $t$ from FMP:

-   $totalStockholdersEquity_{i,t}$
-   $totalAssets_{i,t}$
-   $totalDebt_{i,t}$ (or $shortTermDebt_{i,t} + longTermDebt_{i,t}$ if the $totalDebt$ field is not available)

-   $netIncome_{i,t}$
-   $revenue_{i,t}$
-   $grossProfit_{i,t}$
-   $freeCashFlow_{i,t}$
-   Shares outstanding $Shares_{i,t}$

We also compute:

$$  
\text{MarketCap}_{i,t} = P_{i,t} \times \text{Shares}_{i,t}  
$$

From these we construct standard value and quality metrics.

**Value style metrics**

1.  **Book to price**

$$  
BP_{i,t} = \frac{\text{totalStockholdersEquity}_{i,t}}{\text{MarketCap}_{i,t}}  
$$

2.  **Earnings to price**

$$  
EP_{i,t} = \frac{\text{netIncome}_{i,t}}{\text{MarketCap}_{i,t}}  
$$

3.  **Free cash flow to price**

$$  
FCFP_{i,t} = \frac{\text{freeCashFlow}_{i,t}}{\text{MarketCap}_{i,t}}  
$$

Higher values mean the stock is “cheaper” relative to its fundamentals.

**Quality style metrics**

4.  **Return on equity (ROE)**

$$  
ROE_{i,t} = \frac{\text{netIncome}_{i,t}}{\text{totalStockholdersEquity}_{i,t}}  
$$

5.  **Return on assets (ROA)**

$$  
ROA_{i,t} = \frac{\text{netIncome}_{i,t}}{\text{totalAssets}_{i,t}}  
$$

6.  **Gross margin (GM)**

$$  
GM_{i,t} = \frac{\text{grossProfit}_{i,t}}{\text{revenue}_{i,t}}  
$$

7.  **Free cash flow margin (FCFMargin)**

$$  
FCFMargin_{i,t} = \frac{\text{freeCashFlow}_{i,t}}{\text{revenue}_{i,t}}  
$$

8.  **Leverage (Lev)**

$$  
Leverage_{i,t} = \frac{\text{totalDebt}_{i,t}}{\text{totalStockholdersEquity}_{i,t}}  
$$

9.  **Inverse leverage (LevInv)**

$$  
LevInv_{i,t} = - Leverage_{i,t}  
$$

Lower leverage (safer balance sheet) gives a higher LevInv.



**3. Factor Scores per Stock**

We define four factors for each stock on each date $t$:

-   Value
-   Quality
-   Momentum (12 minus 1 month)
-   Low Volatility

Each factor is built from an intuitive combination of standardized metrics.

**3.1 Cross sectional z score**

Given a raw metric $x_{i,t}$ across all stocks $i$ at date $t$:

$$  
\bar{x}_t = \frac{1}{U} \sum_{i=1}^U x_{i,t},  
\quad  
s_t(x) = \sqrt{\frac{1}{U-1}\sum_{i=1}^U (x_{i,t} - \bar{x}_t)^2}  
$$

The z score is:

$$  
z_{i,t}(x) = \frac{x_{i,t} - \bar{x}_t}{s_t(x)}  
$$

We use this operator on each of the raw fundamental or statistical metrics so that each component has mean zero and unit variance at each date.

----------

**3.2 Value factor**

Use three valuation ratios:

-   $BP_{i,t}, EP_{i,t}, FCFP_{i,t}$

Standardized:

$$  
z_{i,t}(BP), \quad z_{i,t}(EP), \quad z_{i,t}(FCFP)  
$$

Value factor score:


$$  
Value_{i,t} = \frac{1}{3} \Big[z_{i,t}(BP) + z_{i,t}(EP) + z_{i,t}(FCFP)\Big]
$$

Higher Value indicates cheaper stocks relative to peer fundamentals.

----------

**3.3 Quality factor**

Use five quality and balance sheet metrics:

-   $ROE_{i,t}, ROA_{i,t}, GM_{i,t}, FCFMargin_{i,t}, LevInv_{i,t}$

Standardized:

$$  
z_{i,t}(ROE), z_{i,t}(ROA), z_{i,t}(GM), z_{i,t}(FCFMargin), z_{i,t}(LevInv)  
$$

Quality factor score:

$$  
Quality_{i,t} = \frac{1}{5}  
\Big(  
z_{i,t}(ROE) + z_{i,t}(ROA) + z_{i,t}(GM) + z_{i,t}(FCFMargin) + z_{i,t}(LevInv)  
\Big)  
$$

Higher Quality means the company looks stronger on profitability, margins, and leverage.

----------

**3.4 Momentum factor (12 minus 1)**

We approximate 12 months as 252 trading days and 1 month as 21 trading days.

Raw momentum for stock $i$ at time $t$:

$$  
MomRaw_{i,t} = \ln P_{i,t-21} - \ln P_{i,t-252}  
$$

This is the log return from 12 months ago up to 1 month ago (excluding the most recent month).

Standardize:

$$  
Momentum_{i,t} = z\big(MomRaw_{i,t}\big)  
$$

Higher Momentum means stronger performance over the prior year excluding the most recent month.

----------

**3.5 Low Volatility factor**

Define realized volatility over a 63 day window:

$$  
vol_{i,t} = \text{standard deviation of } { r_{i,t-62}, \dots, r_{i,t} }  
$$

Then invert and standardize:

$$  
LowVol_{i,t} = -z\big(vol_{i,t}\big)  
$$

Low volatility stocks get higher LowVol scores; high volatility stocks get lower scores.

**3.6 Industry / sector adjustment:**  

In the current implementation, each stock’s individual factor scores for all four factors are built to be sector or industry neutral whenever that information is available from FMP. After merging fundamentals, the code checks whether a 'sector' column exists; if so, it groups stocks by sector. If 'sector' is missing but 'industry' is present, it groups by industry instead. Only if both are missing does it treat the entire universe as a single group. 

Within each group, it then converts the raw building blocks for each factor into relative scores using percentile ranks or groupwise z scores. 

For VALUE, it computes book to price, earnings to price, and free cash flow to price, converts each of those to z scores within sector or industry, averages them into a single 'value_raw' measure, and finally maps 'value_raw' to a 0 to 1 percentile rank within that group so that 1 corresponds to the cheapest names in that sector. For QUALITY, it takes ROE, ROA, margins, and leverage, converts each to a percentile rank within sector or industry (with lower leverage inverted so that lower leverage gets a higher score), and then averages those ranks to form the 0 to 1 'quality_score'. The same grouping logic applies to LOW VOL and MOMENTUM. For LOW VOL, the model first estimates each stock’s 60 day realized volatility from daily returns, then ranks that volatility within its sector or industry and inverts the percentile so that low volatility stocks receive scores closer to 1. For MOMENTUM, it computes each stock’s 60 day price change and then converts that momentum into a percentile rank within sector or industry, where higher recent performance corresponds to a higher 'momentum_score'.

In short, whenever sector or industry labels are present, a stock’s factor score describes how it ranks compared to peers in the same sector or industry, rather than to the entire universe. Only if no sector or industry metadata is available do these percentile ranks default to being computed cross sectionally over all names together.


----------

**4. Factor Portfolios and Daily Factor Premia**

Once we have factor scores, we construct long short factor portfolios and derive their daily returns.

**4.1 Factor portfolio construction**

For each factor $f \in {\text{Value, Quality, Momentum, LowVol}}$ and date $t$:

1.  Rank stocks by $Score_{f,i,t}$ (for example $Value_{i,t}$ for the Value factor).
2.  Define a long set $L_f(t)$ as the top part of the distribution (currently fixed at top 30 percent of scores).
3.  Define a short set $S_f(t)$ as the bottom part (currently fixed at bottom 30 percent).

Weights (equal weighted):

$$  
w_{i,f,t} =  
\begin{cases}  
\frac{1}{|L_f(t)|}, & i \in L_f(t) \  
\newline -\frac{1}{|S_f(t)|}, & i \in S_f(t) \  
\newline 0, & \text{otherwise}  
\end{cases}  
$$

These weights sum to approximately zero (long and short dollar exposure roughly offset).

The factor portfolio is rebalanced with this procedure at the chosen frequency (in the current implementation, rebalancing is done at every date where inputs are updated, typically daily or when new scores are available).

----------

**4.2 Daily factor premium**

Daily factor premium for factor $f$ on day $t$:

$$  
fp_{f,t} = \sum_i w_{i,f,t} , r_{i,t}  
$$

This is the realized return of the long short factor portfolio.

----------

**4.3 Forward H day factor premium (target)**

We want to forecast the **average factor premium over the next H days**.

For factor $f$ and date $t$:

$$  
y_{f,t} = \frac{1}{H} \sum_{k=1}^{H} fp_{f,t+k}  
$$

This is the target variable used in factor premia forecasting.

Because we use overlapping windows, the target series is smoother and we get more training points, at the cost of some autocorrelation in the target.

----------

**5. Macro Features for Factor Forecasting**

The factor premia forecasting model uses a small, interpretable set of macro time series that summarize interest rate conditions, the shape of the yield curve, corporate credit risk, and market volatility. On each trading day $t$, we construct the following features:

$$  
\text{Macro features on day } t =  
\big(  
\text{rates\_level}_t,\  
\text{rates\_1m\_change}_t,\  
\text{term\_spread10y2y}_t,\  
\newline \text{term\_spread10y3m}_t,\  
\text{credit\_spread\_level}_t,\  
\text{credit\_spread\_1m\_change}_t,\  
\text{vix\_close}_t,\  
\text{vix\_percentile}_t  
\big).  
$$

These are built or extracted from raw inputs pulled from FMP data:

-   Treasury yields at 3 months, 2 years, 5 years, and 10 years:  
    $$  
    R_{3M,t},\ R_{2Y,t},\ R_{5Y,t},\ R_{10Y,t}  
    $$
-   Prices of bond and credit ETFs:

	-   $\text{TLT}_t$: long duration Treasuries
	-   $\text{HYG}_t$: high yield corporate credit
	-   $\text{LQD}_t$: investment grade corporate credit

-   VIX closing level:  
    $$  
    \text{VIX}_t  
    $$

Below, we define each macro feature mathematically, explain what economic concept it captures, and describe how it can affect factor premia.

----------

**5.1 Rates level**

**Definition**

On each day $t$, the model builds a single composite “rate level” feature that summarizes the overall level of risk-free yields across short and intermediate maturities. Concretely, it takes a simple average of whatever yields are available among the 3-month, 2-year, 5-year, and 10-year Treasuries:
$$  
\text{rates\_level}_t  
= \text{average of } { R_{3M,t}, R_{2Y,t}, R_{5Y,t}, R_{10Y,t} }  
$$

If any of these yields are missing on a given day, the model uses the average of the yields that are available.

These yields are pulled from the FMP Treasury endpoint, which provides nominal, constant maturity Treasury yields (not TIPS yields and not auction stop out rates).

**Economic meaning**

This is a summary of the overall level of interest rates, which reflects the general cost of capital and the stance of monetary policy.

-   High $\text{rates\_level}_t$: tighter policy, higher discount rates, more pressure on valuations.
-   Low $\text{rates\_level}_t$: looser policy, lower discount rates, more support for valuations.

**Typical effect on factors**

-   Higher rate levels tend to hurt long duration equity styles like growth and quality, since more of their value is in distant cash flows.
-   Value can be relatively less sensitive or sometimes even benefit on a relative basis, because cheaper stocks are less affected by discount rate changes.
-   A high rates level often coincides with tighter financial conditions and can lead to weaker factor premia in riskier styles.

----------

**5.2 One month change in rates**

**Definition**

We capture the recent speed and direction of rate changes via the one month (approximately 21 trading day) change:

$$  
\text{rates\_1m\_change}_t  
= \text{rates\_level}_t - \text{rates\_level}_{t-21}.  
$$

**Economic meaning**

This measures how quickly monetary conditions are tightening or easing.

-   Large positive values: sharp recent rate hikes.
-   Large negative values: rapid easing.

**Typical effect on factors**

-   Rapid increases in interest rates generally hurt growth, quality, and momentum factors because higher discount rates reduce the present value of long duration cash flows and tighten financial conditions, which weakens investor appetite for these styles.
-   Large hikes frequently precede or coincide with higher volatility and more unstable factor premia.
-   Persistent tightening tends to reduce risk appetite in general, which can change the relative performance of factors like Momentum and Low Vol.

----------

**5.3 Term spreads: 10Y minus 2Y and 10Y minus 3M**

**Definition**

We use two different slopes of the Treasury yield curve:

$$  
\text{term\_spread10y2y}_t = R_{10Y,t} - R_{2Y,t}  
$$  
$$  
\text{term\_spread10y3m}_t = R_{10Y,t} - R_{3M,t}.  
$$

The 10Y, 2Y, and 3M yields come from the same **FMP Treasury constant maturity yield** series used in the rates level calculation.
    
For each date $t$, both spreads are computed as the simple difference between the relevant yields on that same day (for example $R_{10Y,t} - R_{2Y,t}$​).
    
If one leg of the spread is missing (for example 10Y present but 2Y missing), the spread for that date is set to missing.
    
Later, when we construct features and targets, any rows with missing macro features (including missing spreads) are dropped before fitting the models. This avoids ad hoc interpolation of curve data.

**Economic meaning**

These measure the steepness or inversion of the yield curve and are classic recession indicators.

-   Positive spreads: steep curve, normally associated with healthy growth expectations.
-   Negative spreads (inversion): markets expect future rate cuts and higher recession risk.

**Typical effect on factors**

-   Inverted curves often favor more defensive styles such as Quality and Low Vol, since investors anticipate slower growth and seek resilience.
-   Value and more cyclical names can struggle when the curve is heavily inverted, since recession risk is priced in.
-   The model can learn patterns like “when the curve is inverted, certain factors become less attractive or more volatile.”

----------

**5.4 Credit spread level**

**Definition (conceptual)**

The goal of this feature is to summarize the gap between corporate borrowing costs and Treasury yields. Conceptually, one can think of it as an average spread between corporate bond yields and Treasury yields:

$\text{credit\_spread\_level}_t \approx y_{\text{corp},t} - y_{\text{Treasury},t}.$

In the current implementation, the code uses the macro credit series returned directly by FMP rather than constructing the spread from ETF prices.

The macro module calls FMP’s credit spread endpoint, which provides a broad High Yield corporate spread series (for example a high yield option adjusted spread in percent).
    
The tool uses that high yield spread series as $\text{credit\_spread\_level}_t$ whenever it is available.
    
If the high yield spread series is missing, the code falls back to whichever broad corporate spread level FMP exposes in the same endpoint (for example an investment grade OAS or BAA type series).
    
These spreads are already expressed as levels in percentage points, so the model uses them directly without further transformation for the main “level” feature.
    
 As with the Treasury data, credit spreads are merged to the equity trading calendar by date. Days where the spread is missing are later dropped when we construct the clean modeling frame.

**Economic meaning**

This captures how stressed corporate bond markets are:

-   Low values: tight spreads, favorable credit conditions.
-   High values: wide spreads, higher perceived default risk and tighter funding.

**Typical effect on factors**

When credit spreads are wide:

-   Risk appetite is generally lower.
-   High beta, growth, and momentum names tend to struggle.
-   Defensive factors and higher quality balance sheets often hold up better.
-   Factor premia can become more volatile, and riskier styles may have lower expected premia.

----------

**5.5 One month change in credit spreads**

**Definition**

We also track how quickly credit risk is changing:

$$  
\text{credit\_spread\_1m\_change}_t  
= \text{credit\_spread\_level}_t - \text{credit\_spread\_level}_{t-21}.  
$$

This is computed directly from the FMP credit spread level series described above.

If the level is missing on either $t$ or $t−21$, the one month change is missing and that row is later dropped when we build the modeling dataset.

**Economic meaning**

This measures short term deterioration or improvement in credit conditions.

-   Positive values: spreads have widened over the past month, indicating rising credit stress.
-   Negative values: spreads have tightened, indicating improving credit markets.

**Typical effect on factors**

-   Sudden widening is often an early sign of market stress.
-   Factor premia, especially in risk seeking factors such as Momentum, can become more fragile or experience drawdowns when spreads widen quickly.
-   Defensive styles and Low Vol often gain relative strength when credit spreads are widening.

----------

**5.6 VIX close**

**Definition**

The closing level of the CBOE VIX index on day $t$:

$$  
\text{vix\_close}_t = \text{VIX}_t.  
$$
The tool calls the FMP VIX endpoint, which returns daily VIX data that mirrors the official CBOE VIX index, but is served through the FMP API.
    
We use the daily closing VIX value as $\text{vix\_close}_t$

VIX values are merged to the equity trading calendar by date. Market holidays naturally have no VIX observations; if a trading day has no VIX print, that row is later dropped when missing macro features are removed.

    

**Economic meaning**

VIX is a forward looking measure of expected equity market volatility implied by S&P 500 options.

-   Low $\text{vix\_close}_t$: calm, low fear environment.
-   High $\text{vix\_close}_t$: stressed, high uncertainty environment.

**Typical effect on factors**

-   High VIX typically reduces demand for riskier factor exposures.
-   Low Vol, Quality, and other defensive styles tend to become more attractive when VIX is elevated.
-   Momentum can be vulnerable in very high VIX environments, where sharp reversals are common.

----------

**5.7 VIX percentile**

**Definition**

To capture how unusual today’s volatility is relative to the recent past, we compute a rolling percentile of VIX over the past 252 trading days:

$$  
\text{vix\_percentile}_t  
= \frac{\text{rank of } \text{VIX}_t \text{ among }  
{\text{VIX}_{t-251}, \dots, \text{VIX}_t}}{252}.  
$$

So $\text{vix\_percentile}_t = 0.80$ means that only 20 percent of the last 252 days had higher VIX levels than today.

**Economic meaning**

This normalizes volatility to the recent range:

-   High percentile (near 1): VIX is high compared to the last year, consistent with persistent stress.
-   Low percentile (near 0): VIX is low compared to the last year, consistent with sustained calm.

**Typical effect on factors**

-   When the VIX percentile is high, it means volatility has been high for a while. In these environments, investors usually prefer more defensive, lower risk stocks and factors.
    
-   When the percentile is low, conditions have been calm and investors are more willing to take risk. 

The model uses this history to learn how each factor’s expected return tends to change between “high volatility” and “low volatility” regimes.

----------

**5.8 How macro features enter the factor forecasting model**

For each factor $f \in {\text{VALUE}, \text{QUALITY}, \text{MOMENTUM}, \text{LOW\_VOL}}$ and each date $t$, the feature vector used for forecasting the forward $H$ day average factor premium includes:

-   The macro features above on date $t$:  
    $$  
    \text{rates\_level}_t,\ \text{rates\_1m\_change}_t,\  
    \text{term\_spread10y2y}_t,\ \text{term\_spread10y3m}_t,\  
    \newline \text{credit\_spread\_level}_t,\ \text{credit\_spread\_1m\_change}_t,\  
    \text{vix\_close}_t,\ \text{vix\_percentile}_t  
    $$
-   Lagged factor premiums and moving averages for that factor:  
    $$  
    fp_{f,t-1},\ fp_{f,t-5},\ fp_{f,t-21},\ fp_{f,t-63},\  
    \text{MA}_{21}(fp_{f})_t,\ \text{MA}_{63}(fp_{f})_t  
    $$  
    (as implemented in prepare_features_targets).

The AR(1) baseline uses the forward target’s own lag, while the Ridge, Lasso, and Random Forest models use the full macro plus lagged factor feature set to predict the forward $H$ day factor premium:

$$  
y_{f,t} = \frac{1}{H} \sum_{k=1}^{H} fp_{f,t+k}.  
$$

The AR(1) baseline uses only the forward target’s own lag $y_{f,t−1}$.

The Ridge, Lasso, and Random Forest models use the full set of macro features plus lagged factor returns and moving averages as inputs to predict $y_{f,t}$.

The macro variables provide the “state of the world” in which the factors operate, and the model learns statistical relationships such as:

-   Value premia tend to be higher in certain rate or curve environments.
-   Low Vol premia strengthen when volatility is elevated.
-   Momentum behaves differently when credit spreads are widening versus tightening.
This makes the factor forecasts sensitive to both micro (factor history) and macro (rates, credit, volatility) conditions in a transparent and economically intuitive way.
----------

### 6.1 Feature construction per factor

For each factor $f$:

* Start from the merged `modeling_frame`, which contains daily rows with:

* Date

* Macro features above

* Factor premium time series $fp_{f,t}$

* Build feature and target data as follows:

1. **Macro features**:

All variables listed in Section 5.

2. **Lagged factor premiums**:

For lags $L \in \{1, 5, 21, 63\}$:

$$
fp_{f,t-L}
$$

Each lag becomes a separate feature column.

3. **Moving averages of factor premiums**:

* 21 day moving average: $MA_{21,f,t}$

* 63 day moving average: $MA_{63,f,t}$

4. **Target**:

$$
y_{f,t} = \frac{1}{H} \sum_{k=1}^{H} fp_{f,t+k}
$$

We drop rows where either features or target are missing. The result is a clean feature matrix $X_f$ and target vector $y_f$.

---

**6.2 Walk forward validation: ensemble vs AR(1)**

For each factor we run walk_forward_validation:

-   Let $N = \text{len}(X_f)$
-   Choose number of splits n_splits = 5
-   Define a test window length:

$$  
\text{test\_size} = \left\lfloor\frac{N}{n\_splits + 1}\right\rfloor  
$$

For fold $i = 0,1,\dots,n\_splits-1$:

1.  Training indices: $0$ to train_end minus one, where:

$$  
\text{train\_end} = (i+1)\times \text{test\_size}  
$$

2.  Test indices: train_end to test_end minus one, where:

$$  
\text{test\_end} = \min(\text{train\_end} + \text{test\_size}, N)  
$$

We require at least 50 training points and 10 test points per fold. If the data is too short, that fold is skipped.

For each fold:

1.  **Ensemble training**

-   Standardize $X_{\text{train}}$ with StandardScaler.
-   Fit:

-   Ridge regression
-   Lasso regression
-   Random Forest regressor

-   Predict on $X_{\text{test}}$.
-   Ensemble prediction is the average of the three model predictions.

3.  **AR(1) baseline**

-   Define $y_{\text{train}} = y_f$ on training indices.
-   Define $y_{\text{train, lag}} = y_{\text{train shifted by 1}}$.
-   Fit:

$$  
y_{\text{train}} = a_f + b_f y_{\text{train, lag}} + \varepsilon  
$$

using a simple np.polyfit regression when enough non null points exist; otherwise fallback to a constant mean model.

-   For the test set, define $y_{\text{test, lag}}$ as the lagged target and fill missing lags with the last observed training value.
-   Compute AR(1) predictions:

$$  
\hat{y}_{\text{test}}^{AR(1)} = a_f + b_f y_{\text{test, lag}}  
$$

3.  **Metrics**

For each model (Ridge, Lasso, Random Forest, Ensemble, AR(1)) we compute:

-   RMSE (root mean squared error)
-   MAE (mean absolute error)
-   Hit rate, defined as:

$$  
\text{hit rate} = \text{mean}\big(\text{sign}(\hat{y}) = \text{sign}(y)\big)  
$$

where sign(0) is treated neutrally.

Fold level metrics are stored and then averaged across folds. The final stored summary for factor $f$ is:

-   ensemble_rmse, ensemble_mae, ensemble_hit_rate
-   ar1_rmse, ar1_mae, ar1_hit_rate

These feed the **Factor signal validation (walk forward)** table in the UI.

----------

**6.3 Ensemble models and feature importance**

While AR(1) is the primary forecast used in the main factor table, the tool also stores the full trained ensemble models and a Random Forest feature importance table.

For each factor $f$:

-   After training on the full data $X_f, y_f$, we compute a feature importance DataFrame with columns:

	-   feature (macro or lagged factor variable name)
	-   ridge_coef (absolute value of Ridge coefficient)
	-   lasso_coef (absolute value of Lasso coefficient)
	-   rf_importance (Random Forest feature importance)

-   This is sorted by rf_importance and the top k features (where k is the user selected number of drivers, usually 3 or 5) are exposed in the app as **Top drivers per factor**.

These RF importances are described in the UI as “how much this input helped reduce forecast error in the random forest model”, explicitly framed as **predictive associations**, not causal relationships.

----------

**6.4 Forecasting at the latest date**

For the latest date $T$, forecast_next performs:

1.  Build $X_f, y_f$ as above.
2.  Train the ensemble models on all rows.
3.  Take the last row of $X_f$ as the current state $X_{f,T}$.
4.  Standardize it and get model outputs:

$$  
\hat{y}_{f,T}^{ridge}, \quad \hat{y}_{f,T}^{lasso}, \quad \hat{y}_{f,T}^{rf}  
$$

5.  Ensemble forecast:

$$  
\hat{y}_{f,T}^{ensemble} = \frac{1}{3} \left(  
\hat{y}_{f,T}^{ridge} + \hat{y}_{f,T}^{lasso} + \hat{y}_{f,T}^{rf}  
\right)  
$$

6.  AR(1) forecast at $T$:

In parallel to walk forward, we also fit AR(1) on the full available forward premium series $y_f$:

$$  
y_{f,t} = a_f + b_f y_{f,t-1} + \varepsilon_t  
$$

Then the AR(1) forecast of the next H day premium is:

$$  
\hat{y}_{f,T}^{AR(1)} = a_f + b_f y_{f,T-1}  
$$

7.  Package into a dictionary:

	-   $ensemble\_forecast = \hat{y}_{f,T}^{ensemble}$
	-   $ar1\_forecast = \hat{y}_{f,T}^{AR(1)}$
	-   $model\_forecasts$ (per model)
	-   $top\_drivers$: top RF features as described above
	-   $forecast\_horizon\_days = H$
	-   $forecast\_date = date\_T$

The app then uses $ar1\_forecast$ in the main Factor premia table, and exposes $ensemble\_forecast$ and the detailed breakdown in a separate expander.

----------

**7. Universe and Stock Level Factor Scores (Display)**

As an explanatory check, the app provides an expander for:

-   **Universe size** and ticker list.
-   **Portfolio sizes by factor**: for each factor, how many names are in the long leg plus the short leg.
-   **Sample stock level factor scores** for up to 30 tickers:

	-   ticker, value_score, quality_score, momentum_score, lowvol_score

This gives the sponsors a direct view of how individual stocks are being scored.

----------

**8. Stock Level Alpha Model**

The tool predicts a stock’s expected H day return (alpha) based on its fundamentals and price behavior.

**8.1 H day alpha target per stock**

For each stock $i$ and day $t$, raw H day forward average log return:

$$  
\alpha_{i,t} = \frac{1}{H} \sum_{k=1}^{H} r_{i,t+k}  
$$

This is an absolute return forecast, not explicitly adjusted for factor exposures in the current implementation.

----------

**8.2 Feature set for alpha**

For each stock $i$ at date $t$, we build a feature vector $X_{i,t}$ using:

**Fundamental features**

Directly from the fundamental metrics:

-   Value style: $BP_{i,t}, EP_{i,t}, FCFP_{i,t}$
-   Quality style: $ROE_{i,t}, ROA_{i,t}, GM_{i,t}, FCFMargin_{i,t}, Lev_{i,t}, LevInv_{i,t}$

These may be standardized per date so that each feature has a similar scale.

**Technical features**

Based on past returns:

1.  Momentum windows (cumulative log returns):

-   5 day:

$$  
mom_{i,5,t} = \sum_{k=0}^{4} r_{i,t-k}  
$$

-   21 day:

$$  
mom_{i,21,t} = \sum_{k=0}^{20} r_{i,t-k}  
$$

-   63 day:

$$  
mom_{i,63,t} = \sum_{k=0}^{62} r_{i,t-k}  
$$

2.  Volatility windows (realized volatility):

-   5 day:

$$  
vol_{i,5,t} = \text{standard deviation of } {r_{i,t-4},\dots,r_{i,t}}  
$$

-   21 day:

$$  
vol_{i,21,t} = \text{standard deviation of } {r_{i,t-20},\dots,r_{i,t}}  
$$

-   63 day:

$$  
vol_{i,63,t} = \text{standard deviation of } {r_{i,t-62},\dots,r_{i,t}}  
$$

All features are standardized across the cross section (or across observations in the training set) so coefficients are comparable.

----------

**8.3 Lasso regression for alpha and “fundamental_score”**

The main alpha model for each stock is a **Lasso regression**:

$$  
\alpha_{i,t} = \beta_0 + \sum_j \beta_j X_{i,t}^{(j)} + \varepsilon_{i,t}  
$$

with penalty:

$$  
\lambda \sum_j |\beta_j|  
$$

This penalty shrinks many coefficients to zero, leaving a sparse set of features that drive the prediction.

The model is fit on a rolling sample of historical stock level data, using roughly two years of lookback for robustness (for example 2 × 252 trading days).

After fitting and storing the model, for the most recent date $T$ we compute:

$$  
\widehat{\alpha}_{i,T} = \beta_0 + \sum_j \beta_j X_{i,T}^{(j)}  
$$

The app converts this to percent:

$$  
\text{expected\_alpha\_\%}_i = 100 \times \widehat{\alpha}_{i,T}  
$$

and stores this in alpha_preds[i]["expected_alpha"].

**Fundamental score**

Internally, the AlphaPredictor also constructs a compressed **fundamental_score** for each stock as a summary of its value and quality profile.

A simple way to define it consistent with the factor logic is:

$$  
fundamental\_score_{i,t} = \frac{Value_{i,t} + Quality_{i,t}}{2}  
$$

That is, the average of the stock’s standardized Value and Quality scores at that date. This is what is exposed in the **Alpha predictions (top 10)** table’s fundamental_score column, giving a one number summary of how attractive the stock looks on fundamentals alone.

(_If your implementation uses a slightly different blend, the conceptual interpretation is the same: a composite score built from value and quality style metrics._)

**Top drivers and coefficients**

For interpretability, AlphaPredictor also gathers the top features with nonzero Lasso coefficients. For each stock in the top 10 by expected alpha:

-   It lists the features with the largest absolute coefficients.
-   Each coefficient $\beta_j$ is linked to the standardized feature $X_{i,T}^{(j)}$.

Because features are standardized, the meaning is:

-   If $\beta_j > 0$: a one standard deviation increase in feature $j$ is associated with an increase of $\beta_j$ in expected H day return.
-   If $\beta_j < 0$: a higher value of that feature is associated with lower expected H day return.

These are associations learned from historical data, not causal effects.

----------

**8.4 Alpha predictions capped to 100 tickers**

To keep computation reasonable and UI clean, the app only builds alpha predictions for at most 100 tickers:

$$  
\text{cap} = \min(100, U)  
$$

In the code:

-   Only the first cap tickers in the universe (after sorting) are passed into the alpha prediction loop.
-   The optimized portfolio is then further restricted to tickers that both have alpha predictions and adequate return history to estimate covariance.

----------

**8.5 Alpha signal summary (top vs bottom decile)**

In the UI, after computing all expected alphas, the app computes a simple “signal strength” summary:

1.  Sort all stocks by expected_alpha_% descending.
2.  Let ( n ) be the number of stocks with alpha predictions. If $n \ge 30$:

-   Let $k = \max(\lfloor 0.10 n \rfloor, 3)$.
-   Compute:

-   Top group mean = average expected_alpha_% of the top $k$ stocks.
-   Bottom group mean = average expected_alpha_% of the bottom $k$ stocks.

-   Spread:

$$  
\text{Spread} = \text{Top mean} - \text{Bottom mean}  
$$

This is what the app displays as:

Top decile mean X percent, bottom decile Y percent, spread Z percent.

It gives a quick sense of how much separation the alpha model creates between “best” and “worst” names.

----------

**9. Optimized Unified Portfolio**

This feature is new since the last sponsor meeting. It converts the stock level alphas into a practical, constrained long/short portfolio.

**9.1 Expected return vector from alpha**

From alpha_preds, we build:

$$  
\mu_i = \widehat{\alpha}_{i,T}  
$$

for all tickers $i$ where alpha predictions are available.

In practice, we use a subset opt_tickers:

-   Start from the universe tickers.
-   Keep only those with alpha predictions.

Later, we will intersect this set with the tickers that have sufficient return history to build a covariance matrix.

----------

**9.2 Covariance matrix with Ledoit Wolf shrinkage**

We call UnifiedPortfolioOptimizer.build_covariance(price_data, tickers, lookback_days=252):

1.  Filter price_data to the opt_tickers.
2.  Sort by date.
3.  Determine the returns column to use:

	-   If log_returns exists, use that.
	-   Otherwise use returns.

5.  Restrict to the last lookback_days distinct dates.
6.  Pivot into a date by ticker matrix of returns.
7.  Identify tickers with at least 50 non null observations in this window; others are dropped.
8.  Apply **Ledoit Wolf** covariance estimation on the remaining matrix:

$$  
\Sigma = \text{LedoitWolf}(R_{window})  
$$

producing a symmetric $N \times N$ matrix of pairwise covariances.

8.  Return $\Sigma$ and the valid_tickers used.

This robust shrinkage estimator avoids wild covariance estimates that would destabilize the portfolio.

----------

**9.3 Aligning alphas and covariance**

In the app:

-   If $\Sigma$ is empty or fewer than 2 valid tickers remain, we skip portfolio construction.
-   Otherwise, we intersect:

$$  
\text{common} = \text{valid\_tickers} \cap \text{alpha tickers}  
$$

-   Build:

$$  
\mu_{\text{use}} = \mu[\text{common}], \quad \Sigma_{\text{use}} = \Sigma[\text{common}, \text{common}]  
$$

This ensures that every stock in the portfolio has both an expected return estimate and a robust risk estimate.

----------

**9.4 Optimization logic and constraints**

UnifiedPortfolioOptimizer.optimize(mu, Sigma, long_only=False) uses a heuristic risk adjusted scoring approach.

Let $n =$ number of tickers in common, and label them $i = 1,\dots,n$.

1.  Extract variance for each stock:

$$  
\sigma_i^2 = \Sigma_{ii}  
$$

Replace any non positive variance with a small positive $10^{-6}$ to avoid division problems.

2.  Compute risk adjusted scores:

$$  
score_i = \frac{\mu_i}{\sqrt{\sigma_i^2}}  
$$

3.  Split into positive and negative scores:

$$  
pos_i = \max(score_i, 0), \quad neg_i = \max(-score_i, 0)  
$$

4.  Long and short target gross exposures:

-   Set target gross exposure $G = \min(\text{max\_gross}, 1.0)$.
-   Half to longs, half to shorts (dollar neutral):

$$  
G_L = \frac{G}{2}, \quad G_S = \frac{G}{2}  
$$

5.  Long weights:

If $\sum_i pos_i > 0$:

$$  
w_i^{long} = \frac{pos_i}{\sum_j pos_j} \times G_L  
$$

6.  Short weights:

If $\sum_i neg_i > 0$:

$$  
w_i^{short} = - \frac{neg_i}{\sum_j neg_j} \times G_S  
$$

7.  Combined raw weights:

$$  
w_i^{raw} = w_i^{long} + w_i^{short}  
$$

8.  Per name cap:

The tool uses a configurable max_weight. In the UI we are currently using:

$$  
|w_i| \le 0.10  
$$

Implemented via a clip:

$$  
w_i^{cap} = \text{clip}(w_i^{raw}, -0.10, 0.10)  
$$

9.  Gross exposure cap:

Let:

$$  
G_{raw} = \sum_i |w_i^{cap}|  
$$

If $G_{raw} > \text{max\_gross}$ (which is currently set to 1.5), rescale:

$$  
w_i = w_i^{cap} \times \frac{\text{max\_gross}}{G_{raw}}  
$$

This ensures:

$$  
\sum_i |w_i| \le 1.5  
$$

What this achieves:

-   Stocks with higher risk adjusted alpha scores $score_i$ get larger long weights.
-   Stocks with strongly negative scores get larger short weights.
-   Very volatile stocks are penalized because their score is divided by volatility.
-   No single name can dominate the portfolio (10 percent cap).
-   The portfolio cannot take on excessive gross leverage (total long plus short exposure capped at 1.5).

**Fallback if covariance is missing**

If for some reason Sigma is missing or empty:

-   The optimizer falls back to a simple alpha only scheme:
	- $score_i = mu_i$
	-   Construct longs from positive scores and shorts from negative scores, normalize, then apply weight caps and gross cap.

In typical usage, with enough data, we always obtain a valid covariance matrix and use the risk adjusted approach.

----------

**9.5 Portfolio level expected H day return**

Once we have the final weights $w_i$ and expected H day returns $\mu_i$, the portfolio’s expected H day return is:

$$  
\alpha_{port} = \sum_i w_i \mu_i  
$$

The app converts this to percent:

$$  
\text{Portfolio expected H day alpha} = 100 \times \alpha_{port}  
$$

and labels it using the actual horizon used in the last pipeline run (for example “Portfolio expected 21 day alpha”).

----------

**9.6 Displayed unified portfolio table**

The unified portfolio section shows:

-   ticker
-   weight: the final portfolio weight $w_i$
-   side:

-   Long if $w_i > 0$
-   Short if $w_i < 0$
-   Flat if $w_i = 0$

-   expected_alpha_%: the stock’s own expected H day alpha $100 \times \mu_i$

The caption explains that the optimizer is maximizing risk adjusted expected return, using Ledoit Wolf covariance to estimate risk, and that it enforces a 10 percent per name cap with a 1.5 times gross exposure cap.

----------

**10. What Each Section in the App Is Doing**

With all of this in mind, here is how each visible section maps to the methodology above.

1.  **Factor premia forecasts**

-   Uses ar1_forecast from the AR(1) model for each factor.
-   Shows:

$$
\text{Expected Premium \% (AR(1))} = 100 \times \hat{y}_{f,T}^{\text{AR(1)}}
$$

-   Sorted by expected premium from highest to lowest.
-   Caption explains that these are forecasts of long short factor premia over the next H days.

2.  **Top drivers per factor**

-   Uses Random Forest feature importances on the factor forecasting features for each factor.
-   Shows the top k macro variables (rates, spreads, VIX) that historically helped predict that factor’s H day premium.
-   Caption clarifies that these are predictive associations, not causal effects.

4.  **Factor signal validation (walk forward)**

-   Summarizes AR(1) and ensemble performance across time via walk forward backtesting.
-   Shows out of sample hit rate, RMSE, and MAE for each factor.
-   AR(1) metrics are shown in the main table; ensemble metrics are available in an expander titled “Show machine learning ensemble factor forecasts (Ridge, Lasso, Random Forest)”.

6.  **Universe and stock level factor scores (details)**

-   Displays the universe tickers and size.
-   Shows factor portfolio sizes (number of names in each factor’s long and short legs).
-   Shows sample factor scores for up to 30 stocks: Value, Quality, Momentum, Low Vol.
-   This allows sponsors to sanity check how individual names are being classified.

8.  **Optimized unified portfolio**

-   Displays the portfolio weights obtained by combining stock level alpha predictions with the Ledoit Wolf covariance matrix and the weight constraints.
-   Shows which names are long and which are short.
-   Provides a clear caption describing what is optimized, what risk estimate is used, and what constraints are in place (10 percent per name, 1.5 gross).
-   Shows the portfolio’s overall expected H day alpha.

10.  **Alpha predictions (top 10)**

-   Lists the 10 stocks with the highest expected H day alpha from the Lasso model.
-   Shows expected_alpha_% and a composite fundamental_score for each.
-   Provides a caption explaining that alpha is determined from a regression on fundamental and technical features, trained on roughly two years of history.
-   For each of these top 10 names, an expander shows the top features and Lasso coefficients driving that prediction, along with a caption explaining what the coefficients mean.
