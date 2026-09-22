# Blood Bank Inventory Optimization

## Complete Findings, Methodology & Recommendations

**Business Question:** What factors contribute to blood shortages and
wastage, and can historical demand and inventory patterns be used to
predict future blood demand and shortage risk so blood banks can
optimize inventory, reduce wastage, prevent stockouts, and improve blood
availability?

**Scope:** 149 facility IDs, 10,000 facility-month records, January
2021--December 2024, 8 blood groups, and 6 blood products. All records
are labelled `national_blood_centre` in the source dataset.

**Data provenance note:** The source exhibits patterns consistent with a
synthetic/simulated blood-bank supply dataset. Results should therefore
be presented as a portfolio demonstration of analytical methodology and
decision-support design, not as validated evidence about a real national
blood service.

**Companion files:** `blood_bank_inventory_watchlist.csv` (final
decision-support watchlist), `inventory_optimization_combo_level.csv`,
`inventory_action_system.csv`, `demand_forecast_model_comparison.csv`,
`demand_forecast_next6months.csv`, `shortage_model_comparison.csv`,
`shortage_threshold_tuning.csv`, and `models/model_card.json`.

------------------------------------------------------------------------

## 1. Headline Numbers

  ----------------------------------------------------------------------
  Metric                             Value
  ---------------------------------- -----------------------------------
  Demand fulfillment rate            56.89% (1,422,289 transfused ÷
                                     2,499,963 requested)

  Overall wastage rate               16.52% (330,163 units lost to
                                     expiry + TTI discard)

  Shortage rate                      25.54% of all facility-months

  Supply-demand gap                  −501,368 units (system collects
                                     \~20% less than requested)

  Recorded patient-service impact    713 `maternal_deaths_no_blood` and
                                     6,312
                                     `surgical_cancellations_no_blood`
                                     events in the dataset

  Avg shortage duration              3.79 days, when a shortage occurred
  ----------------------------------------------------------------------

## 2. The Three Findings That Matter Most

**Finding 1 --- The system has a persistent, uniform \~20% collection
shortfall, not scattered local problems.** Every blood group and product
shows nearly the same proportional gap (19.3%--20.7%). *Evidence:* SQL
aggregation across all groups/products; confirmed at the facility ×
blood group × product grain in the inventory optimization model (100% of
3,596 combinations show a positive reorder gap, 0% show surplus).
*Implication:* the fix is national collection capacity, not group- or
product-targeted intervention.

**Finding 2 --- Facility identity is the dominant risk factor; blood
group, product, and season are not.** Facility-level shortage rates
range from 12.5% to 40.5% (a \>3x spread), while blood group and product
effects on every outcome tested are statistically negligible (Cramér's V
≤ 0.015, well below the "negligible effect" threshold of 0.1).
*Implication:* resource allocation should target the specific high-risk
facilities identified in the watchlist, not broad category-level policy.

**Finding 3 --- Shortage timing cannot be predicted in advance from the
available data --- confirmed three independent ways.** Correlation
analysis (near-zero across the board), seven formal hypothesis tests
(all effect sizes negligible), and a full 5-algorithm machine-learning
comparison (Logistic Regression, Decision Tree, Random Forest, Gradient
Boosting, XGBoost, with threshold tuning and SMOTE resampling) all
converge on ROC-AUC ≈ 0.50 --- no better than chance. *Implication:*
investment should shift from predictive early-warning systems toward
capacity-based safety stock and rapid facility-level response. See
`models/model_card.json` for the exported model metadata, performance,
features, and deployment warning.

## 3. Inventory Optimization Methodology

**Planning unit:** facility × blood group × product (3,596 combinations)
--- the natural operational reorder unit.

**Forecast Demand:** each combination's own historical mean monthly
demand. Per-record demand shows no exploitable trend or seasonality
beyond its average (a simple historical-mean forecast has MAE of just
0.80 units against a mean of \~250), so this is the defensible choice.

**Demand variability (safety-stock input):** each combination's own
standard deviation where ≥5 historical observations exist (18.7% of
combinations); otherwise **shrunk to the pooled blood-group × product
level std**, since the median combination has only 2 observations ---
too thin to estimate variance reliably alone.

**Safety Stock = 1.65 × σ(demand)** --- a \~95% service-level
assumption, chosen because blood is a critical, life-safety commodity,
justifying a higher default than a typical retail service level. This
assumption is explicit, not hidden in the formula.

**Recommended Stock = Forecast Demand + Safety Stock**

**Current Inventory (proxy):** average historical monthly collection for
that combination. **This is a proxy, not a literal stock balance** ---
no running inventory-ledger field exists in the source data. An earlier
attempt used avg(collected − transfused) as a "leftover residual" proxy,
but this produced a degenerate 100%-reorder/0%-surplus result for the
wrong reason (comparing a tiny monthly leftover against a full month's
demand). Reframing to "recommended stock vs. average collection
achieved" turns this into a genuinely useful **collection-target gap**
metric instead.

**Inventory Gap = Recommended Stock − Current Inventory (proxy)**
**Reorder Quantity = MAX(Inventory Gap, 0)** **Surplus Inventory =
MAX(−Inventory Gap, 0)**

**Shortage Risk Proxy:** the watchlist field
`predicted_shortage_probability` contains the facility's **empirical
historical shortage rate**. Despite the legacy column name, it is not a
machine-learning probability forecast. The classifier is deliberately
excluded from the operational watchlist because its chronological 2024
holdout ROC-AUC is 0.489. This distinction prevents false predictive
precision.

## 4. Result: 100% Reorder Need, 0% Surplus --- Verified as a Real Finding, Not a Bug

Every one of the 3,596 combinations shows a positive reorder gap. This
was checked carefully rather than accepted at face value --- it is the
direct, expected consequence of Finding 1 (the system-wide \~20%
collection shortfall applies almost everywhere), now confirmed at the
granular level. There is no meaningful surplus segment in this system to
find. What *does* vary meaningfully is the **size** of the reorder gap
(23--143 units across combinations), which is what makes the ranked
watchlist useful: it identifies *where* additional collection effort
matters most, even though the underlying shortfall is universal.

## 5. Expiry Risk --- A Reframed, More Useful Metric

A surplus-based expiry-risk flag (short shelf life + surplus inventory)
was tried first, per the standard blueprint approach --- but it never
fires, since 0% of combinations show surplus. Rather than ship an
uninformative all-"Low" column, a **wastage-rate-based attention flag**
was built instead, since wastage rate *does* vary meaningfully at the
facility level (14.75%--19.2%): combinations in the top wastage tercile
AND with shelf life ≤ 42 days are flagged **High** (677 combinations);
top-tercile wastage alone is flagged **Moderate** (1,992); the remainder
is **Low** (927).

## 6. Recommended Action System

  -----------------------------------------------------------------------
  Action                  Count                   Evidence-linked trigger
  ----------------------- ----------------------- -----------------------
  Reorder Soon            2,078                   Positive reorder gap,
                                                  below the top quartile

  Reorder Immediately     899                     Reorder gap ≥ top
                                                  quartile (86.5 units)

  Investigate High        345                     Wastage attention flag
  Wastage                                         = High

  Monitor Closely         141                     Facility shortage rate
                                                  \> 1.3× the system
                                                  average

  Prioritize Near-Expiry  133                     High wastage AND shelf
  Usage                                           life ≤ 5 days
                                                  (platelets)
  -----------------------------------------------------------------------

**A real bug was caught and fixed while building this system:** an early
version of the rule engine checked the wastage-flag condition before the
general large-gap condition, which meant some of the largest-gap
combinations were mislabeled "Investigate High Wastage" instead of
"Reorder Immediately." The priority order was corrected so that a large
reorder gap always takes precedence --- appropriate for a life-safety
commodity where under-stocking is the more urgent failure mode.

**Deliberately absent, and why:** "Reduce Excess Collection," "Adequate
Stock," and "Redistribute Inventory" do not appear. Forcing them to fire
would misrepresent the data --- Finding 1 established 0 of 3,596
combinations have surplus, and redistribution requires a genuine surplus
position to redistribute *from*, which does not exist here.

**Secondary evidence-linked flags** (surfaced alongside the primary
action, with an explicit caution attached): Increase Staffing (453
combinations, facility staff ≤ 25th percentile AND above-average
shortage rate), Review Cold Chain (680, facility maintenance ≤ 25th
percentile AND elevated wastage), Improve Crossmatch Ops (463, facility
turnaround ≥ 75th percentile AND above-average shortage rate). **Caution
restated directly:** staffing, cold-chain, and turnaround all showed
negligible effect sizes in the aggregate statistical tests --- these
flags identify facilities poor on both the metric and the outcome
simultaneously (a reasonable manual-review trigger), not validated
causal fixes.

## 7. Final Business Recommendations

### Immediate (0--1 month)

-   Use the watchlist as an analytical prioritization tool: review the
    899 facility × blood-group × product combinations labelled "Reorder
    Immediately." These are combinations, not 899 unique facilities.
-   Investigate the 15 highest-risk facilities directly (led by facility
    BB_0086 at 40.5% shortage rate and 81 surgical cancellations).
-   Do not deploy the current shortage classifier as an early-warning
    system. The available features do not provide validated
    forward-looking discrimination; prioritize capacity-based buffers
    and improved data collection instead.

### Short-term (1--6 months)

-   Investigate the 677 high-wastage-attention combinations individually
    --- since no system-wide driver (shelf life, cold chain,
    supply-demand balance) explains wastage variation, root causes are
    likely facility-specific.
-   Prioritize near-expiry usage protocols at the 133 flagged
    combinations (concentrated in short-shelf-life platelets).
-   Build the safety-stock methodology above into standard monthly
    ordering.
-   Review staffing at the 453 flagged facility-combinations --- framed
    as a modest, not transformative, lever.

### Medium-term (6--18 months)

-   Evaluate collection-capacity expansion against the observed \~20%
    aggregate collection-to-demand shortfall. Because the dataset
    appears synthetic/simulated, validate the magnitude with real
    operational data before making investment decisions.
-   Refresh the demand forecast quarterly (ARIMA(1,1,1), 1.45% MAPE on
    chronological holdout) --- its confidence interval widens quickly,
    so frequent refreshes matter more than model sophistication.

### Strategic (18+ months)

-   Invest in real-time inventory-balance tracking --- the single
    biggest data gap found across this entire analysis, which forced
    every proxy-based workaround described in Section 3.
-   Fix the region-classification data-governance issue: all 149
    facility IDs show more than one `region_type` across their records,
    so region cannot be treated as a stable facility attribute.
-   Institutionalize the facility-risk-ranking approach as a recurring
    quarterly report.

### Explicitly NOT recommended, with reasons stated

-   **Facility redistribution based on this model** --- no modeled
    surplus positions exist under the collection-proxy methodology. This
    does not prove that real physical surplus never exists because the
    source lacks a true inventory ledger.
-   **Equipment investment** --- all 149 facilities show identical
    equipment-capability flags (constant = 1 across the board); there is
    no variation to base an investment case on.
-   **Blood-group-targeted management programs** --- every blood-group
    effect size tested was negligible (Cramér's V ≤ 0.015).

## 8. Visual Evidence and Chart Alignment

The 20 project charts are aligned with the SQL/Python analytical
workflow and should be retained as the visual evidence layer of the
project. They support the findings below without changing the underlying
conclusions.

### Exploratory and operational evidence

1.  **01_univariate_distributions.png --- Univariate Distributions:
    Volume & Rate Variables.** Requested, collected, and transfused
    units are broadly concentrated around their central ranges. Wastage,
    expiry, and fulfillment rates also show stable unimodal
    distributions. This chart supports the descriptive profile of the
    10,000-record analytical dataset; it does not establish causality.

2.  **02_demand_by_blood_group.png --- Total Demand by Blood Group.**
    O-positive has by far the largest aggregate requested volume,
    followed by A-positive and B-positive. This supports the blood-group
    demand analysis. Aggregate volume should not be confused with
    shortage risk.

3.  **03_shortage_rate_by_blood_group.png --- Shortage Rate by Blood
    Group.** Blood-group shortage rates cluster closely around the
    overall 25.54% rate. B-negative and O-negative are slightly higher,
    but the narrow spread is consistent with the statistical finding
    that blood group has negligible practical association with shortage
    occurrence.

4.  **04_wastage_rate_by_blood_group.png --- Average Wastage Rate by
    Blood Group.** Average wastage rates are tightly grouped around the
    system average. This supports the conclusion that blood group is not
    a strong practical discriminator of wastage.

5.  **05_demand_by_product.png --- Total Demand by Product.** Aggregate
    requested volume is very similar across the six blood products. This
    is consistent with the project finding that product category is not
    the principal source of the system-wide supply problem.

6.  **06_shortage_rate_by_product.png --- Shortage Rate by Product.**
    Product-level shortage rates are also tightly clustered near
    25%--26%, reinforcing the finding that product identity has little
    practical discriminatory power for shortage occurrence.

7.  **07_monthly_demand_trend.png --- Monthly Demand Trend.** The raw
    national total rises strongly as facility-record coverage increases,
    while average demand per facility-record remains essentially flat
    around 250 units. Therefore the apparent national growth must not be
    interpreted as pure underlying demand growth. This coverage effect
    is central to interpreting the forecasting results.

8.  **08_monthly_shortage_trend.png --- Monthly Shortage Rate Trend.**
    Monthly shortage rates fluctuate around the 25.54% overall average
    without a stable directional pattern. This is consistent with the
    statistical analysis finding weak/negligible calendar-month effects.

9.  **09_monthly_wastage_trend.png --- Monthly Wastage Rate Trend.**
    Monthly wastage remains close to the 16.52% overall level, with
    relatively modest variation. The chart supports the conclusion that
    wastage is persistent rather than confined to a single period.

10. **10_supply_vs_demand_scatter.png --- Units Collected vs Units
    Requested.** Most facility-month observations fall below the
    collected=requested reference line. This visually supports the
    negative aggregate supply-demand gap and persistent collection
    shortfall.

11. **11_inventory_status_vs_wastage.png --- Wastage by Inventory
    Status.** Mean wastage is very similar across Adequate, Recovered,
    Low Stock, and Critical status groups. Inventory status therefore
    does not show a strong separation in average wastage within this
    dataset.

12. **12_staffing_vs_shortage.png --- Facility Average Staff Count vs
    Shortage Rate.** The facility-level trend is weakly negative (r ≈
    −0.20): facilities with higher average staffing tend to have
    somewhat lower historical shortage rates. This is an association,
    not evidence that staffing changes cause shortage reduction.

13. **13_turnaround_vs_shortage_boxplot.png --- Crossmatch Turnaround:
    Shortage vs No-Shortage.** The two distributions overlap heavily.
    This visually agrees with the formal tests showing no meaningful
    relationship between crossmatch turnaround and shortage occurrence.

14. **14_coldchain_vs_wastage.png --- Cold-Chain Maintenance vs Average
    Wastage Rate.** Mean wastage is almost identical between the two
    groups in this dataset. The chart therefore does not support a claim
    that the recorded cold-chain indicator is a major wastage driver.

15. **15_shelflife_vs_expiry.png --- Shelf Life vs Average Expiry
    Rate.** The three shelf-life categories have very similar mean
    expiry rates. The project should therefore avoid claiming a strong
    shelf-life effect from these data.

16. **16_facility_performance_shortage.png --- Facility Shortage Rate:
    Lowest vs Highest.** Facility-level rates show substantial
    separation, from roughly 12.5% at the low end to about 40.5% at the
    high end. This is the strongest visual support for prioritizing
    facility-level historical risk rather than broad blood-group/product
    targeting.

17. **17_correlation_heatmap.png --- Correlation Matrix.** Operational
    volume variables have expected structural relationships, but
    `shortage_in_last_month` has near-zero correlations with the
    candidate forward-looking numerical predictors. The 0.65 correlation
    with `shortage_days_last_month` is expected because shortage days
    describe the same-period shortage event; this field is excluded from
    the predictive model to prevent target leakage. Correlation is
    descriptive and does not imply causation.

### Forecasting and machine-learning evidence

18. **18_demand_forecast_comparison.png --- Actual vs Forecast
    Methods.** The chart shows the chronological July--December 2024
    holdout and compares Naive, Seasonal Naive, Holt-Winters, and
    SARIMA-style forecasts. It must be interpreted together with the
    coverage warning in Chart 07: aggregate demand is partly driven by
    changing record coverage. The saved model-comparison table, not
    visual proximity alone, determines model selection. ARIMA(1,1,1) is
    retained as the best realistic no-look-ahead forward model in the
    project pipeline.

19. **19_shortage_model_roc_pr_curves.png --- Shortage Prediction
    Diagnostic Curves.** Logistic Regression ROC-AUC is 0.489,
    essentially indistinguishable from a random classifier (0.50), while
    the precision-recall curve remains near the no-skill baseline. This
    directly supports the decision not to deploy the classifier as an
    operational shortage early-warning model.

20. **20_shap_feature_importance.png --- Mean Absolute SHAP Values.**
    Lagged wastage, turnaround, demand, staffing, historical facility
    shortage rate, and month receive the largest mean absolute SHAP
    values in the displayed Random Forest explanation. However, because
    the underlying classifier performs at approximately ROC-AUC 0.50,
    these values must **not** be presented as reliable shortage drivers.
    They describe how a weak model allocated its predictions, not
    validated causal or predictive determinants.

### Chart-level conclusion

Taken together, the visual evidence supports the same story as the SQL,
statistical tests, forecasting outputs, and operational files:

-   demand exceeds collections at the system level;
-   shortage and wastage rates are persistent;
-   blood-group and product differences in shortage/wastage rates are
    comparatively small;
-   facility-level historical shortage performance varies substantially;
-   aggregate monthly demand is affected by changing facility-record
    coverage;
-   available lagged and operational features do not support a reliable
    shortage classifier;
-   inventory decisions should therefore use transparent historical
    demand, variability-based safety stock, observed facility risk, and
    explicit operational rules rather than unsupported machine-learning
    predictions.

The charts are analytical evidence, not independent sources. All GitHub
captions, README statements, and dashboard commentary should preserve
these qualifications.

------------------------------------------------------------------------

## 9. Reproducibility and File Alignment

The project follows a traceable workflow:

**Raw CSV → PostgreSQL cleaning and feature engineering → cleaned CSV →
Python validation/EDA → statistical testing → demand forecasting →
shortage-model comparison → inventory optimization → action system →
operational watchlist.**

The cleaned analytical file contains **10,000 rows and 49 columns**: 36
source fields, the monthly `report_month` field, and 12 engineered
analytical fields. The Python workflow independently recalculates the
principal SQL KPIs and engineered measures rather than simply copying
SQL outputs.

The demand-forecast comparison contains ten candidate approaches. The
best fully autonomous model in the saved comparison is **ARIMA(1,1,1)**
with MAE 874.4, RMSE 1,175.5, MAPE 1.45%, and sMAPE 1.44% on the
chronological holdout. A GBM result with future record counts known is
retained only as an optimistic, assumption-dependent benchmark and is
not treated as the deployable winner.

The six-month ARIMA output forecasts aggregate requested units for
January--June 2025, beginning at **61,952 units in January 2025**, with
uncertainty intervals widening over the forecast horizon.

The shortage-model comparison contains five classifiers. Their ROC-AUC
values remain approximately random: Decision Tree 0.5090, Random Forest
0.5010, Gradient Boosting 0.4933, XGBoost 0.4899, and Logistic
Regression 0.4890. The exported Logistic Regression artifact is
therefore retained for reproducibility only, not operational deployment.

The inventory optimization output contains **3,596 facility ×
blood-group × product combinations**. Historical mean demand is used at
this granular planning level, while pooled blood-group × product
variability is used when a combination has fewer than five observations.
The final action counts reconcile exactly to all 3,596 combinations:
2,078 Reorder Soon, 899 Reorder Immediately, 345 Investigate High
Wastage, 141 Monitor Closely, and 133 Prioritize Near-Expiry Usage.

------------------------------------------------------------------------

## 10. Key Limitations Carried Into Every Number Above

-   No true inventory-balance field exists --- "current inventory"
    throughout this document is a collection-based proxy.
-   `region_type` is not a stable per-facility attribute --- treat
    regional breakdowns as descriptive of individual records only.
-   The shortage-prediction model is provided for reproducibility, not
    deployment (see `models/model_card.json`).
-   This dataset shows patterns consistent with synthetic/simulated data
    generation (uniform cross-group rates, near-zero shortage
    predictability, a near-perfect mechanical relationship between
    facility-record count and total demand). The *methodology*
    demonstrated here transfers to real operational data; the specific
    *magnitudes* reported should be validated against it first.
