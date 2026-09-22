# Blood Bank Inventory Optimization & Demand Forecasting

> **End-to-end healthcare supply-chain analytics project using
> PostgreSQL, Python, statistical analysis, time-series forecasting,
> machine learning diagnostics, inventory optimization, and Power
> BI-ready data modeling.**

## Executive Summary

Blood banks must balance two competing risks: **insufficient blood
availability** and **avoidable wastage of perishable inventory**. This
project analyzes historical blood-bank operations to identify where
shortages and wastage occur, evaluate whether future demand and shortage
risk can be predicted, and translate the evidence into transparent
inventory actions.

The project covers **10,000 facility-month records**, **149
facilities**, **8 blood groups**, **6 blood products**, and the period
**2021-2024**. The analysis combines SQL-based data engineering, Python
EDA and statistical testing, chronological demand forecasting,
shortage-classification experiments, model explainability, and
operational safety-stock/reorder logic.

A key result is deliberately reported without overstating the
machine-learning component: the shortage classifier achieved
approximately **ROC-AUC 0.49 on the chronological 2024 holdout**, so it
is **not recommended for operational deployment**. The project therefore
uses evidence-based inventory rules and facility-level historical
performance rather than presenting an unreliable classifier as a
production solution.

## Business Question

**What factors contribute to blood shortages and wastage, and can
historical demand and inventory patterns be used to predict future blood
demand and shortage risk so blood banks can optimize inventory, reduce
wastage, prevent stockouts, and improve blood availability?**

## Business Objectives

-   Quantify demand, collections, transfusions, shortages, expiry, and
    wastage.
-   Identify blood groups, products, facilities, and operational
    conditions associated with elevated inventory pressure.
-   Evaluate whether historical demand supports defensible forward
    forecasting.
-   Test whether available lagged and operational variables can predict
    future shortage occurrence without target leakage.
-   Build transparent inventory recommendations using historical demand
    variability, safety stock, inventory position, and facility-level
    risk.
-   Produce reusable outputs for operational monitoring and Power BI
    reporting.

## Dataset Overview

  Attribute                       Scope
  ------------------------- -----------
  Analytical records             10,000
  Facilities                        149
  Time period                 2021-2024
  Blood groups                        8
  Blood products                      6
  Clean analytical fields            49
  Shortage records                2,554
  Non-shortage records            7,446

The source data contains facility, blood-group, product, collection,
request, transfusion, inventory, expiry, discard, shortage, staffing,
cold-chain, and crossmatch-turnaround information.

> **Data provenance note:** patterns in the dataset are consistent with
> synthetic or simulated data generation. The workflow is intended to
> demonstrate professional analytical methodology; operational
> conclusions and magnitudes should be revalidated before application to
> real blood-bank operations.

## Analytical Workflow

``` text
Raw CSV
   |
   v
PostgreSQL Data Audit & Cleaning
   |
   v
Feature Engineering & SQL Analysis
   |
   v
Clean Analytical Dataset
   |
   v
Python Validation & EDA
   |
   +--> Statistical Testing
   |
   +--> Demand Forecasting
   |
   +--> Shortage Classification Diagnostics
   |
   +--> Model Explainability
   |
   v
Inventory Optimization & Safety-Stock Logic
   |
   v
Operational Watchlist + Power BI-Ready Star Schema
```

## Key Performance Indicators

  KPI                                   Result
  ------------------------------- ------------
  Total units collected              1,998,595
  Total units requested              2,499,963
  Total units transfused             1,422,289
  Total units expired                  233,767
  TTI-discarded units                   96,396
  Total wastage                        330,163
  Overall wastage rate                  16.52%
  Overall expiry rate                   11.70%
  Demand fulfillment rate               56.89%
  Overall shortage rate                 25.54%
  Average shortage days                   0.97
  Average crossmatch turnaround     1.12 hours

## Key Findings

### 1. Persistent supply-demand imbalance

Total collections were approximately **501,368 units below total
requests**, meaning collections were about **20% below requested
volume** at aggregate level. The supply-vs-demand analysis shows that
this gap is systemic rather than confined to one blood group or product.

### 2. Demand volume differs strongly by blood group, but shortage rates do not

O-positive accounts for the largest aggregate demand, followed by
A-positive and B-positive. However, shortage rates across blood groups
remain clustered around the **25.54% overall shortage rate**. High
demand volume therefore should not be treated as equivalent to high
shortage probability.

### 3. Product category is not a strong shortage discriminator

Requested volume is broadly similar across the six product categories,
and product-level shortage rates remain close to one another. Product
identity alone provides limited operational discrimination for shortage
risk in this dataset.

### 4. Facility-level performance is the most actionable segmentation

Historical shortage rates vary substantially across facilities,
approximately **12.5%-40.5%** among the displayed lowest/highest
facilities. This variation is materially larger than the differences
observed across blood groups or products and supports facility-specific
monitoring and intervention.

### 5. Wastage is persistent

The overall wastage rate is **16.52%**, with monthly values remaining
relatively close to that level. Blood-group, cold-chain-status,
shelf-life, and inventory-status comparisons show limited separation in
this dataset, so the analysis does not overstate these variables as
strong wastage drivers.

### 6. Aggregate demand growth is affected by changing facility coverage

Raw monthly demand totals rise over time, but average demand per
facility-record remains essentially flat around **250 units**. This
indicates that part of the apparent aggregate growth comes from
increasing facility-record coverage. Forecasting results must therefore
be interpreted with this coverage effect in mind.

### 7. Shortage timing is not reliably predictable from the available features

The project evaluates multiple classification approaches using a
chronological holdout. Logistic Regression produced approximately
**ROC-AUC 0.489**, and the broader model comparison did not establish a
useful operational early-warning signal.

This is treated as a valid analytical result, not hidden as a modeling
failure. The available features do not contain sufficient predictive
signal for defensible shortage deployment.

## Statistical Analysis

The project uses hypothesis testing and effect-size interpretation to
separate statistically detectable patterns from operationally meaningful
ones. The analyses cover relationships involving:

-   blood group and shortage occurrence;
-   product and shortage occurrence;
-   staffing and facility shortage performance;
-   crossmatch turnaround and shortage;
-   cold-chain maintenance and wastage;
-   shelf life and expiry;
-   calendar effects and shortage behavior.

Where effects are negligible, the project explicitly reports them as
negligible rather than converting statistical noise into business
recommendations.

## Demand Forecasting

Demand forecasting is evaluated chronologically rather than with a
random train/test split. The workflow compares baseline and statistical
time-series methods, including:

-   Naive forecast
-   Seasonal Naive
-   Holt-Winters
-   ARIMA/SARIMA-family approaches

The final forward pipeline retains an **ARIMA(1,1,1)** model as the best
realistic no-look-ahead forecasting option used in the project outputs.

Because facility coverage changes across the historical period,
aggregate forecast values are accompanied by an explicit limitation:
part of the total-demand movement reflects changing record coverage
rather than pure per-facility demand growth.

## Shortage Prediction: Model Governance

The shortage-prediction experiment is intentionally documented as a
**diagnostic model**, not a deployed AI solution.

### Leakage controls

Same-period variables that directly describe a shortage event,
particularly `shortage_days_last_month` and `shortage_cause`, are
excluded from the forward-looking predictor set. Lagged and historical
features are used where appropriate.

### Model evaluation

The project compares multiple classification approaches and evaluates
performance using chronological test data, ROC-AUC, accuracy, recall,
precision-recall behavior, threshold tuning, and resampling experiments.

The final model card records approximately:

  -----------------------------------------------------------------------
  Metric                                                           Result
  ------------------------------ ----------------------------------------
  Test ROC-AUC                                                      0.489

  Test accuracy                                                     0.515

  Test recall                                                       0.454

  Deployment recommendation                   **Do not deploy as shortage
                                                    early-warning model**
  -----------------------------------------------------------------------

The diagnostic ROC curve lies close to the random-classifier baseline.
This result is reinforced by near-zero predictor correlations and
negligible practical effects in the statistical tests.

### Explainability limitation

SHAP values are included to demonstrate model-explainability workflow.
Because the underlying classifier has approximately random
discriminatory performance, SHAP magnitudes **must not be interpreted as
validated causal drivers or reliable shortage predictors**. They only
describe how the weak model allocated its predictions.

## Inventory Optimization & Decision Support

Because the classification model is not operationally reliable, the
project shifts from unsupported prediction to transparent decision
support.

The inventory framework combines:

-   historical demand;
-   demand variability;
-   current/available inventory;
-   safety-stock logic;
-   inventory gaps;
-   historical facility shortage performance;
-   expiry/wastage indicators;
-   explicit action rules.

Operational recommendations include categories such as:

-   **Reorder Immediately**
-   **Reorder Soon**
-   **Monitor**
-   **Adequate Stock**
-   **Surplus / Expiry Risk**

The resulting watchlist contains **3,596 facility x blood-group x
product combinations** for prioritization and review.

## Visual Analytics

The repository contains **20 numbered analytical charts** covering EDA,
operations, forecasting, model diagnostics, and explainability.

Key visuals include:

  ----------------------------------------------------------------------------
  Chart                                    Purpose
  ---------------------------------------- -----------------------------------
  `01_univariate_distributions.png`        Distribution of demand,
                                           collections, transfusions, wastage,
                                           expiry, and fulfillment

  `02_demand_by_blood_group.png`           Aggregate demand by blood group

  `03_shortage_rate_by_blood_group.png`    Blood-group shortage comparison

  `04_wastage_rate_by_blood_group.png`     Wastage comparison by blood group

  `05_demand_by_product.png`               Product demand

  `06_shortage_rate_by_product.png`        Product shortage comparison

  `07_monthly_demand_trend.png`            Raw demand vs per-facility-record
                                           trend

  `08_monthly_shortage_trend.png`          Monthly shortage-rate behavior

  `09_monthly_wastage_trend.png`           Monthly wastage behavior

  `10_supply_vs_demand_scatter.png`        Facility-month collection/request
                                           gap

  `12_staffing_vs_shortage.png`            Facility staffing vs historical
                                           shortage

  `16_facility_performance_shortage.png`   Highest/lowest facility shortage
                                           rates

  `17_correlation_heatmap.png`             Numerical correlation structure

  `18_demand_forecast_comparison.png`      Chronological forecasting
                                           comparison

  `19_shortage_model_roc_pr_curves.png`    ROC and precision-recall
                                           diagnostics

  `20_shap_feature_importance.png`         Explainability diagnostic for the
                                           weak classifier
  ----------------------------------------------------------------------------

> Visualizations are evidence for the analysis, not independent proof of
> causality. Interpretations in the documentation preserve this
> distinction.

## Business Recommendations

1.  **Prioritize collection capacity and supply planning.** The
    aggregate collection deficit is the clearest system-level
    constraint.
2.  **Use facility-specific monitoring.** Facility shortage performance
    varies more meaningfully than blood-group or product shortage rates.
3.  **Adopt variability-based safety stock.** Reorder buffers should
    reflect historical demand variability rather than relying on a weak
    shortage classifier.
4.  **Track wastage and expiry continuously.** Persistent wastage
    warrants routine monitoring even though the available categorical
    variables do not strongly explain it.
5.  **Improve data collection for predictive modeling.** Add richer
    operational drivers such as lead time, donor availability, emergency
    events, replenishment timing, reservation/crossmatch status,
    inventory age profile, transfers between facilities, and actual
    stockout timestamps.
6.  **Do not deploy the current shortage classifier.** Reassess
    predictive modeling only after stronger forward-looking operational
    features become available.

## Business Impact

The project converts raw blood-bank records into a reproducible
decision-support workflow that can help an operations team:

-   identify facilities with persistent shortage exposure;
-   quantify supply-demand gaps;
-   monitor expiry and wastage;
-   estimate forward demand;
-   calculate safety-stock and reorder requirements;
-   prioritize inventory actions through a facility/product/blood-group
    watchlist;
-   distinguish actionable evidence from weak or non-predictive signals.

## Technology Stack

  Layer                     Tools
  ------------------------- -----------------------------------------------
  Database & SQL            PostgreSQL 16
  Programming               Python
  Data manipulation         pandas, NumPy
  Visualization             Matplotlib
  Statistics                SciPy
  Machine learning          scikit-learn, XGBoost, imbalanced-learn
  Time-series forecasting   statsmodels
  Explainability            SHAP
  Model persistence         joblib
  BI preparation            Power BI-ready star schema, DAX specification
  Development environment   VS Code / Jupyter Notebook
  Version control           Git & GitHub

## Repository Structure

``` text
blood-bank-inventory-optimization/
|
|-- README.md
|-- Blood Bank Inventory Optimization document.md
|-- SQL blood bank analysis.sql
|-- Python blood bank eda forecasting modeling.py
|-- Blood bank clean.csv
|
|-- Charts/
|   |-- 01_univariate_distributions.png
|   |-- 02_demand_by_blood_group.png
|   |-- ...
|   |-- 19_shortage_model_roc_pr_curves.png
|   `-- 20_shap_feature_importance.png
|
|-- models/
|   |-- model_card.json
|   |-- shortage_prediction_model.pkl
|   |-- shortage_prediction_scaler.pkl
|   `-- shortage_prediction_feature_names.pkl
|
|-- outputs/
|   |-- blood_bank_inventory_watchlist.csv
|   |-- inventory_optimization_combo_level.csv
|   |-- inventory_action_system.csv
|   |-- shortage_model_comparison.csv
|   `-- shortage_threshold_tuning.csv
|
|-- Executive dashboard/
|   |-- fact_blood_bank.csv
|   |-- dim_date.csv
|   |-- dim_facility.csv
|   |-- dim_blood_group.csv
|   |-- dim_product.csv
|   |-- demand_forecast.csv
|   |-- watchlist.csv
|   |-- dashboard_build_specification.md
|   |-- dax_measures.md
|   |-- slicer_configuration.md
|   `-- executive_overview_mockup.html
|
`-- notebooks/
    `-- blood bank analysis.ipynb
```

## How to Reproduce the Project

### 1. PostgreSQL pipeline

Create the database and run the SQL script:

``` bash
createdb blood_bank_project
psql -d blood_bank_project -f "SQL blood bank analysis.sql"
```

Export the cleaned table after the SQL pipeline completes:

``` bash
psql -d blood_bank_project -c "\copy blood_bank_clean TO 'Blood bank clean.csv' WITH (FORMAT csv, HEADER true);"
```

### 2. Python environment

Install the required packages:

``` bash
pip install pandas numpy matplotlib scikit-learn scipy statsmodels xgboost shap imbalanced-learn joblib
```

Run the analysis from the repository root:

``` bash
python "Python blood bank eda forecasting modeling.py"
```

The pipeline regenerates the analytical charts, model artifacts,
operational outputs, and Power BI-ready files.

### 3. Power BI

A `.pbix` file is **not yet included**. The repository instead contains
the prepared star-schema CSVs, DAX measures, slicer configuration,
dashboard build specification, and an HTML Executive Overview mockup.

To complete the dashboard:

1.  Import the CSV files from `Executive dashboard/`.
2.  Create the documented relationships.
3.  Add the measures from `dax_measures.md`.
4.  Configure the slicers using `slicer_configuration.md`.
5.  Build each page according to `dashboard_build_specification.md`.

This allows the analytical repository to be published now while keeping
the future Power BI implementation reproducible.

## Reproducibility & Analytical Integrity

This project follows several practices intended to make the work
auditable:

-   raw and cleaned stages are separated;
-   SQL and Python outputs are cross-validated;
-   feature engineering is documented;
-   time-series evaluation uses chronological holdouts;
-   same-period shortage descriptors are excluded from forward-looking
    prediction;
-   weak model performance is retained and reported;
-   model artifacts and model-card metadata are preserved;
-   analytical charts are numbered and tied to specific findings;
-   operational recommendations are separated from unsupported causal
    claims.

## Limitations

-   The dataset appears synthetic/simulated and is not a substitute for
    validated hospital operational data.
-   Monthly aggregate records do not provide unit-level blood-bag
    collection and expiry timestamps, so the project does not implement
    true FEFO unit-level scheduling.
-   Changing facility coverage affects aggregate demand totals.
-   The available predictors do not provide reliable forward shortage
    discrimination.
-   Observed associations should not be interpreted as causal effects.
-   Inventory recommendations are analytical decision-support rules, not
    a clinically validated blood-allocation protocol.

## Future Improvements

-   Add a completed interactive Power BI `.pbix` dashboard.
-   Introduce richer real-world operational data and true replenishment
    lead times.
-   Model demand at carefully validated facility/product/blood-group
    grains where history is sufficiently dense.
-   Add rolling backtests and forecast-monitoring metrics.
-   Incorporate unit-level expiry dates for FEFO inventory optimization.
-   Evaluate mathematical optimization once procurement, transfer,
    holding, shortage, and wastage cost constraints are available.
-   Deploy an interactive monitoring application only after the
    underlying data and models meet operational validation standards.

## Project Deliverables

-   PostgreSQL data preparation and analytical pipeline
-   Clean feature-engineered dataset
-   Python EDA, statistical analysis, forecasting, and modeling pipeline
-   20 analytical visualizations
-   Demand-forecast outputs
-   Shortage-model comparison and diagnostics
-   Model card and reproducibility artifacts
-   Inventory optimization outputs
-   3,596-row operational watchlist
-   Power BI-ready star schema and dashboard specification
-   Full aligned project documentation

## Documentation

For the complete methodology, evidence, chart-by-chart interpretation,
model limitations, and recommendations, see:

**`Blood Bank Inventory Optimization document.md`**

------------------------------------------------------------------------

### Portfolio Note

This project demonstrates not only SQL, Python, forecasting, machine
learning, and BI preparation, but also **analytical judgment**: a model
that does not generalize is not presented as a successful deployment.
The final recommendations follow the evidence available in the data and
use transparent decision rules where predictive modeling is not
sufficiently reliable.
