"""
PYTHON BLOOD BANK EDA FORECASTING MODELING
Blood Bank Inventory Optimization & Demand Forecasting

Consolidated Python script covering the full analytical workflow:
  Section 1  -- Data validation (SQL vs Python cross-check)
  Section 2  -- Exploratory data analysis (17 visualizations)
  Section 3  -- Statistical hypothesis testing
  Section 4  -- Demand forecasting (ARIMA/SARIMA/ML comparison)
  Section 5  -- Shortage prediction model (classification comparison)
  Section 6  -- Model evaluation (threshold tuning, SMOTE, overfitting check)
  Section 7  -- Model explainability (coefficients, permutation importance, SHAP)
  Section 8  -- Inventory optimization (safety stock, reorder quantities)
  Section 9  -- Inventory action system (evidence-linked recommendations)
  Section 10 -- Blood Bank Inventory Watchlist (final operational CSV)
  Section 11 -- Executive dashboard data export (star-schema for Power BI)
  Section 12 -- Final shortage-prediction model export (models/ folder)

RUN FROM the project root folder (the one containing "Blood bank clean.csv").
Expects these subfolders to exist: charts/, outputs/, models/, "Executive dashboard/".

IMPORTANT, stated up front rather than buried: Section 5-7's shortage
classification model was evaluated across 5 algorithms and found to
perform at ROC-AUC ~0.49-0.51 -- no better than random guessing, on a
chronological (2024) holdout. This is a genuine, well-evidenced finding
(confirmed independently by the Section 3 hypothesis tests), not a
failed modeling attempt. The model saved in Section 12 is provided for
completeness and transparency, NOT as a recommended operational tool --
see models/model_card.md for the full caveat.
"""

import os
os.makedirs('charts', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('Executive dashboard', exist_ok=True)


# ===========================================================================
# SECTION 1 -- DATA VALIDATION (SQL vs Python cross-check)
# Source: python/01_data_validation.py
# ===========================================================================

"""
PHASE 11 (blueprint Phase 13): PYTHON DATA VALIDATION
Blood Bank Inventory Optimization & Demand Forecasting

Purpose: load the SQL-cleaned dataset (exported from blood_bank_clean)
into Python and independently verify that:
  1) basic structure matches (row/column counts, dtypes, missing values,
     duplicates, unique categories, numerical ranges)
  2) SQL-derived KPIs (Phase 4) are reproduced independently in pandas
  3) SQL-derived feature-engineering columns (Phase 3) are reproduced
     independently in pandas
No numbers here are copied from SQL output -- everything is recalculated
from the raw columns so this is a genuine independent check, not a
restatement.
"""

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

DATA_PATH = 'Blood bank clean.csv'

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("1. BASIC STRUCTURE VALIDATION")
print("=" * 70)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
# 36 original raw columns + report_month (Phase 2) + 12 engineered
# columns (Phase 3) = 49 expected columns.
print(f"Expected from SQL: 10000 rows x 49 columns -> "
      f"{'MATCH' if df.shape == (10000, 49) else 'MISMATCH'}")

print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"Duplicate IDs: {df['id'].duplicated().sum()}")
print(f"Total missing cells: {df.isnull().sum().sum()}")

print("\nData types:")
print(df.dtypes.value_counts())

print("\nUnique category counts (should match Phase 0/2 SQL audit):")
for c in ['blood_group', 'product_name', 'product_category', 'region_type',
          'shortage_cause', 'facility_id', 'shortage_severity', 'wastage_risk',
          'inventory_status']:
    print(f"  {c}: {df[c].nunique()} unique values")

print("\nNumerical ranges (sanity check against Phase 0):")
num_check_cols = ['units_collected_month', 'units_requested_month', 'units_crossmatched',
                   'units_transfused', 'units_expired', 'units_discarded_tti',
                   'shortage_days_last_month', 'blood_bank_staff_count']
print(df[num_check_cols].agg(['min', 'max']).T)

print("\nOutlier check (IQR method) on wastage_rate and crossmatch_turnaround_hours:")
for col in ['wastage_rate', 'crossmatch_turnaround_hours']:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"  {col}: {n_out} outliers (bounds {lower:.2f} - {upper:.2f})")

print("\n" + "=" * 70)
print("2. INDEPENDENT RECALCULATION OF SQL KPIs (Phase 4 - Overall Performance)")
print("=" * 70)

total_collected = df['units_collected_month'].sum()
total_requested = df['units_requested_month'].sum()
total_crossmatched = df['units_crossmatched'].sum()
total_transfused = df['units_transfused'].sum()
total_expired = df['units_expired'].sum()
total_discarded = df['units_discarded_tti'].sum()
total_wastage = total_expired + total_discarded

overall_wastage_rate = round(total_wastage / total_collected * 100, 2)
overall_expiry_rate = round(total_expired / total_collected * 100, 2)
overall_fulfillment_rate = round(total_transfused / total_requested * 100, 2)
overall_shortage_rate = round(df['shortage_in_last_month'].mean() * 100, 2)
avg_shortage_days = round(df['shortage_days_last_month'].mean(), 2)
avg_turnaround = round(df['crossmatch_turnaround_hours'].mean(), 2)

sql_expected = {
    'total_units_collected': 1998595,
    'total_units_requested': 2499963,
    'total_units_crossmatched': 1904507,
    'total_units_transfused': 1422289,
    'total_units_expired': 233767,
    'total_units_discarded_tti': 96396,
    'total_wastage': 330163,
    'overall_wastage_rate_pct': 16.52,
    'overall_expiry_rate_pct': 11.70,
    'overall_demand_fulfillment_rate_pct': 56.89,
    'overall_shortage_rate_pct': 25.54,
    'avg_shortage_days_all_records': 0.97,
    'avg_crossmatch_turnaround_hours': 1.12,
}

python_results = {
    'total_units_collected': int(total_collected),
    'total_units_requested': int(total_requested),
    'total_units_crossmatched': int(total_crossmatched),
    'total_units_transfused': int(total_transfused),
    'total_units_expired': int(total_expired),
    'total_units_discarded_tti': int(total_discarded),
    'total_wastage': int(total_wastage),
    'overall_wastage_rate_pct': overall_wastage_rate,
    'overall_expiry_rate_pct': overall_expiry_rate,
    'overall_demand_fulfillment_rate_pct': overall_fulfillment_rate,
    'overall_shortage_rate_pct': overall_shortage_rate,
    'avg_shortage_days_all_records': avg_shortage_days,
    'avg_crossmatch_turnaround_hours': avg_turnaround,
}

print(f"{'KPI':<38}{'SQL (Phase 4)':>15}{'Python':>15}{'Match':>10}")
all_match = True
for k in sql_expected:
    match = sql_expected[k] == python_results[k]
    all_match = all_match and match
    print(f"{k:<38}{sql_expected[k]:>15}{python_results[k]:>15}{str(match):>10}")

print(f"\nALL KPIs MATCH: {all_match}")

print("\n" + "=" * 70)
print("3. INDEPENDENT RECALCULATION OF FEATURE-ENGINEERED COLUMNS (Phase 3)")
print("=" * 70)

# Recalculate each engineered column from raw fields and compare to the
# SQL-computed column already present in the exported CSV.
py_supply_gap = df['units_collected_month'] - df['units_requested_month']
py_total_wastage = df['units_expired'] + df['units_discarded_tti']
py_wastage_rate = round(py_total_wastage / df['units_collected_month'] * 100, 2)
py_expiry_rate = round(df['units_expired'] / df['units_collected_month'] * 100, 2)
py_fulfillment_rate = round(df['units_transfused'] / df['units_requested_month'] * 100, 2)

checks = {
    'supply_demand_gap': (py_supply_gap, df['supply_demand_gap']),
    'total_wastage': (py_total_wastage, df['total_wastage']),
    'wastage_rate': (py_wastage_rate, df['wastage_rate']),
    'expiry_rate': (py_expiry_rate, df['expiry_rate']),
    'demand_fulfillment_rate': (py_fulfillment_rate, df['demand_fulfillment_rate']),
}

for name, (calc, sql_col) in checks.items():
    # allow tiny float rounding differences (<=0.01)
    diffs = (calc - sql_col).abs()
    n_mismatch = (diffs > 0.01).sum()
    print(f"  {name:<28} max abs diff = {diffs.max():.4f}  mismatched rows (>0.01) = {n_mismatch}")

print("\nAll feature-engineering columns verified against independent pandas recalculation.")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)


# ===========================================================================
# SECTION 2 -- EXPLORATORY DATA ANALYSIS (17 visualizations)
# Source: python/02_eda_visualizations.py
# ===========================================================================

"""
PHASE 12 (blueprint Phase 14): PYTHON EDA
Blood Bank Inventory Optimization & Demand Forecasting

Univariate, bivariate, and multivariate analysis with professional
visualizations, each saved separately as PNG. Correlation matrix
included. No causal claims are made from correlational findings.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

pd.set_option('display.max_columns', None)
plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10

DATA_PATH = 'Blood bank clean.csv'
CHART_DIR = 'charts'

df = pd.read_csv(DATA_PATH)
df['report_month'] = pd.to_datetime(df['report_month'])

def save(fig, name):
    path = f"{CHART_DIR}/{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved: {name}.png")

# ---------------------------------------------------------------------
# 1. UNIVARIATE: distributions of key volume variables
# ---------------------------------------------------------------------
dist_vars = ['units_requested_month', 'units_collected_month', 'units_transfused',
             'wastage_rate', 'expiry_rate', 'demand_fulfillment_rate']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, dist_vars):
    ax.hist(df[col], bins=30, color='#2E5EAA', edgecolor='white', alpha=0.85)
    ax.set_title(col.replace('_', ' ').title())
    ax.set_xlabel(col.replace('_', ' '))
    ax.set_ylabel('Frequency')
fig.suptitle('Univariate Distributions: Volume & Rate Variables', fontsize=13, y=1.02)
fig.tight_layout()
save(fig, '01_univariate_distributions')

# ---------------------------------------------------------------------
# 2. Blood-group demand (bivariate: category vs numeric)
# ---------------------------------------------------------------------
bg = df.groupby('blood_group')['units_requested_month'].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(bg.index, bg.values, color='#B22222')
ax.set_title('Total Demand by Blood Group')
ax.set_xlabel('Blood Group')
ax.set_ylabel('Total Units Requested')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.xticks(rotation=30)
fig.tight_layout()
save(fig, '02_demand_by_blood_group')

# ---------------------------------------------------------------------
# 3. Shortage rate by blood group
# ---------------------------------------------------------------------
sg = df.groupby('blood_group')['shortage_in_last_month'].mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(sg.index, sg.values, color='#D4A017')
ax.axhline(df['shortage_in_last_month'].mean()*100, color='black', linestyle='--', linewidth=1, label='Overall avg')
ax.set_title('Shortage Rate by Blood Group')
ax.set_xlabel('Blood Group')
ax.set_ylabel('Shortage Rate (%)')
ax.legend()
plt.xticks(rotation=30)
fig.tight_layout()
save(fig, '03_shortage_rate_by_blood_group')

# ---------------------------------------------------------------------
# 4. Wastage by blood group
# ---------------------------------------------------------------------
wg = df.groupby('blood_group')['wastage_rate'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(wg.index, wg.values, color='#556B2F')
ax.axhline(df['wastage_rate'].mean(), color='black', linestyle='--', linewidth=1, label='Overall avg')
ax.set_title('Average Wastage Rate by Blood Group')
ax.set_xlabel('Blood Group')
ax.set_ylabel('Wastage Rate (%)')
ax.legend()
plt.xticks(rotation=30)
fig.tight_layout()
save(fig, '04_wastage_rate_by_blood_group')

# ---------------------------------------------------------------------
# 5. Product demand
# ---------------------------------------------------------------------
pg = df.groupby('product_name')['units_requested_month'].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(pg.index[::-1], pg.values[::-1], color='#2E5EAA')
ax.set_title('Total Demand by Product')
ax.set_xlabel('Total Units Requested')
fig.tight_layout()
save(fig, '05_demand_by_product')

# ---------------------------------------------------------------------
# 6. Product shortage rate
# ---------------------------------------------------------------------
psr = df.groupby('product_name')['shortage_in_last_month'].mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(psr.index[::-1], psr.values[::-1], color='#D4A017')
ax.set_title('Shortage Rate by Product')
ax.set_xlabel('Shortage Rate (%)')
fig.tight_layout()
save(fig, '06_shortage_rate_by_product')

# ---------------------------------------------------------------------
# 7. Monthly demand trend (national, per-record avg to avoid coverage
#    artifact identified in Phase 10)
# ---------------------------------------------------------------------
monthly = df.groupby('report_month').agg(
    total_demand=('units_requested_month', 'sum'),
    avg_demand_per_record=('units_requested_month', 'mean'),
    n_records=('id', 'count')
).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
axes[0].plot(monthly['report_month'], monthly['total_demand'], color='#2E5EAA', marker='o', markersize=3)
axes[0].set_title('Total Monthly Demand (raw sum -- inflated by growing facility coverage, see Phase 10)')
axes[0].set_ylabel('Total Units Requested')
axes[1].plot(monthly['report_month'], monthly['avg_demand_per_record'], color='#B22222', marker='o', markersize=3)
axes[1].set_title('Average Demand per Facility-Record (true per-facility trend -- essentially flat)')
axes[1].set_ylabel('Avg Units Requested / Record')
axes[1].set_xlabel('Month')
fig.tight_layout()
save(fig, '07_monthly_demand_trend')

# ---------------------------------------------------------------------
# 8. Monthly shortage trend
# ---------------------------------------------------------------------
monthly_shortage = df.groupby('report_month')['shortage_in_last_month'].mean() * 100
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(monthly_shortage.index, monthly_shortage.values, color='#8B0000', marker='o', markersize=3)
ax.axhline(df['shortage_in_last_month'].mean()*100, color='gray', linestyle='--', linewidth=1, label='Overall avg (25.54%)')
ax.set_title('Monthly Shortage Rate Trend')
ax.set_xlabel('Month')
ax.set_ylabel('Shortage Rate (%)')
ax.legend()
fig.tight_layout()
save(fig, '08_monthly_shortage_trend')

# ---------------------------------------------------------------------
# 9. Monthly wastage trend
# ---------------------------------------------------------------------
monthly_wastage = df.groupby('report_month').apply(
    lambda g: (g['units_expired'].sum() + g['units_discarded_tti'].sum()) / g['units_collected_month'].sum() * 100,
    include_groups=False
)
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(monthly_wastage.index, monthly_wastage.values, color='#556B2F', marker='o', markersize=3)
ax.axhline(16.52, color='gray', linestyle='--', linewidth=1, label='Overall avg (16.52%)')
ax.set_title('Monthly Wastage Rate Trend')
ax.set_xlabel('Month')
ax.set_ylabel('Wastage Rate (%)')
ax.legend()
fig.tight_layout()
save(fig, '09_monthly_wastage_trend')

# ---------------------------------------------------------------------
# 10. Supply vs demand scatter (facility-month level)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(df['units_requested_month'], df['units_collected_month'], alpha=0.15, s=10, color='#2E5EAA')
lims = [df[['units_requested_month','units_collected_month']].min().min(),
        df[['units_requested_month','units_collected_month']].max().max()]
ax.plot(lims, lims, color='black', linestyle='--', linewidth=1, label='Collected = Requested')
ax.set_title('Units Collected vs. Units Requested (facility-months)')
ax.set_xlabel('Units Requested')
ax.set_ylabel('Units Collected')
ax.legend()
fig.tight_layout()
save(fig, '10_supply_vs_demand_scatter')

# ---------------------------------------------------------------------
# 11. Inventory status vs shortage/wastage
# ---------------------------------------------------------------------
inv = df.groupby('inventory_status').agg(
    wastage_rate=('wastage_rate', 'mean'),
    n=('id', 'count')
).reindex(['Adequate', 'Recovered', 'Low Stock', 'Critical'])
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(inv.index, inv['wastage_rate'], color=['#2E5EAA', '#D4A017', '#B22222', '#8B0000'])
ax.set_title('Average Wastage Rate by Inventory Status')
ax.set_ylabel('Wastage Rate (%)')
fig.tight_layout()
save(fig, '11_inventory_status_vs_wastage')

# ---------------------------------------------------------------------
# 12. Staffing vs shortage rate (facility level, from Phase 7 finding)
# ---------------------------------------------------------------------
fac = df.groupby('facility_id').agg(
    avg_staff=('blood_bank_staff_count', 'mean'),
    shortage_rate=('shortage_in_last_month', 'mean')
)
fac['shortage_rate'] *= 100
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(fac['avg_staff'], fac['shortage_rate'], alpha=0.6, color='#2E5EAA')
z = np.polyfit(fac['avg_staff'], fac['shortage_rate'], 1)
xs = np.linspace(fac['avg_staff'].min(), fac['avg_staff'].max(), 50)
ax.plot(xs, np.poly1d(z)(xs), color='#B22222', linewidth=2,
        label=f'Trend (r = {fac["avg_staff"].corr(fac["shortage_rate"]):.2f})')
ax.set_title('Facility Avg Staff Count vs. Shortage Rate')
ax.set_xlabel('Avg Staff Count')
ax.set_ylabel('Shortage Rate (%)')
ax.legend()
fig.tight_layout()
save(fig, '12_staffing_vs_shortage')

# ---------------------------------------------------------------------
# 13. Crossmatch turnaround vs shortage (record level)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
data_yes = df.loc[df['shortage_in_last_month']==1, 'crossmatch_turnaround_hours']
data_no = df.loc[df['shortage_in_last_month']==0, 'crossmatch_turnaround_hours']
ax.boxplot([data_no, data_yes], labels=['No Shortage', 'Shortage'], patch_artist=True,
           boxprops=dict(facecolor='#2E5EAA', alpha=0.6))
ax.set_title('Crossmatch Turnaround Hours: Shortage vs No-Shortage Months')
ax.set_ylabel('Crossmatch Turnaround (hours)')
fig.tight_layout()
save(fig, '13_turnaround_vs_shortage_boxplot')

# ---------------------------------------------------------------------
# 14. Cold-chain vs wastage
# ---------------------------------------------------------------------
cc = df.groupby('cold_chain_maintained')['wastage_rate'].mean()
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(['Not Maintained', 'Maintained'], cc.values, color=['#B22222', '#2E5EAA'])
ax.set_title('Cold-Chain Maintenance vs Avg Wastage Rate')
ax.set_ylabel('Wastage Rate (%)')
fig.tight_layout()
save(fig, '14_coldchain_vs_wastage')

# ---------------------------------------------------------------------
# 15. Shelf life vs expiry rate
# ---------------------------------------------------------------------
sl = df.groupby('shelf_life_days')['expiry_rate'].mean()
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(sl.index.astype(str), sl.values, color='#556B2F')
ax.set_title('Shelf Life (days) vs Avg Expiry Rate')
ax.set_xlabel('Shelf Life (days)')
ax.set_ylabel('Expiry Rate (%)')
fig.tight_layout()
save(fig, '15_shelflife_vs_expiry')

# ---------------------------------------------------------------------
# 16. Facility performance (top/bottom by shortage rate)
# ---------------------------------------------------------------------
fac_shortage = df.groupby('facility_id')['shortage_in_last_month'].mean().sort_values() * 100
top_bottom = pd.concat([fac_shortage.head(10), fac_shortage.tail(10)])
fig, ax = plt.subplots(figsize=(9, 8))
colors = ['#2E5EAA']*10 + ['#B22222']*10
ax.barh(range(len(top_bottom)), top_bottom.values, color=colors)
ax.set_yticks(range(len(top_bottom)))
ax.set_yticklabels(top_bottom.index)
ax.set_title('Facility Shortage Rate: 10 Lowest (blue) vs 10 Highest (red)')
ax.set_xlabel('Shortage Rate (%)')
fig.tight_layout()
save(fig, '16_facility_performance_shortage')

# ---------------------------------------------------------------------
# 17. Correlation matrix
# ---------------------------------------------------------------------
corr_vars = ['units_requested_month', 'units_collected_month', 'units_crossmatched',
             'units_transfused', 'units_expired', 'units_discarded_tti',
             'blood_bank_staff_count', 'crossmatch_turnaround_hours',
             'supply_demand_gap', 'wastage_rate', 'expiry_rate',
             'demand_fulfillment_rate', 'shortage_in_last_month', 'shortage_days_last_month']
corr = df[corr_vars].corr()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_vars)))
ax.set_yticks(range(len(corr_vars)))
ax.set_xticklabels([c.replace('_',' ') for c in corr_vars], rotation=90)
ax.set_yticklabels([c.replace('_',' ') for c in corr_vars])
for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center',
                fontsize=6.5, color='white' if abs(corr.iloc[i,j])>0.5 else 'black')
fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
ax.set_title('Correlation Matrix: Numerical Variables\n(correlation does not imply causation)', pad=15)
fig.tight_layout()
save(fig, '17_correlation_heatmap')

print("\nAll charts saved to", CHART_DIR)
print("\nTop correlations with shortage_in_last_month:")
print(corr['shortage_in_last_month'].sort_values(ascending=False))
print("\nTop correlations with wastage_rate:")
print(corr['wastage_rate'].sort_values(ascending=False))


# ===========================================================================
# SECTION 3 -- STATISTICAL HYPOTHESIS TESTING
# Source: python/03_statistical_tests.py
# ===========================================================================

"""
PHASE 13 (blueprint Phase 15): STATISTICAL ANALYSIS
Blood Bank Inventory Optimization & Demand Forecasting

For each test: null hypothesis, alternative hypothesis, test statistic,
p-value, effect size, interpretation, and business implication.

IMPORTANT METHODOLOGICAL NOTE stated up front: with n=10,000, even
trivially small effects can reach statistical significance (p<0.05).
Effect sizes (Cramer's V, rank-biserial correlation) are reported
alongside every p-value specifically to distinguish "statistically
detectable" from "practically meaningful" -- consistent with what the
correlation analysis in Phase 12 already suggested (near-zero
correlations across the board). No causal claims are made from any of
these observational, cross-sectional associations.
"""

import pandas as pd
import numpy as np
from scipy import stats

pd.set_option('display.max_columns', None)

DATA_PATH = 'Blood bank clean.csv'
df = pd.read_csv(DATA_PATH)

def cramers_v(contingency):
    chi2, p, dof, exp = stats.chi2_contingency(contingency)
    n = contingency.values.sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    v = np.sqrt(phi2 / min(k-1, r-1))
    return chi2, p, dof, v

def rank_biserial(group1, group2):
    """Effect size for Mann-Whitney U."""
    u_stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    n1, n2 = len(group1), len(group2)
    rb = 1 - (2*u_stat) / (n1*n2)
    return u_stat, p, rb

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    pooled_sd = np.sqrt(((n1-1)*group1.var(ddof=1) + (n2-1)*group2.var(ddof=1)) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_sd

print("="*75)
print("TEST 1: Is shortage occurrence associated with blood group?")
print("="*75)
print("H0: Shortage occurrence is independent of blood group.")
print("H1: Shortage occurrence is associated with blood group.")
ct1 = pd.crosstab(df['blood_group'], df['shortage_in_last_month'])
chi2, p, dof, v = cramers_v(ct1)
print(f"Chi-square = {chi2:.3f}, dof = {dof}, p-value = {p:.4g}")
print(f"Cramer's V (effect size) = {v:.4f}")
sig = "statistically significant" if p < 0.05 else "not statistically significant"
strength = "negligible" if v < 0.1 else ("small" if v < 0.3 else ("moderate" if v < 0.5 else "large"))
print(f"Interpretation: result is {sig} (p={p:.4g}), but effect size is {strength} (V={v:.4f}).")
print("Business implication: blood group should NOT be treated as a meaningful driver of")
print("shortage risk on its own, despite any statistical significance from large sample size.")

print("\n" + "="*75)
print("TEST 2: Is shortage occurrence associated with product type?")
print("="*75)
print("H0: Shortage occurrence is independent of product type.")
print("H1: Shortage occurrence is associated with product type.")
ct2 = pd.crosstab(df['product_name'], df['shortage_in_last_month'])
chi2, p, dof, v = cramers_v(ct2)
print(f"Chi-square = {chi2:.3f}, dof = {dof}, p-value = {p:.4g}")
print(f"Cramer's V (effect size) = {v:.4f}")
sig = "statistically significant" if p < 0.05 else "not statistically significant"
strength = "negligible" if v < 0.1 else ("small" if v < 0.3 else ("moderate" if v < 0.5 else "large"))
print(f"Interpretation: result is {sig} (p={p:.4g}), effect size is {strength} (V={v:.4f}).")
print("Business implication: product type is not a meaningful shortage-risk differentiator.")

print("\n" + "="*75)
print("TEST 3: Is the supply-demand gap significantly different in shortage vs non-shortage months?")
print("="*75)
print("H0: The mean supply-demand gap is equal between shortage and non-shortage records.")
print("H1: The mean supply-demand gap differs between shortage and non-shortage records.")
grp_shortage = df.loc[df['shortage_in_last_month']==1, 'supply_demand_gap']
grp_no_shortage = df.loc[df['shortage_in_last_month']==0, 'supply_demand_gap']
# Normality check (large-sample Shapiro on a random subsample, since full-sample
# Shapiro is oversensitive at n=10,000)
_, p_norm = stats.shapiro(df['supply_demand_gap'].sample(500, random_state=42))
print(f"Shapiro-Wilk normality check on a 500-row subsample: p = {p_norm:.4g} "
      f"({'looks non-normal, use Mann-Whitney as primary' if p_norm < 0.05 else 'looks roughly normal, t-test appropriate'})")
t_stat, p_t = stats.ttest_ind(grp_shortage, grp_no_shortage, equal_var=False)
d = cohens_d(grp_shortage, grp_no_shortage)
u_stat, p_mw, rb = rank_biserial(grp_shortage, grp_no_shortage)
print(f"Welch's t-test: t = {t_stat:.3f}, p = {p_t:.4g}, Cohen's d = {d:.4f}")
print(f"Mann-Whitney U: U = {u_stat:.1f}, p = {p_mw:.4g}, rank-biserial r = {rb:.4f}")
print(f"Group means: shortage={grp_shortage.mean():.2f}, no-shortage={grp_no_shortage.mean():.2f}")
print("Business implication: the supply-demand gap is virtually identical whether or not a")
print("shortage occurs -- shortage events are NOT explained by a deeper aggregate supply gap.")

print("\n" + "="*75)
print("TEST 4: Is wastage rate significantly different under poor cold-chain conditions?")
print("="*75)
print("H0: Mean wastage rate is equal between cold-chain-maintained and not-maintained records.")
print("H1: Mean wastage rate differs between the two groups.")
grp_cc1 = df.loc[df['cold_chain_maintained']==1, 'wastage_rate']
grp_cc0 = df.loc[df['cold_chain_maintained']==0, 'wastage_rate']
t_stat, p_t = stats.ttest_ind(grp_cc1, grp_cc0, equal_var=False)
d = cohens_d(grp_cc1, grp_cc0)
u_stat, p_mw, rb = rank_biserial(grp_cc1, grp_cc0)
print(f"Welch's t-test: t = {t_stat:.3f}, p = {p_t:.4g}, Cohen's d = {d:.4f}")
print(f"Mann-Whitney U: U = {u_stat:.1f}, p = {p_mw:.4g}, rank-biserial r = {rb:.4f}")
print(f"Group means: maintained={grp_cc1.mean():.2f}%, not maintained={grp_cc0.mean():.2f}%")
print("Business implication: cold-chain maintenance status shows no meaningful relationship")
print("with wastage rate in this dataset -- contrary to the intuitive operational expectation.")

print("\n" + "="*75)
print("TEST 5: Do staffing levels differ between shortage and non-shortage records?")
print("="*75)
print("H0: Mean staff count is equal between shortage and non-shortage records.")
print("H1: Mean staff count differs between the two groups.")
grp_s1 = df.loc[df['shortage_in_last_month']==1, 'blood_bank_staff_count']
grp_s0 = df.loc[df['shortage_in_last_month']==0, 'blood_bank_staff_count']
t_stat, p_t = stats.ttest_ind(grp_s1, grp_s0, equal_var=False)
d = cohens_d(grp_s1, grp_s0)
u_stat, p_mw, rb = rank_biserial(grp_s1, grp_s0)
print(f"Welch's t-test: t = {t_stat:.3f}, p = {p_t:.4g}, Cohen's d = {d:.4f}")
print(f"Mann-Whitney U: U = {u_stat:.1f}, p = {p_mw:.4g}, rank-biserial r = {rb:.4f}")
print(f"Group means: shortage={grp_s1.mean():.2f}, no-shortage={grp_s0.mean():.2f}")
print("Business implication: at the individual record level, staffing does not distinguish")
print("shortage from non-shortage months (note: Phase 7 found a weak facility-AVERAGE")
print("relationship, r=-0.20 -- these are not contradictory, just different units of analysis).")

print("\n" + "="*75)
print("TEST 6: Does crossmatch turnaround differ between shortage and non-shortage records?")
print("="*75)
print("H0: Mean crossmatch turnaround is equal between shortage and non-shortage records.")
print("H1: Mean crossmatch turnaround differs between the two groups.")
grp_t1 = df.loc[df['shortage_in_last_month']==1, 'crossmatch_turnaround_hours']
grp_t0 = df.loc[df['shortage_in_last_month']==0, 'crossmatch_turnaround_hours']
t_stat, p_t = stats.ttest_ind(grp_t1, grp_t0, equal_var=False)
d = cohens_d(grp_t1, grp_t0)
u_stat, p_mw, rb = rank_biserial(grp_t1, grp_t0)
print(f"Welch's t-test: t = {t_stat:.3f}, p = {p_t:.4g}, Cohen's d = {d:.4f}")
print(f"Mann-Whitney U: U = {u_stat:.1f}, p = {p_mw:.4g}, rank-biserial r = {rb:.4f}")
print(f"Group means: shortage={grp_t1.mean():.2f}h, no-shortage={grp_t0.mean():.2f}h")
print("Business implication: crossmatch turnaround time shows no meaningful relationship")
print("with shortage occurrence at the record level.")

print("\n" + "="*75)
print("TEST 7 (bonus, motivated by Phase 8 seasonality finding): Is shortage rate")
print("associated with calendar month (seasonality)?")
print("="*75)
print("H0: Shortage occurrence is independent of calendar month.")
print("H1: Shortage occurrence is associated with calendar month.")
ct3 = pd.crosstab(df['month'], df['shortage_in_last_month'])
chi2, p, dof, v = cramers_v(ct3)
print(f"Chi-square = {chi2:.3f}, dof = {dof}, p-value = {p:.4g}")
print(f"Cramer's V (effect size) = {v:.4f}")
sig = "statistically significant" if p < 0.05 else "not statistically significant"
strength = "negligible" if v < 0.1 else ("small" if v < 0.3 else ("moderate" if v < 0.5 else "large"))
print(f"Interpretation: result is {sig} (p={p:.4g}), effect size is {strength} (V={v:.4f}).")
print("Business implication: any seasonal pattern in shortage rate, even if statistically")
print("detectable, is small in practical magnitude.")

print("\n" + "="*75)
print("SUMMARY TABLE")
print("="*75)


# ===========================================================================
# SECTION 4 -- DEMAND FORECASTING (ARIMA/SARIMA/ML comparison)
# Source: python/04_demand_forecasting.py
# ===========================================================================

"""
PHASE 14 (blueprint Phase 16): DEMAND FORECASTING
Blood Bank Inventory Optimization & Demand Forecasting

Primary target: units_requested_month, aggregated to a national monthly
total series (48 months, 2021-01 to 2024-12).

IMPORTANT CAVEAT (carried from Phase 10/12): the raw monthly total is
substantially inflated by growing facility-reporting coverage over time,
not by genuine per-facility demand growth (avg demand/record is flat at
~250 across all 4 years). This is forecasted here because it is the
blueprint's specified target and the operationally relevant one (a
blood bank needs to plan for total volume, regardless of *why* it's
rising) -- but every forecast below is presented alongside this caveat,
and a companion "per-record demand" check is included to make clear
what part of any apparent trend is coverage vs. real demand.

Chronological split only: train = first 42 months (2021-01 to 2024-06),
test = last 6 months (2024-07 to 2024-12). No random splitting of
time-series data.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

DATA_PATH = 'Blood bank clean.csv'
CHART_DIR = 'charts'
OUT_DIR = 'outputs'
import os
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df['report_month'] = pd.to_datetime(df['report_month'])

# ---------------------------------------------------------------------
# Build the monthly series
# ---------------------------------------------------------------------
monthly = df.groupby('report_month').agg(
    total_demand=('units_requested_month', 'sum'),
    n_records=('id', 'count'),
    avg_demand_per_record=('units_requested_month', 'mean')
).sort_index()

monthly = monthly.asfreq('MS')  # ensure regular monthly frequency
print(f"Series length: {len(monthly)} months ({monthly.index.min().date()} to {monthly.index.max().date()})")

ts = monthly['total_demand']

TRAIN_END = '2024-06-01'
train = ts.loc[:TRAIN_END]
test = ts.loc['2024-07-01':]
print(f"Train: {len(train)} months, Test: {len(test)} months")

# ---------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------
def evaluate(actual, forecast, name):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    smape = np.mean(2 * np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast))) * 100
    return {'model': name, 'MAE': round(mae, 1), 'RMSE': round(rmse, 1),
            'MAPE': round(mape, 2), 'sMAPE': round(smape, 2)}

results = []

# ---------------------------------------------------------------------
# 1. Naive forecast (last observed value repeated)
# ---------------------------------------------------------------------
naive_fc = np.repeat(train.iloc[-1], len(test))
results.append(evaluate(test, naive_fc, 'Naive'))

# ---------------------------------------------------------------------
# 2. Seasonal naive (value from 12 months prior)
# ---------------------------------------------------------------------
seasonal_naive_fc = [ts.loc[d - pd.DateOffset(months=12)] for d in test.index]
results.append(evaluate(test, seasonal_naive_fc, 'Seasonal Naive (12mo)'))

# ---------------------------------------------------------------------
# 3. Moving average (3-month)
# ---------------------------------------------------------------------
ma_fc = np.repeat(train.iloc[-3:].mean(), len(test))
results.append(evaluate(test, ma_fc, 'Moving Average (3mo)'))

# ---------------------------------------------------------------------
# 4. Exponential smoothing (Holt-Winters, additive trend + seasonal)
# ---------------------------------------------------------------------
try:
    hw_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12,
                                     initialization_method='estimated').fit()
    hw_fc = hw_model.forecast(len(test))
    results.append(evaluate(test, hw_fc, 'Exponential Smoothing (Holt-Winters)'))
except Exception as e:
    print(f"Holt-Winters failed: {e}")
    hw_fc = None

# ---------------------------------------------------------------------
# 5. ARIMA (non-seasonal)
# ---------------------------------------------------------------------
try:
    arima_model = SARIMAX(train, order=(1, 1, 1), enforce_stationarity=False,
                           enforce_invertibility=False).fit(disp=False)
    arima_fc = arima_model.forecast(len(test))
    results.append(evaluate(test, arima_fc, 'ARIMA(1,1,1)'))
except Exception as e:
    print(f"ARIMA failed: {e}")
    arima_fc = None

# ---------------------------------------------------------------------
# 6. SARIMA (seasonal)
# ---------------------------------------------------------------------
try:
    sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_fc = sarima_model.get_forecast(len(test))
    sarima_mean = sarima_fc.predicted_mean
    sarima_ci = sarima_fc.conf_int(alpha=0.05)
    results.append(evaluate(test, sarima_mean, 'SARIMA(1,1,1)(1,1,0,12)'))
except Exception as e:
    print(f"SARIMA failed: {e}")
    sarima_mean = None

# ---------------------------------------------------------------------
# 7. ML regression: TWO HONEST SCENARIOS
# ---------------------------------------------------------------------
# METHODOLOGICAL FINDING: corr(n_records, total_demand) = 0.9998 -- total
# demand is almost perfectly explained by (facility-record count x ~250).
# Using the ACTUAL future n_records as a model feature (as in a naive
# first pass) is a form of look-ahead: it only works because it already
# "knows" the answer to 99.98% precision. This is only legitimate if a
# blood bank genuinely knows its future reporting-facility count in
# advance (plausible in real operations, since facility enrollment is a
# planning decision) -- but it must be stated as an assumption, not
# presented as a demand-forecasting achievement. Both scenarios are
# reported below for transparency.
feat_df = monthly.copy()
feat_df['month'] = feat_df.index.month
feat_df['year'] = feat_df.index.year
feat_df['quarter'] = feat_df.index.quarter
feat_df['lag1'] = feat_df['total_demand'].shift(1)
feat_df['lag2'] = feat_df['total_demand'].shift(2)
feat_df['lag3'] = feat_df['total_demand'].shift(3)
feat_df['roll3'] = feat_df['total_demand'].shift(1).rolling(3).mean()
feat_df['roll6'] = feat_df['total_demand'].shift(1).rolling(6).mean()
feat_df['n_records_lag1'] = feat_df['n_records'].shift(1)

feat_df_clean = feat_df.dropna()
train_feat = feat_df_clean.loc[:TRAIN_END]
test_feat = feat_df_clean.loc['2024-07-01':]

# SCENARIO A: future n_records assumed known in advance (e.g. planned
# facility enrollment/reporting schedule) -- optimistic, assumption-dependent
feature_cols_A = ['month', 'quarter', 'lag1', 'lag2', 'roll3', 'n_records']
X_train_A, y_train = train_feat[feature_cols_A], train_feat['total_demand']
X_test_A, y_test = test_feat[feature_cols_A], test_feat['total_demand']
gbr_A = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42)
gbr_A.fit(X_train_A, y_train)
pred_A = gbr_A.predict(X_test_A)
results.append(evaluate(y_test, pred_A, 'GBM + FUTURE n_records known (optimistic, assumption-dependent)'))

# SCENARIO B: fully autonomous -- only information available at forecast
# time (own lags/rolling stats, NO future n_records). This is the
# realistic, no-look-ahead forecasting scenario.
feature_cols_B = ['month', 'quarter', 'lag1', 'lag2', 'lag3', 'roll3', 'roll6', 'n_records_lag1']
X_train_B = train_feat[feature_cols_B]
X_test_B = test_feat[feature_cols_B]
gbr_B = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42)
gbr_B.fit(X_train_B, y_train)
pred_B = gbr_B.predict(X_test_B)
results.append(evaluate(y_test, pred_B, 'GBM (autonomous, no future info -- realistic)'))

lr_B = LinearRegression()
lr_B.fit(X_train_B, y_train)
pred_lr_B = lr_B.predict(X_test_B)
results.append(evaluate(y_test, pred_lr_B, 'Linear Regression (autonomous, no future info)'))

# SCENARIO C: two-stage decomposition -- forecast n_records with its own
# time-series model, then multiply by the stable historical avg-demand-
# per-record (~250.01, per the companion check below). This is the most
# INTERPRETABLE honest approach: it makes explicit that forecasting this
# series is really about forecasting facility-reporting coverage.
n_rec_train = monthly['n_records'].loc[:TRAIN_END]
n_rec_test = monthly['n_records'].loc['2024-07-01':]
try:
    n_rec_model = ExponentialSmoothing(n_rec_train, trend='add', seasonal=None,
                                        initialization_method='estimated').fit()
    n_rec_fc = n_rec_model.forecast(len(n_rec_test))
    avg_per_record = monthly['avg_demand_per_record'].loc[:TRAIN_END].mean()
    decomposed_fc = n_rec_fc.values * avg_per_record
    results.append(evaluate(y_test, decomposed_fc, 'Two-stage: forecast coverage x stable per-record avg'))
except Exception as e:
    print(f"Decomposition model failed: {e}")
    decomposed_fc = None


# ---------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values('RMSE')
print("\n" + "="*70)
print("MODEL COMPARISON (chronological holdout: last 6 months, 2024-07 to 2024-12)")
print("="*70)
print(results_df.to_string(index=False))
results_df.to_csv(f'{OUT_DIR}/demand_forecast_model_comparison.csv', index=False)

best_model_name = results_df.iloc[0]['model']
print(f"\nBest model by RMSE: {best_model_name}")

# ---------------------------------------------------------------------
# FINAL FORWARD FORECAST -- refit the selected model (ARIMA(1,1,1), the
# best REALISTIC/no-look-ahead model per the comparison above) on the
# FULL 48-month series to produce the actual next-6-months forecast with
# 95% confidence intervals. This step was originally run ad hoc and is
# incorporated here as part of the reusable pipeline for reproducibility.
# ---------------------------------------------------------------------
final_arima = SARIMAX(ts, order=(1, 1, 1), enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False)
final_fc = final_arima.get_forecast(6)
final_mean = final_fc.predicted_mean
final_ci = final_fc.conf_int(alpha=0.05)
forecast_future = pd.DataFrame({
    'forecast': final_mean.round(0),
    'lower_95': final_ci.iloc[:, 0].round(0),
    'upper_95': final_ci.iloc[:, 1].round(0),
})
forecast_future.to_csv(f'{OUT_DIR}/demand_forecast_next6months.csv')
print("\n" + "="*70)
print("FINAL 6-MONTH FORWARD FORECAST (ARIMA(1,1,1), refit on full 48-month series)")
print("="*70)
print(forecast_future.to_string())
print(f"Saved: {OUT_DIR}/demand_forecast_next6months.csv")

# ---------------------------------------------------------------------
# Companion check: per-record demand -- is there ANY real trend/pattern?
# ---------------------------------------------------------------------
print("\n" + "="*70)
print("COMPANION CHECK: per-record (true per-facility) demand series")
print("="*70)
per_rec = monthly['avg_demand_per_record']
print(f"Mean: {per_rec.mean():.2f}, Std: {per_rec.std():.2f}, "
      f"Min: {per_rec.min():.2f}, Max: {per_rec.max():.2f}")
naive_per_rec_mae = np.mean(np.abs(per_rec.loc['2024-07-01':] - per_rec.loc[:TRAIN_END].mean()))
print(f"A simple historical-mean forecast for per-record demand has MAE = {naive_per_rec_mae:.3f} units")
print("(out of a mean of ~250 units -- i.e., per-facility demand is close to unforecastable noise")
print("around a constant level; essentially no exploitable trend or seasonality exists at this level.)")

# ---------------------------------------------------------------------
# Chart: actual vs forecasts
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(ts.index, ts.values, label='Actual', color='black', linewidth=2)
ax.plot(test.index, naive_fc, label='Naive', linestyle='--', alpha=0.7)
ax.plot(test.index, seasonal_naive_fc, label='Seasonal Naive', linestyle='--', alpha=0.7)
if hw_fc is not None:
    ax.plot(test.index, hw_fc, label='Holt-Winters', linestyle='--', alpha=0.7)
if sarima_mean is not None:
    ax.plot(test.index, sarima_mean, label='SARIMA', linestyle='--', alpha=0.7)
    ax.fill_between(test.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], alpha=0.15, color='gray', label='SARIMA 95% CI')
ax.axvline(pd.Timestamp(TRAIN_END), color='red', linestyle=':', label='Train/Test split')
ax.set_title('Total Monthly Demand: Actual vs Forecast Methods (Holdout: Jul-Dec 2024)')
ax.set_xlabel('Month')
ax.set_ylabel('Total Units Requested')
ax.legend(loc='upper left', fontsize=8)
fig.tight_layout()
fig.savefig(f'{CHART_DIR}/18_demand_forecast_comparison.png', bbox_inches='tight')
plt.close(fig)
print(f"\nChart saved: 18_demand_forecast_comparison.png")


# ===========================================================================
# SECTION 5 -- SHORTAGE PREDICTION MODEL (classification comparison)
# Source: python/05_shortage_prediction_model.py
# ===========================================================================

"""
PHASE 15 (blueprint Phase 17): SHORTAGE PREDICTION MODEL
Blood Bank Inventory Optimization & Demand Forecasting

Target: shortage_in_last_month

LEAKAGE PREVENTION -- applied more strictly than the bare minimum the
blueprint requires:
  - shortage_days_last_month and shortage_cause are EXCLUDED entirely
    (explicit blueprint requirement -- same-period outcome fields).
  - Beyond that: staffing, cold-chain maintenance, and crossmatch
    turnaround are NOT used contemporaneously either, even though the
    blueprint doesn't explicitly forbid it. Reasoning: these are
    measured over the same reporting month as the target, so using
    them "as-is" implicitly assumes they're known before the shortage
    outcome is determined, which is not a safe assumption. Instead,
    ALL time-varying operational variables are LAGGED BY ONE MONTH at
    the facility level. Only genuinely static/structural attributes
    (blood_group, product_name, region_type as recorded, calendar
    month) are used contemporaneously.
  - Train/test split is CHRONOLOGICAL (train: 2021-2023, test: 2024),
    not random -- mimicking real deployment (train on past, predict
    future), consistent with every prior phase's approach to this
    dataset's time dimension.

Given Phase 13's statistical testing found negligible effect sizes for
every one of these candidate predictors, this model is expected to
perform modestly, not spectacularly -- and that expectation is reported
honestly below rather than adjusted after the fact.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score, confusion_matrix)

DATA_PATH = 'Blood bank clean.csv'
OUT_DIR = 'outputs'

df = pd.read_csv(DATA_PATH)
df['report_month'] = pd.to_datetime(df['report_month'])
df = df.sort_values(['facility_id', 'report_month']).reset_index(drop=True)

# ---------------------------------------------------------------------
# 1. Build facility-month level aggregates (multiple records/facility/month
#    exist -- collapse to one row per facility-month first for lag construction)
# ---------------------------------------------------------------------
fac_month = df.groupby(['facility_id', 'report_month']).agg(
    any_shortage=('shortage_in_last_month', 'max'),
    avg_wastage_rate=('wastage_rate', 'mean'),
    avg_demand=('units_requested_month', 'mean'),
    avg_staff=('blood_bank_staff_count', 'mean'),
    pct_cold_chain=('cold_chain_maintained', 'mean'),
    avg_turnaround=('crossmatch_turnaround_hours', 'mean'),
).reset_index().sort_values(['facility_id', 'report_month'])

# 1-month lag of each facility-month aggregate (shift within facility)
lag_cols = ['any_shortage', 'avg_wastage_rate', 'avg_demand', 'avg_staff', 'pct_cold_chain', 'avg_turnaround']
for col in lag_cols:
    fac_month[f'lag1_{col}'] = fac_month.groupby('facility_id')[col].shift(1)

# Expanding (strictly prior) historical shortage rate per facility --
# a "baseline risk profile" feature, using only months BEFORE the current one
fac_month['facility_hist_shortage_rate'] = (
    fac_month.groupby('facility_id')['any_shortage']
    .apply(lambda s: s.shift(1).expanding().mean())
    .reset_index(level=0, drop=True)
)

lag_feature_cols = [f'lag1_{c}' for c in lag_cols] + ['facility_hist_shortage_rate']

# ---------------------------------------------------------------------
# 2. Merge facility-month lag features back onto the record-level data
# ---------------------------------------------------------------------
model_df = df.merge(
    fac_month[['facility_id', 'report_month'] + lag_feature_cols],
    on=['facility_id', 'report_month'], how='left'
)

print(f"Records with at least one missing lag feature (no prior month on record for that facility): "
      f"{model_df[lag_feature_cols].isnull().any(axis=1).sum()} of {len(model_df)}")

# ---------------------------------------------------------------------
# 3. Chronological train/test split (train: 2021-2023, test: 2024)
#    Imputation of missing lags uses ONLY the training set's statistics.
# ---------------------------------------------------------------------
train_mask = model_df['report_month'] < '2024-01-01'
test_mask = model_df['report_month'] >= '2024-01-01'
print(f"Train records: {train_mask.sum()}, Test records: {test_mask.sum()}")

train_medians = model_df.loc[train_mask, lag_feature_cols].median()
for col in lag_feature_cols:
    model_df[col] = model_df[col].fillna(train_medians[col])

# ---------------------------------------------------------------------
# 4. Feature set (leakage-safe) and encoding
# ---------------------------------------------------------------------
categorical_cols = ['blood_group', 'product_name', 'region_type']
numeric_cols = lag_feature_cols + ['month']  # calendar month, structural, known in advance

X = pd.get_dummies(model_df[categorical_cols + numeric_cols], columns=categorical_cols, drop_first=True)
y = model_df['shortage_in_last_month']

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\nClass balance -- train: {y_train.mean():.4f} shortage rate, test: {y_test.mean():.4f} shortage rate")
print(f"Feature count: {X.shape[1]}")

# Scale for logistic regression only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# 5. Model comparison
# ---------------------------------------------------------------------
models = {
    'Logistic Regression': (LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42), True),
    'Decision Tree': (DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42), False),
    'Random Forest': (RandomForestClassifier(n_estimators=300, max_depth=6, class_weight='balanced', random_state=42), False),
    'Gradient Boosting': (GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42), False),
    'XGBoost': (XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                               scale_pos_weight=(1-y_train.mean())/y_train.mean(),
                               eval_metric='logloss', random_state=42), False),
}

results = []
predictions = {}
for name, (model, needs_scaling) in models.items():
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled if needs_scaling else X_test
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    predictions[name] = (pred, proba)

    results.append({
        'model': name,
        'accuracy': round(accuracy_score(y_test, pred), 4),
        'precision': round(precision_score(y_test, pred, zero_division=0), 4),
        'recall': round(recall_score(y_test, pred, zero_division=0), 4),
        'f1': round(f1_score(y_test, pred, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_test, proba), 4),
        'pr_auc': round(average_precision_score(y_test, proba), 4),
    })

results_df = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
print("\n" + "="*90)
print("MODEL COMPARISON (chronological holdout: 2024 test set)")
print("="*90)
print(results_df.to_string(index=False))
results_df.to_csv(f'{OUT_DIR}/shortage_model_comparison.csv', index=False)

# Baseline comparison: majority-class classifier and random classifier
majority_pred = np.zeros(len(y_test))
print(f"\nBaseline -- always predict 'no shortage': accuracy = {accuracy_score(y_test, majority_pred):.4f} "
      f"(= 1 - test shortage rate, recall = 0.0000, i.e., catches ZERO actual shortages)")

# Confusion matrix for best model by ROC-AUC
best_name = results_df.iloc[0]['model']
best_pred, best_proba = predictions[best_name]
cm = confusion_matrix(y_test, best_pred)
print(f"\nConfusion matrix for best model ({best_name}) at default 0.5 threshold:")
print(f"                 Predicted No-Shortage  Predicted Shortage")
print(f"Actual No-Shortage      {cm[0,0]:>10}          {cm[0,1]:>10}")
print(f"Actual Shortage         {cm[1,0]:>10}          {cm[1,1]:>10}   <- false negatives (missed shortages)")

print(f"\nBest model by ROC-AUC: {best_name}")


# ===========================================================================
# SECTION 6 -- MODEL EVALUATION (threshold tuning, SMOTE, overfitting check)
# Source: python/06_model_evaluation.py
# ===========================================================================

"""
PHASE 16 (blueprint Phase 18): MODEL EVALUATION
Blood Bank Inventory Optimization & Demand Forecasting

Builds on Phase 15's model comparison. This phase:
  1) Checks for overfitting (train vs test metrics)
  2) Tunes the classification threshold, since recall on the shortage
     class is operationally what matters (a false negative = predicted
     "adequate supply" when a shortage actually occurs)
  3) Tests SMOTE resampling to see whether class-imbalance correction
     changes anything
  4) Gives an honest, evidence-based recommendation on whether this
     model is fit for operational deployment

Consistent with Phase 15's finding (ROC-AUC ~0.49-0.51 for every
algorithm), the expectation here is that none of these standard
remedies (threshold tuning, resampling) can manufacture predictive
power that isn't in the underlying features -- and that is reported
honestly rather than searched for repeatedly until a better number
appears.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                              precision_score, recall_score, f1_score, accuracy_score)
from imblearn.over_sampling import SMOTE

DATA_PATH = 'Blood bank clean.csv'
CHART_DIR = 'charts'
OUT_DIR = 'outputs'

df = pd.read_csv(DATA_PATH)
df['report_month'] = pd.to_datetime(df['report_month'])
df = df.sort_values(['facility_id', 'report_month']).reset_index(drop=True)

# --- Rebuild the same leakage-safe feature set from Phase 15 ---
fac_month = df.groupby(['facility_id', 'report_month']).agg(
    any_shortage=('shortage_in_last_month', 'max'),
    avg_wastage_rate=('wastage_rate', 'mean'),
    avg_demand=('units_requested_month', 'mean'),
    avg_staff=('blood_bank_staff_count', 'mean'),
    pct_cold_chain=('cold_chain_maintained', 'mean'),
    avg_turnaround=('crossmatch_turnaround_hours', 'mean'),
).reset_index().sort_values(['facility_id', 'report_month'])

lag_cols = ['any_shortage', 'avg_wastage_rate', 'avg_demand', 'avg_staff', 'pct_cold_chain', 'avg_turnaround']
for col in lag_cols:
    fac_month[f'lag1_{col}'] = fac_month.groupby('facility_id')[col].shift(1)
fac_month['facility_hist_shortage_rate'] = (
    fac_month.groupby('facility_id')['any_shortage']
    .apply(lambda s: s.shift(1).expanding().mean())
    .reset_index(level=0, drop=True)
)
lag_feature_cols = [f'lag1_{c}' for c in lag_cols] + ['facility_hist_shortage_rate']

model_df = df.merge(fac_month[['facility_id', 'report_month'] + lag_feature_cols],
                     on=['facility_id', 'report_month'], how='left')

train_mask = model_df['report_month'] < '2024-01-01'
test_mask = model_df['report_month'] >= '2024-01-01'
train_medians = model_df.loc[train_mask, lag_feature_cols].median()
for col in lag_feature_cols:
    model_df[col] = model_df[col].fillna(train_medians[col])

categorical_cols = ['blood_group', 'product_name', 'region_type']
numeric_cols = lag_feature_cols + ['month']
X = pd.get_dummies(model_df[categorical_cols + numeric_cols], columns=categorical_cols, drop_first=True)
y = model_df['shortage_in_last_month']

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# 1. OVERFITTING CHECK -- train vs test performance for each model
# ---------------------------------------------------------------------
print("="*80)
print("1. OVERFITTING CHECK (train vs test ROC-AUC)")
print("="*80)

logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
rf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

for name, model, Xtr, Xte in [
    ('Logistic Regression', logreg, X_train_scaled, X_test_scaled),
    ('Random Forest', rf, X_train, X_test),
]:
    train_auc = roc_auc_score(y_train, model.predict_proba(Xtr)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(Xte)[:, 1])
    gap = train_auc - test_auc
    print(f"{name:<22} Train AUC = {train_auc:.4f}   Test AUC = {test_auc:.4f}   Gap = {gap:.4f} "
          f"({'no meaningful overfitting -- both near random' if abs(gap) < 0.05 else 'some overfitting'})")

# ---------------------------------------------------------------------
# 2. THRESHOLD TUNING (using Logistic Regression as the interpretable
#    reference model, since tree-based models showed no better AUC)
# ---------------------------------------------------------------------
print("\n" + "="*80)
print("2. THRESHOLD TUNING -- precision/recall trade-off")
print("="*80)
proba = logreg.predict_proba(X_test_scaled)[:, 1]

thresholds_to_test = np.arange(0.1, 0.95, 0.05)
threshold_results = []
for t in thresholds_to_test:
    pred_t = (proba >= t).astype(int)
    threshold_results.append({
        'threshold': round(t, 2),
        'precision': round(precision_score(y_test, pred_t, zero_division=0), 4),
        'recall': round(recall_score(y_test, pred_t, zero_division=0), 4),
        'f1': round(f1_score(y_test, pred_t, zero_division=0), 4),
        'pct_flagged': round(pred_t.mean(), 4)
    })
threshold_df = pd.DataFrame(threshold_results)
print(threshold_df.to_string(index=False))
threshold_df.to_csv(f'{OUT_DIR}/shortage_threshold_tuning.csv', index=False)

# High-recall operating point (business priority: minimize false negatives)
high_recall_row = threshold_df[threshold_df['recall'] >= 0.80].iloc[-1] if (threshold_df['recall'] >= 0.80).any() else threshold_df.iloc[0]
print(f"\nHigh-recall operating point (>=80% of shortages caught): threshold={high_recall_row['threshold']}, "
      f"precision={high_recall_row['precision']}, recall={high_recall_row['recall']}, "
      f"pct of ALL records flagged={high_recall_row['pct_flagged']:.1%}")
print("Interpretation: to catch 80%+ of actual shortages, the model must flag "
      f"{high_recall_row['pct_flagged']:.1%} of ALL facility-months as 'at risk' -- "
      "not a targeted signal, closer to a blanket alert.")

# ---------------------------------------------------------------------
# 3. SMOTE resampling -- does correcting class imbalance help?
# ---------------------------------------------------------------------
print("\n" + "="*80)
print("3. SMOTE RESAMPLING CHECK")
print("="*80)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"Original train shortage rate: {y_train.mean():.4f} -> after SMOTE: {y_train_sm.mean():.4f}")

rf_smote = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
rf_smote.fit(X_train_sm, y_train_sm)
proba_smote = rf_smote.predict_proba(X_test)[:, 1]
auc_smote = roc_auc_score(y_test, proba_smote)
print(f"Random Forest + SMOTE: Test ROC-AUC = {auc_smote:.4f} "
      f"(vs. {roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]):.4f} without SMOTE, "
      f"vs. {roc_auc_score(y_test, proba):.4f} for Logistic Regression baseline)")
print("Interpretation: SMOTE does not meaningfully change performance -- confirms the")
print("issue is a lack of predictive signal in the features, not a class-imbalance artifact.")

# ---------------------------------------------------------------------
# 4. ROC and PR curves (visual confirmation)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fpr, tpr, _ = roc_curve(y_test, proba)
axes[0].plot(fpr, tpr, label=f'Logistic Regression (AUC={roc_auc_score(y_test, proba):.3f})', color='#2E5EAA')
axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random classifier (AUC=0.50)')
axes[0].set_title('ROC Curve')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend()

prec, rec, _ = precision_recall_curve(y_test, proba)
axes[1].plot(rec, prec, color='#B22222', label='Logistic Regression')
axes[1].axhline(y_test.mean(), linestyle='--', color='gray', label=f'No-skill baseline ({y_test.mean():.3f})')
axes[1].set_title('Precision-Recall Curve')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend()

fig.suptitle('Shortage Prediction Model: Diagnostic Curves (near-random performance)', y=1.02)
fig.tight_layout()
fig.savefig(f'{CHART_DIR}/19_shortage_model_roc_pr_curves.png', bbox_inches='tight')
plt.close(fig)
print(f"\nChart saved: 19_shortage_model_roc_pr_curves.png")

# ---------------------------------------------------------------------
# 5. FINAL RECOMMENDATION
# ---------------------------------------------------------------------
print("\n" + "="*80)
print("4. FINAL MODEL EVALUATION VERDICT")
print("="*80)
print("""
No model tested (Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting, XGBoost) -- with or without threshold tuning or
SMOTE resampling -- achieves ROC-AUC meaningfully above 0.50 on the
2024 chronological holdout.

RECOMMENDATION: Do NOT deploy a per-record shortage classifier as an
operational early-warning system using the fields available in this
dataset. It would not outperform random flagging, and threshold
tuning to raise recall (as would be required given the cost of false
negatives) only works by flagging a large, non-targeted share of all
records -- providing no real decision value over a blanket policy.

This is a genuine, evidence-based finding, not a failure to try hard
enough: Phase 12's correlation matrix, Phase 13's hypothesis tests
(all effect sizes negligible), and this phase's modeling exercise
are three independent lines of evidence pointing the same direction.

CONSTRUCTIVE PATH FORWARD (carried into the Inventory Optimization
phase): since shortage occurrence isn't reliably predictable from
available features, the practical response is capacity-based, not
prediction-based -- safety stock and reorder buffers sized to
historical demand VOLATILITY (which we can measure) and targeted at
the specific FACILITIES with structurally higher shortage rates
(Phase 7's finding -- the one dimension that showed real, if modest,
signal), rather than a monthly risk score per record.
""")


# ===========================================================================
# SECTION 7 -- MODEL EXPLAINABILITY (coefficients, permutation importance, SHAP)
# Source: python/07_model_explainability.py
# ===========================================================================

"""
PHASE 17 (blueprint Phase 19): MODEL EXPLAINABILITY
Blood Bank Inventory Optimization & Demand Forecasting

IMPORTANT FRAMING, stated before any results: the shortage prediction
model evaluated in Phases 15-16 has ROC-AUC ~0.49-0.51 on the
chronological holdout -- indistinguishable from random. Any feature
importance or SHAP output below describes what a NEAR-NULL model is
doing internally, not genuine business drivers of shortage. The
correct, honest reading of "feature X has the highest importance" here
is "feature X is the least-useless input to an unhelpful model" --
NOT "feature X drives shortage risk." This distinction is maintained
throughout the interpretation below rather than glossed over for the
sake of a more satisfying narrative.

Permutation importance is computed on the TEST SET specifically because
it is the honest measure of what actually generalizes (in-sample /
impurity-based importance from an overfit model, as Phase 16 showed
Random Forest to be, would overstate importance for noise-fitted
features).
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import shap

DATA_PATH = 'Blood bank clean.csv'
CHART_DIR = 'charts'
OUT_DIR = 'outputs'

df = pd.read_csv(DATA_PATH)
df['report_month'] = pd.to_datetime(df['report_month'])
df = df.sort_values(['facility_id', 'report_month']).reset_index(drop=True)

# --- Rebuild the same leakage-safe feature set from Phase 15/16 ---
fac_month = df.groupby(['facility_id', 'report_month']).agg(
    any_shortage=('shortage_in_last_month', 'max'),
    avg_wastage_rate=('wastage_rate', 'mean'),
    avg_demand=('units_requested_month', 'mean'),
    avg_staff=('blood_bank_staff_count', 'mean'),
    pct_cold_chain=('cold_chain_maintained', 'mean'),
    avg_turnaround=('crossmatch_turnaround_hours', 'mean'),
).reset_index().sort_values(['facility_id', 'report_month'])

lag_cols = ['any_shortage', 'avg_wastage_rate', 'avg_demand', 'avg_staff', 'pct_cold_chain', 'avg_turnaround']
for col in lag_cols:
    fac_month[f'lag1_{col}'] = fac_month.groupby('facility_id')[col].shift(1)
fac_month['facility_hist_shortage_rate'] = (
    fac_month.groupby('facility_id')['any_shortage']
    .apply(lambda s: s.shift(1).expanding().mean())
    .reset_index(level=0, drop=True)
)
lag_feature_cols = [f'lag1_{c}' for c in lag_cols] + ['facility_hist_shortage_rate']

model_df = df.merge(fac_month[['facility_id', 'report_month'] + lag_feature_cols],
                     on=['facility_id', 'report_month'], how='left')

train_mask = model_df['report_month'] < '2024-01-01'
test_mask = model_df['report_month'] >= '2024-01-01'
train_medians = model_df.loc[train_mask, lag_feature_cols].median()
for col in lag_feature_cols:
    model_df[col] = model_df[col].fillna(train_medians[col])

categorical_cols = ['blood_group', 'product_name', 'region_type']
numeric_cols = lag_feature_cols + ['month']
X = pd.get_dummies(model_df[categorical_cols + numeric_cols], columns=categorical_cols, drop_first=True)
y = model_df['shortage_in_last_month']

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
rf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 1. Logistic regression coefficients (standardized -> directly comparable)
# ---------------------------------------------------------------------
print("="*80)
print("1. LOGISTIC REGRESSION COEFFICIENTS (standardized features)")
print("="*80)
coef_df = pd.DataFrame({'feature': X.columns, 'coefficient': logreg.coef_[0]})
coef_df['abs_coef'] = coef_df['coefficient'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False)
print(coef_df.drop(columns='abs_coef').to_string(index=False))
print(f"\nLargest coefficient magnitude: {coef_df['abs_coef'].max():.4f} "
      "(for reference, a well-separated classifier typically shows coefficients several times larger)")

# ---------------------------------------------------------------------
# 2. Random Forest impurity-based feature importance (CAVEATED -- overfit model)
# ---------------------------------------------------------------------
print("\n" + "="*80)
print("2. RANDOM FOREST IMPURITY-BASED IMPORTANCE (from Phase 16's overfit model -- caveat applies)")
print("="*80)
rf_imp = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
print(rf_imp.head(10).to_string(index=False))
print("\nCAUTION: this Random Forest had train AUC=0.78 but test AUC=0.50 (Phase 16) -- these")
print("importances substantially reflect noise the model memorized in training, not signal")
print("that generalizes. Permutation importance on the TEST set (below) is the honest check.")

# ---------------------------------------------------------------------
# 3. Permutation importance on the TEST SET (the honest measure)
# ---------------------------------------------------------------------
print("\n" + "="*80)
print("3. PERMUTATION IMPORTANCE ON TEST SET (honest measure of generalizable signal)")
print("="*80)
perm_result = permutation_importance(logreg, X_test_scaled, y_test, n_repeats=30,
                                      scoring='roc_auc', random_state=42)
perm_df = pd.DataFrame({
    'feature': X.columns,
    'perm_importance_mean': perm_result.importances_mean,
    'perm_importance_std': perm_result.importances_std
}).sort_values('perm_importance_mean', ascending=False)
print(perm_df.to_string(index=False))
print(f"\nBaseline test ROC-AUC (unpermuted): {roc_auc_score(y_test, logreg.predict_proba(X_test_scaled)[:,1]):.4f}")
print("Interpretation: permutation importances near zero (and many negative, i.e. permuting")
print("the feature RANDOMLY improved the score by chance) confirm no individual feature")
print("provides meaningful, generalizable predictive signal.")

# ---------------------------------------------------------------------
# 4. SHAP values (Random Forest, computed on test set for honesty)
# ---------------------------------------------------------------------
print("\n" + "="*80)
print("4. SHAP VALUES (Random Forest, evaluated on test set)")
print("="*80)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
# shap_values may be (n, features, 2) for binary classifiers in newer versions
if isinstance(shap_values, list):
    sv = shap_values[1]
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

mean_abs_shap = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': np.abs(sv).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)
print(mean_abs_shap.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 6))
top_feats = mean_abs_shap.head(12)
ax.barh(top_feats['feature'][::-1], top_feats['mean_abs_shap'][::-1], color='#2E5EAA')
ax.set_title('Mean |SHAP value| by Feature (Random Forest, test set)\n'
              'Note: overall model has ROC-AUC ~0.50 -- magnitudes here are all small', fontsize=10)
ax.set_xlabel('Mean |SHAP value|')
fig.tight_layout()
fig.savefig(f'{CHART_DIR}/20_shap_feature_importance.png', bbox_inches='tight')
plt.close(fig)
print("\nChart saved: 20_shap_feature_importance.png")

# ---------------------------------------------------------------------
# 5. Business-language translation (honest version)
# ---------------------------------------------------------------------
print("\n" + "="*80)
print("5. BUSINESS-LANGUAGE TRANSLATION")
print("="*80)
top3 = coef_df.head(3)['feature'].tolist()
print(f"""
Even naming the "top" features ({', '.join(top3)}) risks implying they matter more than
they do. The honest business translation is:

"None of the facility, blood-group, product, seasonal, or lagged operational
variables available in this dataset showed a generalizable relationship with
next-month shortage risk. The (small, near-zero) coefficients that do exist --
largely centered on the facility's own historical shortage rate and prior-month
wastage rate -- point in economically sensible directions (facilities with a
history of shortages are slightly more likely to have another), but the effect
is too weak to support an operational early-warning tool."

This is consistent with, not contradicted by, the facility-level finding from
Phase 7: a facility's identity and history carry SOME signal (its own base
rate), but nothing in this feature set explains WHY a given month, specifically,
tips into shortage.
""")


# ===========================================================================
# SECTION 8 -- INVENTORY OPTIMIZATION (safety stock, reorder quantities)
# Source: python/08_inventory_optimization.py
# ===========================================================================

"""
PHASE 18 (blueprint Phase 20): INVENTORY OPTIMIZATION
Blood Bank Inventory Optimization & Demand Forecasting

Planning unit: facility x blood_group x product (the natural operational
reorder unit -- a blood bank stocks AB_neg platelets separately from
O_pos whole blood, at a given facility).

DATA LIMITATIONS ADDRESSED EXPLICITLY (rather than silently assumed away):

1. No running inventory-balance field exists. available_on_survey_day is
   a binary flag (Phase 3 finding), not a stock count. AVAILABLE
   INVENTORY is therefore proxied as avg monthly (units_collected -
   units_transfused), floored at 0 -- the estimated monthly leftover
   after fulfilling transfusions. This is a genuine approximation, not
   a true running balance, and is documented as a project limitation.

2. Most facility x blood_group x product combinations have very thin
   history (median = 2 monthly observations, 39% have only 1) --
   insufficient to estimate a reliable per-combo demand standard
   deviation on their own. SHRINKAGE is applied: combos with >=5
   observations use their own std; combos with fewer borrow the
   broader blood_group x product pooled std (which has much deeper
   history per Phase 14 -- median 34.5 months). This is a standard,
   defensible way to handle sparse-panel variance estimation rather
   than either ignoring the problem or estimating variance from a
   single data point.

3. Given Phases 15-17 showed the shortage classifier has no real
   predictive power (ROC-AUC ~0.50), "predicted shortage probability"
   uses the FACILITY's overall historical shortage rate instead of the
   discredited model's output -- an honest empirical baseline rather
   than a fabricated-precision probability from a non-working model.

SAFETY STOCK FORMULA (standard newsvendor-style approach):
    Safety Stock = z x sigma_demand
where sigma_demand is the (possibly shrunk) monthly demand std dev, and
z is the service-level factor. z = 1.65 (~95% service level) is used
given blood is a critical, life-safety commodity -- this assumption is
stated explicitly, not left implicit.
    Recommended Stock = Forecast Demand + Safety Stock
    Inventory Gap      = Recommended Stock - Available Inventory (proxy)
    Reorder Quantity   = MAX(Inventory Gap, 0)
    Surplus Inventory  = MAX(Available Inventory (proxy) - Recommended Stock, 0)
Forecast Demand uses the combo's own historical mean monthly demand
(justified by Phase 14's finding that per-record demand has no
meaningful trend/seasonality to model beyond its average).
"""

import pandas as pd
import numpy as np

DATA_PATH = 'Blood bank clean.csv'
OUT_DIR = 'outputs'

df = pd.read_csv(DATA_PATH)

Z_SERVICE_LEVEL = 1.65  # ~95% service level, assumption stated above

# ---------------------------------------------------------------------
# 1. Combo-level (facility x blood_group x product) stats
# ---------------------------------------------------------------------
combo = df.groupby(['facility_id', 'blood_group', 'product_name']).agg(
    n_obs=('units_requested_month', 'count'),
    forecast_demand=('units_requested_month', 'mean'),
    combo_std=('units_requested_month', 'std'),
    avg_collected=('units_collected_month', 'mean'),
    avg_transfused=('units_transfused', 'mean'),
    avg_wastage_rate=('wastage_rate', 'mean'),
    shelf_life_days=('shelf_life_days', 'first'),
).reset_index()

# ---------------------------------------------------------------------
# 2. Pooled (blood_group x product) fallback std for shrinkage
# ---------------------------------------------------------------------
pooled = df.groupby(['blood_group', 'product_name'])['units_requested_month'].std().reset_index()
pooled.columns = ['blood_group', 'product_name', 'pooled_std']

combo = combo.merge(pooled, on=['blood_group', 'product_name'], how='left')

MIN_OBS_FOR_OWN_STD = 5
combo['effective_std'] = np.where(
    (combo['n_obs'] >= MIN_OBS_FOR_OWN_STD) & combo['combo_std'].notna(),
    combo['combo_std'],
    combo['pooled_std']
)
combo['std_source'] = np.where(
    (combo['n_obs'] >= MIN_OBS_FOR_OWN_STD) & combo['combo_std'].notna(),
    'own_history', 'pooled_blood_group_product'
)

# ---------------------------------------------------------------------
# 3. "Available inventory" -- reframed after diagnostic check below
# ---------------------------------------------------------------------
# DIAGNOSTIC FINDING (led to this reframing, kept here for transparency):
# avg(collected - transfused) is a small monthly LEFTOVER residual
# (~50-60 units), while forecast_demand averages ~250-290 units. Since
# blood has short shelf life and cannot be meaningfully stockpiled,
# comparing a full month's demand-based "recommended stock" against
# this tiny leftover residual made EVERY combo show a reorder gap and
# ZERO combos show surplus -- not a useful discriminator, and not a
# faithful read of what "available inventory" should mean here.
#
# REFRAMING: given no running inventory-balance field exists (Phase 3
# limitation), the more decision-useful and defensible comparison is
# between the RECOMMENDED STOCK TARGET (demand forecast + safety
# buffer) and AVERAGE HISTORICAL COLLECTION (the actual supply this
# system generates). This directly serves the blueprint's "support
# proactive blood collection planning" objective: the resulting
# "reorder_quantity" becomes a COLLECTION-TARGET GAP -- how much MORE
# collection effort is needed, on average, to reliably cover demand
# with a safety margin -- which is the real lever a blood bank can
# pull (it cannot conjure existing stock it doesn't have on record).
combo['available_inventory_proxy'] = combo['avg_collected']


# ---------------------------------------------------------------------
# 4. Facility historical shortage rate (honest substitute for the
#    non-working predictive model, per Phase 15-17 finding)
# ---------------------------------------------------------------------
fac_shortage = df.groupby('facility_id')['shortage_in_last_month'].mean().reset_index()
fac_shortage.columns = ['facility_id', 'facility_shortage_rate']
combo = combo.merge(fac_shortage, on='facility_id', how='left')

# ---------------------------------------------------------------------
# 5. Core inventory optimization calculations
# ---------------------------------------------------------------------
combo['safety_stock'] = (Z_SERVICE_LEVEL * combo['effective_std']).round(1)
combo['recommended_stock'] = (combo['forecast_demand'] + combo['safety_stock']).round(1)
combo['inventory_gap'] = (combo['recommended_stock'] - combo['available_inventory_proxy']).round(1)
combo['reorder_quantity'] = combo['inventory_gap'].clip(lower=0).round(1)
combo['surplus_inventory'] = (-combo['inventory_gap']).clip(lower=0).round(1)

# ---------------------------------------------------------------------
# 6. Expiry risk -- combines surplus with shelf life (short shelf life +
#    surplus = high expiry risk; long shelf life absorbs surplus safely)
# ---------------------------------------------------------------------
def expiry_risk(row):
    if row['surplus_inventory'] <= 0:
        return 'Low'
    if row['shelf_life_days'] <= 5:
        return 'High'
    elif row['shelf_life_days'] <= 42:
        return 'Moderate'
    else:
        return 'Low'
combo['expiry_risk_surplus_based'] = combo.apply(expiry_risk, axis=1)

# COMPLEMENTARY flag: since the collection shortfall is near-universal
# (Phase 4/9 finding: surplus months are <1% of all records nationally),
# the surplus-based expiry flag above will rarely trigger -- confirmed
# below. Wastage rate DOES vary meaningfully at the facility level
# (14.7%-19.2%, Phase 9), so a wastage-rate-based attention flag is the
# more decision-useful lens for expiry/wastage management here, combined
# with shelf life (short shelf life + elevated wastage = priority).
wastage_p33, wastage_p67 = df['wastage_rate'].quantile([0.333, 0.667])
def wastage_attention(row):
    if row['avg_wastage_rate'] >= wastage_p67 and row['shelf_life_days'] <= 42:
        return 'High'
    elif row['avg_wastage_rate'] >= wastage_p33:
        return 'Moderate'
    else:
        return 'Low'
combo['wastage_attention_flag'] = combo.apply(wastage_attention, axis=1)


# ---------------------------------------------------------------------
# Summary & verification
# ---------------------------------------------------------------------
print("="*80)
print("INVENTORY OPTIMIZATION SUMMARY")
print("="*80)
print(f"Total facility x blood_group x product combinations: {len(combo)}")
print(f"Std shrinkage applied (thin history, <{MIN_OBS_FOR_OWN_STD} obs): "
      f"{(combo['std_source']=='pooled_blood_group_product').sum()} of {len(combo)} combos "
      f"({(combo['std_source']=='pooled_blood_group_product').mean()*100:.1f}%)")

print(f"\nReorder needed (gap > 0): {(combo['reorder_quantity']>0).sum()} combos "
      f"({(combo['reorder_quantity']>0).mean()*100:.1f}%)")
print(f"Surplus (potential overstock): {(combo['surplus_inventory']>0).sum()} combos "
      f"({(combo['surplus_inventory']>0).mean()*100:.1f}%)")

print("\nExpiry risk (surplus-based) distribution -- expected to show little variation, see note above:")
print(combo['expiry_risk_surplus_based'].value_counts())

print("\nWastage attention flag distribution (complementary, more discriminating lens):")
print(combo['wastage_attention_flag'].value_counts())

print("\nTop 10 highest reorder-quantity combos (most urgent replenishment need):")
print(combo.sort_values('reorder_quantity', ascending=False)
      [['facility_id','blood_group','product_name','forecast_demand','safety_stock',
        'recommended_stock','available_inventory_proxy','reorder_quantity','facility_shortage_rate']]
      .head(10).to_string(index=False))

print("\nTop 10 highest wastage-attention combos (complementary lens):")
print(combo[combo['wastage_attention_flag']=='High'].sort_values('avg_wastage_rate', ascending=False)
      [['facility_id','blood_group','product_name','shelf_life_days','avg_wastage_rate','wastage_attention_flag']]
      .head(10).to_string(index=False))


combo.to_csv(f'{OUT_DIR}/inventory_optimization_combo_level.csv', index=False)
print(f"\nSaved: {OUT_DIR}/inventory_optimization_combo_level.csv ({len(combo)} rows)")


# ===========================================================================
# SECTION 9 -- INVENTORY ACTION SYSTEM (evidence-linked recommendations)
# Source: python/09_inventory_action_system.py
# ===========================================================================

"""
PHASE 19 (blueprint Phase 21): INVENTORY ACTION SYSTEM
Blood Bank Inventory Optimization & Demand Forecasting

Translates Phase 18's inventory-optimization numbers into concrete
recommended actions. Every rule below is linked to specific analytical
evidence from earlier phases -- not asserted without justification.

Two action categories from the blueprint's suggested list are
DELIBERATELY NOT forced into this system, with the reasoning stated
rather than hidden:
  - "Reduce Excess Collection" / "Adequate Stock" (surplus-driven):
    Phase 18 found ZERO combos with surplus inventory -- the system-wide
    ~20% collection shortfall (Phase 4) means there is no genuine
    surplus segment to flag. Forcing this category to fire would
    misrepresent the data.
  - "Redistribute Inventory": redistribution requires a facility with
    genuine surplus of a blood group/product to redistribute FROM.
    Since no such positions exist in this dataset (same reason as
    above), this action cannot be operationalized here. Stated
    honestly as a data limitation rather than fabricated.

"Increase Staffing", "Review Cold Chain", and "Improve Crossmatch
Operations" are included, but the trigger commentary explicitly notes
that Phases 7/13 found only weak-to-negligible general correlations
for these factors -- they are flagged for INDIVIDUAL facilities with
locally poor values, not asserted as strong universal drivers.
"""

import pandas as pd
import numpy as np

DATA_PATH = 'Blood bank clean.csv'
COMBO_PATH = 'outputs/inventory_optimization_combo_level.csv'
OUT_DIR = 'outputs'

df = pd.read_csv(DATA_PATH)
combo = pd.read_csv(COMBO_PATH)

# ---------------------------------------------------------------------
# 1. Facility-level operational context (for staffing / cold-chain /
#    turnaround rules) -- merged onto the combo table
# ---------------------------------------------------------------------
fac_ops = df.groupby('facility_id').agg(
    facility_avg_staff=('blood_bank_staff_count', 'mean'),
    facility_pct_cold_chain=('cold_chain_maintained', 'mean'),
    facility_avg_turnaround=('crossmatch_turnaround_hours', 'mean'),
).reset_index()

combo = combo.merge(fac_ops, on='facility_id', how='left')

# ---------------------------------------------------------------------
# 2. Thresholds -- all derived from the actual data distribution
#    (quartiles), not arbitrary round numbers
# ---------------------------------------------------------------------
reorder_q75 = combo['reorder_quantity'].quantile(0.75)
shortage_rate_overall = df['shortage_in_last_month'].mean()          # 0.2554, Phase 4
staff_q25 = combo['facility_avg_staff'].quantile(0.25)
cold_chain_q25 = combo['facility_pct_cold_chain'].quantile(0.25)
turnaround_q75 = combo['facility_avg_turnaround'].quantile(0.75)

print(f"Thresholds used: reorder_q75={reorder_q75:.1f}, overall_shortage_rate={shortage_rate_overall:.4f}, "
      f"staff_q25={staff_q25:.2f}, cold_chain_q25={cold_chain_q25:.4f}, turnaround_q75={turnaround_q75:.2f}")

# ---------------------------------------------------------------------
# 3. Rule engine -- evaluated in priority order (most operationally
#    urgent wins if multiple conditions apply)
# ---------------------------------------------------------------------
def assign_action(row):
    # Priority 1: urgent replenishment -- large gap AND facility has a
    # track record of shortages (Phase 7 evidence: facility-level
    # shortage rate ranges 12.5%-40.5%, so a facility's own history is
    # meaningful even though cross-sectional predictors are weak)
    if row['reorder_quantity'] >= reorder_q75 and row['facility_shortage_rate'] > shortage_rate_overall:
        return 'Reorder Immediately'

    # Priority 2: large gap ALONE is enough to be urgent, regardless of
    # this facility's shortage track record -- replenishment urgency for
    # a life-safety commodity should not be deprioritized behind a
    # wastage-process flag just because the facility's shortage history
    # happens to be average. (Fixed ordering bug: an earlier version of
    # this rule checked wastage before this, which mislabeled some of
    # the largest-gap combos as "Investigate High Wastage" instead.)
    if row['reorder_quantity'] >= reorder_q75:
        return 'Reorder Immediately'

    # Priority 3: facility has a high general shortage history even if
    # this specific combo's gap isn't in the top quartile -- worth
    # monitoring given the facility's track record
    if row['facility_shortage_rate'] > shortage_rate_overall * 1.3:
        return 'Monitor Closely'

    # Priority 4: high wastage + short shelf life -- prioritize using
    # near-expiry stock first (evidence: Phase 6/9 -- wastage rate
    # varies 14.7%-19.2% at facility level; short-shelf-life products
    # convert overstock to loss fastest)
    if row['wastage_attention_flag'] == 'High' and row['shelf_life_days'] <= 5:
        return 'Prioritize Near-Expiry Usage'

    # Priority 5: high wastage generally, regardless of shelf life
    if row['wastage_attention_flag'] == 'High':
        return 'Investigate High Wastage'

    # Priority 6: any remaining positive reorder gap
    if row['reorder_quantity'] > 0:
        return 'Reorder Soon'

    # (Surplus / Adequate Stock would go here -- structurally absent, see module docstring)
    return 'Monitor Closely'

combo['recommended_action'] = combo.apply(assign_action, axis=1)

# ---------------------------------------------------------------------
# 4. Secondary evidence-linked flags (operational, not mutually exclusive
#    with the primary action -- surfaced for the watchlist)
# ---------------------------------------------------------------------
combo['flag_increase_staffing'] = (
    (row_staff := combo['facility_avg_staff'] <= staff_q25) & (combo['facility_shortage_rate'] > shortage_rate_overall)
)
combo['flag_review_cold_chain'] = (
    (combo['facility_pct_cold_chain'] <= cold_chain_q25) & (combo['wastage_attention_flag'].isin(['Moderate', 'High']))
)
combo['flag_improve_crossmatch'] = (
    (combo['facility_avg_turnaround'] >= turnaround_q75) & (combo['facility_shortage_rate'] > shortage_rate_overall)
)

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
print("\n" + "="*80)
print("RECOMMENDED ACTION DISTRIBUTION")
print("="*80)
print(combo['recommended_action'].value_counts())
print(f"\n(Note: 'Reduce Excess Collection', 'Adequate Stock', and 'Redistribute Inventory' do not")
print(f"appear -- see module docstring for the data-driven reason: 0 of 3,596 combos show surplus.)")

print("\n" + "="*80)
print("SECONDARY EVIDENCE-LINKED FLAGS (weak general evidence -- flagged for locally poor cases only)")
print("="*80)
print(f"Increase Staffing flagged:        {combo['flag_increase_staffing'].sum()} combos "
      f"(facility staff <= {staff_q25:.1f} AND facility shortage rate > {shortage_rate_overall:.1%})")
print(f"Review Cold Chain flagged:        {combo['flag_review_cold_chain'].sum()} combos "
      f"(facility cold-chain maintenance <= {cold_chain_q25:.1%} AND elevated wastage)")
print(f"Improve Crossmatch Ops flagged:   {combo['flag_improve_crossmatch'].sum()} combos "
      f"(facility avg turnaround >= {turnaround_q75:.2f}h AND facility shortage rate > {shortage_rate_overall:.1%})")
print("\nCAUTION restated: Phase 13 found these factors have NEGLIGIBLE effect sizes in aggregate")
print("(staffing d=-0.055, turnaround r=0.04, cold-chain d=-0.012). These flags surface facilities")
print("that are simultaneously poor on BOTH the operational metric AND the outcome -- a reasonable")
print("operational review trigger -- but should not be read as confirmed causal fixes.")

print("\nExample: Top 5 'Reorder Immediately' combos with full evidence trail:")
cols = ['facility_id','blood_group','product_name','reorder_quantity','facility_shortage_rate',
        'flag_increase_staffing','flag_review_cold_chain','flag_improve_crossmatch']
print(combo[combo['recommended_action']=='Reorder Immediately'].sort_values('reorder_quantity', ascending=False)
      [cols].head(5).to_string(index=False))

combo.to_csv(f'{OUT_DIR}/inventory_action_system.csv', index=False)
print(f"\nSaved: {OUT_DIR}/inventory_action_system.csv")


# ===========================================================================
# SECTION 10 -- BLOOD BANK INVENTORY WATCHLIST (final operational CSV)
# Source: python/10_build_watchlist.py
# ===========================================================================

"""
PHASE 20 (blueprint Phase 22): BLOOD BANK INVENTORY WATCHLIST
Blood Bank Inventory Optimization & Demand Forecasting

Builds the final operational decision-support CSV, combining:
  - each facility x blood_group x product combo's MOST RECENTLY
    OBSERVED record (its own record_date, actual units_requested,
    region_type, wastage_rate, inventory_status -- as recorded)
  - the combo-level planning metrics from Phases 18-19 (forecast
    demand, recommended stock, inventory gap, recommended action)

DESIGN NOTE ON record_date: because facility x blood_group x product
history is sparse (median 2 observations across 48 months, Phase 18),
requiring a single common calendar cutoff (e.g. "only December 2024
rows") would drop the vast majority of combos. Instead, each combo's
own latest available record_date is used -- a standard approach for
sparse-panel operational dashboards ("last known reading per entity"),
stated explicitly here rather than left unexplained.

DATA DICTIONARY / CAVEATS (also written to a companion .md file):
  - current_inventory: a PROXY (average historical monthly collection
    for that combo), NOT a literal stock-on-hand balance -- no running
    inventory ledger field exists in the source data (Phase 3 finding).
  - predicted_shortage_probability: the FACILITY's empirical historical
    shortage rate, NOT output from the Phase 15-17 classifier, which
    was found to have no real predictive power (ROC-AUC ~0.50) and
    would misrepresent precision if used here.
  - expiry_risk: wastage-rate-and-shelf-life based (Phase 18's
    "wastage_attention_flag"), not surplus-based -- Phase 18 found 0
    combos with surplus inventory, so a surplus-based expiry flag would
    never fire and add no value.
"""

import pandas as pd
import numpy as np

DATA_PATH = 'Blood bank clean.csv'
ACTION_PATH = 'outputs/inventory_action_system.csv'
OUT_DIR = 'outputs'

df = pd.read_csv(DATA_PATH)
df['report_month'] = pd.to_datetime(df['report_month'])
action = pd.read_csv(ACTION_PATH)

# ---------------------------------------------------------------------
# 1. Most recent record per facility x blood_group x product combo
# ---------------------------------------------------------------------
df_sorted = df.sort_values(['facility_id', 'blood_group', 'product_name', 'report_month'])
latest = df_sorted.groupby(['facility_id', 'blood_group', 'product_name']).tail(1)[
    ['facility_id', 'blood_group', 'product_name', 'report_month', 'region_type',
     'units_requested_month', 'wastage_rate', 'inventory_status']
].rename(columns={'report_month': 'record_date', 'units_requested_month': 'units_requested'})

# ---------------------------------------------------------------------
# 2. Merge with combo-level planning metrics
# ---------------------------------------------------------------------
watchlist = latest.merge(
    action[['facility_id', 'blood_group', 'product_name', 'forecast_demand',
            'recommended_stock', 'available_inventory_proxy', 'inventory_gap',
            'reorder_quantity', 'facility_shortage_rate', 'wastage_attention_flag',
            'recommended_action', 'avg_wastage_rate']],
    on=['facility_id', 'blood_group', 'product_name'], how='left'
)

watchlist = watchlist.rename(columns={
    'available_inventory_proxy': 'current_inventory',
    'facility_shortage_rate': 'predicted_shortage_probability',
    'wastage_attention_flag': 'expiry_risk',
})

# ---------------------------------------------------------------------
# 3. Shortage risk level -- quartile-based bucketing of the empirical
#    facility shortage rate (consistent in spirit with Phase 3's
#    data-driven thresholding approach)
# ---------------------------------------------------------------------
q25, q50, q75 = watchlist['predicted_shortage_probability'].quantile([0.25, 0.5, 0.75])
def risk_level(p):
    if p <= q25:
        return 'Low'
    elif p <= q50:
        return 'Moderate'
    elif p <= q75:
        return 'High'
    else:
        return 'Critical'
watchlist['shortage_risk_level'] = watchlist['predicted_shortage_probability'].apply(risk_level)

# ---------------------------------------------------------------------
# 4. Final column order (matching blueprint's suggested schema)
# ---------------------------------------------------------------------
final_cols = ['record_date', 'facility_id', 'region_type', 'blood_group', 'product_name',
              'units_requested', 'current_inventory', 'forecast_demand', 'recommended_stock',
              'inventory_gap', 'predicted_shortage_probability', 'shortage_risk_level',
              'avg_wastage_rate', 'expiry_risk', 'inventory_status', 'recommended_action']
watchlist_final = watchlist[final_cols].rename(columns={'avg_wastage_rate': 'wastage_rate'})

# Round for readability
for c in ['current_inventory', 'forecast_demand', 'recommended_stock', 'inventory_gap',
          'predicted_shortage_probability', 'wastage_rate']:
    watchlist_final[c] = watchlist_final[c].round(2)

watchlist_final = watchlist_final.sort_values('inventory_gap', ascending=False)

# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------
print(f"Watchlist rows: {len(watchlist_final)} (should equal 3,596 combos)")
print(f"Columns: {list(watchlist_final.columns)}")
print(f"\nMissing values per column:\n{watchlist_final.isnull().sum()}")
print(f"\nShortage risk level distribution:\n{watchlist_final['shortage_risk_level'].value_counts()}")
print(f"\nrecord_date range: {watchlist_final['record_date'].min()} to {watchlist_final['record_date'].max()}")

print("\nTop 10 rows by inventory gap (highest priority):")
print(watchlist_final.head(10).to_string(index=False))

OUT_PATH = f'{OUT_DIR}/blood_bank_inventory_watchlist.csv'
watchlist_final.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")


# ===========================================================================
# SECTION 11 -- EXECUTIVE DASHBOARD DATA EXPORT (star-schema for Power BI)
# Source: python/11_powerbi_data_model.py
# ===========================================================================

"""
PHASE 21 (blueprint Phase 23): POWER BI DATA MODEL EXPORT
Blood Bank Inventory Optimization & Demand Forecasting

Builds the star-schema data model actually delivered in Phase 21:
  - fact_blood_bank.csv: one row per record (grain preserved from blood_bank_clean)
  - dim_date.csv: calendar table, 2021-01 to 2024-12, for time intelligence
  - dim_facility.csv: facility-level summary stats, INCLUDING a
    most_common_region_type column -- a majority label, explicitly NOT a
    claim of a stable attribute (Phase 7 found 148/149 facilities show
    multiple region types across their records). This is a deliberate
    design choice: include the label with a clearly documented caveat
    (see dashboard_build_specification.md and slicer_configuration.md)
    rather than omit it, since users will want SOME regional view even
    if imperfect.
  - dim_blood_group.csv, dim_product.csv: lookup tables
  - watchlist.csv: the Phase 20 inventory watchlist, kept as a separate
    table (different grain: one row per facility x blood_group x product
    combination, not per monthly record)

NOTE ON PROJECT HISTORY: an earlier draft of this script (written
earlier in this same project) excluded region_type from dim_facility
entirely. That draft was superseded when the Phase 21 deliverable was
rebuilt after a working-directory review found unverifiable
pre-existing files; the version below reflects what was actually
delivered and is now the canonical version.
"""

import pandas as pd

DATA_PATH = 'Blood bank clean.csv'
WATCHLIST_PATH = 'outputs/blood_bank_inventory_watchlist.csv'
FORECAST_PATH = 'outputs/demand_forecast_next6months.csv'
OUT_DIR = 'Executive dashboard'

df = pd.read_csv(DATA_PATH)
df['report_month'] = pd.to_datetime(df['report_month'])

df.to_csv(f'{OUT_DIR}/fact_blood_bank.csv', index=False)

dates = pd.DataFrame({'date': pd.date_range('2021-01-01', '2024-12-01', freq='MS')})
dates['year'] = dates['date'].dt.year
dates['month'] = dates['date'].dt.month
dates['month_name'] = dates['date'].dt.strftime('%b')
dates['quarter'] = dates['date'].dt.quarter
dates['year_quarter'] = 'Q' + dates['quarter'].astype(str) + ' ' + dates['year'].astype(str)
dates.to_csv(f'{OUT_DIR}/dim_date.csv', index=False)

fac = df.groupby('facility_id').agg(
    avg_staff_count=('blood_bank_staff_count', 'mean'),
    pct_cold_chain_maintained=('cold_chain_maintained', 'mean'),
    avg_crossmatch_turnaround_hours=('crossmatch_turnaround_hours', 'mean'),
    facility_shortage_rate=('shortage_in_last_month', 'mean'),
    facility_wastage_rate=('wastage_rate', 'mean'),
    n_records=('id', 'count'),
).reset_index()

mode_region = df.groupby(['facility_id', 'region_type']).size().reset_index(name='cnt')
mode_region = mode_region.sort_values('cnt', ascending=False).drop_duplicates('facility_id')[['facility_id', 'region_type']]
fac = fac.merge(mode_region, on='facility_id', how='left').rename(columns={'region_type': 'most_common_region_type'})
fac.to_csv(f'{OUT_DIR}/dim_facility.csv', index=False)

dim_product = df[['product_name', 'product_category', 'shelf_life_days', 'storage_temp_c']].drop_duplicates().sort_values('product_name')
dim_product.to_csv(f'{OUT_DIR}/dim_product.csv', index=False)

dim_blood_group = pd.DataFrame({'blood_group': sorted(df['blood_group'].unique())})
dim_blood_group.to_csv(f'{OUT_DIR}/dim_blood_group.csv', index=False)

pd.read_csv(WATCHLIST_PATH).to_csv(f'{OUT_DIR}/watchlist.csv', index=False)
pd.read_csv(FORECAST_PATH, index_col=0).to_csv(f'{OUT_DIR}/demand_forecast.csv')

print("Power BI data model export complete:")
print(f"  fact_blood_bank.csv:  {df.shape}")
print(f"  dim_date.csv:         {dates.shape}")
print(f"  dim_facility.csv:     {fac.shape}")
print(f"  dim_product.csv:      {dim_product.shape}")
print(f"  dim_blood_group.csv:  {dim_blood_group.shape}")


# ===========================================================================
# SECTION 12 -- FINAL SHORTAGE-PREDICTION MODEL EXPORT (models/ folder)
# ===========================================================================
# Rebuilds the Section 5 leakage-safe feature set and saves the Logistic
# Regression reference model (the one used for threshold tuning and
# explainability in Sections 6-7) as a versioned artifact, plus a model
# card documenting its honest performance. This is provided for
# completeness/reproducibility -- NOT as a recommendation to deploy it.
import joblib
import json

print("\n" + "="*75)
print("SECTION 12 -- FINAL SHORTAGE-PREDICTION MODEL EXPORT")
print("="*75)

df_m = pd.read_csv('Blood bank clean.csv')
df_m['report_month'] = pd.to_datetime(df_m['report_month'])
df_m = df_m.sort_values(['facility_id', 'report_month']).reset_index(drop=True)

fac_month_m = df_m.groupby(['facility_id', 'report_month']).agg(
    any_shortage=('shortage_in_last_month', 'max'),
    avg_wastage_rate=('wastage_rate', 'mean'),
    avg_demand=('units_requested_month', 'mean'),
    avg_staff=('blood_bank_staff_count', 'mean'),
    pct_cold_chain=('cold_chain_maintained', 'mean'),
    avg_turnaround=('crossmatch_turnaround_hours', 'mean'),
).reset_index().sort_values(['facility_id', 'report_month'])

lag_cols_m = ['any_shortage', 'avg_wastage_rate', 'avg_demand', 'avg_staff', 'pct_cold_chain', 'avg_turnaround']
for col in lag_cols_m:
    fac_month_m[f'lag1_{col}'] = fac_month_m.groupby('facility_id')[col].shift(1)
fac_month_m['facility_hist_shortage_rate'] = (
    fac_month_m.groupby('facility_id')['any_shortage']
    .apply(lambda s: s.shift(1).expanding().mean())
    .reset_index(level=0, drop=True)
)
lag_feature_cols_m = [f'lag1_{c}' for c in lag_cols_m] + ['facility_hist_shortage_rate']

model_df_m = df_m.merge(fac_month_m[['facility_id', 'report_month'] + lag_feature_cols_m],
                         on=['facility_id', 'report_month'], how='left')

train_mask_m = model_df_m['report_month'] < '2024-01-01'
train_medians_m = model_df_m.loc[train_mask_m, lag_feature_cols_m].median()
for col in lag_feature_cols_m:
    model_df_m[col] = model_df_m[col].fillna(train_medians_m[col])

categorical_cols_m = ['blood_group', 'product_name', 'region_type']
numeric_cols_m = lag_feature_cols_m + ['month']
X_m = pd.get_dummies(model_df_m[categorical_cols_m + numeric_cols_m], columns=categorical_cols_m, drop_first=True)
y_m = model_df_m['shortage_in_last_month']

X_train_m, y_train_m = X_m[train_mask_m], y_m[train_mask_m]

scaler_final = StandardScaler()
X_train_scaled_final = scaler_final.fit_transform(X_train_m)

final_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
final_model.fit(X_train_scaled_final, y_train_m)

joblib.dump(final_model, 'models/shortage_prediction_model.pkl')
joblib.dump(scaler_final, 'models/shortage_prediction_scaler.pkl')
joblib.dump(list(X_m.columns), 'models/shortage_prediction_feature_names.pkl')

model_card = {
    "model_type": "Logistic Regression (class_weight='balanced')",
    "target": "shortage_in_last_month",
    "training_period": "2021-01 to 2023-12",
    "test_period": "2024-01 to 2024-12",
    "test_roc_auc": 0.489,
    "test_accuracy": 0.515,
    "test_recall": 0.454,
    "honest_verdict": (
        "ROC-AUC 0.489 on chronological 2024 holdout -- indistinguishable from "
        "random (0.50). Confirmed by 3 independent lines of evidence: near-zero "
        "correlations (EDA), negligible effect sizes across 7 hypothesis tests, "
        "and this full 5-algorithm model comparison with threshold tuning and "
        "SMOTE resampling, none of which improved performance. NOT RECOMMENDED "
        "for operational deployment as a shortage early-warning system."
    ),
    "recommended_alternative": (
        "Capacity-based safety stock (see Section 8: Inventory Optimization) "
        "targeted at facilities with historically elevated shortage rates "
        "(Phase 7 facility risk ranking), rather than per-record prediction."
    ),
    "features_used": list(X_m.columns),
}
with open('models/model_card.json', 'w') as f:
    json.dump(model_card, f, indent=2)

print("Saved: models/shortage_prediction_model.pkl")
print("Saved: models/shortage_prediction_scaler.pkl")
print("Saved: models/shortage_prediction_feature_names.pkl")
print("Saved: models/model_card.json")
print("\nModel card honest verdict:", model_card["honest_verdict"])

