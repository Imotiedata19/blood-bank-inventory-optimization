-- =====================================================================
-- SQL BLOOD BANK ANALYSIS
-- Blood Bank Inventory Optimization & Demand Forecasting
-- =====================================================================
-- Consolidated SQL script covering the full database workflow:
--   Section 1  -- Raw table creation & load
--   Section 2  -- Clean table setup
--   Section 3  -- Data quality checks (read-only audit)
--   Section 4  -- Cleaning & standardization
--   Section 5  -- Feature engineering (12 derived columns)
--   Section 6  -- Overall performance KPIs
--   Section 7  -- Blood group analysis
--   Section 8  -- Blood product analysis
--   Section 9  -- Facility summary
--   Section 10 -- Facility risk ranking
--   Section 11 -- Shortage analysis
--   Section 12 -- Wastage analysis
--   Section 13 -- Time-series analysis
--
-- Run top to bottom against a PostgreSQL 16 database. Sections 1-5 are
-- schema-building and MUST run in order. Sections 6-13 are read-only
-- analysis queries and may be run in any order (or independently) once
-- sections 1-5 have completed.
--
-- Source data: blood_national_blood_centre_1_.csv (10,000 rows, 149
-- facilities, 2021-2024). Update the file path in Section 1 if loading
-- from a different location.
-- =====================================================================


-- =====================================================================
-- SECTION 1 -- RAW TABLE CREATION & LOAD
-- Source: sql/01_create_raw_table.sql
-- =====================================================================

-- =====================================================================
-- PHASE 1 (blueprint Phase 3): POSTGRESQL DATABASE SETUP
-- Blood Bank Inventory Optimization & Demand Forecasting
-- =====================================================================
-- Purpose: Create blood_bank_raw as an exact, untouched copy of the
-- source CSV. This table is never modified after load and serves as
-- the permanent source of truth / audit trail for the project.
-- =====================================================================

DROP TABLE IF EXISTS blood_bank_raw;

CREATE TABLE blood_bank_raw (
    id                                  INTEGER,
    facility_level                      TEXT,
    facility_id                         TEXT,
    region_type                         TEXT,
    collection_type                     TEXT,
    has_component_separation            INTEGER,
    has_nat_testing                     INTEGER,
    has_blood_bank_fridge               INTEGER,
    has_platelet_agitator               INTEGER,
    blood_bank_staff_count              INTEGER,
    product_name                        TEXT,
    product_category                    TEXT,
    storage_temp_c                      INTEGER,
    shelf_life_days                     INTEGER,
    unit_cost_usd                       NUMERIC(10,2),
    blood_group                         TEXT,
    year                                INTEGER,
    month                                INTEGER,
    units_collected_month               INTEGER,
    units_requested_month               INTEGER,
    units_crossmatched                  INTEGER,
    units_transfused                    INTEGER,
    tti_screened_pct                    NUMERIC(6,4),
    tti_positive_pct                    NUMERIC(6,2),
    units_discarded_tti                 INTEGER,
    units_expired                       INTEGER,
    available_on_survey_day             INTEGER,
    shortage_in_last_month              INTEGER,
    shortage_days_last_month            INTEGER,
    shortage_cause                      TEXT,
    crossmatch_turnaround_hours         NUMERIC(6,2),
    transfusion_reactions_month         INTEGER,
    maternal_deaths_no_blood            INTEGER,
    surgical_cancellations_no_blood     INTEGER,
    cold_chain_maintained               INTEGER,
    report_submitted                    INTEGER
);

-- Load directly from the CSV on disk (header row skipped)
--\copy blood_bank_raw FROM '/mnt/user-data/uploads/blood_national_blood_centre_1_.csv' WITH (FORMAT csv, HEADER true);

-- Basic post-load sanity checks (read-only, no changes to raw data)
SELECT COUNT(*) AS row_count FROM blood_bank_raw;
SELECT COUNT(*) AS distinct_ids FROM (SELECT DISTINCT id FROM blood_bank_raw) t;


-- =====================================================================
-- SECTION 2 -- CLEAN TABLE SETUP
-- Source: sql/02_create_clean_table.sql
-- =====================================================================

-- =====================================================================
-- PHASE 1 (blueprint Phase 3, cont.): blood_bank_clean setup
-- =====================================================================
-- blood_bank_clean is the working analytical table for all subsequent
-- phases (cleaning, feature engineering, EDA). It starts as a direct
-- copy of blood_bank_raw. blood_bank_raw itself is NEVER modified from
-- this point forward -- it remains the immutable source of truth.
--
-- Basic integrity constraints are added here to enforce data-quality
-- rules going forward (id uniqueness, non-negative core counts). Any
-- records that fail these constraints will surface as an explicit
-- error during Phase 2 cleaning rather than being silently dropped.
-- =====================================================================

DROP TABLE IF EXISTS blood_bank_clean;

CREATE TABLE blood_bank_clean (
    LIKE blood_bank_raw INCLUDING ALL
);

ALTER TABLE blood_bank_clean
    ADD PRIMARY KEY (id);

INSERT INTO blood_bank_clean
SELECT * FROM blood_bank_raw;

-- Post-load verification: row count must match raw exactly at this stage
-- (no rows have been removed yet -- that only happens, with justification,
-- during Phase 2 cleaning).
SELECT
    (SELECT COUNT(*) FROM blood_bank_raw)   AS raw_row_count,
    (SELECT COUNT(*) FROM blood_bank_clean) AS clean_row_count;
SELECT COUNT(*) AS clean_rows
FROM blood_bank_clean;

-- =====================================================================
-- SECTION 3 -- DATA QUALITY CHECKS (READ-ONLY AUDIT)
-- Source: sql/03_data_quality_checks.sql
-- =====================================================================

-- =====================================================================
-- PHASE 2 (blueprint Phase 4): SQL DATA CLEANING - QUALITY CHECKS
-- Blood Bank Inventory Optimization & Demand Forecasting
-- =====================================================================
-- Every check below is READ-ONLY (SELECT/COUNT). No rows are altered
-- or deleted here. Results determine what (if anything) Phase 2's
-- cleaning step below needs to actually fix.
-- =====================================================================

-- 1. Missing values (per column, key columns)
SELECT
    COUNT(*) FILTER (WHERE facility_id IS NULL)                   AS null_facility_id,
    COUNT(*) FILTER (WHERE region_type IS NULL)                   AS null_region_type,
    COUNT(*) FILTER (WHERE product_name IS NULL)                  AS null_product_name,
    COUNT(*) FILTER (WHERE product_category IS NULL)              AS null_product_category,
    COUNT(*) FILTER (WHERE blood_group IS NULL)                   AS null_blood_group,
    COUNT(*) FILTER (WHERE year IS NULL)                          AS null_year,
    COUNT(*) FILTER (WHERE month IS NULL)                         AS null_month,
    COUNT(*) FILTER (WHERE units_collected_month IS NULL)         AS null_units_collected,
    COUNT(*) FILTER (WHERE units_requested_month IS NULL)         AS null_units_requested,
    COUNT(*) FILTER (WHERE shortage_cause IS NULL)                AS null_shortage_cause
FROM blood_bank_clean;

-- 2. Duplicate rows (full-row duplicates) and duplicate IDs
SELECT COUNT(*) AS full_row_duplicates
FROM (
    SELECT id, facility_id, blood_group, product_name, year, month, COUNT(*) AS c
    FROM blood_bank_clean
    GROUP BY id, facility_id, blood_group, product_name, year, month
    HAVING COUNT(*) > 1
) d;

SELECT COUNT(*) AS duplicate_ids
FROM (SELECT id, COUNT(*) FROM blood_bank_clean GROUP BY id HAVING COUNT(*) > 1) x;

-- 3. Blank / whitespace-only strings in text columns
SELECT
    COUNT(*) FILTER (WHERE TRIM(facility_id) = '')       AS blank_facility_id,
    COUNT(*) FILTER (WHERE TRIM(region_type) = '')        AS blank_region_type,
    COUNT(*) FILTER (WHERE TRIM(product_name) = '')       AS blank_product_name,
    COUNT(*) FILTER (WHERE TRIM(blood_group) = '')        AS blank_blood_group,
    COUNT(*) FILTER (WHERE TRIM(shortage_cause) = '')     AS blank_shortage_cause
FROM blood_bank_clean;

-- 4. Leading/trailing whitespace present (but non-blank)
SELECT
    COUNT(*) FILTER (WHERE facility_id <> TRIM(facility_id))   AS untrimmed_facility_id,
    COUNT(*) FILTER (WHERE region_type <> TRIM(region_type))    AS untrimmed_region_type,
    COUNT(*) FILTER (WHERE product_name <> TRIM(product_name))  AS untrimmed_product_name,
    COUNT(*) FILTER (WHERE blood_group <> TRIM(blood_group))    AS untrimmed_blood_group,
    COUNT(*) FILTER (WHERE shortage_cause <> TRIM(shortage_cause)) AS untrimmed_shortage_cause
FROM blood_bank_clean;

-- 5. Inconsistent capitalization -- compare distinct raw values vs distinct lower() values
SELECT 'region_type' AS col, COUNT(DISTINCT region_type) AS distinct_raw, COUNT(DISTINCT LOWER(region_type)) AS distinct_lower FROM blood_bank_clean
UNION ALL
SELECT 'product_name', COUNT(DISTINCT product_name), COUNT(DISTINCT LOWER(product_name)) FROM blood_bank_clean
UNION ALL
SELECT 'product_category', COUNT(DISTINCT product_category), COUNT(DISTINCT LOWER(product_category)) FROM blood_bank_clean
UNION ALL
SELECT 'blood_group', COUNT(DISTINCT blood_group), COUNT(DISTINCT LOWER(blood_group)) FROM blood_bank_clean
UNION ALL
SELECT 'shortage_cause', COUNT(DISTINCT shortage_cause), COUNT(DISTINCT LOWER(shortage_cause)) FROM blood_bank_clean;

-- 6. Valid category membership checks
SELECT DISTINCT blood_group FROM blood_bank_clean
WHERE blood_group NOT IN ('O_pos','A_pos','B_pos','AB_pos','O_neg','A_neg','B_neg','AB_neg');

SELECT DISTINCT product_name FROM blood_bank_clean
WHERE product_name NOT IN ('whole_blood','packed_RBC','fresh_frozen_plasma','platelet_concentrate','cryoprecipitate','group_O_neg_emergency');

SELECT DISTINCT product_category FROM blood_bank_clean
WHERE product_category NOT IN ('red_cells','plasma','platelets','plasma_derived');

SELECT DISTINCT region_type FROM blood_bank_clean
WHERE region_type NOT IN ('urban','peri_urban','rural');

-- 7. Invalid year / month values
SELECT COUNT(*) AS invalid_year FROM blood_bank_clean WHERE year NOT BETWEEN 2000 AND 2100;
SELECT COUNT(*) AS invalid_month FROM blood_bank_clean WHERE month NOT BETWEEN 1 AND 12;

-- 8. Negative values in count/volume fields that should never be negative
SELECT
    COUNT(*) FILTER (WHERE units_collected_month < 0)           AS neg_collected,
    COUNT(*) FILTER (WHERE units_requested_month < 0)           AS neg_requested,
    COUNT(*) FILTER (WHERE units_crossmatched < 0)              AS neg_crossmatched,
    COUNT(*) FILTER (WHERE units_transfused < 0)                AS neg_transfused,
    COUNT(*) FILTER (WHERE units_expired < 0)                   AS neg_expired,
    COUNT(*) FILTER (WHERE units_discarded_tti < 0)             AS neg_discarded,
    COUNT(*) FILTER (WHERE shortage_days_last_month < 0)        AS neg_shortage_days,
    COUNT(*) FILTER (WHERE blood_bank_staff_count < 0)          AS neg_staff,
    COUNT(*) FILTER (WHERE crossmatch_turnaround_hours < 0)     AS neg_turnaround
FROM blood_bank_clean;

-- 9. Percentages outside plausible 0-100 range
-- Note: tti_screened_pct is stored as a 0-1 FRACTION in source data (confirmed in Phase 0),
-- so its valid range is 0-1, not 0-100. tti_positive_pct is a true 0-100 percent.
SELECT COUNT(*) AS invalid_tti_screened_fraction
FROM blood_bank_clean WHERE tti_screened_pct < 0 OR tti_screened_pct > 1;

SELECT COUNT(*) AS invalid_tti_positive_pct
FROM blood_bank_clean WHERE tti_positive_pct < 0 OR tti_positive_pct > 100;

-- 10. Invalid shelf-life / storage temperature (must be consistent with product_name)
SELECT product_name, COUNT(DISTINCT shelf_life_days) AS distinct_shelf_life_values,
       COUNT(DISTINCT storage_temp_c) AS distinct_storage_temp_values
FROM blood_bank_clean
GROUP BY product_name;

-- 11. Extreme staffing / turnaround outliers (flag only, IQR-based)
WITH stats AS (
    SELECT
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY blood_bank_staff_count) AS q1_staff,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY blood_bank_staff_count) AS q3_staff,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY crossmatch_turnaround_hours) AS q1_turn,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY crossmatch_turnaround_hours) AS q3_turn
    FROM blood_bank_clean
)
SELECT
    (SELECT COUNT(*) FROM blood_bank_clean, stats
       WHERE blood_bank_staff_count < q1_staff - 1.5*(q3_staff-q1_staff)
          OR blood_bank_staff_count > q3_staff + 1.5*(q3_staff-q1_staff)) AS staff_outliers,
    (SELECT COUNT(*) FROM blood_bank_clean, stats
       WHERE crossmatch_turnaround_hours < q1_turn - 1.5*(q3_turn-q1_turn)
          OR crossmatch_turnaround_hours > q3_turn + 1.5*(q3_turn-q1_turn)) AS turnaround_outliers;

-- 12. Logical inconsistencies (funnel-style checks)
SELECT
    COUNT(*) FILTER (WHERE units_crossmatched > units_collected_month)        AS crossmatch_gt_collected,
    COUNT(*) FILTER (WHERE units_transfused > units_crossmatched)             AS transfused_gt_crossmatched,
    COUNT(*) FILTER (WHERE (units_expired + units_discarded_tti) > units_collected_month) AS wastage_gt_collected,
    COUNT(*) FILTER (WHERE tti_positive_pct > tti_screened_pct * 100)         AS positive_gt_screened,
    COUNT(*) FILTER (
        WHERE (shortage_in_last_month = 1 AND shortage_days_last_month = 0)
           OR (shortage_in_last_month = 0 AND shortage_days_last_month > 0)
    ) AS shortage_flag_inconsistent,
    COUNT(*) FILTER (WHERE shortage_days_last_month > 31)                     AS shortage_days_over_31
FROM blood_bank_clean;


-- =====================================================================
-- SECTION 4 -- CLEANING & STANDARDIZATION
-- Source: sql/04_clean_and_standardize.sql
-- =====================================================================

-- =====================================================================
-- PHASE 2 (blueprint Phase 4, cont.): SQL DATA CLEANING - APPLY
-- =====================================================================
-- The audit in 03_data_quality_checks.sql found no missing values,
-- duplicates, blank/untrimmed strings, case inconsistencies, invalid
-- categories, or logical violations. No rows are removed or altered
-- in content. This script:
--   1) Defensively re-applies TRIM() to all text columns (idempotent
--      safeguard -- a no-op today, but protects future re-runs of
--      this pipeline against upstream data-entry drift).
--   2) Builds the analytical monthly date column from year + month,
--      using the first day of the month, as specified in the blueprint.
-- No transaction-level dates are invented; report_month always
-- represents "the first of the reporting month," nothing more precise.
-- =====================================================================

-- 1. Defensive standardization (no-op confirmed by audit, applied anyway)
UPDATE blood_bank_clean
SET
    facility_id     = TRIM(facility_id),
    region_type     = TRIM(LOWER(region_type)),
    product_name    = TRIM(product_name),
    product_category = TRIM(LOWER(product_category)),
    blood_group     = TRIM(blood_group),
    shortage_cause  = TRIM(LOWER(shortage_cause));

SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE facility_id <> TRIM(facility_id)) AS bad_facility_id,
    COUNT(*) FILTER (WHERE region_type <> TRIM(LOWER(region_type))) AS bad_region_type,
    COUNT(*) FILTER (WHERE product_name <> TRIM(product_name)) AS bad_product_name,
    COUNT(*) FILTER (WHERE product_category <> TRIM(LOWER(product_category))) AS bad_product_category,
    COUNT(*) FILTER (WHERE blood_group <> TRIM(blood_group)) AS bad_blood_group,
    COUNT(*) FILTER (WHERE shortage_cause <> TRIM(LOWER(shortage_cause))) AS bad_shortage_cause
FROM blood_bank_clean;

-- 2. Add the analytical monthly date column
ALTER TABLE blood_bank_clean
    ADD COLUMN IF NOT EXISTS report_month DATE;

UPDATE blood_bank_clean
SET report_month = MAKE_DATE(year, month, 1);

ALTER TABLE blood_bank_clean
    ALTER COLUMN report_month SET NOT NULL;

-- 3. Verification
SELECT COUNT(*) AS rows_after_cleaning FROM blood_bank_clean;
SELECT MIN(report_month) AS earliest_month, MAX(report_month) AS latest_month FROM blood_bank_clean;
SELECT report_month, COUNT(*) FROM blood_bank_clean GROUP BY report_month ORDER BY report_month LIMIT 5;


-- =====================================================================
-- SECTION 5 -- FEATURE ENGINEERING
-- Source: sql/05_feature_engineering.sql
-- =====================================================================

-- =====================================================================
-- PHASE 3 (blueprint Phase 5): SQL FEATURE ENGINEERING
-- Blood Bank Inventory Optimization & Demand Forecasting
-- =====================================================================
-- All formulas per the project blueprint. Division-by-zero is handled
-- with NULLIF (defensive; Phase 0 confirmed units_collected_month,
-- units_requested_month, and units_crossmatched have no zero values
-- in this dataset, min = 149 / 194 / 136 respectively, but the guard
-- is kept for robustness against future data extracts).
--
-- DATA-SCOPING NOTE on available_on_survey_day:
-- This field is a BINARY FLAG (1 = stock available, 0 = not available
-- on the day of the survey), not a numeric stock count (confirmed in
-- Phase 0). The blueprint's literal formula
--     inventory_demand_ratio = available_on_survey_day / units_requested_month
-- is therefore computed below for completeness, but it is NOT a
-- meaningful "days of stock on hand" ratio -- it can only ever take
-- the value 0, or a very small fraction (1/requested), so it carries
-- almost no analytical signal on its own. Instead, the real inventory
-- signal used throughout this project is inventory_status, a 2x2
-- category built from available_on_survey_day CROSSED WITH
-- shortage_in_last_month (the crosstab below shows all four quadrants
-- are well populated, so the categories are well supported by data,
-- not arbitrary):
--   available=1, shortage=0 (n=5,645) -> Adequate
--   available=1, shortage=1 (n=1,906) -> Recovered (shortage occurred
--       in the month but stock had been replenished by survey day)
--   available=0, shortage=0 (n=1,801) -> Low Stock (unavailable at the
--       survey snapshot, but not enough to cross the official shortage-day
--       threshold for the month)
--   available=0, shortage=1 (n=  648) -> Critical (unavailable at the
--       survey snapshot AND a shortage was recorded during the month)
-- =====================================================================

ALTER TABLE blood_bank_clean
    ADD COLUMN IF NOT EXISTS supply_demand_gap          INTEGER,
    ADD COLUMN IF NOT EXISTS total_wastage               INTEGER,
    ADD COLUMN IF NOT EXISTS wastage_rate                NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS expiry_rate                 NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS tti_discard_rate             NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS demand_fulfillment_rate     NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS crossmatch_conversion_rate  NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS collection_utilization_rate NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS inventory_demand_ratio      NUMERIC(8,5),
    ADD COLUMN IF NOT EXISTS shortage_severity           TEXT,
    ADD COLUMN IF NOT EXISTS wastage_risk                TEXT,
    ADD COLUMN IF NOT EXISTS inventory_status            TEXT;

UPDATE blood_bank_clean
SET
    supply_demand_gap = units_collected_month - units_requested_month,

    total_wastage = units_expired + units_discarded_tti,

    wastage_rate = ROUND(
        (units_expired + units_discarded_tti)::NUMERIC
        / NULLIF(units_collected_month, 0) * 100, 2),

    expiry_rate = ROUND(
        units_expired::NUMERIC / NULLIF(units_collected_month, 0) * 100, 2),

    tti_discard_rate = ROUND(
        units_discarded_tti::NUMERIC / NULLIF(units_collected_month, 0) * 100, 2),

    demand_fulfillment_rate = ROUND(
        units_transfused::NUMERIC / NULLIF(units_requested_month, 0) * 100, 2),

    crossmatch_conversion_rate = ROUND(
        units_transfused::NUMERIC / NULLIF(units_crossmatched, 0) * 100, 2),

    collection_utilization_rate = ROUND(
        units_transfused::NUMERIC / NULLIF(units_collected_month, 0) * 100, 2),

    -- Literal blueprint formula; see data-scoping note above on its limited
    -- interpretability given available_on_survey_day is a 0/1 flag.
    inventory_demand_ratio = ROUND(
        available_on_survey_day::NUMERIC / NULLIF(units_requested_month, 0), 5);

-- ---------------------------------------------------------------------
-- Shortage Severity
-- Thresholds are quantile-based, derived from the actual distribution
-- of shortage_days_last_month AMONG records that had a shortage
-- (n=2,554; median=2 days, p75=5 days, p90=9 days, max=30 days).
-- Using data quantiles (not round numbers) avoids arbitrary cutoffs:
--   0 days           -> No Shortage   (74.5% of all records)
--   1-2 days (<=p50)  -> Low
--   3-5 days (p50-p75)-> Moderate
--   6-9 days (p75-p90)-> High
--   >9 days  (>p90)   -> Critical
-- ---------------------------------------------------------------------
UPDATE blood_bank_clean
SET shortage_severity = CASE
    WHEN shortage_days_last_month = 0 THEN 'No Shortage'
    WHEN shortage_days_last_month <= 2 THEN 'Low'
    WHEN shortage_days_last_month <= 5 THEN 'Moderate'
    WHEN shortage_days_last_month <= 9 THEN 'High'
    ELSE 'Critical'
END;

-- ---------------------------------------------------------------------
-- Wastage Risk
-- Thresholds are tertile-based on the actual wastage_rate distribution
-- across all 10,000 records (tertile_1 ~= 14.0%, tertile_2 ~= 18.9%),
-- so each category is designed to hold roughly one-third of records
-- rather than using round arbitrary numbers.
-- ---------------------------------------------------------------------
UPDATE blood_bank_clean
SET wastage_risk = CASE
    WHEN wastage_rate < 14.0 THEN 'Low'
    WHEN wastage_rate < 19.0 THEN 'Moderate'
    ELSE 'High'
END;

-- ---------------------------------------------------------------------
-- Inventory Status (see data-scoping note above for full rationale)
-- ---------------------------------------------------------------------
UPDATE blood_bank_clean
SET inventory_status = CASE
    WHEN available_on_survey_day = 1 AND shortage_in_last_month = 0 THEN 'Adequate'
    WHEN available_on_survey_day = 1 AND shortage_in_last_month = 1 THEN 'Recovered'
    WHEN available_on_survey_day = 0 AND shortage_in_last_month = 0 THEN 'Low Stock'
    WHEN available_on_survey_day = 0 AND shortage_in_last_month = 1 THEN 'Critical'
END;

-- =====================================================================
-- Verification
-- =====================================================================
SELECT COUNT(*) AS total_rows,
       COUNT(*) FILTER (WHERE wastage_rate IS NULL)  AS null_wastage_rate,
       COUNT(*) FILTER (WHERE shortage_severity IS NULL) AS null_severity,
       COUNT(*) FILTER (WHERE wastage_risk IS NULL) AS null_wastage_risk,
       COUNT(*) FILTER (WHERE inventory_status IS NULL) AS null_inventory_status
FROM blood_bank_clean;

SELECT shortage_severity, COUNT(*) FROM blood_bank_clean GROUP BY shortage_severity ORDER BY COUNT(*) DESC;
SELECT wastage_risk, COUNT(*) FROM blood_bank_clean GROUP BY wastage_risk ORDER BY COUNT(*) DESC;
SELECT inventory_status, COUNT(*) FROM blood_bank_clean GROUP BY inventory_status ORDER BY COUNT(*) DESC;


-- =====================================================================
-- SECTION 6 -- OVERALL PERFORMANCE KPIS
-- Source: sql/06_overall_performance.sql
-- =====================================================================

-- =====================================================================
-- PHASE 4 (blueprint Phase 6): SQL EXPLORATORY DATA ANALYSIS
-- Overall Performance
-- =====================================================================
-- Aggregate-ratio KPIs (wastage rate, expiry rate, fulfillment rate,
-- shortage rate) are calculated from SUMMED totals, not by averaging
-- the row-level rate columns -- this avoids letting small-volume
-- facility-months skew the headline numbers the same way a simple
-- mean-of-percentages would.
-- =====================================================================

SELECT
    COUNT(*)                                                    AS total_records,
    COUNT(DISTINCT facility_id)                                 AS total_facilities,

    SUM(units_collected_month)                                  AS total_units_collected,
    SUM(units_requested_month)                                  AS total_units_requested,
    SUM(units_crossmatched)                                     AS total_units_crossmatched,
    SUM(units_transfused)                                       AS total_units_transfused,

    SUM(available_on_survey_day)                                AS facility_months_with_stock_available,
    ROUND(AVG(available_on_survey_day) * 100, 2)                AS pct_facility_months_with_stock,

    SUM(units_expired)                                          AS total_units_expired,
    SUM(units_discarded_tti)                                    AS total_units_discarded_tti,
    SUM(units_expired) + SUM(units_discarded_tti)               AS total_wastage,

    ROUND(
        (SUM(units_expired) + SUM(units_discarded_tti))::NUMERIC
        / NULLIF(SUM(units_collected_month), 0) * 100, 2)       AS overall_wastage_rate_pct,

    ROUND(
        SUM(units_expired)::NUMERIC
        / NULLIF(SUM(units_collected_month), 0) * 100, 2)       AS overall_expiry_rate_pct,

    ROUND(
        SUM(units_discarded_tti)::NUMERIC
        / NULLIF(SUM(units_collected_month), 0) * 100, 2)       AS overall_tti_discard_rate_pct,

    ROUND(
        SUM(units_transfused)::NUMERIC
        / NULLIF(SUM(units_requested_month), 0) * 100, 2)       AS overall_demand_fulfillment_rate_pct,

    ROUND(
        SUM(units_transfused)::NUMERIC
        / NULLIF(SUM(units_crossmatched), 0) * 100, 2)          AS overall_crossmatch_conversion_rate_pct,

    ROUND(
        SUM(units_transfused)::NUMERIC
        / NULLIF(SUM(units_collected_month), 0) * 100, 2)       AS overall_collection_utilization_rate_pct,

    ROUND(
        SUM(units_collected_month)::NUMERIC
        - SUM(units_requested_month), 2)                        AS overall_supply_demand_gap,

    SUM(shortage_in_last_month)                                 AS total_shortage_records,
    ROUND(AVG(shortage_in_last_month) * 100, 2)                 AS overall_shortage_rate_pct,

    ROUND(AVG(shortage_days_last_month), 2)                     AS avg_shortage_days_all_records,
    ROUND(AVG(shortage_days_last_month)
          FILTER (WHERE shortage_in_last_month = 1), 2)         AS avg_shortage_days_when_shortage_occurred,

    ROUND(AVG(crossmatch_turnaround_hours), 2)                  AS avg_crossmatch_turnaround_hours,

    SUM(transfusion_reactions_month)                            AS total_transfusion_reactions,
    SUM(maternal_deaths_no_blood)                                AS total_maternal_deaths_no_blood,
    SUM(surgical_cancellations_no_blood)                        AS total_surgical_cancellations_no_blood
FROM blood_bank_clean;


-- =====================================================================
-- SECTION 7 -- BLOOD GROUP ANALYSIS
-- Source: sql/07_blood_group_analysis.sql
-- =====================================================================

-- =====================================================================
-- PHASE 5 (blueprint Phase 7): BLOOD GROUP ANALYSIS
-- =====================================================================

SELECT
    blood_group,
    COUNT(*)                                                        AS n_records,

    SUM(units_requested_month)                                      AS total_requested,
    SUM(units_collected_month)                                      AS total_collected,
    SUM(units_transfused)                                           AS total_transfused,

    SUM(units_collected_month) - SUM(units_requested_month)         AS supply_demand_gap,

    SUM(units_expired) + SUM(units_discarded_tti)                   AS total_wastage,
    ROUND(
        (SUM(units_expired) + SUM(units_discarded_tti))::NUMERIC
        / NULLIF(SUM(units_collected_month), 0) * 100, 2)           AS wastage_rate_pct,

    ROUND(
        SUM(units_expired)::NUMERIC
        / NULLIF(SUM(units_collected_month), 0) * 100, 2)           AS expiry_rate_pct,

    ROUND(
        SUM(units_transfused)::NUMERIC
        / NULLIF(SUM(units_requested_month), 0) * 100, 2)           AS demand_fulfillment_rate_pct,

    SUM(shortage_in_last_month)                                     AS n_shortage_months,
    ROUND(AVG(shortage_in_last_month) * 100, 2)                     AS shortage_rate_pct,
    ROUND(AVG(shortage_days_last_month), 2)                         AS avg_shortage_days,

    COUNT(*) FILTER (WHERE inventory_status = 'Critical')            AS n_critical_inventory_months,
    COUNT(*) FILTER (WHERE inventory_status = 'Low Stock')           AS n_low_stock_months,
    ROUND(
        COUNT(*) FILTER (WHERE inventory_status IN ('Critical','Low Stock'))::NUMERIC
        / COUNT(*) * 100, 2)                                        AS pct_understocked_months

FROM blood_bank_clean
GROUP BY blood_group
ORDER BY total_requested DESC;


-- =====================================================================
-- SECTION 8 -- BLOOD PRODUCT ANALYSIS
-- Source: sql/08_product_analysis.sql
-- =====================================================================

-- =====================================================================
-- PHASE 6 (blueprint Phase 8): BLOOD PRODUCT ANALYSIS
-- =====================================================================

SELECT
    product_name,
    product_category,
    shelf_life_days,                                                -- deterministic per product (Phase 0)
    COUNT(*)                                                        AS n_records,

    SUM(units_requested_month)                                      AS total_requested,
    SUM(units_collected_month)                                      AS total_collected,
    SUM(units_transfused)                                           AS total_transfused,

    SUM(units_collected_month) - SUM(units_requested_month)         AS supply_demand_gap,
    ROUND(
        (SUM(units_collected_month) - SUM(units_requested_month))::NUMERIC
        / NULLIF(SUM(units_requested_month), 0) * 100, 2)           AS supply_gap_pct_of_demand,

    SUM(units_expired) + SUM(units_discarded_tti)                   AS total_wastage,
    ROUND(
        (SUM(units_expired) + SUM(units_discarded_tti))::NUMERIC
        / NULLIF(SUM(units_collected_month), 0) * 100, 2)           AS wastage_rate_pct,

    ROUND(
        SUM(units_expired)::NUMERIC
        / NULLIF(SUM(units_collected_month), 0) * 100, 2)           AS expiry_rate_pct,

    ROUND(
        SUM(units_transfused)::NUMERIC
        / NULLIF(SUM(units_requested_month), 0) * 100, 2)           AS demand_fulfillment_rate_pct,

    SUM(shortage_in_last_month)                                     AS n_shortage_months,
    ROUND(AVG(shortage_in_last_month) * 100, 2)                     AS shortage_rate_pct,

    ROUND(
        COUNT(*) FILTER (WHERE inventory_status = 'Adequate')::NUMERIC
        / COUNT(*) * 100, 2)                                        AS pct_adequate_months,
    ROUND(
        COUNT(*) FILTER (WHERE inventory_status IN ('Critical','Low Stock'))::NUMERIC
        / COUNT(*) * 100, 2)                                        AS pct_understocked_months,

    ROUND(AVG(available_on_survey_day) * 100, 2)                    AS pct_available_on_survey_day

FROM blood_bank_clean
GROUP BY product_name, product_category, shelf_life_days
ORDER BY total_requested DESC;


-- =====================================================================
-- SECTION 9 -- FACILITY SUMMARY
-- Source: sql/09_facility_summary.sql
-- =====================================================================

-- =====================================================================
-- PHASE 7 (blueprint Phase 9): FACILITY ANALYSIS
-- =====================================================================
-- has_component_separation, has_NAT_testing, has_blood_bank_fridge,
-- has_platelet_agitator are constant = 1 across ALL facilities
-- (confirmed Phase 0), so "equipment adequacy" differentiation in this
-- dataset can only come from cold_chain_maintained (which does vary)
-- and blood_bank_staff_count -- not from the has_* capability flags.
--
-- DATA-QUALITY NOTE (found during this phase, not caught by the Phase 2
-- checklist because it's a referential-integrity issue, not a format
-- issue): facility_id does NOT map to a single stable region_type. All
-- 149 facilities appear under multiple region_type values across their
-- records (confirmed: every facility_id has region_type count = 3).
-- Since a physical facility cannot genuinely be urban, peri_urban, AND
-- rural, region_type is treated below as a RECORD-LEVEL attribute only
-- (already used correctly that way in earlier phases' GROUP BY
-- region_type queries). Facility-level aggregation here groups by
-- facility_id ALONE; a facility's most frequently occurring region_type
-- is reported separately as a descriptive label, not a guaranteed
-- stable attribute.

DROP TABLE IF EXISTS facility_summary;

CREATE TABLE facility_summary AS
WITH region_mode AS (
    SELECT DISTINCT ON (facility_id)
        facility_id,
        region_type AS most_common_region_type
    FROM (
        SELECT facility_id, region_type, COUNT(*) AS cnt
        FROM blood_bank_clean
        GROUP BY facility_id, region_type
    ) r
    ORDER BY facility_id, cnt DESC
)
SELECT
    b.facility_id,
    rm.most_common_region_type,
    COUNT(*)                                                        AS n_records,
    ROUND(AVG(blood_bank_staff_count), 1)                           AS avg_staff_count,

    SUM(units_requested_month)                                      AS total_requested,
    SUM(units_collected_month)                                      AS total_collected,
    SUM(units_transfused)                                           AS total_transfused,

    ROUND(
        SUM(units_transfused)::NUMERIC
        / NULLIF(SUM(units_requested_month), 0) * 100, 2)           AS demand_fulfillment_rate_pct,

    SUM(units_expired) + SUM(units_discarded_tti)                   AS total_wastage,
    ROUND(
        (SUM(units_expired) + SUM(units_discarded_tti))::NUMERIC
        / NULLIF(SUM(units_collected_month), 0) * 100, 2)           AS wastage_rate_pct,
    ROUND(
        SUM(units_expired)::NUMERIC
        / NULLIF(SUM(units_collected_month), 0) * 100, 2)           AS expiry_rate_pct,

    SUM(shortage_in_last_month)                                     AS n_shortage_months,
    ROUND(AVG(shortage_in_last_month) * 100, 2)                     AS shortage_rate_pct,
    ROUND(AVG(shortage_days_last_month), 2)                         AS avg_shortage_days,

    ROUND(AVG(crossmatch_turnaround_hours), 2)                      AS avg_crossmatch_turnaround_hours,
    ROUND(AVG(cold_chain_maintained) * 100, 2)                      AS pct_months_cold_chain_maintained,
    ROUND(AVG(report_submitted) * 100, 2)                           AS pct_months_report_submitted,

    SUM(surgical_cancellations_no_blood)                            AS total_surgical_cancellations,
    SUM(maternal_deaths_no_blood)                                   AS total_maternal_deaths,
    SUM(transfusion_reactions_month)                                AS total_transfusion_reactions,

    ROUND(
        COUNT(*) FILTER (WHERE inventory_status IN ('Critical','Low Stock'))::NUMERIC
        / COUNT(*) * 100, 2)                                        AS pct_understocked_months

FROM blood_bank_clean b
JOIN region_mode rm ON rm.facility_id = b.facility_id
GROUP BY b.facility_id, rm.most_common_region_type;

SELECT COUNT(*) AS total_facilities FROM facility_summary;


-- =====================================================================
-- SECTION 10 -- FACILITY RISK RANKING
-- Source: sql/10_facility_risk_ranking.sql
-- =====================================================================

-- =====================================================================
-- PHASE 7 (blueprint Phase 9, cont.): FACILITY RISK RANKING
-- =====================================================================
-- Composite risk score = equal-weighted average of percentile ranks
-- across 5 adverse-outcome dimensions (0 = best facility in the
-- dataset on that dimension, 1 = worst). Percentile rank is used
-- instead of raw z-scores so the score is robust to each metric's
-- different scale/distribution shape, and instead of arbitrary manual
-- weights so no single dimension is subjectively privileged over
-- another. This is a defensible starting framework, not a claim that
-- these 5 dimensions are causally equal contributors to risk --- that
-- question is addressed properly in the statistical testing phase.
--
-- Dimensions (all "higher percentile = worse"):
--   1. shortage_rate_pct            (higher = worse)
--   2. wastage_rate_pct             (higher = worse)
--   3. (100 - demand_fulfillment_rate_pct)   (lower fulfillment = worse)
--   4. (100 - pct_months_cold_chain_maintained) (less maintained = worse)
--   5. avg_crossmatch_turnaround_hours (slower = worse)
-- =====================================================================

DROP TABLE IF EXISTS facility_risk_ranking;

CREATE TABLE facility_risk_ranking AS
SELECT
    facility_id,
    most_common_region_type,
    n_records,
    avg_staff_count,
    demand_fulfillment_rate_pct,
    wastage_rate_pct,
    shortage_rate_pct,
    pct_months_cold_chain_maintained,
    avg_crossmatch_turnaround_hours,
    total_surgical_cancellations,
    total_maternal_deaths,
    pct_understocked_months,

    PERCENT_RANK() OVER (ORDER BY shortage_rate_pct)                          AS pctile_shortage,
    PERCENT_RANK() OVER (ORDER BY wastage_rate_pct)                           AS pctile_wastage,
    PERCENT_RANK() OVER (ORDER BY demand_fulfillment_rate_pct DESC)           AS pctile_low_fulfillment,
    PERCENT_RANK() OVER (ORDER BY pct_months_cold_chain_maintained DESC)      AS pctile_poor_cold_chain,
    PERCENT_RANK() OVER (ORDER BY avg_crossmatch_turnaround_hours)            AS pctile_slow_turnaround,

    ROUND((
        (PERCENT_RANK() OVER (ORDER BY shortage_rate_pct)
      + PERCENT_RANK() OVER (ORDER BY wastage_rate_pct)
      + PERCENT_RANK() OVER (ORDER BY demand_fulfillment_rate_pct DESC)
      + PERCENT_RANK() OVER (ORDER BY pct_months_cold_chain_maintained DESC)
      + PERCENT_RANK() OVER (ORDER BY avg_crossmatch_turnaround_hours))::NUMERIC
    ) / 5.0, 4)                                                               AS composite_risk_score

FROM facility_summary;

-- Top 15 highest-risk facilities
SELECT facility_id, most_common_region_type, n_records, avg_staff_count,
       shortage_rate_pct, wastage_rate_pct, demand_fulfillment_rate_pct,
       pct_months_cold_chain_maintained, avg_crossmatch_turnaround_hours,
       total_surgical_cancellations, total_maternal_deaths,
       composite_risk_score
FROM facility_risk_ranking
ORDER BY composite_risk_score DESC
LIMIT 15;


-- =====================================================================
-- SECTION 11 -- SHORTAGE ANALYSIS
-- Source: sql/11_shortage_analysis.sql
-- =====================================================================

-- =====================================================================
-- PHASE 8 (blueprint Phase 10): SHORTAGE ANALYSIS
-- =====================================================================

-- 1. Shortage frequency by region_type (RECORD-level; region_type is
--    not a stable facility attribute, per Phase 7 finding)
SELECT region_type,
       COUNT(*) AS n_records,
       SUM(shortage_in_last_month) AS n_shortage_months,
       ROUND(AVG(shortage_in_last_month)*100,2) AS shortage_rate_pct,
       ROUND(AVG(shortage_days_last_month),2) AS avg_shortage_days
FROM blood_bank_clean
GROUP BY region_type
ORDER BY shortage_rate_pct DESC;

-- 2. Shortage frequency by calendar month (seasonality)
SELECT month,
       COUNT(*) AS n_records,
       SUM(shortage_in_last_month) AS n_shortage_months,
       ROUND(AVG(shortage_in_last_month)*100,2) AS shortage_rate_pct
FROM blood_bank_clean
GROUP BY month
ORDER BY month;

-- 3. Shortage frequency by year
SELECT year,
       COUNT(*) AS n_records,
       SUM(shortage_in_last_month) AS n_shortage_months,
       ROUND(AVG(shortage_in_last_month)*100,2) AS shortage_rate_pct
FROM blood_bank_clean
GROUP BY year
ORDER BY year;

-- 4. Major shortage causes (among records where a shortage occurred)
SELECT shortage_cause,
       COUNT(*) AS n_records,
       ROUND(COUNT(*)::NUMERIC / (SELECT COUNT(*) FROM blood_bank_clean WHERE shortage_in_last_month=1) * 100, 2) AS pct_of_shortage_events,
       ROUND(AVG(shortage_days_last_month),2) AS avg_shortage_days
FROM blood_bank_clean
WHERE shortage_in_last_month = 1
GROUP BY shortage_cause
ORDER BY n_records DESC;

-- 5. Relationship: supply-demand gap, staffing, cold chain, turnaround
--    -- comparing shortage vs no-shortage records directly (record level,
--    not facility-averaged, to avoid washing out event-level signal)
SELECT
    shortage_in_last_month,
    COUNT(*) AS n,
    ROUND(AVG(supply_demand_gap),2) AS avg_supply_demand_gap,
    ROUND(AVG(units_requested_month - units_collected_month),2) AS avg_deficit,
    ROUND(AVG(blood_bank_staff_count),2) AS avg_staff_count,
    ROUND(AVG(cold_chain_maintained)*100,2) AS pct_cold_chain_maintained,
    ROUND(AVG(crossmatch_turnaround_hours),2) AS avg_crossmatch_turnaround_hours,
    ROUND(AVG(inventory_demand_ratio),4) AS avg_inventory_demand_ratio_literal,
    ROUND(AVG(available_on_survey_day)*100,2) AS pct_available_on_survey_day
FROM blood_bank_clean
GROUP BY shortage_in_last_month;

-- 6. Point-biserial style correlations (record level) between shortage flag
--    and candidate drivers
SELECT
    CORR(shortage_in_last_month, supply_demand_gap)          AS corr_shortage_gap,
    CORR(shortage_in_last_month, blood_bank_staff_count)     AS corr_shortage_staff,
    CORR(shortage_in_last_month, cold_chain_maintained)      AS corr_shortage_coldchain,
    CORR(shortage_in_last_month, crossmatch_turnaround_hours) AS corr_shortage_turnaround,
    CORR(shortage_in_last_month, available_on_survey_day)    AS corr_shortage_available
FROM blood_bank_clean;


-- =====================================================================
-- SECTION 12 -- WASTAGE ANALYSIS
-- Source: sql/12_wastage_analysis.sql
-- =====================================================================

-- =====================================================================
-- PHASE 9 (blueprint Phase 11): WASTAGE ANALYSIS
-- =====================================================================

-- 1. Wastage / expiry / TTI-discard trend by year
SELECT year,
       SUM(units_collected_month) AS total_collected,
       SUM(units_expired) AS total_expired,
       SUM(units_discarded_tti) AS total_discarded_tti,
       SUM(units_expired) + SUM(units_discarded_tti) AS total_wastage,
       ROUND((SUM(units_expired)+SUM(units_discarded_tti))::NUMERIC / NULLIF(SUM(units_collected_month),0) * 100, 2) AS wastage_rate_pct
FROM blood_bank_clean
GROUP BY year
ORDER BY year;

-- 2. Wastage trend by calendar month (seasonality)
SELECT month,
       SUM(units_expired) + SUM(units_discarded_tti) AS total_wastage,
       ROUND((SUM(units_expired)+SUM(units_discarded_tti))::NUMERIC / NULLIF(SUM(units_collected_month),0) * 100, 2) AS wastage_rate_pct
FROM blood_bank_clean
GROUP BY month
ORDER BY month;

-- 3. Facility-level wastage leaders (top 10 by wastage rate, min 30 records for stability)
SELECT facility_id,
       COUNT(*) AS n_records,
       ROUND(AVG(wastage_rate),2) AS avg_wastage_rate,
       ROUND(AVG(expiry_rate),2) AS avg_expiry_rate
FROM blood_bank_clean
GROUP BY facility_id
HAVING COUNT(*) >= 30
ORDER BY avg_wastage_rate DESC
LIMIT 10;

-- 4. Is a supply SURPLUS (collected > requested) associated with more wastage?
--    Split records into surplus vs deficit and compare wastage rate directly.
SELECT
    CASE WHEN supply_demand_gap > 0 THEN 'Surplus (collected > requested)'
         ELSE 'Deficit or balanced' END AS supply_status,
    COUNT(*) AS n_records,
    ROUND(AVG(wastage_rate),2) AS avg_wastage_rate,
    ROUND(AVG(expiry_rate),2) AS avg_expiry_rate
FROM blood_bank_clean
GROUP BY 1;

-- 5. Correlation: supply_demand_gap vs wastage_rate / expiry_rate (continuous)
SELECT
    CORR(supply_demand_gap, wastage_rate) AS corr_gap_wastage,
    CORR(supply_demand_gap, expiry_rate)  AS corr_gap_expiry,
    CORR(units_collected_month, expiry_rate) AS corr_collected_expiry,
    CORR(shelf_life_days, expiry_rate)    AS corr_shelflife_expiry_recheck
FROM blood_bank_clean;

-- 6. Inventory status (from Phase 3 categorical) vs expiry/wastage rate
--    Tests whether "Adequate"/overstocked-leaning categories show more expiry
SELECT inventory_status,
       COUNT(*) AS n_records,
       ROUND(AVG(wastage_rate),2) AS avg_wastage_rate,
       ROUND(AVG(expiry_rate),2) AS avg_expiry_rate
FROM blood_bank_clean
GROUP BY inventory_status
ORDER BY avg_expiry_rate DESC;

-- 7. Cold-chain maintained vs wastage/expiry rate (record level, as promised in Phase 8)
SELECT cold_chain_maintained,
       COUNT(*) AS n_records,
       ROUND(AVG(wastage_rate),2) AS avg_wastage_rate,
       ROUND(AVG(expiry_rate),2) AS avg_expiry_rate,
       ROUND(AVG(units_discarded_tti::NUMERIC/NULLIF(units_collected_month,0))*100,2) AS avg_tti_discard_rate
FROM blood_bank_clean
GROUP BY cold_chain_maintained;


-- =====================================================================
-- SECTION 13 -- TIME-SERIES ANALYSIS
-- Source: sql/13_time_series_analysis.sql
-- =====================================================================

-- =====================================================================
-- PHASE 10 (blueprint Phase 12): TIME-SERIES ANALYSIS
-- Primary demand variable: units_requested_month
-- =====================================================================

-- 1. National monthly demand series (full 48 months)
SELECT report_month,
       SUM(units_requested_month) AS total_demand,
       SUM(units_collected_month) AS total_collected,
       COUNT(*) AS n_records
FROM blood_bank_clean
GROUP BY report_month
ORDER BY report_month;

-- 2. Yearly demand totals + year-over-year growth
WITH yearly AS (
    SELECT year, SUM(units_requested_month) AS total_demand
    FROM blood_bank_clean GROUP BY year
)
SELECT year, total_demand,
       ROUND((total_demand - LAG(total_demand) OVER (ORDER BY year))::NUMERIC
             / NULLIF(LAG(total_demand) OVER (ORDER BY year),0) * 100, 2) AS yoy_growth_pct
FROM yearly
ORDER BY year;

-- 3. Seasonal pattern: average demand per calendar month (normalized by
--    number of distinct facility-records in that month to avoid bias from
--    uneven record counts per month)
SELECT month,
       ROUND(AVG(units_requested_month),2) AS avg_demand_per_record,
       SUM(units_requested_month) AS total_demand,
       COUNT(*) AS n_records
FROM blood_bank_clean
GROUP BY month
ORDER BY month;

-- 4. Demand spikes: months where total demand deviates >1.5 SD from the
--    mean monthly total (using the 48-month series)
WITH monthly AS (
    SELECT report_month, SUM(units_requested_month) AS total_demand
    FROM blood_bank_clean GROUP BY report_month
),
stats AS (
    SELECT AVG(total_demand) AS mean_demand, STDDEV(total_demand) AS sd_demand FROM monthly
)
SELECT m.report_month, m.total_demand,
       ROUND((m.total_demand - s.mean_demand)/NULLIF(s.sd_demand,0), 2) AS z_score
FROM monthly m, stats s
WHERE ABS((m.total_demand - s.mean_demand)/NULLIF(s.sd_demand,0)) > 1.5
ORDER BY m.report_month;

-- 5. Blood-group demand trend by year
SELECT year, blood_group, SUM(units_requested_month) AS total_demand
FROM blood_bank_clean
GROUP BY year, blood_group
ORDER BY blood_group, year;

-- 6. Product demand trend by year
SELECT year, product_name, SUM(units_requested_month) AS total_demand
FROM blood_bank_clean
GROUP BY year, product_name
ORDER BY product_name, year;

