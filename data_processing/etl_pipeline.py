#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETL Pipeline — WMS Fixed Assets Dataset
=========================================
Transforms the raw inventory Excel file into a clean, feature-engineered
CSV ready for machine-learning modelling.

Input : AGREGATED INVENTORY DATASET 2026.xlsx
Output: wms_dataset_transformed.csv
"""

# ── UTF-8 console output ────────────────────────────────────────────────────
import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_FILE  = r"c:\Users\PC\Documents\Next educacion\TFM\AGREGATED INVENTORY DATASET 2026.xlsx"
OUTPUT_FILE = r"c:\Users\PC\Documents\Next educacion\TFM\wms_dataset_transformed.csv"
REFERENCE_DATE = pd.Timestamp("2026-06-05")

SEPARATOR = "=" * 80


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 1 — Load & Audit                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def phase1_load_and_audit(path: str) -> pd.DataFrame:
    """Load the Excel file, standardise column names, cast types, print audit."""
    print(f"\n{SEPARATOR}")
    print("PHASE 1 — Load & Audit")
    print(SEPARATOR)

    df = pd.read_excel(path, engine="openpyxl")
    print(f"  Rows loaded: {len(df):,}")
    print(f"  Columns loaded: {df.shape[1]}")

    # ── Standardise column names to snake_case ──────────────────────────────
    def to_snake(name: str) -> str:
        name = name.strip()
        name = re.sub(r"[^\w\s]", "", name)          # remove punctuation
        name = re.sub(r"\s+", "_", name)              # spaces → underscores
        name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)  # camelCase split
        return name.lower()

    df.columns = [to_snake(c) for c in df.columns]
    print(f"\n  Standardised columns:\n    {list(df.columns)}")

    # ── Date conversions (string dd-mm-yyyy → datetime64) ───────────────────
    date_cols = ["purchase_date", "data_place_in_service",
                 "last_issued_or_transfer_date", "day_disposed"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%d-%m-%Y", errors="coerce")

    # ── Price to float64 ────────────────────────────────────────────────────
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float64")

    # ── Audit report ────────────────────────────────────────────────────────
    print("\n  ── Dtype / Null / Cardinality Report ──")
    for col in df.columns:
        null_ct  = df[col].isnull().sum()
        null_pct = null_ct / len(df) * 100
        card     = df[col].nunique()
        print(f"    {col:40s}  dtype={str(df[col].dtype):15s}  "
              f"nulls={null_ct:6,} ({null_pct:5.2f}%)  cardinality={card:,}")

    return df


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 2 — Label Standardization                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def phase2_label_standardization(df: pd.DataFrame) -> pd.DataFrame:
    """Fix inconsistent labels in status, condition, and category."""
    print(f"\n{SEPARATOR}")
    print("PHASE 2 — Label Standardization")
    print(SEPARATOR)

    # ── Status ──────────────────────────────────────────────────────────────
    status_map = {
        "In use":   "In Use",
        "In US":    "In Use",
        "Storage ": "In Storage",
        "Storage":  "In Storage",
    }
    df["status"] = df["status"].str.strip().replace(status_map)
    print(f"\n  Status value_counts:\n{df['status'].value_counts(dropna=False).to_string()}")

    # ── Condition ───────────────────────────────────────────────────────────
    condition_map = {
        "worn":   "Worn",
        "wonr":   "Worn",
        "broken": "Broken",
        "Used":   "Worn",
    }
    df["condition"] = df["condition"].str.strip().replace(condition_map)
    print(f"\n  Condition value_counts:\n{df['condition'].value_counts(dropna=False).to_string()}")

    # ── Category — fix misplaced values ─────────────────────────────────────
    valid_categories = {"Hardware", "Tools", "Consumable", "Supplies", "Books/Documents"}
    bad_mask = ~df["category"].isin(valid_categories) & df["category"].notna()
    n_bad = bad_mask.sum()
    print(f"\n  Category: {n_bad} rows with invalid values → imputing with dept mode")

    if n_bad > 0:
        # Compute mode of VALID category per department
        valid_df = df.loc[df["category"].isin(valid_categories)]
        dept_mode = (valid_df.groupby("department")["category"]
                     .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Hardware"))

        for idx in df.loc[bad_mask].index:
            dept = df.at[idx, "department"]
            df.at[idx, "category"] = dept_mode.get(dept, "Hardware")

    print(f"\n  Category value_counts:\n{df['category'].value_counts(dropna=False).to_string()}")

    return df


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 3 — Imputation & Outlier Treatment                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def phase3_imputation_and_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Impute nulls with business logic; treat price outliers via Winsorization."""
    print(f"\n{SEPARATOR}")
    print("PHASE 3 — Imputation & Outlier Treatment")
    print(SEPARATOR)

    # ── 3A. Null imputation ─────────────────────────────────────────────────
    # Status nulls → conditioned on Condition
    status_null_mask = df["status"].isna()
    print(f"\n  Status nulls before imputation: {status_null_mask.sum()}")
    cond_to_status = {
        "Broken": "Needs Repair",
        "Worn":   "Needs Repair",
        "Good":   "In Use",
        "New":    "In Use",
    }
    for cond_val, status_val in cond_to_status.items():
        mask = status_null_mask & (df["condition"] == cond_val)
        df.loc[mask, "status"] = status_val
    # Remaining status nulls (if condition is also null) → 'In Use'
    df["status"] = df["status"].fillna("In Use")
    print(f"  Status nulls after imputation:  {df['status'].isna().sum()}")

    # Condition nulls → conditioned on Status
    cond_null_mask = df["condition"].isna()
    print(f"\n  Condition nulls before imputation: {cond_null_mask.sum()}")
    status_to_cond = {
        "Needs Repair": "Broken",
        "In Use":       "Good",
        "In Storage":   "Good",
    }
    for status_val, cond_val in status_to_cond.items():
        mask = cond_null_mask & (df["status"] == status_val)
        df.loc[mask, "condition"] = cond_val
    df["condition"] = df["condition"].fillna("Good")
    print(f"  Condition nulls after imputation:  {df['condition'].isna().sum()}")

    # ── Binary variables BEFORE dropping columns ────────────────────────────
    df["has_purchase_order"] = df["po_"].notna().astype(int)
    df["has_supplier_info"]  = df["supplier"].notna().astype(int)
    print(f"\n  has_purchase_order distribution:\n{df['has_purchase_order'].value_counts().to_string()}")
    print(f"\n  has_supplier_info distribution:\n{df['has_supplier_info'].value_counts().to_string()}")

    # ── Drop columns ───────────────────────────────────────────────────────
    cols_to_drop = ["part_", "notes"]
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_to_drop, inplace=True)
    print(f"\n  Dropped columns: {existing_to_drop}")

    # ── 3B. Price outlier treatment ─────────────────────────────────────────
    print(f"\n  ── Price Outlier Treatment (Winsorization p1/p99) ──")
    df["price"] = df["price"].round(2)
    price_before = df["price"].describe()
    print(f"  BEFORE:\n{price_before.to_string()}")

    p1  = np.percentile(df["price"].dropna(), 1)
    p99 = np.percentile(df["price"].dropna(), 99)
    print(f"\n  Winsorization bounds: p1={p1:.2f}, p99={p99:.2f}")
    df["price"] = df["price"].clip(lower=p1, upper=p99)

    price_after = df["price"].describe()
    print(f"\n  AFTER:\n{price_after.to_string()}")

    return df


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 4 — Feature Engineering & Encoding                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def phase4_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features, encode categoricals, build target variable."""
    print(f"\n{SEPARATOR}")
    print("PHASE 4 — Feature Engineering & Encoding")
    print(SEPARATOR)

    # ── 4A. Primary Feature Variables ───────────────────────────────────────
    print("\n  4A — Primary features")

    # 1. lifespan_days
    df["lifespan_days"] = (df["day_disposed"] - df["data_place_in_service"]).dt.days
    print(f"    lifespan_days  non-null: {df['lifespan_days'].notna().sum()}")

    # 2. estimated_useful_life_days
    useful_life_map = {
        "Hardware":        1825,
        "Tools":           1095,
        "Supplies":         365,
        "Consumable":       183,
        "Books/Documents": 1825,
    }
    df["estimated_useful_life_days"] = df["category"].map(useful_life_map)
    print(f"    estimated_useful_life_days nulls: {df['estimated_useful_life_days'].isna().sum()}")

    # 3. asset_current_age_days
    df["asset_current_age_days"] = (REFERENCE_DATE - df["data_place_in_service"]).dt.days
    print(f"    asset_current_age_days nulls: {df['asset_current_age_days'].isna().sum()}")

    # 4. linear_depreciation_rate ($/day)
    df["linear_depreciation_rate"] = df["price"] / df["estimated_useful_life_days"]
    print(f"    linear_depreciation_rate nulls: {df['linear_depreciation_rate'].isna().sum()}")

    # 5. stagnation_time_days  (NaN → -1)
    df["stagnation_time_days"] = (REFERENCE_DATE - df["last_issued_or_transfer_date"]).dt.days
    df["stagnation_time_days"] = df["stagnation_time_days"].fillna(-1).astype(int)
    print(f"    stagnation_time_days nulls: {df['stagnation_time_days'].isna().sum()}")

    # 6. depreciation_pct
    df["depreciation_pct"] = (df["asset_current_age_days"] / df["estimated_useful_life_days"]) * 100
    print(f"    depreciation_pct nulls: {df['depreciation_pct'].isna().sum()}")

    # ── 4B. Support Variables ───────────────────────────────────────────────
    print("\n  4B — Support variables")

    # 7. days_to_service
    df["days_to_service"] = (df["data_place_in_service"] - df["purchase_date"]).dt.days
    print(f"    days_to_service nulls: {df['days_to_service'].isna().sum()}")

    # 8. has_been_transferred
    df["has_been_transferred"] = df["last_issued_or_transfer_date"].notna().astype(int)
    print(f"    has_been_transferred distribution:\n{df['has_been_transferred'].value_counts().to_string()}")

    # 9. price_per_unit
    df["price_per_unit"] = df["price"] / df["quantity"]
    print(f"    price_per_unit nulls: {df['price_per_unit'].isna().sum()}")

    # 10. is_high_value
    p75 = df["price_per_unit"].quantile(0.75)
    df["is_high_value"] = (df["price_per_unit"] > p75).astype(int)
    print(f"    is_high_value threshold (p75): {p75:.2f}")
    print(f"    is_high_value distribution:\n{df['is_high_value'].value_counts().to_string()}")

    # ── 4C. Categorical Encoding ────────────────────────────────────────────
    print("\n  4C — Categorical encoding")

    # category → category_encoded (ordinal)
    cat_ord = {"Consumable": 1, "Supplies": 2, "Books/Documents": 2,
               "Tools": 3, "Hardware": 4}
    df["category_encoded"] = df["category"].map(cat_ord)
    print(f"    category_encoded nulls: {df['category_encoded'].isna().sum()}")

    # condition → condition_encoded (ordinal)
    cond_ord = {"New": 0, "Good": 1, "Worn": 2, "Broken": 3}
    df["condition_encoded"] = df["condition"].map(cond_ord)
    print(f"    condition_encoded nulls: {df['condition_encoded'].isna().sum()}")

    # department → one-hot
    dept_dummies = pd.get_dummies(df["department"], drop_first=True, prefix="dept")
    dept_dummies = dept_dummies.astype(int)
    df = pd.concat([df, dept_dummies], axis=1)
    print(f"    department one-hot columns: {list(dept_dummies.columns)}")

    # campus → one-hot
    campus_dummies = pd.get_dummies(df["campus"], drop_first=True, prefix="campus")
    campus_dummies = campus_dummies.astype(int)
    df = pd.concat([df, campus_dummies], axis=1)
    print(f"    campus one-hot columns: {list(campus_dummies.columns)}")

    # ── 4D. Target Variable ────────────────────────────────────────────────
    print("\n  4D — Target variable")
    df["target_needs_repair"] = (df["status"] == "Needs Repair").astype(int)
    print(f"    target_needs_repair distribution:\n{df['target_needs_repair'].value_counts().to_string()}")

    # ── 4E. Final Cleanup ──────────────────────────────────────────────────
    print("\n  4E — Final cleanup")
    cols_to_drop_final = [
        "day_disposed", "po_", "supplier",
        "status", "condition", "category",
        "department", "campus",
        "purchase_date", "data_place_in_service", "last_issued_or_transfer_date",
    ]
    existing_final = [c for c in cols_to_drop_final if c in df.columns]
    df.drop(columns=existing_final, inplace=True)
    print(f"    Dropped columns: {existing_final}")
    print(f"    Remaining columns: {list(df.columns)}")

    return df


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  VALIDATION                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def validate(df: pd.DataFrame) -> None:
    """Print final validation report."""
    print(f"\n{SEPARATOR}")
    print("VALIDATION REPORT")
    print(SEPARATOR)

    # 1. Shape
    print(f"\n  1. Final shape: {df.shape}")

    # 2. Column names
    print(f"\n  2. Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"      {i:3d}. {col}")

    # 3. Null counts
    print(f"\n  3. Null counts per column:")
    null_counts = df.isnull().sum()
    for col, nc in null_counts.items():
        flag = " ⚠️" if nc > 0 else ""
        print(f"      {col:40s}  {nc:6,}{flag}")

    # 4. Target distribution
    print(f"\n  4. Target variable distribution (target_needs_repair):")
    tgt = df["target_needs_repair"].value_counts()
    tgt_pct = df["target_needs_repair"].value_counts(normalize=True) * 100
    for val in tgt.index:
        print(f"      {val}: {tgt[val]:,}  ({tgt_pct[val]:.2f}%)")

    # 5. Summary statistics of engineered features
    feat_cols = ["asset_current_age_days", "linear_depreciation_rate",
                 "stagnation_time_days", "depreciation_pct"]
    print(f"\n  5. Summary statistics of key engineered features:")
    print(df[feat_cols].describe().to_string())


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def main() -> None:
    print("\n" + "▓" * 80)
    print("  WMS FIXED ASSETS — ETL PIPELINE")
    print("▓" * 80)

    df = phase1_load_and_audit(INPUT_FILE)
    df = phase2_label_standardization(df)
    df = phase3_imputation_and_outliers(df)
    df = phase4_feature_engineering(df)

    # ── Export ──────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  ✅ Dataset exported to: {OUTPUT_FILE}")

    validate(df)

    print(f"\n{'▓' * 80}")
    print("  ETL PIPELINE COMPLETE")
    print("▓" * 80 + "\n")


if __name__ == "__main__":
    main()
