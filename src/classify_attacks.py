"""
classify_attacks.py
-------------------
Loads the Petro Ivaniuk Kaggle dataset (missile_attacks_daily.csv) and
produces a clean, classified version focused on long-range drone attacks.

Dataset: https://www.kaggle.com/datasets/pityfm/massive-missile-attacks-on-ukraine
File:    missile_attacks_daily.csv

Key fields used:
  time_start   — date of the attack
  model        — weapon type (text, needs keyword matching)
  launched     — number of weapons launched
  destroyed    — number of weapons intercepted/destroyed

Classification logic
--------------------
Three tiers:
  1. SHAHED_DRONE  — Shahed-136, Shahed-131, Geran-2 (Iranian-origin loitering munitions)
  2. OTHER_DRONE   — other UAV types (Lancet excluded — it's a close-range system)
  3. MISSILE       — all ballistic/cruise missiles (Kalibr, Kh-101, Iskander, etc.)

For this analysis the primary focus is SHAHED_DRONE as these are:
  a) Russia's main long-range strike drone
  b) Weather-sensitive (subsonic, low-altitude, optically guided terminal phase)
  c) The system for which your interception data is richest

Output: data/attacks_classified.csv
"""

import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_PATH  = DATA_DIR / "missile_attacks_daily.csv"
OUTPUT_PATH = DATA_DIR / "attacks_classified.csv"

# ── Keyword classification maps ────────────────────────────────────────────────

SHAHED_KEYWORDS = [
    "shahed", "geran", "geran-2", "shahid",
    "shahed-136", "shahed-131", "shahed136", "shahed131",
]

MISSILE_KEYWORDS = [
    "kalibr", "kh-101", "kh-555", "kh-22", "kh-32",
    "kh-47", "kinzhal", "iskander", "oniks", "zircon",
    "kh-59", "kh-69", "tochka", "x-101", "x-555",
    "cruise missile", "ballistic",
]

# Exclude from drone category — not long-range strategic systems
EXCLUDE_KEYWORDS = [
    "lancet", "orlan", "supercam", "merlin", "zala",
]


def classify_model(model_str: str) -> str:
    """
    Returns 'SHAHED', 'MISSILE', 'OTHER_DRONE', or 'UNKNOWN'.
    """
    if pd.isna(model_str):
        return "UNKNOWN"
    s = str(model_str).lower().strip()

    # Exclusions first
    if any(kw in s for kw in EXCLUDE_KEYWORDS):
        return "EXCLUDED"

    if any(kw in s for kw in SHAHED_KEYWORDS):
        return "SHAHED"

    if any(kw in s for kw in MISSILE_KEYWORDS):
        return "MISSILE"

    # Remaining UAV-type entries
    if any(kw in s for kw in ["drone", "uav", "fpv", "shahab"]):
        return "OTHER_DRONE"

    return "UNKNOWN"


def load_and_classify(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ── Standardise column names ──
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Rename common variants
    rename_map = {
        "time_start": "date",
        "start_time": "date",
        "attack_date": "date",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # ── Parse dates ──
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    # ── Numeric coercion ──
    for col in ["launched", "destroyed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Classify weapon type ──
    df["weapon_class"] = df["model"].apply(classify_model)

    # ── Compute interception rate per row ──
    df["interception_rate"] = (df["destroyed"] / df["launched"]).clip(0, 1)

    # Flag rows with suspicious data (destroyed > launched)
    df["data_quality_flag"] = df["destroyed"] > df["launched"]

    print(f"Loaded {len(df):,} rows")
    print(f"\nWeapon class distribution:")
    print(df["weapon_class"].value_counts().to_string())
    print(f"\nDate range: {df['date'].min().date()} → {df['date'].max().date()}")

    return df


def aggregate_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to one row per date per weapon class.
    Also produces a 'SHAHED_ONLY' daily summary for the main analysis.
    """
    agg = df.groupby(["date", "weapon_class"]).agg(
        total_launched  = ("launched",  "sum"),
        total_destroyed = ("destroyed", "sum"),
        n_records       = ("launched",  "count"),
    ).reset_index()

    agg["interception_rate"] = (
        agg["total_destroyed"] / agg["total_launched"]
    ).clip(0, 1)

    return agg


def build_shahed_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Primary analysis dataframe: one row per attack date, Shahed drones only.
    Days with no Shahed launches are excluded (we only analyse days
    where Shaheds were actually deployed).
    """
    shaheds = df[df["weapon_class"] == "SHAHED"].copy()

    daily = shaheds.groupby("date").agg(
        launched        = ("launched",  "sum"),
        destroyed       = ("destroyed", "sum"),
        attack_records  = ("launched",  "count"),
    ).reset_index()

    daily["interception_rate"] = (
        daily["destroyed"] / daily["launched"]
    ).clip(0, 1)

    # Month + year columns for temporal grouping
    daily["year"]       = daily["date"].dt.year
    daily["month"]      = daily["date"].dt.month
    daily["year_month"] = daily["date"].dt.to_period("M")

    print(f"\n── Shahed daily summary ──")
    print(f"  Attack days: {len(daily)}")
    print(f"  Total launched:  {daily['launched'].sum():,}")
    print(f"  Total destroyed: {daily['destroyed'].sum():,}")
    print(f"  Mean interception rate: {daily['interception_rate'].mean():.1%}")
    print(f"  Median: {daily['interception_rate'].median():.1%}")
    print(f"  Std dev: {daily['interception_rate'].std():.3f}")

    return daily


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {INPUT_PATH}\n"
            "Download missile_attacks_daily.csv from:\n"
            "https://www.kaggle.com/datasets/pityfm/massive-missile-attacks-on-ukraine\n"
            "and place it in the data/ directory."
        )

    df = load_and_classify(INPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Full classified dataset saved → {OUTPUT_PATH}")

    # Shahed-specific daily summary
    shahed_daily = build_shahed_daily(df)
    shahed_path = DATA_DIR / "shahed_daily.csv"
    shahed_daily.to_csv(shahed_path, index=False)
    print(f"✓ Shahed daily summary saved → {shahed_path}")

    # Full aggregation by date + class (useful for missile comparison)
    agg = aggregate_by_date(df)
    agg_path = DATA_DIR / "attacks_by_date_class.csv"
    agg.to_csv(agg_path, index=False)
    print(f"✓ Aggregated by date+class saved → {agg_path}")


if __name__ == "__main__":
    main()
