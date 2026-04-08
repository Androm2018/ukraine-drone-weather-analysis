"""
operational_analysis.py
-----------------------
Five operational analyses of Russian missile/drone attacks on Ukraine.

1. Saturation Threshold    — at what launch volume does interception rate collapse?
2. Salvo Composition       — do mixed salvos (Shahed + missiles) lower interception rates?
3. Weapon Mix Evolution    — how has Russia's weapon ratio shifted over time?
4. Geographic Dispersion   — is Russia spreading attacks across more oblasts over time?
5. Target Category Rotation— does Russia rotate between target types systematically?

Data: Petro Ivaniuk Kaggle dataset (missile_attacks_daily.csv)
"""

import ast
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("outputs/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
UKRAINE_BLUE   = "#005BBB"
UKRAINE_YELLOW = "#FFD700"
RED            = "#B71C1C"
GREY           = "#616161"
GREEN          = "#2E7D32"
ORANGE         = "#E65100"
PURPLE         = "#6A1B9A"
BG             = "#FAFAFA"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})

PCT = mticker.FuncFormatter(lambda x, _: f"{x:.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    path = DATA_DIR / "missile_attacks_daily.csv"
    if not path.exists():
        raise FileNotFoundError("Place missile_attacks_daily.csv in data/")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Parse date from time_start
    df["date"] = pd.to_datetime(
        df["time_start"].str[:10], errors="coerce"
    ).dt.normalize()
    df = df.dropna(subset=["date"])

    # Numeric
    for col in ["launched", "destroyed", "is_shahed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["interception_rate"] = (df["destroyed"] / df["launched"]).clip(0, 1)

    # Classify weapon type
    def classify(model):
        if pd.isna(model):
            return "UNKNOWN"
        m = str(model).lower()
        if any(k in m for k in ["shahed", "geran", "shahid"]):
            return "SHAHED"
        if any(k in m for k in ["kalibr", "kh-101", "x-101", "kh-555", "x-555",
                                  "kh-22", "kh-32", "kh-47", "kinzhal",
                                  "iskander", "oniks", "zircon", "kh-59",
                                  "kh-69", "x-59", "x-69"]):
            return "MISSILE"
        if any(k in m for k in ["uav", "drone", "unknown uav"]):
            return "OTHER_UAV"
        return "OTHER"

    df["weapon_class"] = df["model"].apply(classify)

    # Year + month
    df["year"]       = df["date"].dt.year
    df["month"]      = df["date"].dt.month
    df["year_month"] = df["date"].dt.to_period("M")

    print(f"Loaded {len(df):,} rows | {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def parse_affected_regions(val) -> list:
    """Parse the affected_region column (stored as string repr of list)."""
    if pd.isna(val) or val == "":
        return []
    try:
        result = ast.literal_eval(str(val))
        if isinstance(result, list):
            return [r.replace(" oblast", "").strip() for r in result]
    except Exception:
        pass
    return []


def parse_destroyed_details(val) -> dict:
    """Parse destroyed_details dict (south/east/north/west breakdown)."""
    if pd.isna(val) or val == "{}":
        return {}
    try:
        # Replace NaN with null for safe parsing
        cleaned = str(val).replace("NaN", "null")
        result = ast.literal_eval(cleaned)
        if isinstance(result, dict):
            return {k: v for k, v in result.items() if v is not None}
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: SATURATION THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════

def analysis_saturation(df: pd.DataFrame):
    """
    At what launch volume does Ukraine's air defence start to fail?
    Tests for a non-linear threshold above which interception rates collapse.
    """
    print("\n═══ ANALYSIS 1: SATURATION THRESHOLD ═══")

    # Daily aggregation — all weapons
    daily = df.groupby("date").agg(
        launched        = ("launched",          "sum"),
        destroyed       = ("destroyed",         "sum"),
        shahed_launched = ("is_shahed",          "sum"),
    ).reset_index()
    daily["interception_rate"] = (daily["destroyed"] / daily["launched"]).clip(0, 1)
    daily = daily[daily["launched"] >= 5].copy()

    # Bin by launch volume
    bins   = [0, 30, 60, 100, 150, 200, 300, 400, 600, 2000]
    labels = ["1-30", "31-60", "61-100", "101-150",
              "151-200", "201-300", "301-400", "401-600", "600+"]
    daily["volume_bin"] = pd.cut(daily["launched"], bins=bins, labels=labels)

    summary = daily.groupby("volume_bin", observed=True).agg(
        mean_rate = ("interception_rate", "mean"),
        n         = ("interception_rate", "count"),
        std       = ("interception_rate", "std"),
    ).reset_index()
    summary["sem"]      = summary["std"] / np.sqrt(summary["n"])
    summary["ci95_low"] = (summary["mean_rate"] - 1.96 * summary["sem"]).clip(0)
    summary["ci95_high"]= (summary["mean_rate"] + 1.96 * summary["sem"]).clip(0, 1)

    print(summary[["volume_bin", "n", "mean_rate"]].to_string(index=False))

    # ── Find threshold via piecewise — where does rate start dropping? ──
    # Simple approach: find the bin where rate drops most sharply
    summary["rate_drop"] = summary["mean_rate"].diff().fillna(0)
    threshold_bin = summary.loc[summary["rate_drop"].idxmin(), "volume_bin"]
    print(f"\nSharpest interception rate drop at volume bin: {threshold_bin}")

    # ── Chart ──
    fig, ax = plt.subplots(figsize=(11, 5))

    colors = [UKRAINE_BLUE if r >= 0.70 else ORANGE if r >= 0.55 else RED
              for r in summary["mean_rate"]]

    bars = ax.bar(
        range(len(summary)),
        summary["mean_rate"] * 100,
        color=colors, width=0.65,
        edgecolor="white", linewidth=1.2, zorder=3,
    )
    ax.errorbar(
        range(len(summary)),
        summary["mean_rate"] * 100,
        yerr=[(summary["mean_rate"] - summary["ci95_low"]) * 100,
              (summary["ci95_high"] - summary["mean_rate"]) * 100],
        fmt="none", color="#333", capsize=4, linewidth=1.5, zorder=4,
    )

    for i, (bar, (_, row)) in enumerate(zip(bars, summary.iterrows())):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{row['mean_rate']:.0%}\n(n={int(row['n'])})",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary["volume_bin"].astype(str), rotation=30, ha="right")
    ax.set_xlabel("Total Weapons Launched per Attack Night")
    ax.set_ylabel("Mean Interception Rate (%)")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, 110)
    ax.set_title("Air Defence Saturation: Interception Rate vs. Launch Volume\n"
                 "Identifies the threshold above which Ukrainian AD is overwhelmed",
                 pad=12)
    ax.axhline(daily["interception_rate"].mean() * 100, color=GREY,
               linestyle="--", linewidth=1, alpha=0.6)
    ax.text(len(summary) - 0.5,
            daily["interception_rate"].mean() * 100 + 1.5,
            f"Overall mean: {daily['interception_rate'].mean():.0%}",
            ha="right", fontsize=9, color=GREY)

    patches = [
        mpatches.Patch(color=UKRAINE_BLUE, label="≥70% interception (manageable)"),
        mpatches.Patch(color=ORANGE,       label="55-70% (degraded)"),
        mpatches.Patch(color=RED,          label="<55% (overwhelmed)"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    path = OUTPUT_DIR / "op1_saturation_threshold.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

    return daily, summary


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: SALVO COMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def analysis_salvo_composition(df: pd.DataFrame):
    """
    Do mixed salvos (Shahed + cruise missiles on same night) produce lower
    interception rates than single-weapon attacks?
    Tests the 'Shahed as decoy/saturation' hypothesis.
    """
    print("\n═══ ANALYSIS 2: SALVO COMPOSITION ═══")

    # Identify weapon classes present per attack night
    date_classes = df.groupby("date")["weapon_class"].apply(
        lambda x: set(x.dropna())
    ).reset_index()
    date_classes.columns = ["date", "classes"]

    def salvo_type(classes):
        has_shahed  = "SHAHED"  in classes
        has_missile = "MISSILE" in classes
        if has_shahed and has_missile:
            return "MIXED (Shahed + Missile)"
        elif has_shahed:
            return "Shahed Only"
        elif has_missile:
            return "Missile Only"
        else:
            return "Other/Unknown"

    date_classes["salvo_type"] = date_classes["classes"].apply(salvo_type)

    # Daily totals
    daily = df.groupby("date").agg(
        launched  = ("launched",  "sum"),
        destroyed = ("destroyed", "sum"),
    ).reset_index()
    daily["interception_rate"] = (daily["destroyed"] / daily["launched"]).clip(0, 1)

    merged = daily.merge(date_classes[["date", "salvo_type"]], on="date")

    summary = merged.groupby("salvo_type").agg(
        mean_rate     = ("interception_rate", "mean"),
        median_rate   = ("interception_rate", "median"),
        n             = ("interception_rate", "count"),
        std           = ("interception_rate", "std"),
        mean_launched = ("launched",          "mean"),
    ).reset_index()
    summary["sem"]       = summary["std"] / np.sqrt(summary["n"])
    summary["ci95_low"]  = (summary["mean_rate"] - 1.96 * summary["sem"]).clip(0)
    summary["ci95_high"] = (summary["mean_rate"] + 1.96 * summary["sem"]).clip(0, 1)

    print(summary[["salvo_type", "n", "mean_rate", "mean_launched"]].to_string(index=False))

    # Statistical test: Mixed vs Shahed Only
    mixed  = merged[merged["salvo_type"] == "MIXED (Shahed + Missile)"]["interception_rate"]
    shahed = merged[merged["salvo_type"] == "Shahed Only"]["interception_rate"]
    if len(mixed) >= 5 and len(shahed) >= 5:
        t, p = stats.ttest_ind(mixed, shahed)
        print(f"\n  Mixed vs Shahed Only: t={t:.3f}, p={p:.4f}")
        print(f"  Mixed mean:        {mixed.mean():.1%}")
        print(f"  Shahed-only mean:  {shahed.mean():.1%}")
        delta = (shahed.mean() - mixed.mean()) * 100
        direction = "LOWER" if delta > 0 else "HIGHER"
        print(f"  Mixed salvos produce {abs(delta):.1f}pp {direction} interception rates")

    # ── Chart: two panels ──
    order = ["Shahed Only", "MIXED (Shahed + Missile)", "Missile Only", "Other/Unknown"]
    summary["salvo_type"] = pd.Categorical(summary["salvo_type"], categories=order, ordered=True)
    summary = summary.sort_values("salvo_type").dropna(subset=["mean_rate"])

    colors_map = {
        "Shahed Only":               UKRAINE_BLUE,
        "MIXED (Shahed + Missile)":  RED,
        "Missile Only":              ORANGE,
        "Other/Unknown":             GREY,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: interception rate
    bars = ax1.bar(
        range(len(summary)),
        summary["mean_rate"] * 100,
        color=[colors_map.get(s, GREY) for s in summary["salvo_type"]],
        width=0.55, edgecolor="white", linewidth=1.2, zorder=3,
    )
    ax1.errorbar(
        range(len(summary)), summary["mean_rate"] * 100,
        yerr=[(summary["mean_rate"] - summary["ci95_low"]) * 100,
              (summary["ci95_high"] - summary["mean_rate"]) * 100],
        fmt="none", color="#333", capsize=4, linewidth=1.5, zorder=4,
    )
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5,
                 f"{row['mean_rate']:.0%}\n(n={int(row['n'])})",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_xticks(range(len(summary)))
    ax1.set_xticklabels(
        [s.replace(" (", "\n(") for s in summary["salvo_type"].astype(str)],
        fontsize=9
    )
    ax1.yaxis.set_major_formatter(PCT)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Mean Interception Rate (%)")
    ax1.set_title("Interception Rate by Salvo Type")
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: mean launch volume
    ax2.bar(
        range(len(summary)),
        summary["mean_launched"],
        color=[colors_map.get(s, GREY) for s in summary["salvo_type"]],
        width=0.55, edgecolor="white", linewidth=1.2, zorder=3,
    )
    for i, (_, row) in enumerate(summary.iterrows()):
        ax2.text(i, row["mean_launched"] + 2,
                 f"{row['mean_launched']:.0f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_xticks(range(len(summary)))
    ax2.set_xticklabels(
        [s.replace(" (", "\n(") for s in summary["salvo_type"].astype(str)],
        fontsize=9
    )
    ax2.set_ylabel("Mean Weapons Launched per Night")
    ax2.set_title("Launch Volume by Salvo Type")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Salvo Composition Analysis: Tests the 'Shahed as Decoy' Hypothesis\n"
                 "Do mixed salvos produce lower interception rates than Shahed-only attacks?",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / "op2_salvo_composition.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

    return merged, summary


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: WEAPON MIX EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def analysis_weapon_mix(df: pd.DataFrame):
    """
    How has Russia's weapon mix (Shahed vs cruise missile vs ballistic) evolved?
    Tracks the shift toward cheaper, mass-producible systems over time.
    """
    print("\n═══ ANALYSIS 3: WEAPON MIX EVOLUTION ═══")

    monthly = df.groupby(["year_month", "weapon_class"])["launched"].sum().reset_index()
    pivot   = monthly.pivot_table(
        index="year_month", columns="weapon_class",
        values="launched", aggfunc="sum", fill_value=0
    )

    # Compute share of total
    pivot_share = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_share.index = pivot_share.index.to_timestamp()

    print("\nFinal 6 months weapon share (%):")
    print(pivot_share.tail(6).round(1).to_string())

    # ── Chart: stacked area ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    class_colors = {
        "SHAHED":    UKRAINE_BLUE,
        "MISSILE":   RED,
        "OTHER_UAV": ORANGE,
        "OTHER":     GREY,
        "UNKNOWN":   "#BDBDBD",
    }

    # Panel 1: absolute volume
    classes = [c for c in ["SHAHED", "MISSILE", "OTHER_UAV", "OTHER", "UNKNOWN"]
               if c in pivot.columns]
    ax1.stackplot(
        pivot.index.to_timestamp(),
        [pivot[c] for c in classes],
        labels=classes,
        colors=[class_colors[c] for c in classes],
        alpha=0.85,
    )
    ax1.set_ylabel("Weapons Launched per Month")
    ax1.set_title("Russian Weapon Mix: Absolute Volume Over Time")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.2)

    # Panel 2: percentage share
    ax2.stackplot(
        pivot_share.index,
        [pivot_share[c] for c in classes if c in pivot_share.columns],
        labels=classes,
        colors=[class_colors[c] for c in classes if c in pivot_share.columns],
        alpha=0.85,
    )
    ax2.set_ylabel("Share of Total Launches (%)")
    ax2.set_title("Russian Weapon Mix: Proportional Share Over Time")
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_formatter(PCT)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    path = OUTPUT_DIR / "op3_weapon_mix_evolution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

    return pivot, pivot_share


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: GEOGRAPHIC DISPERSION
# ══════════════════════════════════════════════════════════════════════════════

def analysis_geographic_dispersion(df: pd.DataFrame):
    """
    Is Russia spreading attacks across more oblasts over time?
    Dispersion forces Ukraine to spread AD assets thinner.
    Uses the 'affected_region' column (parsed list of oblasts per attack).
    """
    print("\n═══ ANALYSIS 4: GEOGRAPHIC DISPERSION ═══")

    col = "affected_region"
    if col not in df.columns:
        print("  'affected_region' column not found — skipping")
        return None, None

    df = df.copy()
    df["regions_list"] = df[col].apply(parse_affected_regions)
    df["n_regions"]    = df["regions_list"].apply(len)

    # Only rows where we have region data
    has_regions = df[df["n_regions"] > 0].copy()
    print(f"  Rows with region data: {len(has_regions)}")

    # Monthly mean number of oblasts affected per attack
    monthly = has_regions.groupby("year_month").agg(
        mean_regions = ("n_regions", "mean"),
        max_regions  = ("n_regions", "max"),
        n_attacks    = ("n_regions", "count"),
    ).reset_index()
    monthly.index = monthly["year_month"].dt.to_timestamp()

    # Most frequently targeted oblasts overall
    all_regions = [r for sublist in has_regions["regions_list"] for r in sublist]
    region_counts = pd.Series(all_regions).value_counts().head(15)
    print("\nTop 10 most targeted oblasts:")
    print(region_counts.head(10).to_string())

    # ── Chart: two panels ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: dispersion over time
    dates = monthly["year_month"].dt.to_timestamp()
    ax1.plot(dates, monthly["mean_regions"],
             color=UKRAINE_BLUE, linewidth=2, label="Mean oblasts per attack")
    ax1.fill_between(dates,
                     monthly["mean_regions"] - monthly["mean_regions"].std(),
                     monthly["mean_regions"] + monthly["mean_regions"].std(),
                     alpha=0.15, color=UKRAINE_BLUE)
    rolling = monthly["mean_regions"].rolling(3, center=True).mean()
    ax1.plot(dates, rolling,
             color=UKRAINE_YELLOW, linewidth=2.5, linestyle="--",
             label="3-month rolling mean")
    ax1.set_ylabel("Mean Number of Oblasts Affected per Attack")
    ax1.set_title("Geographic Dispersion Over Time\nHigher = Russia spreading attacks wider")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    # Panel 2: most targeted oblasts (horizontal bar)
    region_counts.head(12).sort_values().plot(
        kind="barh", ax=ax2,
        color=UKRAINE_BLUE, edgecolor="white", linewidth=0.8,
    )
    ax2.set_xlabel("Number of Attack Events")
    ax2.set_title("Most Frequently Targeted Oblasts\n(all attacks, 2022–present)")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "op4_geographic_dispersion.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

    return monthly, region_counts


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: TARGET CATEGORY ROTATION
# ══════════════════════════════════════════════════════════════════════════════

def analysis_target_rotation(df: pd.DataFrame):
    """
    Does Russia rotate systematically between target categories?
    Uses the 'target' field to classify attack focus and tracks
    how the mix changes month-by-month.
    """
    print("\n═══ ANALYSIS 5: TARGET CATEGORY ROTATION ═══")

    df = df.copy()

    def classify_target(val):
        if pd.isna(val):
            return "UNKNOWN"
        v = str(val).lower()
        # Energy infrastructure
        if any(k in v for k in ["energy", "power", "electric", "thermal",
                                  "heating", "grid"]):
            return "ENERGY"
        # Military / defence
        if any(k in v for k in ["military", "army", "air base", "airfield",
                                  "depot", "ammunition", "logistics"]):
            return "MILITARY"
        # Urban / civilian
        if any(k in v for k in ["city", "urban", "residential", "civilian",
                                  "hospital", "school"]):
            return "URBAN/CIVILIAN"
        # Broad geographic targets — classify by region name presence
        # These are "Ukraine", "south", "Kyiv oblast" etc — strategic area attacks
        if any(k in v for k in ["ukraine", "south", "east", "north", "west",
                                  "oblast", "region"]):
            return "AREA/STRATEGIC"
        return "OTHER"

    df["target_category"] = df["target"].apply(classify_target)

    # Monthly breakdown
    monthly = df.groupby(["year_month", "target_category"])["launched"].sum().reset_index()
    pivot   = monthly.pivot_table(
        index="year_month", columns="target_category",
        values="launched", aggfunc="sum", fill_value=0
    )
    pivot_share = pivot.div(pivot.sum(axis=1), axis=0) * 100

    print("\nTarget category distribution (all time):")
    total_by_cat = df.groupby("target_category")["launched"].sum().sort_values(ascending=False)
    print(total_by_cat.to_string())

    # ── Detect rotation: rolling 2-month dominant category ──
    pivot_share_ts = pivot_share.copy()
    pivot_share_ts.index = pivot_share_ts.index.to_timestamp()
    dominant = pivot_share_ts.idxmax(axis=1)

    # Count transitions between dominant categories
    transitions = (dominant != dominant.shift()).sum()
    print(f"\n  Dominant category changed {transitions} times across {len(dominant)} months")
    print(f"  (Higher = more systematic rotation)")

    # ── Chart ──
    cat_colors = {
        "ENERGY":          RED,
        "MILITARY":        UKRAINE_BLUE,
        "AREA/STRATEGIC":  ORANGE,
        "URBAN/CIVILIAN":  PURPLE,
        "OTHER":           GREY,
        "UNKNOWN":         "#BDBDBD",
    }

    cats = [c for c in cat_colors if c in pivot_share_ts.columns]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # Panel 1: stacked area share
    ax1.stackplot(
        pivot_share_ts.index,
        [pivot_share_ts[c] for c in cats],
        labels=cats,
        colors=[cat_colors[c] for c in cats],
        alpha=0.85,
    )
    ax1.set_ylabel("Share of Launches by Target Category (%)")
    ax1.set_title("Target Category Mix Over Time")
    ax1.set_ylim(0, 100)
    ax1.yaxis.set_major_formatter(PCT)
    ax1.legend(loc="upper left", fontsize=9, ncol=2)
    ax1.grid(alpha=0.2)

    # Panel 2: dominant category per month as colour band
    cmap = {cat: color for cat, color in cat_colors.items()}
    for i, (month, cat) in enumerate(dominant.items()):
        ax2.axvspan(month, month + pd.DateOffset(months=1),
                    color=cmap.get(cat, GREY), alpha=0.7)

    ax2.set_ylabel("")
    ax2.set_yticks([])
    ax2.set_title("Dominant Target Category per Month\n"
                  "(Colour = which category received most launches that month)")

    patches = [mpatches.Patch(color=cat_colors[c], label=c) for c in cats]
    ax2.legend(handles=patches, fontsize=9, loc="upper left", ncol=3)

    plt.tight_layout()
    path = OUTPUT_DIR / "op5_target_rotation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

    return pivot_share_ts, dominant


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  UKRAINE ATTACK DATA — OPERATIONAL ANALYSES")
    print("═" * 60)

    df = load_data()

    daily_sat, summary_sat     = analysis_saturation(df)
    merged_sal, summary_sal    = analysis_salvo_composition(df)
    pivot_wm,   pivot_wm_share = analysis_weapon_mix(df)
    monthly_geo, region_counts = analysis_geographic_dispersion(df)
    pivot_tgt,   dominant_tgt  = analysis_target_rotation(df)

    print("\n═" * 60)
    print("  ALL ANALYSES COMPLETE")
    print(f"  Charts saved to: {OUTPUT_DIR}")
    print("═" * 60)


if __name__ == "__main__":
    main()
