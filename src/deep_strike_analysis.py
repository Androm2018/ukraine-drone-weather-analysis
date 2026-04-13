import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats
from collections import Counter

warnings.filterwarnings("ignore")

DATA_DIR   = Path("/workspaces/ukraine-drone-weather-analysis/data/telegram")
OUTPUT_DIR = Path("/workspaces/ukraine-drone-weather-analysis/outputs/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BG             = "#FAFAFA"
UKRAINE_BLUE   = "#005BBB"
UKRAINE_YELLOW = "#FFD700"
RED            = "#B71C1C"
GREY           = "#616161"
GREEN          = "#2E7D32"
ORANGE         = "#E65100"
PURPLE         = "#6A1B9A"
TEAL           = "#00695C"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "axes.titlesize": 13,
    "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
})
PCT = mticker.FuncFormatter(lambda x, _: f"{x:.0f}%")


def load_data():
    mod = pd.read_csv(DATA_DIR / "mod_russia_clean.csv", parse_dates=["date"])
    drn = pd.read_csv(DATA_DIR / "dronbomber_v2.csv",    parse_dates=["date"])
    mod["year_month"] = mod["date"].dt.to_period("M")
    drn["year_month"] = drn["date"].dt.to_period("M")
    print(f"mod_russia: {len(mod)} rows | {mod['date'].min().date()} → {mod['date'].max().date()}")
    print(f"dronbomber: {len(drn)} rows | {drn['date'].min().date()} → {drn['date'].max().date()}")
    return mod, drn


# ══════════════════════════════════════════════════════════════════
# DS1 — Russian intercept claim trends
# ══════════════════════════════════════════════════════════════════

def ds1_intercept_trends(mod):
    print("\n── DS1: Russian intercept claim trends ──")

    monthly = mod.groupby("year_month").agg(
        total_claimed   = ("drone_count", "sum"),
        n_reports       = ("drone_count", "count"),
        reports_with_count = ("drone_count", lambda x: (x > 0).sum()),
    ).reset_index()
    monthly["date_dt"] = monthly["year_month"].dt.to_timestamp()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Panel 1: total claimed intercepts per month
    ax1.bar(monthly["date_dt"], monthly["total_claimed"],
            color=RED, alpha=0.8, width=25, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Total Drones Claimed Intercepted")
    ax1.set_title("Russian MoD: Claimed Ukrainian Drone Intercepts Over Russian Territory\n"
                  "⚠ Source: @mod_russia Telegram — treat as propaganda claims, not verified data",
                  pad=10)
    ax1.grid(axis="y", alpha=0.25)

    # Rolling 3-month
    monthly["rolling"] = monthly["total_claimed"].rolling(3, center=True).mean()
    ax1.plot(monthly["date_dt"], monthly["rolling"],
             color=UKRAINE_YELLOW, linewidth=2.5, linestyle="--", label="3-month rolling mean")
    ax1.legend(fontsize=9)

    # Panel 2: number of intercept reports per month
    ax2.bar(monthly["date_dt"], monthly["reports_with_count"],
            color=UKRAINE_BLUE, alpha=0.8, width=25, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Reports with Drone Count")
    ax2.set_title("Number of Daily Intercept Reports per Month")
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = OUTPUT_DIR / "ds1_intercept_trends.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")
    return monthly


# ══════════════════════════════════════════════════════════════════
# DS2 — Confirmed strike tempo (dronbomber)
# ══════════════════════════════════════════════════════════════════

def ds2_confirmed_strikes(drn):
    print("\n── DS2: Confirmed strike tempo ──")

    monthly = drn.groupby("year_month").agg(
        total_strikes   = ("date",       "count"),
        confirmed       = ("confirmed",  "sum"),
        oil_refinery    = ("target_cat", lambda x: (x == "oil_refinery").sum()),
        airbase         = ("target_cat", lambda x: (x == "airbase").sum()),
        ammunition      = ("target_cat", lambda x: (x == "ammunition_depot").sum()),
        radar_ad        = ("target_cat", lambda x: (x == "radar_ad").sum()),
        naval           = ("target_cat", lambda x: (x == "naval_port").sum()),
        command         = ("target_cat", lambda x: (x == "command_control").sum()),
    ).reset_index()
    monthly["date_dt"] = monthly["year_month"].dt.to_timestamp()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Panel 1: total reported strikes
    ax1.bar(monthly["date_dt"], monthly["total_strikes"],
            color=UKRAINE_BLUE, alpha=0.8, width=25, edgecolor="white", linewidth=0.5,
            label="All reported")
    ax1.bar(monthly["date_dt"], monthly["confirmed"],
            color=GREEN, alpha=0.9, width=25, edgecolor="white", linewidth=0.5,
            label="Confirmed")
    ax1.set_ylabel("Strike Events Reported")
    ax1.set_title("Ukrainian Deep Strikes on Russia: Reported Events by Month\n"
                  "Source: @dronbomber OSINT channel (Jun 2024–present)", pad=10)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.25)

    # Panel 2: stacked by target category
    cats = ["oil_refinery","airbase","ammunition","radar_ad","naval","command"]
    colors = [ORANGE, UKRAINE_BLUE, RED, PURPLE, TEAL, GREEN]
    labels = ["Oil Refinery","Airbase","Ammo Depot","Radar/AD","Naval","Command"]

    bottom = np.zeros(len(monthly))
    for cat, color, label in zip(cats, colors, labels):
        vals = monthly[cat].fillna(0).values
        ax2.bar(monthly["date_dt"], vals, bottom=bottom,
                color=color, alpha=0.85, width=25,
                edgecolor="white", linewidth=0.3, label=label)
        bottom += vals

    ax2.set_ylabel("Strike Events by Target Category")
    ax2.set_title("Target Category Breakdown")
    ax2.legend(fontsize=8, ncol=3, loc="upper left")
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = OUTPUT_DIR / "ds2_confirmed_strikes.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")
    return monthly


# ══════════════════════════════════════════════════════════════════
# DS3 — Gap analysis: Russian claims vs confirmed hits
# ══════════════════════════════════════════════════════════════════

def ds3_gap_analysis(mod_monthly, drn_monthly):
    print("\n── DS3: Gap analysis ──")

    # Merge on year_month — overlap period only
    merged = mod_monthly.merge(
        drn_monthly[["year_month","total_strikes","confirmed"]],
        on="year_month", how="inner"
    )
    merged["date_dt"] = merged["year_month"].dt.to_timestamp()

    # Normalise to per-month rates
    # Gap = Russian claims vs confirmed hits
    # Higher gap = larger discrepancy between Russian narrative and OSINT

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    # Panel 1: claimed intercepts vs confirmed strikes
    ax = axes[0]
    ax2_twin = ax.twinx()
    ax.bar(merged["date_dt"], merged["total_claimed"],
           color=RED, alpha=0.7, width=25, label="Russian claimed intercepts")
    ax2_twin.plot(merged["date_dt"], merged["total_strikes"],
                  color=UKRAINE_BLUE, linewidth=2.5, marker="o", markersize=5,
                  label="OSINT confirmed strikes")
    ax.set_ylabel("Claimed Intercepts (bars)", color=RED)
    ax2_twin.set_ylabel("Confirmed Strikes (line)", color=UKRAINE_BLUE)
    ax.set_title("Russian Intercept Claims vs OSINT-Confirmed Strikes\n"
                 "⚠ Scales differ — claims are inflated propaganda figures", pad=10)
    lines1 = [mpatches.Patch(color=RED, label="Russian claimed intercepts"),
              plt.Line2D([0],[0], color=UKRAINE_BLUE, linewidth=2, label="OSINT confirmed strikes")]
    ax.legend(handles=lines1, fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    # Panel 2: ratio — claims per confirmed strike
    axes[1].bar(merged["date_dt"],
                merged["total_claimed"] / merged["total_strikes"].clip(1),
                color=PURPLE, alpha=0.8, width=25, edgecolor="white", linewidth=0.5)
    axes[1].axhline(merged["total_claimed"].sum() / merged["total_strikes"].sum(),
                    color=GREY, linestyle="--", linewidth=1.5,
                    label=f"Overall mean: {merged['total_claimed'].sum()/merged['total_strikes'].sum():.0f}x")
    axes[1].set_ylabel("Claimed : Confirmed Ratio")
    axes[1].set_title("Russian Inflation Factor: Claimed Intercepts per Confirmed Strike\n"
                      "Higher = larger gap between Russian narrative and verified OSINT")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)

    # Panel 3: confirmed rate (confirmed / total reported)
    axes[2].bar(merged["date_dt"],
                (merged["confirmed"] / merged["total_strikes"].clip(1)) * 100,
                color=GREEN, alpha=0.8, width=25, edgecolor="white", linewidth=0.5)
    axes[2].yaxis.set_major_formatter(PCT)
    axes[2].set_ylabel("% of Reports Confirmed")
    axes[2].set_title("OSINT Confirmation Rate: What % of Reported Strikes Were Independently Confirmed?")
    axes[2].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = OUTPUT_DIR / "ds3_gap_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

    # Print headline numbers
    overall_ratio = merged["total_claimed"].sum() / merged["total_strikes"].sum()
    print(f"\n  Overall claimed:confirmed ratio: {overall_ratio:.0f}:1")
    print(f"  Total claimed intercepts: {merged['total_claimed'].sum():,}")
    print(f"  Total confirmed strikes:  {merged['total_strikes'].sum():,}")
    print(f"  Mean confirmation rate:   {(merged['confirmed']/merged['total_strikes'].clip(1)).mean():.1%}")


# ══════════════════════════════════════════════════════════════════
# DS4 — Target category effectiveness
# ══════════════════════════════════════════════════════════════════

def ds4_target_effectiveness(drn):
    print("\n── DS4: Target category effectiveness ──")

    # Score damage
    severity = {"destroyed": 3, "fire": 2, "explosion": 2,
                "damaged": 1, "confirmed": 2, "unknown": 0}
    drn = drn.copy()
    drn["severity"] = drn["damage"].map(severity).fillna(0)

    by_cat = drn.groupby("target_cat").agg(
        n_strikes      = ("date",      "count"),
        confirmed      = ("confirmed", "sum"),
        mean_severity  = ("severity",  "mean"),
        pct_destroyed  = ("damage",    lambda x: (x == "destroyed").mean() * 100),
        pct_fire       = ("damage",    lambda x: (x == "fire").mean() * 100),
    ).reset_index()
    by_cat = by_cat[by_cat["n_strikes"] >= 5].sort_values("mean_severity", ascending=True)

    print(by_cat[["target_cat","n_strikes","confirmed","mean_severity","pct_destroyed"]].to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = [UKRAINE_BLUE] * len(by_cat)

    # Panel 1: mean severity by target
    bars = ax1.barh(by_cat["target_cat"], by_cat["mean_severity"],
                    color=colors, edgecolor="white", linewidth=1, zorder=3)
    for bar, (_, row) in zip(bars, by_cat.iterrows()):
        ax1.text(bar.get_width() + 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f"{row['mean_severity']:.2f} (n={int(row['n_strikes'])})",
                 va="center", fontsize=9)
    ax1.set_xlabel("Mean Damage Severity Score (0=unknown, 1=damaged, 2=fire/explosion, 3=destroyed)")
    ax1.set_title("Strike Effectiveness by Target Category\n(Higher = more severe confirmed damage)")
    ax1.grid(axis="x", alpha=0.3)
    ax1.set_xlim(0, by_cat["mean_severity"].max() * 1.5)

    # Panel 2: confirmed rate by target
    by_cat2 = by_cat.sort_values("confirmed")
    conf_rate = (by_cat2["confirmed"] / by_cat2["n_strikes"] * 100)
    bars2 = ax2.barh(by_cat2["target_cat"], conf_rate,
                     color=GREEN, edgecolor="white", linewidth=1, zorder=3)
    for bar, (_, row) in zip(bars2, by_cat2.iterrows()):
        rate = row["confirmed"] / row["n_strikes"] * 100
        ax2.text(bar.get_width() + 0.5,
                 bar.get_y() + bar.get_height() / 2,
                 f"{rate:.0f}%", va="center", fontsize=9)
    ax2.set_xlabel("% of Strikes Independently Confirmed")
    ax2.set_title("Confirmation Rate by Target Category\n(Higher = better OSINT verification)")
    ax2.grid(axis="x", alpha=0.3)
    ax2.xaxis.set_major_formatter(PCT)

    plt.tight_layout()
    path = OUTPUT_DIR / "ds4_target_effectiveness.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════
# DS5 — Geographic distribution of strikes
# ══════════════════════════════════════════════════════════════════

def ds5_geographic_distribution(mod, drn):
    print("\n── DS5: Geographic distribution ──")

    # mod_russia — which oblasts does Russia report intercepts over?
    mod_oblasts = Counter()
    for o in mod["oblasts"].dropna():
        for item in o.split("|"):
            if item and item != "unknown":
                mod_oblasts[item] += 1

    # dronbomber — which oblasts do confirmed strikes hit?
    drn_oblasts = Counter()
    for o in drn["oblasts"].dropna():
        for item in o.split("|"):
            if item and item != "unknown":
                drn_oblasts[item] += 1

    # Top 15 for each
    mod_top = pd.DataFrame(mod_oblasts.most_common(15),
                           columns=["oblast","mod_count"])
    drn_top = pd.DataFrame(drn_oblasts.most_common(15),
                           columns=["oblast","drn_count"])

    # Merge for comparison
    combined = mod_top.merge(drn_top, on="oblast", how="outer").fillna(0)
    combined = combined.sort_values("mod_count", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Panel 1: mod_russia oblast frequency
    ax1.barh(combined["oblast"], combined["mod_count"],
             color=RED, alpha=0.8, edgecolor="white", linewidth=0.8)
    ax1.set_xlabel("Russian MoD Intercept Reports Mentioning Oblast")
    ax1.set_title("Russian MoD: Most Mentioned Oblasts\n(proxy for Ukrainian drone corridors)")
    ax1.grid(axis="x", alpha=0.3)

    # Panel 2: dronbomber confirmed strikes by oblast
    drn_sorted = drn_top.sort_values("drn_count", ascending=True)
    ax2.barh(drn_sorted["oblast"], drn_sorted["drn_count"],
             color=UKRAINE_BLUE, alpha=0.8, edgecolor="white", linewidth=0.8)
    ax2.set_xlabel("Confirmed/Reported Strike Events")
    ax2.set_title("OSINT Confirmed Strikes by Oblast\n(dronbomber, Jun 2024–present)")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "ds5_geographic_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

    print(f"\n  Top mod_russia oblasts: {list(mod_oblasts.most_common(5))}")
    print(f"  Top dronbomber oblasts: {list(drn_oblasts.most_common(5))}")


# ══════════════════════════════════════════════════════════════════
# DS6 — Temporal patterns
# ══════════════════════════════════════════════════════════════════

def ds6_temporal_patterns(mod, drn):
    print("\n── DS6: Temporal patterns ──")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: mod_russia — intercepts by month of year (seasonal)
    mod["month"] = mod["date"].dt.month
    seasonal_mod = mod[mod["drone_count"] > 0].groupby("month")["drone_count"].sum()
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    winter = [10,11,12,1,2,3]
    colors_m = [RED if m in winter else GREY for m in range(1,13)]
    axes[0,0].bar(range(1,13), seasonal_mod.reindex(range(1,13), fill_value=0),
                  color=colors_m, edgecolor="white", linewidth=0.8)
    axes[0,0].set_xticks(range(1,13))
    axes[0,0].set_xticklabels(months, fontsize=8)
    axes[0,0].set_title("Russian Claimed Intercepts by Month\n(Red = winter)")
    axes[0,0].set_ylabel("Total Claimed Intercepts")
    axes[0,0].grid(axis="y", alpha=0.3)

    # Panel 2: dronbomber — strikes by month of year
    drn["month"] = drn["date"].dt.month
    seasonal_drn = drn.groupby("month").size()
    axes[0,1].bar(range(1,13), seasonal_drn.reindex(range(1,13), fill_value=0),
                  color=[UKRAINE_BLUE if m not in winter else ORANGE for m in range(1,13)],
                  edgecolor="white", linewidth=0.8)
    axes[0,1].set_xticks(range(1,13))
    axes[0,1].set_xticklabels(months, fontsize=8)
    axes[0,1].set_title("Confirmed Strikes on Russia by Month\n(Orange = winter)")
    axes[0,1].set_ylabel("Strike Events")
    axes[0,1].grid(axis="y", alpha=0.3)

    # Panel 3: day of week patterns (mod_russia)
    mod["dow"] = mod["date"].dt.dayofweek
    dow_mod = mod[mod["drone_count"] > 0].groupby("dow")["drone_count"].sum()
    dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    axes[1,0].bar(range(7), dow_mod.reindex(range(7), fill_value=0),
                  color=RED, alpha=0.8, edgecolor="white", linewidth=0.8)
    axes[1,0].set_xticks(range(7))
    axes[1,0].set_xticklabels(dow_labels)
    axes[1,0].set_title("Russian Claimed Intercepts by Day of Week")
    axes[1,0].set_ylabel("Total Claimed Intercepts")
    axes[1,0].grid(axis="y", alpha=0.3)

    # Panel 4: day of week (dronbomber)
    drn["dow"] = drn["date"].dt.dayofweek
    dow_drn = drn.groupby("dow").size()
    axes[1,1].bar(range(7), dow_drn.reindex(range(7), fill_value=0),
                  color=UKRAINE_BLUE, alpha=0.8, edgecolor="white", linewidth=0.8)
    axes[1,1].set_xticks(range(7))
    axes[1,1].set_xticklabels(dow_labels)
    axes[1,1].set_title("Confirmed Strikes on Russia by Day of Week")
    axes[1,1].set_ylabel("Strike Events")
    axes[1,1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "ds6_temporal_patterns.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  UKRAINE DEEP STRIKE ANALYSIS")
    print("=" * 60)

    mod, drn = load_data()

    mod_monthly = ds1_intercept_trends(mod)
    drn_monthly = ds2_confirmed_strikes(drn)
    ds3_gap_analysis(mod_monthly, drn_monthly)
    ds4_target_effectiveness(drn)
    ds5_geographic_distribution(mod, drn)
    ds6_temporal_patterns(mod, drn)

    print("\n" + "=" * 60)
    print("  ALL ANALYSES COMPLETE")
    print(f"  Charts saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
