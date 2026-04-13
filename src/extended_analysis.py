import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import requests
import time
from pathlib import Path
from scipy import stats
from collections import Counter

warnings.filterwarnings("ignore")

DATA_DIR   = Path("/workspaces/ukraine-drone-weather-analysis/data")
TEL_DIR    = DATA_DIR / "telegram"
OUTPUT_DIR = Path("/workspaces/ukraine-drone-weather-analysis/outputs/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BG           = "#FAFAFA"
UKRAINE_BLUE = "#005BBB"
UKRAINE_YELLOW = "#FFD700"
RED          = "#B71C1C"
GREY         = "#616161"
GREEN        = "#2E7D32"
ORANGE       = "#E65100"
PURPLE       = "#6A1B9A"
TEAL         = "#00695C"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "axes.titlesize": 13,
    "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
})
PCT = mticker.FuncFormatter(lambda x, _: f"{x:.0f}%")


# ══════════════════════════════════════════════════════════════════
# ANALYSIS A — AD ATTRITION EFFECT
# ══════════════════════════════════════════════════════════════════

# Confirmed Russian AD system destructions from tracker
# Source: tochnyi.info / ISW / Oryx as documented on unmannedsystemstracker.com
AD_DESTRUCTIONS = [
    {"date": "2025-08-01", "system": "S-350 Vityaz SAM",      "oblast": "crimea",   "notes": "First of 3 S-350 strikes Jul-Feb"},
    {"date": "2025-09-10", "system": "40th C&M Complex HQ",    "oblast": "crimea",   "notes": "Yevpatoriya C2 node destroyed"},
    {"date": "2025-09-01", "system": "Nebo-U radar (x4 hits)", "oblast": "crimea",   "notes": "VHF early warning, repeatedly struck"},
    {"date": "2025-11-01", "system": "S-350 Vityaz x7",        "oblast": "crimea",   "notes": "Peak month — 7 strikes"},
    {"date": "2026-03-09", "system": "5N84A Oborona-14 radar", "oblast": "crimea",   "notes": "Long-range early warning destroyed"},
    {"date": "2026-03-09", "system": "Nebo-U radar x2",        "oblast": "crimea",   "notes": "2 in radomes, Yevpatoriya"},
    {"date": "2026-03-22", "system": "S-400 SAM radar",        "oblast": "moscow",   "notes": "Occupied Donetsk Oblast"},
    {"date": "2025-02-01", "system": "Pantsir stockpile ~50%", "oblast": "unknown",  "notes": "SBU Alpha unit assessment"},
]

def analysis_ad_attrition(drn: pd.DataFrame):
    print("\n══ ANALYSIS A: AD ATTRITION EFFECT ══")

    ad_df = pd.DataFrame(AD_DESTRUCTIONS)
    ad_df["date"] = pd.to_datetime(ad_df["date"])

    # Focus on Crimea — most data
    crimea_strikes = drn[drn["oblasts"].str.contains("crimea", na=False)].copy()
    crimea_strikes["date"] = pd.to_datetime(crimea_strikes["date"])
    crimea_strikes = crimea_strikes.sort_values("date")

    # Crimea AD events
    crimea_ad = ad_df[ad_df["oblast"] == "crimea"].sort_values("date")

    # Monthly hit rate for Crimea
    crimea_strikes["year_month"] = crimea_strikes["date"].dt.to_period("M")
    monthly = crimea_strikes.groupby("year_month").agg(
        total        = ("date",      "count"),
        confirmed    = ("confirmed", "sum"),
        mean_severity= ("damage",    lambda x: x.map({
            "destroyed":3,"fire":2,"explosion":2,
            "confirmed":2,"damaged":1,"unknown":0}).mean())
    ).reset_index()
    monthly["date_dt"]   = monthly["year_month"].dt.to_timestamp()
    monthly["conf_rate"] = monthly["confirmed"] / monthly["total"].clip(1)

    severity = {"destroyed":3,"fire":2,"explosion":2,"confirmed":2,"damaged":1,"unknown":0}
    crimea_strikes["sev"] = crimea_strikes["damage"].map(severity).fillna(0)

    print(f"  Crimea strike events: {len(crimea_strikes)}")
    print(f"  AD destruction events (Crimea): {len(crimea_ad)}")

    # Pre/post analysis around major AD destructions
    # Key event: Sep-Nov 2025 intensive AD destruction campaign
    pre_cutoff  = pd.Timestamp("2025-09-01")
    post_cutoff = pd.Timestamp("2025-09-01")

    pre  = crimea_strikes[crimea_strikes["date"] < pre_cutoff]
    post = crimea_strikes[crimea_strikes["date"] >= post_cutoff]

    pre_sev  = pre["sev"].mean()
    post_sev = post["sev"].mean()
    pre_conf  = pre["confirmed"].mean()
    post_conf = post["confirmed"].mean()

    print(f"\n  Pre Sep-2025 (before intensive AD campaign):")
    print(f"    Mean severity: {pre_sev:.2f}  Confirmation rate: {pre_conf:.1%}  n={len(pre)}")
    print(f"  Post Sep-2025 (after intensive AD campaign):")
    print(f"    Mean severity: {post_sev:.2f}  Confirmation rate: {post_conf:.1%}  n={len(post)}")

    if len(pre) >= 3 and len(post) >= 3:
        t, p = stats.ttest_ind(pre["sev"], post["sev"])
        print(f"  t-test severity pre vs post: t={t:.3f}, p={p:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # Panel 1: monthly severity in Crimea + AD destruction markers
    ax1 = axes[0]
    ax1.bar(monthly["date_dt"], monthly["mean_severity"],
            color=UKRAINE_BLUE, alpha=0.8, width=25, edgecolor="white")

    for _, row in crimea_ad.iterrows():
        ax1.axvline(row["date"], color=RED, linewidth=1.5,
                    linestyle="--", alpha=0.8)
        ax1.text(row["date"], monthly["mean_severity"].max() * 0.95,
                 row["system"][:20], fontsize=7, color=RED,
                 rotation=90, va="top", ha="right")

    ax1.set_ylabel("Mean Damage Severity (Crimea strikes)")
    ax1.set_title("AD Attrition Effect: Crimea Strike Severity vs Russian AD System Destructions\n"
                  "Red dashed lines = confirmed Russian AD system destroyed", pad=10)
    ax1.grid(axis="y", alpha=0.25)
    ax1.axvline(pd.Timestamp("2025-09-01"), color=ORANGE,
                linewidth=2, alpha=0.5, label="Intensive AD campaign begins")
    ax1.legend(fontsize=9)

    # Panel 2: confirmation rate over time
    ax2 = axes[1]
    ax2.plot(monthly["date_dt"], monthly["conf_rate"] * 100,
             color=GREEN, linewidth=2, marker="o", markersize=5)
    ax2.fill_between(monthly["date_dt"], monthly["conf_rate"] * 100,
                     alpha=0.15, color=GREEN)
    ax2.axvline(pd.Timestamp("2025-09-01"), color=ORANGE,
                linewidth=2, alpha=0.5)
    for _, row in crimea_ad.iterrows():
        ax2.axvline(row["date"], color=RED, linewidth=1.5,
                    linestyle="--", alpha=0.6)
    ax2.yaxis.set_major_formatter(PCT)
    ax2.set_ylabel("% Strikes Confirmed (Crimea)")
    ax2.set_title("Confirmation Rate Over Time — Does AD Attrition Improve Strike Verification?")
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    path = OUTPUT_DIR / "ext_a_ad_attrition.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════
# ANALYSIS B — OIL REFINERY CAMPAIGN DEEP DIVE
# ══════════════════════════════════════════════════════════════════

# Known major Russian oil refineries with approximate capacity (kt/year)
REFINERIES = {
    "saratov":    {"name": "Saratov refinery",         "capacity_mt": 7.0},
    "samara":     {"name": "Kuibyshev/Samara complex",  "capacity_mt": 7.5},
    "engels":     {"name": "Engels fuel depot",         "capacity_mt": 2.0},
    "krasnodar":  {"name": "Krasnodar/Afipsky",        "capacity_mt": 6.0},
    "rostov":     {"name": "Novoshakhtinsk",            "capacity_mt": 3.5},
    "bryansk":    {"name": "Bryansk facilities",        "capacity_mt": 2.0},
    "tver":       {"name": "Tver/Kimry",                "capacity_mt": 1.5},
    "moscow":     {"name": "Moscow-area refineries",    "capacity_mt": 12.0},
    "ryazan":     {"name": "Ryazan oil refinery",       "capacity_mt": 7.0},
    "tatarstan":  {"name": "TANECO/Nizhnekamsk",        "capacity_mt": 8.0},
}

def analysis_refinery_campaign(drn: pd.DataFrame):
    print("\n══ ANALYSIS B: OIL REFINERY CAMPAIGN ══")

    ref = drn[drn["target_cat"] == "oil_refinery"].copy()
    ref["date"] = pd.to_datetime(ref["date"])
    ref["year_month"] = ref["date"].dt.to_period("M")

    severity = {"destroyed":3,"fire":2,"explosion":2,"confirmed":2,"damaged":1,"unknown":0}
    ref["sev"] = ref["damage"].map(severity).fillna(0)

    print(f"  Total refinery strike events: {len(ref)}")
    print(f"  Confirmed: {ref['confirmed'].sum()}")
    print(f"  Date range: {ref['date'].min().date()} → {ref['date'].max().date()}")

    # By oblast
    ref_oblasts = Counter()
    for o in ref["oblasts"].dropna():
        for item in o.split("|"):
            if item and item != "unknown":
                ref_oblasts[item] += 1

    print(f"\n  Top targeted oblasts (refineries):")
    for o, n in ref_oblasts.most_common(8):
        cap = REFINERIES.get(o, {}).get("capacity_mt", "?")
        print(f"    {o}: {n} strikes | capacity: {cap} Mt/year")

    # Monthly tempo
    monthly = ref.groupby("year_month").agg(
        strikes      = ("date",      "count"),
        confirmed    = ("confirmed", "sum"),
        mean_sev     = ("sev",       "mean"),
    ).reset_index()
    monthly["date_dt"] = monthly["year_month"].dt.to_timestamp()

    # Damage distribution
    damage_counts = ref["damage"].value_counts()

    # Re-strike analysis — same oblast hit multiple times in same month?
    ref["month_oblast"] = ref["year_month"].astype(str) + "_" + ref["oblasts"].str.split("|").str[0]
    restrike = ref["month_oblast"].value_counts()
    multi_hit = restrike[restrike > 1]
    print(f"\n  Multi-strike events (same oblast, same month): {len(multi_hit)}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: monthly tempo
    axes[0,0].bar(monthly["date_dt"], monthly["strikes"],
                  color=ORANGE, alpha=0.85, width=25, edgecolor="white")
    axes[0,0].bar(monthly["date_dt"], monthly["confirmed"],
                  color=GREEN, alpha=0.9, width=25, edgecolor="white")
    axes[0,0].set_title("Oil Refinery Strike Tempo\nOrange=reported, Green=confirmed")
    axes[0,0].set_ylabel("Strike Events")
    axes[0,0].grid(axis="y", alpha=0.25)
    axes[0,0].tick_params(axis='x', rotation=45)

    # Panel 2: by oblast
    top_oblasts = dict(ref_oblasts.most_common(10))
    sev_by_oblast = {}
    for o in top_oblasts:
        mask = ref["oblasts"].str.contains(o, na=False)
        sev_by_oblast[o] = ref[mask]["sev"].mean()

    ob_df = pd.DataFrame({
        "oblast":   list(top_oblasts.keys()),
        "strikes":  list(top_oblasts.values()),
        "severity": [sev_by_oblast.get(o, 0) for o in top_oblasts]
    }).sort_values("strikes")

    colors_ob = [ORANGE if sev >= 1.5 else UKRAINE_BLUE
                 for sev in ob_df["severity"]]
    axes[0,1].barh(ob_df["oblast"], ob_df["strikes"],
                   color=colors_ob, edgecolor="white", linewidth=0.8)
    for i, (_, row) in enumerate(ob_df.iterrows()):
        axes[0,1].text(row["strikes"] + 0.2, i,
                       f"{row['severity']:.1f}", va="center", fontsize=9)
    axes[0,1].set_xlabel("Strike Events (severity score →)")
    axes[0,1].set_title("Strikes by Oblast\nColour = severity (orange≥1.5, blue<1.5)")
    axes[0,1].grid(axis="x", alpha=0.25)

    # Panel 3: damage distribution
    dmg_labels = ["Unknown","Damaged","Fire/Explosion","Confirmed","Destroyed"]
    dmg_vals = [
        (ref["damage"] == "unknown").sum(),
        (ref["damage"] == "damaged").sum(),
        (ref["damage"].isin(["fire","explosion"])).sum(),
        (ref["damage"] == "confirmed").sum(),
        (ref["damage"] == "destroyed").sum(),
    ]
    dmg_colors = [GREY, TEAL, ORANGE, GREEN, RED]
    bars = axes[1,0].bar(dmg_labels, dmg_vals,
                          color=dmg_colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, dmg_vals):
        axes[1,0].text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.3,
                       str(val), ha="center", fontsize=10, fontweight="bold")
    axes[1,0].set_ylabel("Number of Strike Events")
    axes[1,0].set_title("Damage Outcome Distribution\n(Oil refinery strikes only)")
    axes[1,0].grid(axis="y", alpha=0.25)
    axes[1,0].tick_params(axis='x', rotation=20)

    # Panel 4: monthly severity trend
    monthly["rolling_sev"] = monthly["mean_sev"].rolling(3, center=True).mean()
    axes[1,1].plot(monthly["date_dt"], monthly["mean_sev"],
                   color=ORANGE, linewidth=1.5, alpha=0.6, marker="o", markersize=4)
    axes[1,1].plot(monthly["date_dt"], monthly["rolling_sev"],
                   color=RED, linewidth=2.5, linestyle="--", label="3-month rolling mean")
    axes[1,1].set_ylabel("Mean Damage Severity")
    axes[1,1].set_title("Refinery Strike Severity Over Time\nIs Ukraine getting better at damaging refineries?")
    axes[1,1].legend(fontsize=9)
    axes[1,1].grid(alpha=0.2)
    axes[1,1].tick_params(axis='x', rotation=45)

    plt.suptitle("Ukrainian Oil Refinery Campaign: Deep Dive Analysis",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / "ext_b_refinery_campaign.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════
# ANALYSIS C — WEATHER EFFECT ON UKRAINIAN DEEP STRIKES
# ══════════════════════════════════════════════════════════════════

# Target cities for weather — key confirmed hit locations
WEATHER_CITIES = {
    "moscow":    {"lat": 55.75, "lon": 37.62},
    "tver":      {"lat": 56.86, "lon": 35.90},
    "saratov":   {"lat": 51.53, "lon": 46.03},
    "krasnodar": {"lat": 45.04, "lon": 38.98},
    "crimea":    {"lat": 44.95, "lon": 34.10},
}

def fetch_weather_russia(start_date: str, end_date: str) -> pd.DataFrame:
    cache = DATA_DIR / "russia_weather.csv"
    if cache.exists():
        print("  Loading cached Russia weather data...")
        return pd.read_csv(cache, parse_dates=["date"])

    print(f"  Fetching weather for Russian target cities: {start_date} → {end_date}")
    dfs = []
    for city, coords in WEATHER_CITIES.items():
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude":   coords["lat"],
            "longitude":  coords["lon"],
            "daily":      ["cloudcover_mean","windspeed_10m_max",
                           "precipitation_sum","weathercode"],
            "start_date": start_date,
            "end_date":   end_date,
            "timezone":   "Europe/Moscow",
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            d = r.json()
            df = pd.DataFrame(d["daily"])
            df["date"] = pd.to_datetime(df["time"]).dt.date
            df = df.drop(columns=["time"])
            df.columns = [f"{city}_{c}" if c != "date" else "date"
                          for c in df.columns]
            dfs.append(df)
            print(f"    {city}... done")
            time.sleep(0.4)
        except Exception as e:
            print(f"    {city} FAILED: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.merge(df, on="date", how="outer")
    combined["date"] = pd.to_datetime(combined["date"])

    # National averages
    cloud_cols = [c for c in combined.columns if "cloudcover_mean" in c]
    wind_cols  = [c for c in combined.columns if "windspeed_10m_max" in c]
    precip_cols= [c for c in combined.columns if "precipitation_sum" in c]

    if cloud_cols:  combined["avg_cloud"]  = combined[cloud_cols].mean(axis=1)
    if wind_cols:   combined["avg_wind"]   = combined[wind_cols].mean(axis=1)
    if precip_cols: combined["avg_precip"] = combined[precip_cols].mean(axis=1)

    combined.to_csv(cache, index=False)
    print(f"  Saved → {cache}")
    return combined


def analysis_weather_deepstrike(drn: pd.DataFrame):
    print("\n══ ANALYSIS C: WEATHER EFFECT ON DEEP STRIKES ══")

    drn = drn.copy()
    drn["date"] = pd.to_datetime(drn["date"])

    start = drn["date"].min().strftime("%Y-%m-%d")
    end   = drn["date"].max().strftime("%Y-%m-%d")

    weather = fetch_weather_russia(start, end)
    if weather.empty:
        print("  Weather fetch failed — skipping")
        return

    # Merge
    merged = drn.merge(weather, on="date", how="left")

    severity = {"destroyed":3,"fire":2,"explosion":2,"confirmed":2,"damaged":1,"unknown":0}
    merged["sev"] = merged["damage"].map(severity).fillna(0)

    # Daily aggregation
    daily_strikes = merged.groupby("date").agg(
        n_strikes    = ("date",      "count"),
        mean_sev     = ("sev",       "mean"),
        n_confirmed  = ("confirmed", "sum"),
        avg_cloud    = ("avg_cloud", "first"),
        avg_wind     = ("avg_wind",  "first"),
        avg_precip   = ("avg_precip","first"),
    ).reset_index()

    # All days baseline
    all_days = weather[["date","avg_cloud","avg_wind","avg_precip"]].copy()

    # Strike days vs non-strike days
    strike_dates    = set(daily_strikes["date"].dt.date)
    all_days["is_strike"] = all_days["date"].dt.date.isin(strike_dates)

    strike_days    = all_days[all_days["is_strike"]]
    nonstrike_days = all_days[~all_days["is_strike"]]

    print(f"\n  Strike days: {len(strike_days)} | Non-strike days: {len(nonstrike_days)}")

    for metric, label in [("avg_cloud","Cloud cover (%)"),
                           ("avg_wind", "Wind speed (km/h)"),
                           ("avg_precip","Precipitation (mm)")]:
        s = strike_days[metric].dropna()
        ns = nonstrike_days[metric].dropna()
        if len(s) > 5 and len(ns) > 5:
            t, p = stats.ttest_ind(s, ns)
            print(f"  {label}: strike={s.mean():.1f} vs non-strike={ns.mean():.1f} | p={p:.4f}")

    # Weather bucket
    def bucket(row):
        c = row.get("avg_cloud", np.nan)
        p = row.get("avg_precip", np.nan)
        if pd.isna(c): return "UNKNOWN"
        if c < 40 and (pd.isna(p) or p < 1): return "CLEAR"
        if c >= 70 or (not pd.isna(p) and p >= 5): return "ADVERSE"
        return "OVERCAST"

    daily_strikes["wx_bucket"] = daily_strikes.apply(bucket, axis=1)
    all_days["wx_bucket"]      = all_days.apply(bucket, axis=1)

    # Severity by weather bucket
    bucket_sev = daily_strikes.groupby("wx_bucket").agg(
        mean_sev  = ("mean_sev",  "mean"),
        n_days    = ("mean_sev",  "count"),
        mean_strikes = ("n_strikes","mean"),
    ).reset_index()
    print(f"\n  Strike severity by weather bucket:")
    print(bucket_sev.to_string(index=False))

    # Chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    BKT_COLORS = {"CLEAR": GREEN, "OVERCAST": UKRAINE_BLUE, "ADVERSE": RED, "UNKNOWN": GREY}
    bucket_order = ["CLEAR","OVERCAST","ADVERSE"]

    # Panel 1: cloud cover distribution — strike vs non-strike
    ax = axes[0,0]
    s_cloud = strike_days["avg_cloud"].dropna()
    ns_cloud = nonstrike_days["avg_cloud"].dropna()
    bins = np.linspace(0, 100, 20)
    ax.hist(ns_cloud, bins=bins, alpha=0.5, color=GREY,
            density=True, label=f"Non-strike days (n={len(ns_cloud)})")
    ax.hist(s_cloud, bins=bins, alpha=0.7, color=UKRAINE_BLUE,
            density=True, label=f"Strike days (n={len(s_cloud)})")
    ax.axvline(ns_cloud.mean(), color=GREY, linestyle="--", linewidth=2,
               label=f"Non-strike mean: {ns_cloud.mean():.0f}%")
    ax.axvline(s_cloud.mean(), color=UKRAINE_BLUE, linestyle="--", linewidth=2,
               label=f"Strike mean: {s_cloud.mean():.0f}%")
    ax.set_xlabel("Mean Cloud Cover over Russian targets (%)")
    ax.set_ylabel("Density")
    ax.set_title("Cloud Cover: Strike Days vs All Days\n(over Russian target cities)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Panel 2: wind distribution
    ax2 = axes[0,1]
    s_wind  = strike_days["avg_wind"].dropna()
    ns_wind = nonstrike_days["avg_wind"].dropna()
    bins_w  = np.linspace(0, 60, 20)
    ax2.hist(ns_wind, bins=bins_w, alpha=0.5, color=GREY,
             density=True, label=f"Non-strike days")
    ax2.hist(s_wind, bins=bins_w, alpha=0.7, color=UKRAINE_BLUE,
             density=True, label=f"Strike days")
    ax2.axvline(ns_wind.mean(), color=GREY, linestyle="--", linewidth=2,
                label=f"Non-strike mean: {ns_wind.mean():.1f} km/h")
    ax2.axvline(s_wind.mean(), color=UKRAINE_BLUE, linestyle="--", linewidth=2,
                label=f"Strike mean: {s_wind.mean():.1f} km/h")
    t_w, p_w = stats.ttest_ind(s_wind, ns_wind)
    ax2.set_xlabel("Max Wind Speed over Russian targets (km/h)")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Wind Speed: Strike Days vs All Days\n(t-test p={p_w:.4f})")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    # Panel 3: severity by weather bucket
    ax3 = axes[1,0]
    bkt_plot = bucket_sev[bucket_sev["wx_bucket"].isin(bucket_order)].set_index("wx_bucket").reindex(bucket_order).reset_index().dropna()
    bars = ax3.bar(bkt_plot["wx_bucket"], bkt_plot["mean_sev"],
                   color=[BKT_COLORS[b] for b in bkt_plot["wx_bucket"]],
                   edgecolor="white", linewidth=1.2, zorder=3)
    for bar, (_, row) in zip(bars, bkt_plot.iterrows()):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f"{row['mean_sev']:.2f}\n(n={int(row['n_days'])})",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.set_ylabel("Mean Damage Severity Score")
    ax3.set_title("Deep Strike Damage Severity by Weather Conditions\n(over Russian target cities)")
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_ylim(0, bkt_plot["mean_sev"].max() * 1.3)

    # Panel 4: strike volume by weather bucket
    ax4 = axes[1,1]
    bkt_vol = bucket_sev[bucket_sev["wx_bucket"].isin(bucket_order)].set_index("wx_bucket").reindex(bucket_order).reset_index().dropna()
    bars2 = ax4.bar(bkt_vol["wx_bucket"], bkt_vol["mean_strikes"],
                    color=[BKT_COLORS[b] for b in bkt_vol["wx_bucket"]],
                    edgecolor="white", linewidth=1.2, zorder=3)
    for bar, (_, row) in zip(bars2, bkt_vol.iterrows()):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f"{row['mean_strikes']:.1f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax4.set_ylabel("Mean Strikes per Attack Day")
    ax4.set_title("Launch Volume by Weather Conditions\nDoes Ukraine select specific weather windows?")
    ax4.grid(axis="y", alpha=0.3)

    plt.suptitle("Weather Effect on Ukrainian Deep Strikes into Russia\n"
                 "Weather measured over Russian target cities (Moscow, Tver, Saratov, Krasnodar, Crimea)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / "ext_c_weather_deepstrike.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  EXTENDED DEEP STRIKE ANALYSES")
    print("=" * 60)

    drn = pd.read_csv(TEL_DIR / "dronbomber_v2.csv", parse_dates=["date"])

    analysis_ad_attrition(drn)
    analysis_refinery_campaign(drn)
    analysis_weather_deepstrike(drn)

    print("\n" + "=" * 60)
    print("  ALL DONE — charts saved to outputs/charts/")
    print("=" * 60)


if __name__ == "__main__":
    main()
