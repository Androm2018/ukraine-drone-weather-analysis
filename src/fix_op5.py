import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("outputs/charts")

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
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
})

OBLAST_CATEGORIES = {
    "donetsk":         "FRONTLINE",
    "zaporizhzhia":    "FRONTLINE",
    "zaporizhia":      "FRONTLINE",
    "kherson":         "FRONTLINE",
    "kharkiv":         "FRONTLINE",
    "sumy":            "FRONTLINE",
    "luhansk":         "FRONTLINE",
    "lugansk":         "FRONTLINE",
    "mykolaiv":        "FRONTLINE",
    "nikolaev":        "FRONTLINE",
    "kyiv":            "CAPITAL",
    "kiev":            "CAPITAL",
    "odesa":           "SOUTHERN",
    "odessa":          "SOUTHERN",
    "dnipropetrovsk":  "CENTRAL",
    "dnipro":          "CENTRAL",
    "poltava":         "CENTRAL",
    "kirovohrad":      "CENTRAL",
    "kirovograd":      "CENTRAL",
    "kropyvnytskyi":   "CENTRAL",
    "cherkasy":        "CENTRAL",
    "vinnytsia":       "CENTRAL",
    "vinnitsa":        "CENTRAL",
    "lviv":            "DEEP REAR",
    "ivano-frankivsk": "DEEP REAR",
    "ternopil":        "DEEP REAR",
    "khmelnytskyi":    "DEEP REAR",
    "khmelnitsky":     "DEEP REAR",
    "rivne":           "DEEP REAR",
    "volyn":           "DEEP REAR",
    "zakarpattia":     "DEEP REAR",
    "chernivtsi":      "DEEP REAR",
    "chernihiv":       "NORTHERN",
    "zhytomyr":        "NORTHERN",
}

def classify_target_oblast(target_val):
    if pd.isna(target_val):
        return "UNKNOWN"
    t = str(target_val).lower()
    for oblast, category in OBLAST_CATEGORIES.items():
        if oblast in t:
            return category
    if "south" in t:   return "SOUTHERN"
    if "east"  in t:   return "FRONTLINE"
    if "north" in t:   return "NORTHERN"
    if "west"  in t:   return "DEEP REAR"
    if "ukraine" in t: return "AREA-WIDE"
    return "OTHER"

# Load
df = pd.read_csv(DATA_DIR / "missile_attacks_daily.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df["date"] = pd.to_datetime(df["time_start"].str[:10], errors="coerce").dt.normalize()
df = df.dropna(subset=["date"])
df["launched"]   = pd.to_numeric(df["launched"],  errors="coerce")
df["destroyed"]  = pd.to_numeric(df["destroyed"], errors="coerce")
df["year_month"] = df["date"].dt.to_period("M")
df["month"]      = df["date"].dt.month

df["target_category"] = df["target"].apply(classify_target_oblast)

print("Target category distribution:")
total = df.groupby("target_category")["launched"].sum().sort_values(ascending=False)
total_sum = total.sum()
for cat, n in total.items():
    print(f"  {cat:<20} {n:>8,.0f}  ({n/total_sum:.1%})")

# Monthly pivot
monthly = df.groupby(["year_month", "target_category"])["launched"].sum().reset_index()
pivot   = monthly.pivot_table(
    index="year_month", columns="target_category",
    values="launched", aggfunc="sum", fill_value=0
)
pivot_share    = pivot.div(pivot.sum(axis=1), axis=0) * 100
pivot_share_ts = pivot_share.copy()
pivot_share_ts.index = pivot_share_ts.index.to_timestamp()

dominant    = pivot_share_ts.idxmax(axis=1)
transitions = (dominant != dominant.shift()).sum()
print(f"\nDominant category changed {transitions} times across {len(dominant)} months")

# Seasonal
seasonal     = df.groupby(["month","target_category"])["launched"].sum().reset_index()
seasonal_piv = seasonal.pivot_table(
    index="month", columns="target_category",
    values="launched", aggfunc="sum", fill_value=0
)
seasonal_share = seasonal_piv.div(seasonal_piv.sum(axis=1), axis=0) * 100

print("\nSeasonal breakdown (DEEP REAR % of monthly launches):")
months_labels  = ["","Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
winter_months  = [10, 11, 12, 1, 2, 3]
if "DEEP REAR" in seasonal_share.columns:
    for m, v in seasonal_share["DEEP REAR"].items():
        print(f"  {months_labels[m]}: {v:.1f}%")

# Chart
cat_colors = {
    "FRONTLINE":  RED,
    "CAPITAL":    UKRAINE_BLUE,
    "SOUTHERN":   TEAL,
    "CENTRAL":    ORANGE,
    "DEEP REAR":  PURPLE,
    "NORTHERN":   GREEN,
    "AREA-WIDE":  GREY,
    "OTHER":      "#BDBDBD",
    "UNKNOWN":    "#E0E0E0",
}

cats = [c for c in cat_colors if c in pivot_share_ts.columns]

fig, axes = plt.subplots(3, 1, figsize=(13, 13))

# Panel 1: stacked area
axes[0].stackplot(
    pivot_share_ts.index,
    [pivot_share_ts[c] for c in cats],
    labels=cats,
    colors=[cat_colors[c] for c in cats],
    alpha=0.85,
)
axes[0].set_ylabel("Share of Launches (%)")
axes[0].set_title("Target Category Mix Over Time (Oblast-Based Classification)")
axes[0].set_ylim(0, 100)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
axes[0].legend(loc="upper left", fontsize=9, ncol=3)
axes[0].grid(alpha=0.2)

# Panel 2: dominant category colour band
for month, cat in dominant.items():
    axes[1].axvspan(month, month + pd.DateOffset(months=1),
                    color=cat_colors.get(cat, GREY), alpha=0.75)
axes[1].set_yticks([])
axes[1].set_title("Dominant Target Category per Month")
patches = [mpatches.Patch(color=cat_colors[c], label=c)
           for c in cats if c in dominant.values]
axes[1].legend(handles=patches, fontsize=9, loc="upper left", ncol=3)

# Panel 3: seasonal deep rear
if "DEEP REAR" in seasonal_share.columns:
    colors_seasonal = [RED if m in winter_months else UKRAINE_BLUE
                       for m in seasonal_share.index]
    axes[2].bar(range(1, 13), seasonal_share["DEEP REAR"],
                color=colors_seasonal, edgecolor="white", linewidth=1, zorder=3)
    axes[2].set_xticks(range(1, 13))
    axes[2].set_xticklabels(months_labels[1:])
    axes[2].set_ylabel("Deep Rear Share of Monthly Launches (%)")
    axes[2].set_title("Seasonal Pattern: Deep Rear / Energy Infrastructure Targeting\n"
                      "Red = winter months (Oct–Mar)")
    axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    axes[2].grid(axis="y", alpha=0.3)
    patches2 = [
        mpatches.Patch(color=RED,          label="Winter months (Oct–Mar)"),
        mpatches.Patch(color=UKRAINE_BLUE, label="Summer months (Apr–Sep)"),
    ]
    axes[2].legend(handles=patches2, fontsize=9)

plt.tight_layout()
path = OUTPUT_DIR / "op5_target_rotation_v2.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {path}")
