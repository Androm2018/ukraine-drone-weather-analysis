"""
analysis.py
-----------
Core analysis: does inclement weather reduce Ukraine's Shahed interception rate?

Two analytical approaches:
  1. BUCKET ANALYSIS   — group attack days by weather condition, compare mean
                         interception rates (intuitive, easy to communicate)
  2. OLS REGRESSION    — interception_rate ~ weather variables + controls
                         (accounts for confounders: launch volume, time trend)

Outputs (saved to outputs/charts/):
  fig1_bucket_interception_by_weather.png
  fig2_scatter_cloudcover_vs_rate.png
  fig3_scatter_visibility_vs_rate.png
  fig4_regression_coefficients.png
  fig5_time_trend_with_weather.png
  fig6_launch_volume_by_weather.png   ← tests deliberate exploitation hypothesis

Also prints a full results summary to stdout.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

# ── Try to import statsmodels; fall back to scipy if not available ──
try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Note: statsmodels not installed. Using scipy for regression.")

DATA_DIR    = Path(__file__).parent.parent / "data"
OUTPUT_DIR  = Path(__file__).parent.parent / "outputs" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (restrained, publication-quality) ──
UKRAINE_BLUE   = "#005BBB"
UKRAINE_YELLOW = "#FFD700"
CLEAR_COLOR    = "#2196F3"
CLOUDY_COLOR   = "#90A4AE"
ADVERSE_COLOR  = "#B71C1C"
NEUTRAL_GREY   = "#616161"
BG_COLOR       = "#FAFAFA"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_merged() -> pd.DataFrame:
    """Load and merge Shahed daily + weather data."""
    shahed_path  = DATA_DIR / "shahed_daily.csv"
    weather_path = DATA_DIR / "weather_by_date.csv"

    if not shahed_path.exists():
        raise FileNotFoundError("Run classify_attacks.py first.")
    if not weather_path.exists():
        raise FileNotFoundError("Run fetch_weather.py first.")

    shahed  = pd.read_csv(shahed_path,  parse_dates=["date"])
    weather = pd.read_csv(weather_path, parse_dates=["date"])

    df = shahed.merge(weather, on="date", how="inner")
    print(f"Merged dataset: {len(df)} attack days with weather data")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER BUCKETING
# ══════════════════════════════════════════════════════════════════════════════

def assign_weather_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each attack day to a weather bucket based on night-time conditions
    (operationally most relevant — Shaheds typically arrive at night).

    Buckets:
      CLEAR    — cloudcover < 40% AND visibility > 8000m AND precip < 1mm
      OVERCAST — cloudcover ≥ 60% OR visibility 4000–8000m
      ADVERSE  — fog (visibility < 4000m) OR heavy precipitation (≥ 5mm)
                 OR weathercode indicates fog/rain/snow
    """
    df = df.copy()

    # Primary: use night-time metrics if available
    cloud = df.get("avg_night_cloudcover",   df.get("avg_cloudcover"))
    vis   = df.get("avg_night_visibility",   df.get("avg_visibility"))
    precip= df.get("avg_night_precipitation",df.get("avg_precipitation"))
    code  = df.get("weather_category", pd.Series(["Unknown"] * len(df)))

    adverse_codes = {"Fog", "Rain", "Heavy Rain", "Snow", "Drizzle",
                     "Rain Showers", "Thunderstorm"}

    conditions = []
    for i in df.index:
        cl = cloud.iloc[i]   if cloud  is not None else np.nan
        vi = vis.iloc[i]     if vis    is not None else np.nan
        pr = precip.iloc[i]  if precip is not None else np.nan
        wc = code.iloc[i]    if code   is not None else "Unknown"

        vi_ok = not (vi != vi)  # False if NaN
        if wc in adverse_codes or (vi_ok and vi < 4000) or (pr >= 5):
            conditions.append("ADVERSE")
        elif (cl < 40) and (not vi_ok or vi > 8000) and (pr < 1):
            conditions.append("CLEAR")
        else:
            conditions.append("OVERCAST")

    df["weather_bucket"] = conditions
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def bucket_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compare interception rates across weather buckets."""
    summary = df.groupby("weather_bucket").agg(
        n_days          = ("interception_rate", "count"),
        mean_rate       = ("interception_rate", "mean"),
        median_rate     = ("interception_rate", "median"),
        std_rate        = ("interception_rate", "std"),
        mean_launched   = ("launched",          "mean"),
        total_launched  = ("launched",          "sum"),
    ).reset_index()

    summary["sem"] = summary["std_rate"] / np.sqrt(summary["n_days"])
    summary["ci95_low"]  = summary["mean_rate"] - 1.96 * summary["sem"]
    summary["ci95_high"] = summary["mean_rate"] + 1.96 * summary["sem"]

    # Statistical test: CLEAR vs ADVERSE
    clear   = df[df["weather_bucket"] == "CLEAR"]["interception_rate"].dropna()
    adverse = df[df["weather_bucket"] == "ADVERSE"]["interception_rate"].dropna()

    if len(clear) >= 5 and len(adverse) >= 5:
        t_stat, p_val = stats.ttest_ind(clear, adverse)
        mann_stat, mann_p = stats.mannwhitneyu(clear, adverse, alternative="greater")
        print(f"\n── Statistical Tests: CLEAR vs ADVERSE ──")
        print(f"  t-test:        t={t_stat:.3f}, p={p_val:.4f}")
        print(f"  Mann-Whitney:  U={mann_stat:.0f}, p={mann_p:.4f}")
        print(f"  Clear mean:    {clear.mean():.1%}  (n={len(clear)})")
        print(f"  Adverse mean:  {adverse.mean():.1%}  (n={len(adverse)})")
        print(f"  Difference:    {clear.mean() - adverse.mean():.1%} percentage points")

    return summary


def run_regression(df: pd.DataFrame) -> None:
    """
    OLS regression: interception_rate ~ weather + controls.

    Controls included:
      - log_launched:  launch volume (larger waves may be harder/easier to intercept)
      - time_trend:    days since first attack (accounts for improving Ukrainian AD)
      - year dummies:  structural breaks between years
    """
    df = df.copy()
    df["log_launched"] = np.log1p(df["launched"])
    df["time_trend"]   = (df["date"] - df["date"].min()).dt.days
    df["cloudcover_scaled"] = df["avg_night_cloudcover"]   / 100
    df["visibility_km"]     = df["avg_night_visibility"]   / 1000
    df["precip_mm"]         = df["avg_night_precipitation"]

    if HAS_STATSMODELS:
        formula = ("interception_rate ~ cloudcover_scaled + visibility_km "
                   "+ precip_mm + log_launched + time_trend")
        model = smf.ols(formula, data=df.dropna()).fit()
        print("\n── OLS Regression Results ──")
        print(model.summary2())
        return model
    else:
        # Fallback: individual Pearson correlations
        print("\n── Pearson Correlations with Interception Rate ──")
        weather_vars = {
            "Night cloud cover (%)": "avg_night_cloudcover",
            "Night visibility (m)":  "avg_night_visibility",
            "Night precipitation":   "avg_night_precipitation",
            "Daily precipitation":   "avg_precipitation",
        }
        for label, col in weather_vars.items():
            if col in df.columns:
                valid = df[[col, "interception_rate"]].dropna()
                r, p = stats.pearsonr(valid[col], valid["interception_rate"])
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                print(f"  {label:<30}  r={r:+.3f}  p={p:.4f} {sig}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

BUCKET_COLORS = {
    "CLEAR":    CLEAR_COLOR,
    "OVERCAST": CLOUDY_COLOR,
    "ADVERSE":  ADVERSE_COLOR,
}
BUCKET_ORDER = ["CLEAR", "OVERCAST", "ADVERSE"]


def fig1_bucket_bars(df: pd.DataFrame, summary: pd.DataFrame):
    """Bar chart: mean interception rate by weather bucket with 95% CI."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ordered = summary.set_index("weather_bucket").reindex(BUCKET_ORDER).reset_index()
    ordered = ordered.dropna(subset=["mean_rate"])

    bars = ax.bar(
        ordered["weather_bucket"],
        ordered["mean_rate"] * 100,
        color=[BUCKET_COLORS[b] for b in ordered["weather_bucket"]],
        width=0.55,
        zorder=3,
        edgecolor="white", linewidth=1.5,
    )

    # Error bars (95% CI)
    ax.errorbar(
        ordered["weather_bucket"],
        ordered["mean_rate"] * 100,
        yerr=[(ordered["mean_rate"] - ordered["ci95_low"]) * 100,
              (ordered["ci95_high"] - ordered["mean_rate"]) * 100],
        fmt="none", color="#333", capsize=5, linewidth=1.5, zorder=4,
    )

    # Value labels
    for bar, (_, row) in zip(bars, ordered.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{row['mean_rate']:.1%}\n(n={int(row['n_days'])})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xlabel("Night-time Weather Conditions", labelpad=10)
    ax.set_ylabel("Mean Shahed Interception Rate")
    ax.set_title("Ukraine's Shahed Interception Rate by Weather Conditions\n"
                 "Error bars = 95% confidence intervals", pad=12)
    ax.axhline(df["interception_rate"].mean() * 100, color=NEUTRAL_GREY,
               linestyle="--", linewidth=1, alpha=0.7, zorder=2)
    ax.text(2.45, df["interception_rate"].mean() * 100 + 1,
            f"Overall mean: {df['interception_rate'].mean():.1%}",
            ha="right", fontsize=9, color=NEUTRAL_GREY)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_facecolor(BG_COLOR)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig1_bucket_interception_by_weather.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def fig2_scatter_cloud(df: pd.DataFrame):
    """Scatter: night cloud cover vs interception rate."""
    fig, ax = plt.subplots(figsize=(8, 5))
    col = "avg_night_cloudcover"
    if col not in df.columns:
        col = "avg_cloudcover"

    valid = df[[col, "interception_rate", "launched"]].dropna()
    scatter = ax.scatter(
        valid[col], valid["interception_rate"] * 100,
        c=valid["launched"], cmap="YlOrRd",
        alpha=0.65, s=40, edgecolors="white", linewidth=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="Shaheds launched")

    # Regression line
    m, b, r, p, se = stats.linregress(valid[col], valid["interception_rate"] * 100)
    x_line = np.linspace(valid[col].min(), valid[col].max(), 100)
    ax.plot(x_line, m * x_line + b, color=ADVERSE_COLOR,
            linewidth=2, label=f"OLS fit  r={r:.3f}, p={p:.4f}")

    ax.set_xlabel("Night-time Cloud Cover (%)")
    ax.set_ylabel("Shahed Interception Rate (%)")
    ax.set_title("Night Cloud Cover vs. Shahed Interception Rate\n"
                 "Colour = launch volume")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(alpha=0.25)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig2_scatter_cloudcover_vs_rate.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def fig3_scatter_visibility(df: pd.DataFrame):
    """Scatter: night visibility vs interception rate."""
    col = "avg_night_visibility"
    if col not in df.columns:
        col = "avg_visibility"

    valid = df[[col, "interception_rate"]].dropna()
    valid = valid[valid[col] > 0]

    if len(valid) < 5:
        print(f"  Skipping fig3 — insufficient visibility data (n={len(valid)})")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    vis_km = valid[col] / 1000

    ax.scatter(vis_km, valid["interception_rate"] * 100,
               color=UKRAINE_BLUE, alpha=0.55, s=35,
               edgecolors="white", linewidth=0.5)

    m, b, r, p, _ = stats.linregress(vis_km, valid["interception_rate"] * 100)
    x_line = np.linspace(vis_km.min(), vis_km.max(), 100)
    ax.plot(x_line, m * x_line + b, color=UKRAINE_YELLOW,
            linewidth=2.5, label=f"OLS fit  r={r:.3f}, p={p:.4f}")

    ax.axvline(4, color=ADVERSE_COLOR, linestyle=":", linewidth=1.5,
               alpha=0.7, label="Fog threshold (4 km)")

    ax.set_xlabel("Night-time Visibility (km)")
    ax.set_ylabel("Shahed Interception Rate (%)")
    ax.set_title("Night Visibility vs. Shahed Interception Rate")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(alpha=0.25)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig3_scatter_visibility_vs_rate.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def fig5_time_trend(df: pd.DataFrame):
    """
    Monthly mean interception rate over time, coloured by dominant weather.
    Shows both the improving trend and weather overlay.
    """
    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("year_month").agg(
        mean_rate      = ("interception_rate", "mean"),
        n              = ("interception_rate", "count"),
        adverse_days   = ("weather_bucket",    lambda x: (x == "ADVERSE").sum()),
        clear_days     = ("weather_bucket",    lambda x: (x == "CLEAR").sum()),
    ).reset_index()
    monthly["dom_weather"] = monthly.apply(
        lambda r: "ADVERSE" if r["adverse_days"] > r["clear_days"] else "CLEAR",
        axis=1,
    )
    monthly["date_dt"] = monthly["year_month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["date_dt"], monthly["mean_rate"] * 100,
            color=UKRAINE_BLUE, linewidth=2, zorder=3)

    for _, row in monthly.iterrows():
        ax.scatter(row["date_dt"], row["mean_rate"] * 100,
                   color=BUCKET_COLORS.get(row["dom_weather"], NEUTRAL_GREY),
                   s=55, zorder=4, edgecolors="white", linewidth=0.8)

    # Rolling 3-month trend
    monthly["rolling_mean"] = monthly["mean_rate"].rolling(3, center=True).mean()
    ax.plot(monthly["date_dt"], monthly["rolling_mean"] * 100,
            color=UKRAINE_YELLOW, linewidth=2.5, linestyle="--",
            label="3-month rolling mean", zorder=5)

    # Legend patches
    patches = [
        mpatches.Patch(color=CLEAR_COLOR,   label="Month dominated by clear nights"),
        mpatches.Patch(color=ADVERSE_COLOR, label="Month dominated by adverse nights"),
        plt.Line2D([0], [0], color=UKRAINE_YELLOW, linewidth=2.5,
                   linestyle="--", label="3-month rolling mean"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_ylabel("Mean Monthly Interception Rate (%)")
    ax.set_title("Ukraine's Shahed Interception Rate Over Time\n"
                 "Dots coloured by dominant night-time weather that month")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig5_time_trend_with_weather.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def fig6_launch_volume_by_weather(df: pd.DataFrame):
    """
    Tests the deliberate exploitation hypothesis:
    Does Russia launch MORE Shaheds on adverse weather nights?
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    summary = df.groupby("weather_bucket")["launched"].agg(["mean", "median", "std", "count"])
    summary = summary.reindex(BUCKET_ORDER).dropna()
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])

    bars = ax.bar(
        summary.index,
        summary["mean"],
        color=[BUCKET_COLORS[b] for b in summary.index],
        width=0.55, edgecolor="white", linewidth=1.5, zorder=3,
    )
    ax.errorbar(summary.index, summary["mean"],
                yerr=1.96 * summary["sem"],
                fmt="none", color="#333", capsize=5, linewidth=1.5, zorder=4)

    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{row['mean']:.1f}\n(n={int(row['count'])})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Mean Shaheds Launched per Attack Day")
    ax.set_xlabel("Night-time Weather Conditions")
    ax.set_title("Shahed Launch Volume by Weather Conditions\n"
                 "Tests whether Russia deliberately exploits adverse weather")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig6_launch_volume_by_weather.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  UKRAINE SHAHED WEATHER-INTERCEPTION ANALYSIS")
    print("═" * 60)

    df = load_merged()
    df = assign_weather_bucket(df)

    print(f"\nWeather bucket distribution (attack days):")
    print(df["weather_bucket"].value_counts().to_string())

    summary = bucket_analysis(df)
    print("\n── Bucket summary ──")
    print(summary.to_string(index=False))

    model = run_regression(df)

    print("\n── Generating charts ──")
    fig1_bucket_bars(df, summary)
    fig2_scatter_cloud(df)
    fig3_scatter_visibility(df)
    fig5_time_trend(df)
    fig6_launch_volume_by_weather(df)

    print(f"\n✓ All charts saved to outputs/charts/")
    print("\n── Key finding ──")
    clear_rate   = summary[summary["weather_bucket"] == "CLEAR"]["mean_rate"].values
    adverse_rate = summary[summary["weather_bucket"] == "ADVERSE"]["mean_rate"].values
    if len(clear_rate) and len(adverse_rate):
        delta = (clear_rate[0] - adverse_rate[0]) * 100
        direction = "LOWER" if delta > 0 else "HIGHER"
        print(f"  Interception rate is {abs(delta):.1f} pp {direction} on adverse nights vs clear nights")

    return df, summary, model


if __name__ == "__main__":
    df, summary, model = main()
