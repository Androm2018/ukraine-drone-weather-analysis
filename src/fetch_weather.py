"""
fetch_weather.py
----------------
Pulls historical weather data from the Open-Meteo API (free, no key required)
for each unique attack date in the Kaggle dataset.

Monitoring locations chosen to cover Ukraine's primary target zones:
  - Kyiv       (48.31°N, 30.23°E)  — capital, highest attack frequency
  - Kharkiv    (49.99°N, 36.23°E)  — eastern front proximity
  - Odesa      (46.48°N, 30.73°E)  — southern corridor, maritime drone context
  - Zaporizhzhia (47.84°N, 35.14°E) — front-line city, high exposure

Weather is averaged across all four locations per date to produce a
representative national picture. Individual city data is retained for
sensitivity analysis.

Output: data/weather_by_date.csv
"""

import time
import requests
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

LOCATIONS = {
    "kyiv":          {"lat": 50.45, "lon": 30.52},
    "kharkiv":       {"lat": 49.99, "lon": 36.23},
    "odesa":         {"lat": 46.48, "lon": 30.73},
    "zaporizhzhia":  {"lat": 47.84, "lon": 35.14},
}

# Variables requested from Open-Meteo daily endpoint
DAILY_VARS = [
    "precipitation_sum",        # mm  — total rainfall/snowfall
    "cloudcover_mean",          # %   — mean cloud cover
    "visibility_mean",          # m   — mean visibility (proxy for fog)
    "windspeed_10m_max",        # km/h
    "weathercode",              # WMO code (fog = 45/48, rain = 61-67, snow = 71-77)
]

# Hourly variables to check night-time conditions specifically
# Shaheds typically launch/arrive at night — night window = 20:00–06:00 local (UTC+2/3)
HOURLY_VARS = [
    "cloudcover",
    "visibility",
    "precipitation",
]

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "weather_by_date.csv"

# ── WMO Code Classification ────────────────────────────────────────────────────

def classify_weather_code(code):
    """
    Map WMO weather code to a human-readable category.
    Returns one of: Clear, Partly Cloudy, Overcast, Fog, Drizzle,
                    Rain, Heavy Rain, Snow, Thunderstorm
    """
    if code is None:
        return "Unknown"
    code = int(code)
    if code == 0:
        return "Clear"
    elif code in (1, 2):
        return "Partly Cloudy"
    elif code == 3:
        return "Overcast"
    elif code in (45, 48):
        return "Fog"
    elif code in (51, 53, 55):
        return "Drizzle"
    elif code in (61, 63):
        return "Rain"
    elif code == 65:
        return "Heavy Rain"
    elif code in (71, 73, 75, 77):
        return "Snow"
    elif code in (80, 81, 82):
        return "Rain Showers"
    elif code in (95, 96, 99):
        return "Thunderstorm"
    else:
        return "Other"


def fetch_weather_for_location(name: str, lat: float, lon: float,
                                start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily + hourly weather for a single location over a date range.
    Returns a DataFrame indexed by date.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Europe/Kyiv",
    }

    resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # ── Daily data ──
    daily = pd.DataFrame(data["daily"])
    daily["date"] = pd.to_datetime(daily["time"]).dt.date
    daily = daily.drop(columns=["time"])
    daily.columns = [f"{name}_{c}" if c != "date" else "date"
                     for c in daily.columns]

    # ── Night-time hourly means (20:00–06:00 UTC+2) ──
    hourly = pd.DataFrame(data["hourly"])
    hourly["datetime"] = pd.to_datetime(hourly["time"])
    hourly["date"] = hourly["datetime"].dt.date
    hourly["hour"] = hourly["datetime"].dt.hour
    night = hourly[hourly["hour"].isin(list(range(20, 24)) + list(range(0, 7)))]
    night_agg = night.groupby("date")[HOURLY_VARS].mean().reset_index()
    night_agg.columns = [f"{name}_night_{c}" if c != "date" else "date"
                         for c in night_agg.columns]

    merged = daily.merge(night_agg, on="date", how="left")
    return merged


def fetch_all_locations(attack_dates: list[str]) -> pd.DataFrame:
    """
    For efficiency, fetch one continuous block covering all attack dates
    rather than individual calls per date.
    """
    start_date = min(attack_dates)
    end_date   = max(attack_dates)

    print(f"Fetching weather: {start_date} → {end_date}")
    print(f"Locations: {', '.join(LOCATIONS.keys())}\n")

    all_dfs = []
    for name, coords in LOCATIONS.items():
        print(f"  Fetching {name}...", end=" ")
        df = fetch_weather_for_location(
            name, coords["lat"], coords["lon"], start_date, end_date
        )
        all_dfs.append(df)
        print("done")
        time.sleep(0.5)  # polite rate limiting

    # Merge all locations on date
    combined = all_dfs[0]
    for df in all_dfs[1:]:
        combined = combined.merge(df, on="date", how="outer")

    # ── Compute national averages across all four cities ──
    daily_vars_all = [v for loc in LOCATIONS for v in [
        f"{loc}_precipitation_sum",
        f"{loc}_cloudcover_mean",
        f"{loc}_visibility_mean",
        f"{loc}_windspeed_10m_max",
    ]]

    combined["avg_precipitation"] = combined[[
        f"{loc}_precipitation_sum" for loc in LOCATIONS]].mean(axis=1)
    combined["avg_cloudcover"]    = combined[[
        f"{loc}_cloudcover_mean"   for loc in LOCATIONS]].mean(axis=1)
    combined["avg_visibility"]    = combined[[
        f"{loc}_visibility_mean"   for loc in LOCATIONS]].mean(axis=1)
    combined["avg_windspeed"]     = combined[[
        f"{loc}_windspeed_10m_max" for loc in LOCATIONS]].mean(axis=1)

    # Night-time averages (most operationally relevant for Shahed attacks)
    combined["avg_night_cloudcover"]   = combined[[
        f"{loc}_night_cloudcover"   for loc in LOCATIONS]].mean(axis=1)
    combined["avg_night_visibility"]   = combined[[
        f"{loc}_night_visibility"   for loc in LOCATIONS]].mean(axis=1)
    combined["avg_night_precipitation"] = combined[[
        f"{loc}_night_precipitation" for loc in LOCATIONS]].mean(axis=1)

    # Use Kyiv weather code as primary (capital = most representative)
    combined["weather_category"] = combined["kyiv_weathercode"].apply(
        classify_weather_code)

    # Filter to only attack dates
    combined["date"] = pd.to_datetime(combined["date"])
    attack_dates_dt = pd.to_datetime(attack_dates)
    result = combined[combined["date"].isin(attack_dates_dt)].copy()
    result = result.reset_index(drop=True)

    print(f"\n✓ Weather data ready: {len(result)} attack dates")
    return result


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Load attack dates from the classified attacks file if it exists,
    # otherwise fall back to reading the raw Kaggle CSV
    classified_path = DATA_DIR / "attacks_classified.csv"
    raw_path        = DATA_DIR / "missile_attacks_daily.csv"

    if classified_path.exists():
        attacks = pd.read_csv(classified_path)
        date_col = "date"
    elif raw_path.exists():
        attacks = pd.read_csv(raw_path)
        date_col = "time_start"
    else:
        raise FileNotFoundError(
            "Place missile_attacks_daily.csv in the data/ directory first.\n"
            "Download from: https://www.kaggle.com/datasets/pityfm/"
            "massive-missile-attacks-on-ukraine"
        )

    attacks[date_col] = pd.to_datetime(attacks[date_col]).dt.date.astype(str)
    attack_dates = sorted(attacks[date_col].unique().tolist())

    weather_df = fetch_all_locations(attack_dates)
    weather_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
