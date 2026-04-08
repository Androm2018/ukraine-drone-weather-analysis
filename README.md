# Ukraine Shahed Drone — Weather & Interception Rate Analysis

An OSINT-based quantitative analysis examining whether adverse night-time weather conditions reduce Ukraine's ability to intercept Russian Shahed-136 / Geran-2 long-range attack drones.

## Research Questions

1. **Primary:** Do interception rates fall on nights with poor weather conditions (cloud cover, reduced visibility, rain/fog)?
2. **Secondary (exploitation test):** Does Russia launch larger Shahed waves on adverse-weather nights, suggesting deliberate exploitation of Ukrainian sensor limitations?

## Methodology

### Data Sources
| Dataset | Source | Notes |
|---|---|---|
| Missile/drone attack data | [Kaggle: pityfm/massive-missile-attacks-on-ukraine](https://www.kaggle.com/datasets/pityfm/massive-missile-attacks-on-ukraine) | `missile_attacks_daily.csv` |
| Historical weather | [Open-Meteo Archive API](https://open-meteo.com/) | Free, no API key required |

### Weapon Scope
Analysis is restricted to **Shahed-136 / Geran-2** loitering munitions. These are:
- Russia's primary long-range strike drone platform
- Subsonic and low-altitude — making them sensitive to optical/thermal detection degradation
- The platform with the richest interception dataset

Ballistic missiles (Iskander, Kinzhal) and close-range systems (Lancet) are excluded: weather affects these systems differently and conflating them would obscure the signal.

### Weather Monitoring Locations
Weather is averaged across four cities to produce a representative national picture:

| City | Rationale |
|---|---|
| Kyiv | Capital, highest attack frequency |
| Kharkiv | Eastern proximity, high exposure |
| Odesa | Southern corridor, maritime context |
| Zaporizhzhia | Front-line city, consistent targeting |

### Night-time Window
**20:00–06:00 local time (Europe/Kyiv)** — reflects the documented Shahed attack window. Night-time metrics are used as the primary analytical variable rather than daily averages.

### Weather Buckets
Each attack day is classified into one of three categories based on night-time conditions:

| Bucket | Criteria |
|---|---|
| **CLEAR** | Cloud cover < 40%, visibility > 8 km, precipitation < 1 mm |
| **OVERCAST** | Cloud cover ≥ 60% or visibility 4–8 km |
| **ADVERSE** | Visibility < 4 km (fog threshold), precipitation ≥ 5 mm, or WMO fog/rain/snow code |

### Statistical Methods
- Descriptive comparison of mean interception rates across weather buckets
- Student's t-test and Mann-Whitney U test (CLEAR vs ADVERSE)
- Pearson correlation (continuous weather variables vs interception rate)
- OLS regression with controls: launch volume, time trend, year fixed effects (requires `statsmodels`)

---

## Repository Structure

```
ukraine-drone-weather-analysis/
├── data/
│   └── missile_attacks_daily.csv   ← YOU MUST ADD THIS (Kaggle download)
├── notebooks/
│   └── weather_interception_analysis.ipynb   ← Main analysis
├── src/
│   ├── classify_attacks.py   ← Load & classify Kaggle data
│   ├── fetch_weather.py      ← Pull Open-Meteo weather by date
│   └── analysis.py           ← Regression, bucket analysis, charts
├── outputs/
│   └── charts/               ← PNG figures (auto-generated)
├── requirements.txt
└── README.md
```

---

## Setup & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add the Kaggle dataset
Download `missile_attacks_daily.csv` from:
https://www.kaggle.com/datasets/pityfm/massive-missile-attacks-on-ukraine

Place it at `data/missile_attacks_daily.csv`

### 3. Run the pipeline
```bash
# Step 1: Classify attacks and build Shahed daily summary
python src/classify_attacks.py

# Step 2: Fetch weather data (makes ~4 API calls to Open-Meteo)
python src/fetch_weather.py

# Step 3: Run analysis and generate charts
python src/analysis.py
```

### 4. Or use the Jupyter notebook
```bash
jupyter notebook notebooks/weather_interception_analysis.ipynb
```
The notebook runs all three steps in sequence with narrative commentary.

---

## Output Charts

| File | Description |
|---|---|
| `fig1_bucket_interception_by_weather.png` | Mean interception rate by weather bucket with 95% CI |
| `fig2_scatter_cloudcover_vs_rate.png` | Cloud cover vs interception rate scatter |
| `fig3_scatter_visibility_vs_rate.png` | Visibility vs interception rate scatter |
| `fig5_time_trend_with_weather.png` | Monthly interception rate over time, coloured by weather |
| `fig6_launch_volume_by_weather.png` | Launch volume by weather (exploitation test) |

---

## Limitations

1. **Weather corridor approximation** — national average does not capture weather along specific Shahed flight paths (which vary by launch origin)
2. **Reporting accuracy** — Ukrainian MoD figures are primary source; independent verification is partial
3. **Confounders** — Western AD system deliveries (Patriot, IRIS-T, NASAMS), Ukrainian AD learning curve, and seasonal patterns all affect interception rates independently
4. **Causality** — correlation analysis only; causal claims about Russian deliberate weather exploitation require additional evidence

---

## Related Work

This analysis was developed alongside the [Ukraine War Unmanned Systems Tracker](https://unmannedsystemstracker.com) — a public OSINT platform tracking USV strikes, UGV operations, UAV kill data, and air defence statistics.

---

## Data Attribution

- Attack data: Petro Ivaniuk, *Massive Missile Attacks on Ukraine* (Kaggle, 2022–present)
- Weather data: Open-Meteo, ERA5 reanalysis via Copernicus Climate Change Service

## Licence

Analysis code: MIT  
Data: subject to original source licences (see above)
