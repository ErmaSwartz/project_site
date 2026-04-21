# Transit Access & Scenario Modeling — Northern New Jersey

A census-tract-level analysis of transit accessibility and suppressed demand across Northern New Jersey, combining spatial feature engineering, econometric modeling, and machine learning to identify where infrastructure constrains ridership — and where targeted investment would have the greatest impact.

**[View the live site →](https://ermaswartz.github.io/project_site)**

---

## The Core Finding

Transit usage is driven more by **accessibility** — proximity and network density — than by demographics alone. Areas with high predicted demand but low observed ridership are not low-need areas. They are infrastructure-constrained areas. The gap between what the model expects and what actually happens is an investment opportunity.

---

## What This Project Does

| Step | Method | Output |
|------|--------|--------|
| Spatial feature engineering | GeoPandas · EPSG:3424 | `dist_to_bus`, `dist_to_rail`, `bus_density_2mi` per tract |
| Econometric modeling | OLS Regression | Interpretable coefficient estimates (R² ≈ 0.56) |
| Predictive modeling | Random Forest | Transit share predictions (R² ≈ 0.65, RMSE ≈ 0.06) |
| Gap analysis | Predicted − Actual | Map of underperforming tracts by suppressed demand |
| Scenario modeling | RF counterfactuals | Ridership impact of 4 investment strategies |
| Interactive maps | Folium | Choropleth dashboards with transit layer toggles |

---

## Key Variables

**Target:** `transit_share` — % of commuters using transit (ACS 2024)

**Accessibility features (most important):**
- `bus_density_2mi` — bus stops within a 2-mile radius of tract centroid ← *top predictor*
- `dist_to_bus` — distance to nearest NJ Transit bus stop (meters)
- `dist_to_rail` — distance to nearest commuter rail station (meters)

**Socioeconomic controls:** median income, median age, % Black, % Hispanic, % foreign-born

---

## Scenario Modeling

Four investment strategies were simulated using the trained Random Forest:

1. **Blanket Bus Expansion** — +20% bus density everywhere → modest system-wide gain
2. **Targeted Bus Expansion** — concentrated in high-demand, low-access tracts → stronger efficiency
3. **Rail Expansion** — reduced distance to rail in underserved corridors → high localized impact
4. **Hybrid Strategy** *(best)* — targeted bus + rail combined → highest overall improvement

---

## Data Sources

- **American Community Survey** (ACS 1-Year 2024) — demographics and commute mode shares
- **NJ Transit GIS** — 31,132 bus stops · 165 rail stations · 62 light rail stations
- **US Census TIGER** — NJ census tract boundaries (2025)

---

## Stack

```
Python · GeoPandas · scikit-learn · statsmodels · Folium · Plotly · Pandas · NumPy
```

---

## Project Structure

```
project_site/
├── index.html              # Overview
├── methods.html            # Methodology + model results
├── findings.html           # Six key findings
├── shared.css              # All styles
├── maps/
│   ├── gap.html            # Interactive map dashboard (4 tabs)
│   └── scenarios.html      # Scenario maps + comparison
├── outputs/                # Generated Folium map HTML files
└── scripts/
    └── generate_maps.py    # Reproduces all outputs from raw data
```

---

## Reproducing the Analysis

```bash
# Install dependencies
pip install geopandas pandas folium scikit-learn statsmodels plotly branca

# Generate all map outputs (reads from NJTPA/data/)
python scripts/generate_maps.py

# Preview locally
python3 -m http.server 8000
# open http://localhost:8000
```

---

## Limitations

No service frequency data (headways, reliability), no time-of-day variation, no network routing or travel time modeling. Distance metrics are straight-line from tract centroid. These represent natural extensions — particularly GTFS feed integration for frequency and reliability data.

---

*Built as a portfolio project for NJTPA Systems Planning. Analysis covers Northern New Jersey at the census tract level using 2024 data.*
