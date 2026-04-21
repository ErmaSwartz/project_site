#!/usr/bin/env python3
"""
generate_maps.py
Generates missing map and chart outputs for the NJTPA transit analysis site.

Outputs produced (skips existing unless --force):
  map_actual.html       — choropleth: observed transit share
  map_predicted.html    — choropleth: RF-predicted transit share
  map_access.html       — choropleth: bus stop density within 2 miles
  scatter_fit.html      — Plotly: actual vs predicted, colored by gap
  dist_ridership.html   — Plotly: distance to bus vs transit share

Usage:
  python generate_maps.py
  python generate_maps.py --force
  python generate_maps.py --retrain
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import folium
import numpy as np
import plotly.graph_objects as go

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from sklearn.ensemble import RandomForestRegressor

# ── PATHS ───────────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).resolve().parent
OUTPUT_DIR     = SCRIPT_DIR.parent / "outputs"
# scripts/ → project_site/ → NJTPA/data/
DATA_PATH      = SCRIPT_DIR.parent.parent / "data" / "final_analysis_dataset.gpkg"
MODEL_PATH     = OUTPUT_DIR / "rf_model.pkl"
FEATURES_PATH  = OUTPUT_DIR / "features.json"

MAP_CENTER     = [40.07, -74.72]
MAP_ZOOM       = 9

# Site color palette (matches shared.css)
C_CREAM   = "#fefae0"
C_BRONZE  = "#d4a373"
C_ACCENT  = "#8a6a3e"
C_GREEN   = "#ccd5ae"
C_INK     = "#1e1c18"
C_INK2    = "#3d3a30"
C_MUTED   = "#7a7060"
C_RULE    = "#d8d0bc"
C_GAP     = "#b85c38"

DEFAULT_FEATURES = [
    "median_income", "median_age", "pct_black", "pct_hispanic",
    "pct_foreign_born", "bus_stops", "rail_stations",
    "dist_to_bus", "dist_to_rail", "bus_density_2mi",
]


# ── DATA ────────────────────────────────────────────────────────────────────
def load_data() -> gpd.GeoDataFrame:
    print(f"Loading {DATA_PATH.name} ...")
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: data file not found at {DATA_PATH}")
    gdf = gpd.read_file(DATA_PATH)
    print(f"  {len(gdf)} tracts, CRS: {gdf.crs}")
    gdf = gdf.to_crs(epsg=4326)
    # GEOID must stay a string or Folium's key_on join silently fails
    gdf["GEOID"] = gdf["GEOID"].astype(str)
    print("  Reprojected to EPSG:4326")
    return gdf


# ── MODEL ────────────────────────────────────────────────────────────────────
def load_model(features: list):
    """Load serialized RF model. Try joblib first, fall back to pickle."""
    if HAS_JOBLIB:
        model = joblib.load(MODEL_PATH)
    else:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    print(f"  Loaded RF model ({model.n_estimators} trees, {len(features)} features)")
    return model


def train_model(gdf: gpd.GeoDataFrame, features: list) -> RandomForestRegressor:
    gdf_clean = gdf.dropna(subset=features + ["transit_share"]).copy()
    print(f"  Training on {len(gdf_clean)} clean rows ({len(gdf) - len(gdf_clean)} dropped)")
    X, y = gdf_clean[features], gdf_clean["transit_share"]
    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    print(f"  R² (train): {model.score(X, y):.3f}")
    return model


def add_predictions(gdf: gpd.GeoDataFrame, model, features: list) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    mask = gdf[features].notna().all(axis=1)
    preds = np.full(len(gdf), np.nan)
    preds[mask] = model.predict(gdf.loc[mask, features])
    gdf["predicted_transit_share"] = preds
    gdf["transit_gap"] = gdf["predicted_transit_share"] - gdf["transit_share"]
    print(f"  Predictions generated for {mask.sum()}/{len(gdf)} tracts")
    return gdf


# ── CHOROPLETH MAPS ──────────────────────────────────────────────────────────
def make_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    legend_name: str,
    fill_color: str,
    output_path: Path,
    force: bool = False,
) -> None:
    """
    Create a Folium choropleth and save as standalone HTML.

    Critical: dropna on the value column before passing to Folium — branca's
    StepColormap raises ValueError if any value in data[column] is NaN.
    """
    if output_path.exists() and not force:
        print(f"  Skipping {output_path.name} (already exists; use --force to overwrite)")
        return

    gdf_clean = gdf[["GEOID", column, "geometry"]].dropna(subset=[column]).copy()

    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=gdf_clean,
        data=gdf_clean,
        columns=["GEOID", column],
        key_on="feature.properties.GEOID",
        fill_color=fill_color,
        fill_opacity=0.75,
        line_opacity=0.15,
        line_color="#888888",
        legend_name=legend_name,
    ).add_to(m)

    m.save(str(output_path))
    print(f"  Saved {output_path.name}")


# ── SCATTER: ACTUAL VS PREDICTED ─────────────────────────────────────────────
def make_scatter_fit(
    gdf: gpd.GeoDataFrame, output_path: Path, force: bool = False
) -> None:
    """
    Scatter plot: actual transit share (x) vs predicted (y).
    Color = gap (predicted − actual). Points above y=x line = suppressed demand.
    """
    if output_path.exists() and not force:
        print(f"  Skipping {output_path.name} (already exists)")
        return

    cols = ["GEOID", "NAMELSAD", "transit_share", "predicted_transit_share", "transit_gap"]
    df = gdf[cols].dropna().copy()

    # Symmetric color range capped at 95th percentile to avoid outlier wash
    vmax = float(max(
        abs(df["transit_gap"].quantile(0.05)),
        abs(df["transit_gap"].quantile(0.95)),
    ))

    # y = x reference line
    rng_max = float(df["transit_share"].max() * 1.08)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, rng_max],
        y=[0, rng_max],
        mode="lines",
        line=dict(color=C_RULE, width=1.5, dash="dot"),
        showlegend=False,
        hoverinfo="skip",
        name="y = x",
    ))

    fig.add_trace(go.Scatter(
        x=df["transit_share"],
        y=df["predicted_transit_share"],
        mode="markers",
        marker=dict(
            size=5,
            color=df["transit_gap"],
            colorscale=[
                [0.0,  C_INK2],
                [0.4,  C_RULE],
                [0.5,  C_RULE],
                [0.6,  C_BRONZE],
                [1.0,  C_GAP],
            ],
            cmin=-vmax,
            cmax=vmax,
            colorbar=dict(
                title=dict(
                    text="Gap (Predicted − Actual)",
                    font=dict(size=11, color=C_MUTED, family="IBM Plex Mono, monospace"),
                ),
                tickfont=dict(size=10, color=C_MUTED),
                tickformat=".0%",
                thickness=12,
                len=0.55,
                x=1.01,
            ),
            opacity=0.75,
        ),
        customdata=df[["NAMELSAD", "transit_gap"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Actual: %{x:.1%}<br>"
            "Predicted: %{y:.1%}<br>"
            "Gap: %{customdata[1]:+.1%}<extra></extra>"
        ),
        showlegend=False,
    ))

    fig.update_layout(
        plot_bgcolor=C_CREAM,
        paper_bgcolor=C_CREAM,
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color=C_INK),
        xaxis=dict(
            title=dict(text="Actual Transit Share", font=dict(size=12, color=C_MUTED)),
            tickformat=".0%",
            gridcolor=C_RULE,
            gridwidth=0.5,
            zeroline=False,
            range=[0, rng_max],
        ),
        yaxis=dict(
            title=dict(text="Predicted Transit Share", font=dict(size=12, color=C_MUTED)),
            tickformat=".0%",
            gridcolor=C_RULE,
            gridwidth=0.5,
            zeroline=False,
            range=[0, rng_max],
        ),
        margin=dict(l=65, r=80, t=50, b=65),
        hovermode="closest",
        annotations=[
            dict(
                x=0.01, y=0.99, xref="paper", yref="paper",
                text="Above the line = suppressed demand",
                showarrow=False,
                font=dict(size=10, color=C_MUTED, family="IBM Plex Mono, monospace"),
                align="left",
                bgcolor=C_CREAM,
            ),
            dict(
                x=0.01, y=0.93, xref="paper", yref="paper",
                text="Below the line = overperforming",
                showarrow=False,
                font=dict(size=10, color=C_MUTED, family="IBM Plex Mono, monospace"),
                align="left",
                bgcolor=C_CREAM,
            ),
        ],
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"  Saved {output_path.name}")


# ── SCATTER: DISTANCE VS RIDERSHIP ───────────────────────────────────────────
def make_dist_ridership(
    gdf: gpd.GeoDataFrame, output_path: Path, force: bool = False
) -> None:
    """
    Scatter: distance to nearest bus stop (miles) vs transit commute share.
    Color = bus_density_2mi. Shows nonlinear threshold effect past ~1 mile.
    """
    if output_path.exists() and not force:
        print(f"  Skipping {output_path.name} (already exists)")
        return

    cols = ["dist_to_bus", "transit_share", "bus_density_2mi", "NAMELSAD"]
    df = gdf[cols].dropna().copy()
    # Convert feet → miles for readability (EPSG:3424 = NJ State Plane, US Survey Feet)
    df["dist_miles"] = df["dist_to_bus"] / 5280.0
    THRESHOLD = 1.0  # 1 mile ≈ 5,280 ft — the inflection point

    # Cap x-axis and color scale at high quantiles to avoid outlier distortion
    x_max = float(df["dist_miles"].quantile(0.97))
    density_max = float(df["bus_density_2mi"].quantile(0.95))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["dist_miles"],
        y=df["transit_share"],
        mode="markers",
        marker=dict(
            size=5,
            color=df["bus_density_2mi"],
            colorscale=[
                [0.0, C_RULE],
                [0.3, C_GREEN],
                [0.7, C_BRONZE],
                [1.0, C_ACCENT],
            ],
            cmin=0,
            cmax=density_max,
            colorbar=dict(
                title=dict(
                    text="Bus Stops within 2 mi",
                    font=dict(size=11, color=C_MUTED, family="IBM Plex Mono, monospace"),
                ),
                tickfont=dict(size=10, color=C_MUTED),
                thickness=12,
                len=0.55,
                x=1.01,
            ),
            opacity=0.65,
        ),
        customdata=df[["NAMELSAD", "bus_density_2mi"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Distance to bus: %{x:.2f} mi<br>"
            "Transit share: %{y:.1%}<br>"
            "Bus stops (2 mi): %{customdata[1]:.0f}<extra></extra>"
        ),
        showlegend=False,
    ))

    # 1-mile threshold annotation
    fig.add_vline(
        x=THRESHOLD,
        line=dict(color=C_BRONZE, width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=THRESHOLD + 0.04,
        y=float(df["transit_share"].quantile(0.88)),
        text="1-mile mark",
        showarrow=False,
        font=dict(size=10, color=C_BRONZE, family="IBM Plex Mono, monospace"),
        align="left",
    )

    fig.update_layout(
        plot_bgcolor=C_CREAM,
        paper_bgcolor=C_CREAM,
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color=C_INK),
        xaxis=dict(
            title=dict(
                text="Distance to Nearest Bus Stop (miles)",
                font=dict(size=12, color=C_MUTED),
            ),
            gridcolor=C_RULE,
            gridwidth=0.5,
            zeroline=False,
            range=[0, x_max],
        ),
        yaxis=dict(
            title=dict(
                text="Transit Commute Share",
                font=dict(size=12, color=C_MUTED),
            ),
            tickformat=".0%",
            gridcolor=C_RULE,
            gridwidth=0.5,
            zeroline=False,
        ),
        margin=dict(l=65, r=80, t=50, b=65),
        hovermode="closest",
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"  Saved {output_path.name}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate transit analysis map/chart outputs.")
    p.add_argument("--force",   action="store_true", help="Overwrite existing output files")
    p.add_argument("--retrain", action="store_true", help="Retrain RF model from scratch")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    gdf = load_data()

    # 2. Load or train model
    if not args.retrain and MODEL_PATH.exists() and FEATURES_PATH.exists():
        print("Loading existing RF model...")
        with open(FEATURES_PATH) as f:
            features = json.load(f)
        model = load_model(features)
    else:
        print("Training RF model from scratch...")
        features = DEFAULT_FEATURES
        model = train_model(gdf, features)

    # 3. Add predictions and gap column
    print("Generating predictions...")
    gdf = add_predictions(gdf, model, features)

    print("\nGenerating outputs...")

    # 4. Three choropleth maps
    make_choropleth(
        gdf, "transit_share",
        "Actual Transit Share — % of commuters using transit (ACS 2024)",
        "YlOrRd",
        OUTPUT_DIR / "map_actual.html",
        args.force,
    )
    make_choropleth(
        gdf, "predicted_transit_share",
        "Predicted Transit Share — Random Forest model (R² = 0.65)",
        "YlOrRd",
        OUTPUT_DIR / "map_predicted.html",
        args.force,
    )
    make_choropleth(
        gdf, "bus_density_2mi",
        "Bus Stop Density within 2 Miles — top predictor of ridership",
        "YlGn",
        OUTPUT_DIR / "map_access.html",
        args.force,
    )

    # 5. Two diagnostic Plotly charts
    make_scatter_fit(gdf, OUTPUT_DIR / "scatter_fit.html", args.force)
    make_dist_ridership(gdf, OUTPUT_DIR / "dist_ridership.html", args.force)

    print("\nAll outputs complete.")

    # Report what's in outputs/
    html_files = sorted(OUTPUT_DIR.glob("*.html"))
    print(f"\nOutputs directory ({len(html_files)} HTML files):")
    for f in html_files:
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:<35} {size_kb:>5} KB")


if __name__ == "__main__":
    main()
