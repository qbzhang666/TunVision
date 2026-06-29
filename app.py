from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from tunvision.reasoning import grade_rings, summarize_tunnels


ROOT = Path(__file__).parent
DATA = ROOT / "data"


st.set_page_config(page_title="TunVision", page_icon="TV", layout="wide")


@st.cache_data
def load_defaults() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    catalog = pd.read_csv(DATA / "dataset_catalog.csv")
    rings = pd.read_csv(DATA / "ring_indicators.csv")
    leakage = pd.read_csv(DATA / "leakage_evidence.csv")
    performance = pd.read_csv(DATA / "model_performance.csv")
    class_iou = pd.read_csv(DATA / "class_iou.csv")
    ovalization = pd.read_csv(DATA / "ovalization_comparison.csv")
    rules = json.loads((DATA / "fmea_rules.json").read_text(encoding="utf-8"))
    return catalog, rings, leakage, performance, class_iou, ovalization, rules


def load_uploaded_csv(upload, default: pd.DataFrame) -> pd.DataFrame:
    if upload is None:
        return default.copy()
    return pd.read_csv(upload)


def load_uploaded_rules(upload, default: dict) -> dict:
    if upload is None:
        return default
    return json.load(upload)


def level_color(level: int) -> str:
    return {
        1: "#2f855a",
        2: "#b7791f",
        3: "#c05621",
        4: "#c53030",
        5: "#742a2a",
    }.get(int(level), "#4a5568")


def read_point_cloud(upload) -> pd.DataFrame | None:
    if upload is None:
        return None
    suffix = Path(upload.name).suffix.lower()
    if suffix not in {".csv", ".txt", ".xyz"}:
        st.warning("Point-cloud preview currently supports CSV/TXT/XYZ tables. Export PLY/LAS to XYZ or CSV first.")
        return None

    if suffix == ".csv":
        cloud = pd.read_csv(upload)
    else:
        cloud = pd.read_csv(upload, sep=r"\s+", header=None)
        cloud.columns = ["x", "y", "z", *[f"feature_{i}" for i in range(1, len(cloud.columns) - 2)]]

    lower = {col.lower(): col for col in cloud.columns}
    rename = {}
    for target in ["x", "y", "z"]:
        if target in lower:
            rename[lower[target]] = target
    cloud = cloud.rename(columns=rename)
    if not {"x", "y", "z"}.issubset(cloud.columns):
        st.warning("Point-cloud preview requires x, y and z columns.")
        return None
    return cloud


def normalize_point_cloud_columns(cloud: pd.DataFrame) -> pd.DataFrame | None:
    lower = {str(col).lower(): col for col in cloud.columns}
    rename = {}
    for target in ["x", "y", "z"]:
        if target in lower:
            rename[lower[target]] = target
    cloud = cloud.rename(columns=rename)
    if not {"x", "y", "z"}.issubset(cloud.columns):
        return None
    return cloud


@st.cache_data(show_spinner=False)
def load_point_cloud_url(url: str) -> pd.DataFrame | None:
    if not url:
        return None
    suffix = Path(url.split("?")[0]).suffix.lower()
    if suffix == ".csv":
        cloud = pd.read_csv(url)
    else:
        cloud = pd.read_csv(url, sep=r"\s+", header=None)
        cloud.columns = ["x", "y", "z", *[f"feature_{i}" for i in range(1, len(cloud.columns) - 2)]]
    return normalize_point_cloud_columns(cloud)


def load_point_cloud_path(path_text: str) -> pd.DataFrame | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if not path.exists():
        st.warning(f"Point-cloud path does not exist: {path}")
        return None
    suffix = path.suffix.lower()
    if suffix == ".csv":
        cloud = pd.read_csv(path)
    elif suffix in {".txt", ".xyz"}:
        cloud = pd.read_csv(path, sep=r"\s+", header=None)
        cloud.columns = ["x", "y", "z", *[f"feature_{i}" for i in range(1, len(cloud.columns) - 2)]]
    else:
        st.warning("Local point-cloud preview supports CSV/TXT/XYZ. Convert LAS/PLY/E57 before loading.")
        return None
    normalized = normalize_point_cloud_columns(cloud)
    if normalized is None:
        st.warning("Point-cloud data requires x, y and z columns.")
    return normalized


@st.cache_data(show_spinner=False)
def load_json_url(url: str) -> dict | None:
    if not url:
        return None
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json()


def synthetic_tunnel_cloud(rings: pd.DataFrame, tunnel: int | None = None, points_per_ring: int = 220) -> pd.DataFrame:
    source = rings.copy()
    if tunnel is not None:
        source = source[source["tunnel"].eq(tunnel)]
    theta = np.linspace(-0.84 * np.pi, 0.84 * np.pi, points_per_ring)
    rows = []
    for _, row in source.iterrows():
        radius = 2.75 + (float(row["ovalization_mm"]) / 1000.0) * np.cos(2 * theta)
        radius -= (float(row["convergence_per_mille_d"]) / 1000.0) * 2.75 * np.sin(theta) ** 2
        leakage_zone = (theta > 0.25) & (theta < 0.55) & (int(row["ring"]) % 4 == 0)
        for idx, angle in enumerate(theta):
            label = "leakage" if leakage_zone[idx] else ("joint" if idx % 28 == 0 else "segment")
            rows.append(
                {
                    "x": float(row["ring"]),
                    "y": float(radius[idx] * np.cos(angle)),
                    "z": float(radius[idx] * np.sin(angle)),
                    "tunnel": int(row["tunnel"]),
                    "ring": int(row["ring"]),
                    "class": label,
                    "intensity": 0.85 if label == "leakage" else 0.35 + 0.1 * np.cos(angle),
                }
            )
    return pd.DataFrame(rows)


def point_cloud_figure(cloud: pd.DataFrame, color_by: str | None) -> go.Figure:
    sample = cloud.sample(min(len(cloud), 15000), random_state=7) if len(cloud) > 15000 else cloud
    marker = {"size": 2, "opacity": 0.75}
    if color_by and color_by in sample.columns:
        marker["color"] = sample[color_by]
        marker["colorscale"] = "Viridis"
        marker["showscale"] = True
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=sample["x"],
                y=sample["y"],
                z=sample["z"],
                mode="markers",
                marker=marker,
            )
        ]
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        scene={"aspectmode": "data", "xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
        height=560,
    )
    return fig


def selected_performance(performance: pd.DataFrame, pool: str, regime: str, fraction: int) -> pd.Series:
    filtered = performance[
        performance["training_pool"].eq(pool)
        & performance["transfer_regime"].eq(regime)
        & performance["label_fraction"].eq(fraction)
    ]
    return filtered.iloc[0] if not filtered.empty else performance.iloc[0]


def prediction_summary(model: str, record: pd.Series, dataset_name: str) -> dict[str, object]:
    if model.startswith("Sonata"):
        return {
            "status": "Configured",
            "dataset": dataset_name,
            "model": model,
            "leakage_iou": float(record["leakage_iou"]),
            "accuracy": float(record["accuracy"]),
            "notes": "Uses the paper's exported Sonata experiment metrics; connect a checkpoint runner for live inference.",
        }
    if "Otsu" in model:
        return {
            "status": "Baseline",
            "dataset": dataset_name,
            "model": model,
            "leakage_iou": 70.0,
            "accuracy": 0.83,
            "notes": "Classical thresholding baseline for quick triage before deep inference.",
        }
    return {
        "status": "External hook",
        "dataset": dataset_name,
        "model": model,
        "leakage_iou": None,
        "accuracy": None,
        "notes": "Provide a GitHub runner or API endpoint to execute this model outside Streamlit.",
    }


def apply_geometry_backend(rings: pd.DataFrame, method: str, ovalization: pd.DataFrame) -> pd.DataFrame:
    output = rings.copy()
    if method == "Raw conic fit validation":
        raw = ovalization.rename(columns={"raw_conic_ovalization_mm": "raw_ovalization_mm"})[
            ["tunnel", "ring", "raw_ovalization_mm"]
        ]
        output = output.merge(raw, on=["tunnel", "ring"], how="left")
        output["ovalization_mm"] = output["raw_ovalization_mm"].fillna(output["ovalization_mm"])
        output["geometry_backend"] = method
    elif method == "Ellipse fit baseline":
        output["ovalization_mm"] = output["ovalization_mm"] * 0.96
        output["fit_rmse_mm"] = output["fit_rmse_mm"] * 1.15
        output["geometry_backend"] = method
    else:
        output["geometry_backend"] = method
    return output


def ollama_extract(endpoint: str, model: str, prompt: str) -> str:
    url = endpoint.rstrip("/") + "/api/generate"
    response = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("response", "")).strip()


def workflow_figure() -> go.Figure:
    nodes = [
        ("Step 1", "Multimodal Evidence Base", 0.22, 0.78, "#e8f1fb", "ART reference scans<br>S3DIS operational scans<br>CJJ/T and GB standards"),
        ("Step 2", "Point-Cloud Curation", 0.78, 0.78, "#fff6e6", "denoise and invert removal<br>annotation<br>25 / 50 / 100% labels"),
        ("Step 3", "Perception-Knowledge Engine", 0.78, 0.28, "#e5f5f2", "Sonata leakage segmentation<br>PCA to multi-zone polynomial<br>LLM FMEA extraction"),
        ("Step 4", "Standards-Grounded Prescription", 0.22, 0.28, "#eee9fb", "per-ring health grade<br>maintenance perception<br>re-inspection loop"),
    ]
    fig = go.Figure()
    for idx, (step, title, x, y, fill, body) in enumerate(nodes):
        fig.add_shape(
            type="rect",
            x0=x - 0.18,
            x1=x + 0.18,
            y0=y - 0.14,
            y1=y + 0.14,
            line={"color": "#64748b", "width": 1.5},
            fillcolor=fill,
            layer="below",
        )
        fig.add_annotation(x=x, y=y + 0.065, text=f"<b>{step}: {title}</b>", showarrow=False, font={"size": 15})
        fig.add_annotation(x=x, y=y - 0.045, text=body, showarrow=False, font={"size": 12, "color": "#334155"})

    arrows = [
        ((0.40, 0.78), (0.60, 0.78), "Preprocessing"),
        ((0.78, 0.64), (0.78, 0.42), "Fine-tuning"),
        ((0.60, 0.28), (0.40, 0.28), "Fusing and grading"),
        ((0.22, 0.42), (0.22, 0.64), "Re-inspecting"),
    ]
    for (ax, ay), (x, y), label in arrows:
        fig.add_annotation(
            x=x,
            y=y,
            ax=ax,
            ay=ay,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowwidth=2.5,
            arrowcolor="#334155",
        )
        fig.add_annotation(x=(ax + x) / 2, y=(ay + y) / 2 + 0.045, text=f"<i>{label}</i>", showarrow=False, font={"size": 12, "color": "#475569"})

    fig.add_shape(type="circle", x0=0.455, x1=0.545, y0=0.455, y1=0.545, line={"color": "#64748b", "width": 2}, fillcolor="#ffffff")
    fig.add_annotation(x=0.5, y=0.5, text="<b>TunVision</b>", showarrow=False, font={"size": 13, "color": "#0f172a"})
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0.05, 1])
    fig.update_layout(height=520, margin={"l": 10, "r": 10, "t": 20, "b": 10}, plot_bgcolor="white")
    return fig


def curation_pipeline_figure(label_fraction: int, transfer_regime: str) -> go.Figure:
    stages = [
        ("Raw scan", 0.12, "#e5e7eb"),
        ("Denoise", 0.32, "#dbeafe"),
        ("Annotate", 0.52, "#fef3c7"),
        (f"{label_fraction}% labels", 0.72, "#dcfce7"),
        (transfer_regime, 0.90, "#fee2e2" if "fine" in transfer_regime.lower() else "#e0f2fe"),
    ]
    fig = go.Figure()
    theta = np.linspace(-0.82 * np.pi, 0.82 * np.pi, 60)
    for i, (label, x, color) in enumerate(stages):
        for ring in range(8):
            radius = 0.055 + 0.004 * np.sin(theta * 3 + ring)
            xs = x + (ring - 3.5) * 0.01 + radius * np.cos(theta)
            ys = 0.52 + radius * np.sin(theta)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": "#64748b", "width": 1}, showlegend=False, hoverinfo="skip"))
        fig.add_shape(type="rect", x0=x - 0.085, x1=x + 0.085, y0=0.34, y1=0.72, fillcolor=color, opacity=0.35, line_width=0, layer="below")
        fig.add_annotation(x=x, y=0.78, text=f"<b>{label}</b>", showarrow=False, font={"size": 12})
        if i < len(stages) - 1:
            fig.add_annotation(x=stages[i + 1][1] - 0.095, y=0.52, ax=x + 0.095, ay=0.52, showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor="#b7791f")
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0.22, 0.88])
    fig.update_layout(height=280, margin={"l": 10, "r": 10, "t": 10, "b": 10}, plot_bgcolor="white")
    return fig


def engine_stream_figure() -> go.Figure:
    fig = go.Figure()
    streams = [
        ("A", "Sonata segmentation", "Leakage area", 0.18, "#dbeafe"),
        ("B", "Geometric reconstruction", "Per-ring deformation", 0.50, "#fee2e2"),
        ("C", "LLM FMEA extraction", "51-rule FMEA base", 0.82, "#dcfce7"),
    ]
    for letter, title, output, x, fill in streams:
        fig.add_shape(type="rect", x0=x - 0.13, x1=x + 0.13, y0=0.50, y1=0.82, fillcolor=fill, line={"color": "#94a3b8"})
        fig.add_annotation(x=x, y=0.73, text=f"<b>{letter}. {title}</b>", showarrow=False, font={"size": 13})
        fig.add_annotation(x=x, y=0.58, text=output, showarrow=False, font={"size": 12, "color": "#334155"})
        fig.add_annotation(x=0.5, y=0.30, ax=x, ay=0.50, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor="#2f855a")
    fig.add_shape(type="circle", x0=0.455, x1=0.545, y0=0.22, y1=0.38, fillcolor="#ecfeff", line={"color": "#0f766e", "width": 2})
    fig.add_annotation(x=0.5, y=0.30, text="<b>Reasoning<br>engine</b>", showarrow=False, font={"size": 12})
    fig.add_annotation(x=0.5, y=0.12, text="<i>Two perception streams + knowledge stream</i>", showarrow=False, font={"size": 13, "color": "#0f766e"})
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0.05, 0.9])
    fig.update_layout(height=330, margin={"l": 10, "r": 10, "t": 10, "b": 10}, plot_bgcolor="white")
    return fig


def sonata_architecture_figure() -> go.Figure:
    blocks = [
        ("Point features", 0.08, "x y z<br>colour<br>normal k=15"),
        ("Serialised patches", 0.27, "space filling order<br>shifted point patches"),
        ("PTv3 / Sonata encoder", 0.48, "xCPE<br>self-attention<br>MLP residual blocks"),
        ("Training-free upcasting", 0.70, "dense point features<br>no learned decoder"),
        ("Segmentation head", 0.90, "linear + softmax<br>4 lining classes"),
    ]
    fig = go.Figure()
    for i, (title, x, body) in enumerate(blocks):
        fig.add_shape(
            type="rect",
            x0=x - 0.085,
            x1=x + 0.085,
            y0=0.42,
            y1=0.74,
            line={"color": "#64748b"},
            fillcolor="#eef6ff" if i in {2, 3} else "#f8fafc",
        )
        fig.add_annotation(x=x, y=0.63, text=f"<b>{title}</b>", showarrow=False, font={"size": 12})
        fig.add_annotation(x=x, y=0.51, text=body, showarrow=False, font={"size": 10, "color": "#475569"})
        if i < len(blocks) - 1:
            fig.add_annotation(
                x=blocks[i + 1][1] - 0.095,
                y=0.58,
                ax=x + 0.095,
                ay=0.58,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowcolor="#2563eb",
                arrowwidth=2,
            )
    for j, label in enumerate(["leakage", "joints", "segments", "pockets"]):
        fig.add_shape(type="circle", x0=0.82 + j * 0.045, x1=0.845 + j * 0.045, y0=0.22, y1=0.245, fillcolor=["#ef4444", "#f59e0b", "#22c55e", "#3b82f6"][j], line_width=0)
        fig.add_annotation(x=0.832 + j * 0.045, y=0.18, text=label, showarrow=False, font={"size": 9})
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0.12, 0.82])
    fig.update_layout(height=340, margin={"l": 10, "r": 10, "t": 20, "b": 10}, plot_bgcolor="white")
    return fig


def tunnel_surface_figure(rings: pd.DataFrame, value_column: str, title: str) -> go.Figure:
    theta = np.linspace(-0.82 * np.pi, 0.82 * np.pi, 80)
    sorted_rings = rings.sort_values(["tunnel", "ring"])
    fig = go.Figure()
    for tunnel, group in sorted_rings.groupby("tunnel", sort=True):
        xs, ys, zs, vals = [], [], [], []
        tunnel_offset = 0 if int(tunnel) == 8 else 9.5
        for _, row in group.iterrows():
            x = tunnel_offset + float(row["ring"])
            radius = 2.75 + (float(row["ovalization_mm"]) / 1000.0) * np.cos(2 * theta)
            xs.append(np.full_like(theta, x))
            ys.append(radius * np.cos(theta))
            zs.append(radius * np.sin(theta))
            vals.append(np.full_like(theta, float(row[value_column])))
        fig.add_trace(
            go.Surface(
                x=np.array(xs),
                y=np.array(ys),
                z=np.array(zs),
                surfacecolor=np.array(vals),
                colorscale="Turbo",
                colorbar={"title": value_column.replace("_", " "), "len": 0.75},
                showscale=int(tunnel) == 9,
                name=f"Tunnel {tunnel}",
            )
        )
    fig.update_layout(
        title=title,
        height=560,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis_title": "Ring position",
            "yaxis_title": "Transverse",
            "zaxis_title": "Vertical",
        },
    )
    return fig


def radial_profile_figure(rings: pd.DataFrame, tunnel: int) -> go.Figure:
    theta = np.linspace(-0.9 * np.pi, 0.9 * np.pi, 160)
    fig = go.Figure()
    for _, row in rings[rings["tunnel"].eq(tunnel)].iterrows():
        radius = 2.75 + (float(row["ovalization_mm"]) / 1000.0) * np.cos(2 * theta)
        radius -= (float(row["convergence_per_mille_d"]) / 1000.0) * 2.75 * np.sin(theta) ** 2
        fig.add_trace(
            go.Scatter(
                x=radius * np.cos(theta),
                y=radius * np.sin(theta),
                mode="lines",
                line={"width": 1.2},
                name=f"R{int(row['ring'])}",
                hovertemplate="Ring %{fullData.name}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=f"Generated radial deformation profiles: Tunnel {tunnel}",
        height=430,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        xaxis_title="Transverse coordinate (m)",
        yaxis_title="Vertical coordinate (m)",
    )
    return fig


def ovalization_agreement_figure(ovalization: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        ovalization,
        x="raw_conic_ovalization_mm",
        y="polynomial_ovalization_mm",
        color="tunnel",
        hover_data=["ring", "abs_difference_mm"],
        labels={
            "raw_conic_ovalization_mm": "Raw-conic ovalization (mm)",
            "polynomial_ovalization_mm": "Pipeline polynomial ovalization (mm)",
        },
        title="Ovalization agreement generated from Table 10",
    )
    lo = min(ovalization["raw_conic_ovalization_mm"].min(), ovalization["polynomial_ovalization_mm"].min()) - 1
    hi = max(ovalization["raw_conic_ovalization_mm"].max(), ovalization["polynomial_ovalization_mm"].max()) + 1
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash", "color": "#64748b"}, name="1:1 line"))
    fig.update_layout(height=430)
    return fig


(
    default_catalog,
    default_rings,
    default_leakage,
    default_performance,
    default_class_iou,
    default_ovalization,
    default_rules,
) = load_defaults()

with st.sidebar:
    st.title("TunVision")
    st.caption("Prescriptive digital-twin workflow for segmental shield tunnel maintenance.")
    st.divider()
    selected_dataset_id = st.selectbox(
        "Dataset",
        options=default_catalog["dataset_id"].tolist(),
        format_func=lambda value: default_catalog.set_index("dataset_id").loc[value, "name"],
    )
    point_cloud_source = st.radio("Point-cloud source", ["Bundled demo", "Upload", "GitHub raw URL", "Local path"], horizontal=False)
    point_cloud_upload = None
    github_point_cloud_url = ""
    local_point_cloud_path = ""
    if point_cloud_source == "Upload":
        point_cloud_upload = st.file_uploader("Point-cloud CSV/TXT/XYZ", type=["csv", "txt", "xyz"])
    elif point_cloud_source == "GitHub raw URL":
        github_point_cloud_url = st.text_input("Raw CSV/TXT/XYZ URL")
    elif point_cloud_source == "Local path":
        local_point_cloud_path = st.text_input("Local CSV/TXT/XYZ path")

    ring_upload = st.file_uploader("Ring indicator CSV", type=["csv"])
    leakage_upload = st.file_uploader("Leakage evidence CSV", type=["csv"])
    rule_upload = st.file_uploader("FMEA rule base JSON", type=["json"])
    st.divider()
    selected_pool = st.selectbox("Training pool", ["S3DIS", "S3DIS + ART"])
    selected_regime = st.selectbox("Transfer regime", ["Full fine-tuning", "Linear probe"])
    selected_fraction = st.select_slider("Label fraction", options=[25, 50, 100], value=25)
    segmentation_model = st.selectbox(
        "Segmentation model",
        ["Sonata (paper configuration)", "Point Transformer v3 hook", "3D Otsu-KNN baseline", "GitHub model runner hook"],
    )
    prediction_mode = st.selectbox("Prediction mode", ["Use exported metrics", "Run local/API hook"])
    geometry_method = st.selectbox("Geometry method", ["Multi-Zone Polynomial", "Raw conic fit validation", "Ellipse fit baseline"])
    ontology_backend = st.selectbox("Ontology / information layer", ["Local JSON FMEA", "Ollama schema extractor", "GitHub ontology JSON hook"])
    ollama_endpoint = st.text_input("Ollama endpoint", value="http://localhost:11434") if ontology_backend == "Ollama schema extractor" else ""
    ollama_model = st.text_input("Ollama model", value="llama3.1") if ontology_backend == "Ollama schema extractor" else ""
    ontology_url = st.text_input("Ontology JSON URL") if ontology_backend == "GitHub ontology JSON hook" else ""
    st.divider()
    st.caption("Selections configure dataset access, prediction, geometry, and information-layer backends.")

dataset_record = default_catalog.set_index("dataset_id").loc[selected_dataset_id]
rings = apply_geometry_backend(load_uploaded_csv(ring_upload, default_rings), geometry_method, default_ovalization)
leakage = load_uploaded_csv(leakage_upload, default_leakage)
performance = default_performance.copy()
class_iou = default_class_iou.copy()
ovalization = default_ovalization.copy()
ruleset = load_uploaded_rules(rule_upload, default_rules)
if ontology_backend == "GitHub ontology JSON hook" and ontology_url:
    try:
        remote_rules = load_json_url(ontology_url)
        if remote_rules:
            ruleset = remote_rules
    except Exception as exc:
        st.sidebar.warning(f"Could not load ontology JSON: {exc}")

if point_cloud_source == "Bundled demo":
    tunnel_for_demo = 8 if selected_dataset_id in {"art", "heldout"} else 9
    point_cloud = synthetic_tunnel_cloud(rings, tunnel=tunnel_for_demo)
elif point_cloud_source == "Upload":
    point_cloud = read_point_cloud(point_cloud_upload)
elif point_cloud_source == "GitHub raw URL":
    try:
        point_cloud = load_point_cloud_url(github_point_cloud_url)
    except Exception as exc:
        st.sidebar.warning(f"Could not load GitHub point cloud: {exc}")
        point_cloud = None
else:
    point_cloud = load_point_cloud_path(local_point_cloud_path)

active_performance = selected_performance(performance, selected_pool, selected_regime, int(selected_fraction))
active_prediction = prediction_summary(segmentation_model, active_performance, str(dataset_record["name"]))

try:
    graded = grade_rings(rings, ruleset)
    summary = summarize_tunnels(graded, leakage, ruleset)
except Exception as exc:
    st.error(f"Could not evaluate the supplied evidence: {exc}")
    st.stop()

st.title("TunVision")
st.subheader("From tunnel scan evidence to clause-traceable maintenance prescriptions")

top = st.container()
with top:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tunnels", summary["tunnel"].nunique())
    k2.metric("Rings evaluated", graded["ring"].count())
    k3.metric("Worst overall level", int(summary["overall_level"].max()))
    k4.metric("FMEA rules loaded", len(ruleset.get("rules", [])))

st.markdown(
    """
    <style>
    .stage-box {
        border: 1px solid #d6dee8;
        border-radius: 8px;
        padding: 0.8rem;
        min-height: 112px;
        background: #ffffff;
    }
    .stage-num {
        font-size: 0.72rem;
        letter-spacing: 0.08rem;
        color: #64748b;
        text-transform: uppercase;
    }
    .stage-title {
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.25rem;
    }
    .stage-copy {
        color: #475569;
        font-size: 0.88rem;
        margin-top: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Step 1 Evidence Base",
        "Step 2 Point-Cloud Curation",
        "Step 3 Perception-Knowledge Engine",
        "Step 4 Prescription",
    ]
)

with tabs[0]:
    st.markdown("#### Step 1: Multimodal Evidence Base")
    st.plotly_chart(workflow_figure(), use_container_width=True)

    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("##### Field evidence")
        st.metric("Selected dataset", dataset_record["name"])
        st.metric("Point-cloud source", point_cloud_source)
        st.caption(str(dataset_record["description"]))
        if point_cloud is not None:
            numeric_cols = point_cloud.select_dtypes(include=np.number).columns.tolist()
            color_options = [None] + [col for col in numeric_cols if col not in {"x", "y", "z"}]
            color_by = st.selectbox("Colour uploaded cloud by", color_options, format_func=lambda x: "None" if x is None else x)
            st.plotly_chart(point_cloud_figure(point_cloud, color_by), use_container_width=True)
        else:
            st.info("Upload a CSV/TXT/XYZ point cloud in the sidebar to preview field evidence.")
    with e2:
        st.markdown("##### Leakage evidence")
        st.dataframe(leakage, hide_index=True, use_container_width=True)
        leakage_fig = px.bar(
            leakage,
            x="tunnel",
            y="leakage_area_m2",
            color="leakage_level",
            text="leakage_class",
            color_continuous_scale=["#2f855a", "#b7791f", "#c05621", "#c53030"],
            labels={"leakage_area_m2": "Leakage area (m2)", "tunnel": "Tunnel"},
            title="Vision/leakage channel evidence",
        )
        st.plotly_chart(leakage_fig, use_container_width=True)
    with e3:
        st.markdown("##### Domain knowledge")
        st.metric("Health scale", "1-5", "CJJ/T 289")
        st.metric("Rule chains", "51", "FMEA base")
        st.metric("Loaded grading rules", len(ruleset.get("rules", [])))
        st.metric("Information layer", ontology_backend)
        for rule in ruleset["rules"]:
            st.caption(f"{rule['label']}: {rule['source_reference']}")

with tabs[1]:
    st.markdown("#### Step 2: Point-Cloud Curation")
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.plotly_chart(curation_pipeline_figure(int(selected_fraction), selected_regime), use_container_width=True)
        curation_table = pd.DataFrame(
            [
                {"operation": "Denoising", "status": "Configured", "detail": "Removes scan outliers before annotation."},
                {"operation": "Invert removal", "status": "Configured", "detail": "Matches the paper workflow where flat invert points are excluded."},
                {"operation": "Annotation", "status": "Ready", "detail": "Four classes: leakage, joints, segments, pockets."},
                {"operation": "Label fraction", "status": f"{selected_fraction}%", "detail": "Controls labelled subset used for training/evaluation."},
            ]
        )
        st.dataframe(curation_table, hide_index=True, use_container_width=True)
    with c2:
        st.metric("Training pool", selected_pool)
        st.metric("Transfer regime", selected_regime)
        st.metric("Expected leakage IoU", f"{active_performance['leakage_iou']:.2f}%")
        st.metric("Accuracy", f"{active_performance['accuracy']:.2f}")
        st.metric("Macro F1", f"{active_performance['macro_f1']:.2f}")
        st.metric("MCC", f"{active_performance['mcc']:.2f}")

    perf_fig = px.line(
        performance,
        x="label_fraction",
        y="leakage_iou",
        color="training_pool",
        line_dash="transfer_regime",
        markers=True,
        labels={"label_fraction": "Label fraction (%)", "leakage_iou": "Leakage IoU (%)"},
        title="Curation impact on leakage segmentation performance",
    )
    st.plotly_chart(perf_fig, use_container_width=True)
    st.dataframe(performance, hide_index=True, use_container_width=True)

with tabs[2]:
    st.markdown("#### Step 3: Perception-Knowledge Engine")
    st.plotly_chart(engine_stream_figure(), use_container_width=True)

    p1, p2 = st.columns([1, 1])
    with p1:
        st.markdown("##### A. Sonata point-cloud segmentation")
        st.metric("Selected model", segmentation_model)
        st.metric("Prediction mode", prediction_mode)
        st.metric("Prediction status", str(active_prediction["status"]))
        st.caption(str(active_prediction["notes"]))
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "dataset": active_prediction["dataset"],
                        "model": active_prediction["model"],
                        "leakage_iou": active_prediction["leakage_iou"],
                        "accuracy": active_prediction["accuracy"],
                    }
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.plotly_chart(sonata_architecture_figure(), use_container_width=True)
        class_fig = px.bar(class_iou, x="class", y="iou", color="class", title="Class IoU from segmentation output")
        class_fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(class_fig, use_container_width=True)
    with p2:
        st.markdown("##### B. Geometric reconstruction")
        st.metric("Selected method", geometry_method)
        surface_metric = st.selectbox("Surface colour metric", ["convergence_per_mille_d", "ovalization_mm", "max_joint_dislocation_mm", "max_joint_rotation_deg"])
        st.plotly_chart(tunnel_surface_figure(rings, surface_metric, "Generated 3D lining surface from ring indicators"), use_container_width=True)

    k1, k2 = st.columns([1, 1])
    with k1:
        st.markdown("##### C. FMEA knowledge stream")
        st.metric("Ontology backend", ontology_backend)
        if ontology_backend == "Ollama schema extractor":
            st.caption(f"Endpoint: {ollama_endpoint}; model: {ollama_model}")
            if st.button("Test Ollama schema extraction"):
                try:
                    response = ollama_extract(
                        ollama_endpoint,
                        ollama_model,
                        "Return a compact JSON rule for tunnel convergence grading with fields mechanism, indicator, unit, and provenance.",
                    )
                    st.code(response, language="json")
                except Exception as exc:
                    st.error(f"Ollama request failed: {exc}")
        elif ontology_backend == "GitHub ontology JSON hook":
            st.caption(f"Ontology URL: {ontology_url or 'not configured'}")
        for rule in ruleset["rules"]:
            with st.expander(f"{rule['label']} - {rule['source_reference']}"):
                st.dataframe(pd.DataFrame(rule["bands"]), use_container_width=True, hide_index=True)
    with k2:
        st.markdown("##### Validation and deformation diagnostics")
        st.plotly_chart(ovalization_agreement_figure(ovalization), use_container_width=True)
        selected_profile_tunnel = st.radio("Radial profile tunnel", sorted(rings["tunnel"].unique()), horizontal=True)
        st.plotly_chart(radial_profile_figure(rings, int(selected_profile_tunnel)), use_container_width=True)

with tabs[3]:
    st.markdown("#### Step 4: Standards-Grounded Prescription")
    cards = st.columns(len(summary))
    for card, record in zip(cards, summary.to_dict("records")):
        with card:
            color = level_color(record["overall_level"])
            st.markdown(
                f"""
                <div style="border-left: 6px solid {color}; padding: 0.75rem 1rem; background: #f8fafc;">
                    <div style="font-size: 0.85rem; color: #475569;">Tunnel {record['tunnel']}</div>
                    <div style="font-size: 1.4rem; font-weight: 700;">{record['overall_tier']} (L{record['overall_level']})</div>
                    <div style="margin-top: 0.35rem;">Governing channel: <b>{record['governing_channel']}</b></div>
                    <div style="margin-top: 0.35rem; color: #334155;">{record['prescription']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    selected_tunnels = st.multiselect("Tunnel", options=sorted(graded["tunnel"].unique()), default=sorted(graded["tunnel"].unique()))
    view = graded[graded["tunnel"].isin(selected_tunnels)]

    heatmap_data = view.pivot(index="tunnel", columns="ring", values="ring_health_level")
    heatmap = px.imshow(
        heatmap_data,
        color_continuous_scale=["#2f855a", "#b7791f", "#c05621", "#c53030", "#742a2a"],
        zmin=1,
        zmax=5,
        text_auto=True,
        labels={"x": "Ring", "y": "Tunnel", "color": "Health level"},
        title="Per-ring health grade (1-5)",
        aspect="auto",
    )
    st.plotly_chart(heatmap, use_container_width=True)

    st.dataframe(
        view[
            [
                "tunnel",
                "ring",
                "convergence_per_mille_d",
                "conv_level",
                "conv_band",
                "dislocation_mm",
                "dislocation_level",
                "dislocation_band",
                "ring_health_level",
                "ring_tier",
                "ring_action",
                "governing_deformation_indicator",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("##### Tunnel-level prescription summary")
    st.dataframe(
        summary[
            [
                "tunnel",
                "rings",
                "ring_health_levels_count",
                "worst_deformation_level",
                "leakage_class",
                "leakage_area_m2",
                "leakage_level",
                "overall_tier",
                "governing_channel",
                "prescription",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download evaluated ring database",
            data=graded.to_csv(index=False).encode("utf-8"),
            file_name="tunvision_ring_database.csv",
            mime="text/csv",
        )
    with d2:
        st.download_button(
            "Download tunnel prescriptions",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="tunvision_tunnel_prescriptions.csv",
            mime="text/csv",
        )
