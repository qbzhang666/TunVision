from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tunvision.reasoning import grade_rings, summarize_tunnels


ROOT = Path(__file__).parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"


st.set_page_config(page_title="TunVision", page_icon="TV", layout="wide")


@st.cache_data
def load_defaults() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rings = pd.read_csv(DATA / "ring_indicators.csv")
    leakage = pd.read_csv(DATA / "leakage_evidence.csv")
    performance = pd.read_csv(DATA / "model_performance.csv")
    class_iou = pd.read_csv(DATA / "class_iou.csv")
    rules = json.loads((DATA / "fmea_rules.json").read_text(encoding="utf-8"))
    return rings, leakage, performance, class_iou, rules


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


default_rings, default_leakage, default_performance, default_class_iou, default_rules = load_defaults()

with st.sidebar:
    st.title("TunVision")
    st.caption("Prescriptive digital-twin workflow for segmental shield tunnel maintenance.")
    st.divider()
    point_cloud_upload = st.file_uploader("Point-cloud preview CSV/TXT/XYZ", type=["csv", "txt", "xyz"])
    ring_upload = st.file_uploader("Ring indicator CSV", type=["csv"])
    leakage_upload = st.file_uploader("Leakage evidence CSV", type=["csv"])
    rule_upload = st.file_uploader("FMEA rule base JSON", type=["json"])
    st.divider()
    st.caption("Defaults reproduce the held-out Tunnel 8 and Tunnel 9 examples from the paper draft.")

rings = load_uploaded_csv(ring_upload, default_rings)
leakage = load_uploaded_csv(leakage_upload, default_leakage)
performance = default_performance.copy()
class_iou = default_class_iou.copy()
ruleset = load_uploaded_rules(rule_upload, default_rules)
point_cloud = read_point_cloud(point_cloud_upload)

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
        "Workflow",
        "Point Cloud + ML",
        "Deformation Reconstruction",
        "Decision Dashboard",
        "Ring Evidence",
        "Rule Base",
    ]
)

with tabs[0]:
    st.markdown("#### Paper workflow implemented in TunVision")
    stages = [
        ("Stage 1", "Multimodal Evidence Base", "ART + S3DIS point clouds, image/leakage evidence, and maintenance standards."),
        ("Stage 2", "Preprocessing", "CloudCompare cleaning, ring partitioning, feature preparation, and train/test split."),
        ("Stage 3", "Point-Cloud Segmentation", "Sonata encoder with upcasting and a linear segmentation head for leakage, joints, segments, and pockets."),
        ("Stage 4", "Geometric Evaluation", "PCA alignment, robust fixed-radius fitting, multi-zone polynomial reconstruction, and deformation indicators."),
        ("Stage 5", "Prescriptive Reasoning", "Schema-guided FMEA extraction and max-severity fusion of leakage and deformation channels."),
    ]
    cols = st.columns(5)
    for col, (num, title, copy) in zip(cols, stages):
        with col:
            st.markdown(
                f"""
                <div class="stage-box">
                    <div class="stage-num">{num}</div>
                    <div class="stage-title">{title}</div>
                    <div class="stage-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    image_path = ASSETS / "workflow_page.png"
    if image_path.exists():
        st.image(str(image_path), caption="Rendered workflow page from the manuscript PDF", use_container_width=True)

with tabs[1]:
    st.markdown("#### Point-cloud perception and machine-learning stage")
    st.write(
        "This tab mirrors the paper's perception branch: cleaned tunnel point clouds are segmented by Sonata into "
        "leakage, joints, segments, and pockets. TunVision ingests exported model evidence and can preview XYZ/CSV point clouds."
    )

    left, right = st.columns([1.25, 1])
    with left:
        if point_cloud is not None:
            numeric_cols = point_cloud.select_dtypes(include=np.number).columns.tolist()
            color_options = [None] + [col for col in numeric_cols if col not in {"x", "y", "z"}]
            color_by = st.selectbox("Colour point-cloud preview by", color_options, format_func=lambda x: "None" if x is None else x)
            st.plotly_chart(point_cloud_figure(point_cloud, color_by), use_container_width=True)
        else:
            sonar_page = ASSETS / "sonata_page.png"
            if sonar_page.exists():
                st.image(str(sonar_page), caption="Sonata architecture and point feature flow from the manuscript PDF", use_container_width=True)
            st.info("Upload a CSV/TXT/XYZ point cloud in the sidebar to preview scan evidence here.")

    with right:
        best = performance.sort_values("leakage_iou", ascending=False).iloc[0]
        st.metric("Best leakage IoU", f"{best['leakage_iou']:.2f}%", best["training_pool"])
        st.metric("Best mean accuracy", f"{best['mean_accuracy']:.2f}%")
        st.metric("Classes", "4", "leakage, joints, segments, pockets")
        st.dataframe(class_iou, hide_index=True, use_container_width=True)

    perf_fig = px.line(
        performance,
        x="label_fraction",
        y="leakage_iou",
        color="training_pool",
        line_dash="transfer_regime",
        markers=True,
        labels={"label_fraction": "Label fraction (%)", "leakage_iou": "Leakage IoU (%)"},
        title="Leakage IoU across transfer regimes and label fractions",
    )
    st.plotly_chart(perf_fig, use_container_width=True)
    st.dataframe(performance, hide_index=True, use_container_width=True)

with tabs[2]:
    st.markdown("#### Multi-zone polynomial deformation reconstruction")
    st.write(
        "The deformation branch reduces each reconstructed ring to clearance convergence, ovalization, joint dislocation, "
        "joint rotation, and fit quality before standards-based grading."
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Mean convergence", f"{rings['convergence_per_mille_d'].mean():.2f} per mille D")
    metric_cols[1].metric("Max ovalization", f"{rings['ovalization_mm'].max():.1f} mm")
    metric_cols[2].metric("Max dislocation", f"{rings['max_joint_dislocation_mm'].max():.1f} mm")
    metric_cols[3].metric("Max rotation", f"{rings['max_joint_rotation_deg'].max():.2f} deg")
    metric_cols[4].metric("Mean fit RMSE", f"{rings['fit_rmse_mm'].mean():.2f} mm")

    c1, c2 = st.columns([1, 1])
    with c1:
        deformation_fig = px.line(
            rings,
            x="ring",
            y=["convergence_per_mille_d", "ovalization_mm", "max_joint_dislocation_mm", "max_joint_rotation_deg"],
            facet_col="tunnel",
            markers=True,
            labels={"value": "Value", "variable": "Indicator"},
            title="Per-ring deformation indicators from Table 9",
        )
        st.plotly_chart(deformation_fig, use_container_width=True)
    with c2:
        render_page = ASSETS / "deformation_render_page.png"
        if render_page.exists():
            st.image(str(render_page), caption="3D radial-deviation, dislocation and rotation rendering from the PDF", use_container_width=True)

    image_cols = st.columns(3)
    for col, filename, caption in [
        (image_cols[0], "ovalization_page.png", "Polynomial vs raw-conic ovalization agreement"),
        (image_cols[1], "tunnel8_radial_page.png", "Tunnel 8 radial deformation sections"),
        (image_cols[2], "tunnel9_radial_page.png", "Tunnel 9 radial deformation sections"),
    ]:
        path = ASSETS / filename
        if path.exists():
            col.image(str(path), caption=caption, use_container_width=True)

with tabs[3]:
    st.markdown("#### Tunnel-level prescription")
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
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        graded,
        x="ring",
        y="ring_health_level",
        color="ring_health_level",
        facet_col="tunnel",
        color_continuous_scale=["#2f855a", "#b7791f", "#c05621", "#c53030", "#742a2a"],
        labels={"ring": "Ring", "ring_health_level": "Health level"},
        title="Ring health levels along each tunnel",
    )
    fig.update_yaxes(dtick=1, range=[0, 5])
    st.plotly_chart(fig, use_container_width=True)

    decision_page = ASSETS / "decision_summary_page.png"
    if decision_page.exists():
        st.image(str(decision_page), caption="Decision summary page from the manuscript PDF", use_container_width=True)

with tabs[4]:
    st.markdown("#### Clause-traceable ring grading")
    selected_tunnels = st.multiselect(
        "Tunnel",
        options=sorted(graded["tunnel"].unique()),
        default=sorted(graded["tunnel"].unique()),
    )
    view = graded[graded["tunnel"].isin(selected_tunnels)]
    st.dataframe(
        view[
            [
                "tunnel",
                "ring",
                "convergence_per_mille_d",
                "ovalization_mm",
                "max_joint_dislocation_mm",
                "max_joint_rotation_deg",
                "fit_rmse_mm",
                "conv_level",
                "conv_band",
                "dislocation_mm",
                "dislocation_level",
                "dislocation_band",
                "ring_health_level",
                "ring_tier",
                "governing_deformation_indicator",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    chart = px.line(
        view,
        x="ring",
        y=["convergence_per_mille_d", "dislocation_mm"],
        facet_col="tunnel",
        markers=True,
        labels={"value": "Indicator value", "variable": "Indicator", "ring": "Ring"},
        title="Deformation indicators before standards-based grading",
    )
    st.plotly_chart(chart, use_container_width=True)

with tabs[5]:
    st.markdown("#### Standards-grounded decision rules")
    for rule in ruleset["rules"]:
        with st.expander(f"{rule['label']} - {rule['source_reference']}"):
            st.dataframe(pd.DataFrame(rule["bands"]), use_container_width=True, hide_index=True)
    st.download_button(
        "Download evaluated ring decisions",
        data=graded.to_csv(index=False).encode("utf-8"),
        file_name="tunvision_ring_decisions.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download tunnel prescriptions",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="tunvision_tunnel_prescriptions.csv",
        mime="text/csv",
    )
