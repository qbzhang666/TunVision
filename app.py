from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from tunvision.reasoning import grade_rings, summarize_tunnels


ROOT = Path(__file__).parent
DATA = ROOT / "data"


st.set_page_config(page_title="TunVision", page_icon="TV", layout="wide")


@st.cache_data
def load_defaults() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rings = pd.read_csv(DATA / "ring_indicators.csv")
    leakage = pd.read_csv(DATA / "leakage_evidence.csv")
    rules = json.loads((DATA / "fmea_rules.json").read_text(encoding="utf-8"))
    return rings, leakage, rules


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


default_rings, default_leakage, default_rules = load_defaults()

with st.sidebar:
    st.title("TunVision")
    st.caption("Prescriptive digital-twin workflow for segmental shield tunnel maintenance.")
    st.divider()
    ring_upload = st.file_uploader("Ring indicator CSV", type=["csv"])
    leakage_upload = st.file_uploader("Leakage evidence CSV", type=["csv"])
    rule_upload = st.file_uploader("FMEA rule base JSON", type=["json"])
    st.divider()
    st.caption("Defaults reproduce the held-out Tunnel 8 and Tunnel 9 examples from the paper draft.")

rings = load_uploaded_csv(ring_upload, default_rings)
leakage = load_uploaded_csv(leakage_upload, default_leakage)
ruleset = load_uploaded_rules(rule_upload, default_rules)

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

tabs = st.tabs(["Decision Dashboard", "Ring Evidence", "Perception Context", "Rule Base"])

with tabs[0]:
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

with tabs[1]:
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

with tabs[2]:
    st.markdown("#### Point-cloud perception evidence")
    st.write(
        "TunVision expects segmentation or inspection outputs as evidence. "
        "The manuscript configuration uses Sonata for four classes: leakage, joints, segments, and pockets."
    )
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Leakage IoU", "94.34%", "full fine-tuning")
    p2.metric("Mean accuracy", "98.60%")
    p3.metric("Low-label leakage IoU", "87.59%", "25% labels")
    p4.metric("FMEA chains", "51")
    st.dataframe(leakage, use_container_width=True, hide_index=True)
    st.info(
        "Direct Sonata inference is intentionally separated from this lightweight app. "
        "Export leakage class, area, and source provenance from the perception pipeline, then ingest it here."
    )

with tabs[3]:
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
