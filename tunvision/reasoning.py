from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class GradeResult:
    level: int
    band: str
    source_reference: str


def _matches_band(value: float, band: dict[str, Any]) -> bool:
    lower = band.get("min")
    upper = band.get("max")
    above_lower = True if lower is None else value >= float(lower)
    below_upper = True if upper is None else value <= float(upper)
    if band.get("lower_inclusive") is False and lower is not None:
        above_lower = value > float(lower)
    if band.get("upper_inclusive") is False and upper is not None:
        below_upper = value < float(upper)
    if lower not in (None, 0):
        above_lower = value > float(lower)
        if band.get("lower_inclusive") is True:
            above_lower = value >= float(lower)
    return above_lower and below_upper


def grade_indicator(value: float, rule: dict[str, Any]) -> GradeResult:
    for band in rule["bands"]:
        if _matches_band(value, band):
            return GradeResult(
                level=int(band["level"]),
                band=str(band["band"]),
                source_reference=str(rule["source_reference"]),
            )
    last = rule["bands"][-1]
    return GradeResult(
        level=int(last["level"]),
        band=str(last["band"]),
        source_reference=str(rule["source_reference"]),
    )


def grade_rings(rings: pd.DataFrame, ruleset: dict[str, Any]) -> pd.DataFrame:
    required = {"tunnel", "ring", "convergence_per_mille_d", "dislocation_mm"}
    missing = required.difference(rings.columns)
    if missing:
        raise ValueError(f"Ring indicator data is missing columns: {', '.join(sorted(missing))}")

    rules = {rule["indicator"]: rule for rule in ruleset["rules"]}
    rows: list[dict[str, Any]] = []
    for record in rings.to_dict("records"):
        conv = grade_indicator(float(record["convergence_per_mille_d"]), rules["convergence_per_mille_d"])
        disloc = grade_indicator(float(record["dislocation_mm"]), rules["dislocation_mm"])
        health_level = max(conv.level, disloc.level)
        governing = "Convergence" if conv.level >= disloc.level else "Joint dislocation"
        rows.append(
            {
                **record,
                "conv_level": conv.level,
                "conv_band": conv.band,
                "conv_source": conv.source_reference,
                "dislocation_level": disloc.level,
                "dislocation_band": disloc.band,
                "dislocation_source": disloc.source_reference,
                "ring_health_level": health_level,
                "ring_tier": ruleset["health_scale"][str(health_level)]["tier"],
                "ring_action": ruleset["health_scale"][str(health_level)]["action"],
                "governing_deformation_indicator": governing,
            }
        )
    return pd.DataFrame(rows)


def summarize_tunnels(graded_rings: pd.DataFrame, leakage: pd.DataFrame, ruleset: dict[str, Any]) -> pd.DataFrame:
    required = {"tunnel", "leakage_class", "leakage_area_m2", "leakage_level", "source_reference"}
    missing = required.difference(leakage.columns)
    if missing:
        raise ValueError(f"Leakage evidence is missing columns: {', '.join(sorted(missing))}")

    summaries: list[dict[str, Any]] = []
    leakage_by_tunnel = leakage.set_index("tunnel").to_dict("index")
    for tunnel, group in graded_rings.groupby("tunnel", sort=True):
        worst_deformation = int(group["ring_health_level"].max())
        leakage_record = leakage_by_tunnel.get(tunnel) or leakage_by_tunnel.get(str(tunnel), {})
        leakage_level = int(leakage_record.get("leakage_level", 1))
        overall_level = max(worst_deformation, leakage_level)
        governing_channel = "Leakage" if leakage_level > worst_deformation else "Deformation"
        counts = group["ring_health_level"].value_counts().sort_index()
        count_text = ", ".join(f"L{level} x {count}" for level, count in counts.items())
        summaries.append(
            {
                "tunnel": tunnel,
                "rings": int(group["ring"].nunique()),
                "ring_health_levels_count": count_text,
                "worst_deformation_level": worst_deformation,
                "leakage_class": leakage_record.get("leakage_class", "Not supplied"),
                "leakage_area_m2": float(leakage_record.get("leakage_area_m2", 0)),
                "leakage_level": leakage_level,
                "overall_level": overall_level,
                "overall_tier": ruleset["health_scale"][str(overall_level)]["tier"],
                "prescription": ruleset["health_scale"][str(overall_level)]["action"],
                "governing_channel": governing_channel,
                "leakage_source": leakage_record.get("source_reference", "Not supplied"),
            }
        )
    return pd.DataFrame(summaries)
