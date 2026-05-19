"""Composants Streamlit réutilisables."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.utils import format_currency_eur, format_number, format_percent


def kpi_card(label: str, value: str, help_text: str | None = None) -> None:
    """Bloc KPI uniforme."""
    st.metric(label=label, value=value, help=help_text)


def kpi_row(metrics: list[dict[str, Any]]) -> None:
    """Affiche une ligne de KPIs (label/value/help)."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            kpi_card(m["label"], m["value"], m.get("help"))


def risk_badge(level: str) -> str:
    """Renvoie un badge HTML coloré pour un niveau de risque."""
    palette = {
        "faible": ("#2e7d32", "#e8f5e9"),
        "modéré": ("#ef6c00", "#fff3e0"),
        "élevé": ("#c62828", "#ffebee"),
    }
    color, bg = palette.get(level, ("#37474f", "#eceff1"))
    return (
        f'<span style="background:{bg};color:{color};padding:4px 10px;'
        'border-radius:12px;font-weight:600;font-size:0.85em;">'
        f"Risque {level}</span>"
    )


def format_kpi_dict(value: float, kind: str) -> str:
    """Wrapper pour appliquer le bon formatage selon le type de KPI."""
    if kind == "currency":
        return format_currency_eur(value)
    if kind == "percent":
        return format_percent(value)
    return format_number(value)
