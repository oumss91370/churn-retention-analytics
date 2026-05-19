"""Dashboard Streamlit du système de rétention client (orienté responsable CRM)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from app.components import format_kpi_dict, kpi_card, kpi_row, risk_badge
from app.utils import (
    BUSINESS_LABELS,
    feature_name_to_business_label,
    format_currency_eur,
    format_number,
    format_percent,
    proba_to_risk_level,
    recommendation_from_factors,
    revenue_at_risk,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API = os.environ.get("CHURN_API_URL", "http://127.0.0.1:8000")
RAW_CSV = ROOT / "data" / "raw" / "customer_churn.csv"
FIGURES_DIR = ROOT / "reports" / "figures"

st.set_page_config(
    page_title="Churn Retention Analytics",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stMetric { background-color: #f8f9fb; padding: 12px; border-radius: 8px; }
    section.main > div { padding-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    df["complaint_type"] = df["complaint_type"].fillna("None")
    return df


@st.cache_data(show_spinner=False, ttl=300)
def fetch_model_info(api_url: str) -> dict | None:
    try:
        r = requests.get(f"{api_url}/model-info", timeout=5)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        return None
    return None


@st.cache_data(show_spinner=False, ttl=300)
def fetch_health(api_url: str) -> dict | None:
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        return None
    return None


def call_predict(api_url: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json()
        st.error(f"L'API a renvoyé {r.status_code} : {r.text}")
    except requests.RequestException as exc:
        st.error(f"Impossible de contacter l'API : {exc}")
    return None


@st.cache_data(show_spinner=False)
def score_dataset(api_url: str, sample_size: int = 500) -> pd.DataFrame:
    """Scoring d'un échantillon du dataset pour les vues agrégées.

    On envoie chaque ligne à `/predict`. Pour rester rapide on prend `sample_size`
    clients aléatoires. Les colonnes inutiles (`customer_id`, `churn`) sont
    retirées du payload mais conservées pour l'affichage.
    """
    df = load_dataset().copy()
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    keep_cols = [c for c in df.columns if c not in ("customer_id", "churn")]
    proba: list[float] = []
    risk: list[str] = []
    top: list[list[dict]] = []
    failed = 0
    for _, row in df.iterrows():
        payload = row[keep_cols].to_dict()
        for k, v in list(payload.items()):
            if hasattr(v, "item"):
                payload[k] = v.item()
        result = None
        try:
            r = requests.post(f"{api_url}/predict", json=payload, timeout=5)
            if r.status_code == 200:
                result = r.json()
        except requests.RequestException:
            result = None
        if result is None:
            failed += 1
            proba.append(0.0)
            risk.append("faible")
            top.append([])
        else:
            proba.append(result["churn_probability"])
            risk.append(result["risk_level"])
            top.append(result["top_factors"])

    df["churn_probability"] = proba
    df["risk_level"] = risk
    df["top_factors"] = top
    df["_failed_calls"] = failed
    return df


def page_overview(api_url: str) -> None:
    st.title("Vue d'ensemble — Rétention client")
    df = load_dataset()
    info = fetch_model_info(api_url)
    threshold = float(info["decision_threshold"]) if info else 0.35

    scored = score_dataset(api_url, sample_size=500)
    if scored.empty:
        st.warning("Impossible de récupérer les prédictions. Vérifiez que l'API tourne.")
        return

    failed = int(scored["_failed_calls"].iloc[0]) if "_failed_calls" in scored.columns else 0
    if failed > 0:
        st.warning(f"{failed} appels API ont échoué sur cet échantillon (réessayer).")

    at_risk = scored[scored["churn_probability"] >= threshold]
    revenue_at_risk_total = float((scored["churn_probability"] * scored["total_revenue"]).sum())
    historical_churn = float(df["churn"].mean())
    predicted_churn_rate = float((scored["churn_probability"] >= threshold).mean())

    kpi_row(
        [
            {"label": "Clients (échantillon)", "value": format_kpi_dict(len(scored), "count")},
            {"label": "Clients à risque", "value": format_kpi_dict(len(at_risk), "count")},
            {"label": "Revenu pondéré à risque", "value": format_kpi_dict(revenue_at_risk_total, "currency"), "help": "Σ P(churn) × revenu total"},
            {"label": "Taux churn prédit vs historique", "value": f"{format_percent(predicted_churn_rate)} (vs {format_percent(historical_churn)})"},
        ]
    )

    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Distribution des probabilités de churn")
        hist = pd.cut(
            scored["churn_probability"],
            bins=[0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01],
            labels=["0-15 %", "15-30 %", "30-45 %", "45-60 %", "60-80 %", "80-100 %"],
        ).value_counts().sort_index()
        st.bar_chart(hist)
    with col2:
        st.subheader("Répartition par niveau de risque")
        risk_counts = scored["risk_level"].value_counts().reindex(["faible", "modéré", "élevé"], fill_value=0)
        st.bar_chart(risk_counts)

    st.divider()
    st.subheader("Top 10 segments par revenu pondéré à risque")
    seg = (
        scored.assign(rev_risk=lambda d: d["churn_probability"] * d["total_revenue"])
        .groupby("customer_segment")["rev_risk"]
        .sum()
        .sort_values(ascending=False)
    )
    st.bar_chart(seg)


def page_at_risk(api_url: str) -> None:
    st.title("Clients à risque")
    df = load_dataset()
    info = fetch_model_info(api_url)
    threshold = float(info["decision_threshold"]) if info else 0.35

    scored = score_dataset(api_url, sample_size=500)
    if scored.empty:
        st.warning("API indisponible.")
        return

    col_a, col_b, col_c = st.columns(3)
    segments = ["(tous)"] + sorted(scored["customer_segment"].unique())
    contracts = ["(tous)"] + sorted(scored["contract_type"].unique())
    countries = ["(tous)"] + sorted(scored["country"].unique())
    with col_a:
        seg_filter = st.selectbox("Segment", segments)
    with col_b:
        contract_filter = st.selectbox("Contrat", contracts)
    with col_c:
        country_filter = st.selectbox("Pays", countries)

    min_proba = st.slider("Probabilité minimale", 0.0, 1.0, threshold, 0.05)

    view = scored[scored["churn_probability"] >= min_proba].copy()
    if seg_filter != "(tous)":
        view = view[view["customer_segment"] == seg_filter]
    if contract_filter != "(tous)":
        view = view[view["contract_type"] == contract_filter]
    if country_filter != "(tous)":
        view = view[view["country"] == country_filter]

    view = view.sort_values("churn_probability", ascending=False)

    st.caption(f"{len(view)} clients filtrés")

    def factors_to_str(factors: list[dict]) -> str:
        if not factors:
            return "—"
        return " · ".join(
            f.get("label", f.get("feature", "")) for f in factors[:3]
        )

    display = view.assign(
        proba_pct=lambda d: (d["churn_probability"] * 100).round(1),
        revenu=lambda d: d["total_revenue"].map(format_currency_eur),
        risque=lambda d: d["risk_level"],
        facteurs=lambda d: d["top_factors"].map(factors_to_str),
    )[["customer_id", "customer_segment", "contract_type", "country", "revenu", "proba_pct", "risque", "facteurs"]]
    display.columns = ["ID client", "Segment", "Contrat", "Pays", "Revenu", "P(churn) %", "Risque", "Top 3 facteurs"]

    st.dataframe(display, height=420, use_container_width=True)

    csv_bytes = display.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Exporter en CSV",
        data=csv_bytes,
        file_name="clients_a_risque.csv",
        mime="text/csv",
    )


def page_simulator(api_url: str) -> None:
    st.title("Simulateur individuel")
    df = load_dataset()

    info = fetch_model_info(api_url)
    if info:
        st.caption(f"Modèle : {info.get('model_class','?')} · entraîné le {info.get('training_date','?')} · seuil = {info.get('decision_threshold','?')}")

    with st.form("simulator"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Genre", ["Female", "Male"])
            age = st.slider("Âge", 18, 99, 38)
            country = st.selectbox("Pays", sorted(df["country"].unique()))
            city = st.selectbox("Ville", sorted(df["city"].unique()))
            customer_segment = st.selectbox("Segment", sorted(df["customer_segment"].unique()))
            signup_channel = st.selectbox("Canal d'inscription", sorted(df["signup_channel"].unique()))
            contract_type = st.selectbox("Contrat", sorted(df["contract_type"].unique()))
            tenure_months = st.slider("Ancienneté (mois)", 1, 60, 12)
        with col2:
            monthly_fee = st.number_input("Abonnement mensuel (€)", 10, 200, 45)
            total_revenue = st.number_input("Revenu total cumulé (€)", 10, 100_000, 540)
            payment_method = st.selectbox("Moyen de paiement", sorted(df["payment_method"].unique()))
            discount_applied = st.selectbox("Remise active", ["No", "Yes"])
            price_increase_last_3m = st.selectbox("Hausse de prix récente", ["No", "Yes"])
            monthly_logins = st.slider("Connexions / mois", 0, 60, 22)
            weekly_active_days = st.slider("Jours actifs / semaine", 0, 7, 4)
            avg_session_time = st.slider("Durée moyenne d'une session", 1.0, 60.0, 18.5)
            features_used = st.slider("Fonctionnalités utilisées", 1, 15, 5)
        with col3:
            usage_growth_rate = st.slider("Évolution d'usage", -0.5, 0.5, -0.05, 0.05)
            last_login_days_ago = st.slider("Dernière connexion (jours)", 0, 80, 6)
            payment_failures = st.slider("Échecs de paiement", 0, 6, 1)
            support_tickets = st.slider("Tickets support", 0, 10, 2)
            avg_resolution_time = st.slider("Temps moyen de résolution (h)", 0.0, 80.0, 24.0)
            complaint_type = st.selectbox("Type de plainte", ["None", "Technical", "Billing", "Service"])
            csat_score = st.slider("Score CSAT", 1.0, 5.0, 3.2, 0.1)
            escalations = st.slider("Escalades support", 0, 5, 0)
            email_open_rate = st.slider("Taux d'ouverture e-mail", 0.0, 1.0, 0.42, 0.05)
            marketing_click_rate = st.slider("Taux de clic marketing", 0.0, 1.0, 0.18, 0.05)
            nps_score = st.slider("Score NPS", -100, 100, -10)
            survey_response = st.selectbox("Réponse au sondage", sorted(df["survey_response"].unique()))
            referral_count = st.slider("Parrainages", 0, 10, 1)

        submitted = st.form_submit_button("Calculer la probabilité de churn", type="primary")

    if submitted:
        payload = {
            "gender": gender,
            "age": int(age),
            "country": country,
            "city": city,
            "customer_segment": customer_segment,
            "tenure_months": int(tenure_months),
            "signup_channel": signup_channel,
            "contract_type": contract_type,
            "monthly_logins": int(monthly_logins),
            "weekly_active_days": int(weekly_active_days),
            "avg_session_time": float(avg_session_time),
            "features_used": int(features_used),
            "usage_growth_rate": float(usage_growth_rate),
            "last_login_days_ago": int(last_login_days_ago),
            "monthly_fee": float(monthly_fee),
            "total_revenue": float(total_revenue),
            "payment_method": payment_method,
            "payment_failures": int(payment_failures),
            "discount_applied": discount_applied,
            "price_increase_last_3m": price_increase_last_3m,
            "support_tickets": int(support_tickets),
            "avg_resolution_time": float(avg_resolution_time),
            "complaint_type": complaint_type,
            "csat_score": float(csat_score),
            "escalations": int(escalations),
            "email_open_rate": float(email_open_rate),
            "marketing_click_rate": float(marketing_click_rate),
            "nps_score": int(nps_score),
            "survey_response": survey_response,
            "referral_count": int(referral_count),
        }
        result = call_predict(api_url, payload)
        if result:
            st.divider()
            cols = st.columns([1, 1, 1])
            with cols[0]:
                kpi_card(
                    "Probabilité de churn",
                    f"{result['churn_probability'] * 100:.1f} %".replace(".", ","),
                )
            with cols[1]:
                st.markdown(f"**Niveau de risque**<br>{risk_badge(result['risk_level'])}", unsafe_allow_html=True)
            with cols[2]:
                kpi_card("Seuil opérationnel", f"{result['decision_threshold']:.2f}")

            st.subheader("Facteurs contributifs")
            for factor in result["top_factors"]:
                bullet = "↑" if factor["direction"] == "augmente" else "↓"
                st.markdown(
                    f"- {bullet} **{feature_name_to_business_label(factor['feature'])}** "
                    f"(impact {factor['shap_value']:+.3f})"
                )

            st.divider()
            st.subheader("Recommandation")
            st.info(recommendation_from_factors(result["top_factors"]))


def page_global_factors() -> None:
    st.title("Facteurs globaux du churn")
    st.markdown(
        "Vue d'ensemble des leviers identifiés par le modèle sur l'ensemble du jeu de données. "
        "Les libellés sont traduits en termes métier."
    )

    business_csv = ROOT / "reports" / "shap_business_importance.csv"
    if business_csv.exists():
        df = pd.read_csv(business_csv)
        df = df.sort_values("abs_shap", ascending=False).head(15)
        df["Variable métier"] = df["label"]
        df = df.set_index("Variable métier")["abs_shap"].rename("Impact moyen sur la prédiction")
        st.bar_chart(df)

    cols = st.columns(2)
    with cols[0]:
        bar_path = FIGURES_DIR / "shap_global_bar.png"
        if bar_path.exists():
            st.image(str(bar_path), caption="Top facteurs explicatifs (|SHAP| moyen)")
    with cols[1]:
        beeswarm_path = FIGURES_DIR / "shap_beeswarm.png"
        if beeswarm_path.exists():
            st.image(str(beeswarm_path), caption="Distribution des contributions par client")

    st.divider()
    st.subheader("Recommandations actionnables")
    insights_md = ROOT / "reports" / "insights_metier.md"
    if insights_md.exists():
        st.markdown(insights_md.read_text(encoding="utf-8"))


def main() -> None:
    st.sidebar.title("Atelier Rétention")
    api_url = st.sidebar.text_input("URL de l'API", value=DEFAULT_API)
    health = fetch_health(api_url)
    if health and health.get("model_loaded"):
        st.sidebar.success(f"API connectée · modèle {health.get('model_version')}")
    else:
        st.sidebar.error("API indisponible ou modèle non chargé")

    pages = {
        "Vue d'ensemble": lambda: page_overview(api_url),
        "Clients à risque": lambda: page_at_risk(api_url),
        "Simulateur": lambda: page_simulator(api_url),
        "Facteurs globaux": page_global_factors,
    }
    choice = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[choice]()

    st.sidebar.divider()
    st.sidebar.caption("Projet M1 — Système intelligent de rétention client")


if __name__ == "__main__":
    main()
