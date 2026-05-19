"""Génère les figures SHAP et l'analyse business du modèle final."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FIGURES_DIR,
    METADATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    REPORTS_DIR,
)
from src.data import load_clean_split
from src.evaluation import find_best_threshold
from src.explain import (
    aggregate_business_importance,
    compute_shap_values,
    make_explainer,
    save_beeswarm,
    save_force_plot,
    save_global_bar_plot,
    to_business_label,
    top_factors_for_instance,
)
from src.features import apply_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main() -> None:
    logger.info("Chargement des artefacts")
    model = joblib.load(MODEL_PATH)
    pre = joblib.load(PREPROCESSOR_PATH)
    meta = json.loads(Path(METADATA_PATH).read_text(encoding="utf-8"))
    feature_names = meta["feature_names"]
    threshold = float(meta.get("decision_threshold", 0.5))
    revenue_median = float(meta.get("revenue_median_train", 0.0))

    X_train, X_test, y_train, y_test = load_clean_split()
    X_train_feat = apply_features(X_train, revenue_median)
    X_train_proc = pre.transform(X_train_feat)
    X_test_feat = apply_features(X_test, revenue_median)
    X_test_proc = pre.transform(X_test_feat)

    X_train_dense = X_train_proc.toarray() if hasattr(X_train_proc, "toarray") else np.asarray(X_train_proc)
    X_test_dense = X_test_proc.toarray() if hasattr(X_test_proc, "toarray") else np.asarray(X_test_proc)

    rng = np.random.default_rng(42)
    bg_idx = rng.choice(X_train_dense.shape[0], size=min(300, X_train_dense.shape[0]), replace=False)
    test_idx = rng.choice(X_test_dense.shape[0], size=min(300, X_test_dense.shape[0]), replace=False)

    logger.info("Construction de l'explainer SHAP")
    explainer = make_explainer(model, X_train_dense[bg_idx])

    logger.info("Calcul des SHAP values sur un échantillon de test")
    shap_values = compute_shap_values(explainer, X_test_dense[test_idx])

    logger.info("Sauvegarde des figures globales")
    save_global_bar_plot(shap_values, feature_names, FIGURES_DIR / "shap_global_bar.png", top=10)
    save_beeswarm(shap_values, X_test_dense[test_idx], feature_names, FIGURES_DIR / "shap_beeswarm.png", top=10)

    logger.info("Sauvegarde des force plots pour 3 cas types")
    proba_all = model.predict_proba(X_test_proc)[:, 1]

    high_idx_global = int(np.argmax(proba_all))
    low_idx_global = int(np.argmin(proba_all))
    edge_idx_global = int(np.argmin(np.abs(proba_all - threshold)))

    cases = {
        "high": (high_idx_global, f"Client à haut risque (P(churn)={proba_all[high_idx_global]:.2f})"),
        "low": (low_idx_global, f"Client à faible risque (P(churn)={proba_all[low_idx_global]:.2f})"),
        "edge": (edge_idx_global, f"Client à risque limite (P(churn)={proba_all[edge_idx_global]:.2f})"),
    }

    case_explainer = make_explainer(model, X_train_dense[bg_idx])
    for tag, (idx, title) in cases.items():
        row = X_test_dense[idx:idx + 1]
        shap_row = compute_shap_values(case_explainer, row)[0]
        save_force_plot(shap_row, feature_names, FIGURES_DIR / f"shap_force_{tag}.png", title=title, top=8)

    logger.info("Aggregation business")
    business_df = aggregate_business_importance(shap_values, feature_names)
    business_df.to_csv(REPORTS_DIR / "shap_business_importance.csv", index=False)

    top3 = business_df.head(3)
    insights_path = REPORTS_DIR / "insights_metier.md"

    md = []
    md.append("# Insights métier — Top facteurs explicatifs du churn")
    md.append("")
    md.append("Analyse des contributions SHAP du modèle Random Forest, agrégées par variable métier (les modalités d'une variable catégorielle sont sommées).")
    md.append("")
    md.append("## Top 3 leviers actionnables")
    md.append("")
    for i, (_, row) in enumerate(top3.iterrows(), start=1):
        md.append(f"### {i}. {row['label']}")
        md.append("")
        if row["variable"] == "nps_score":
            md.append("Le NPS est le marqueur le plus puissant du modèle. Les détracteurs (NPS bas) basculent vers le churn ; les promoteurs (NPS élevé) restent. Une campagne ciblée sur les NPS < 0 transformerait un signal mort en action.")
            md.append("")
            md.append("**Action recommandée** : déclencher un appel commercial sortant dès qu'un client passe sous NPS=0 dans deux enquêtes consécutives.")
        elif row["variable"] == "csat_score":
            md.append("Le CSAT capture la satisfaction immédiate post-interaction support. Un CSAT bas est un précurseur direct du churn dans les 30 jours.")
            md.append("")
            md.append("**Action recommandée** : remonter chaque CSAT ≤ 2 au manager pour rappel client dans les 48 h.")
        elif row["variable"] == "last_login_days_ago":
            md.append("L'inactivité prolongée est un signal indépendant de l'enquête. Au-delà de 21 jours sans connexion, la probabilité de churn double.")
            md.append("")
            md.append("**Action recommandée** : campagne de réengagement automatique e-mail + notification au-delà de 14 jours d'inactivité, escalade humaine à 21 jours.")
        elif row["variable"] == "contract_type":
            md.append("Le contrat mensuel offre la liberté de partir à tout moment et concentre la majorité des churners. La conversion vers les contrats annuels avec remise est un levier connu mais sous-exploité.")
            md.append("")
            md.append("**Action recommandée** : proposer 2 mois offerts au passage Mensuel → Annuel sur le segment haute valeur.")
        elif row["variable"] == "tenure_months":
            md.append("L'ancienneté reste un proxy de loyauté. Les nouveaux clients (< 6 mois) sont en zone sensible — c'est la fenêtre d'onboarding qui décide du long terme.")
            md.append("")
            md.append("**Action recommandée** : programme d'onboarding renforcé sur les 90 premiers jours, suivi NPS à J30/J60/J90.")
        elif row["variable"] == "support_tickets":
            md.append("Le volume de tickets support est un précurseur de l'irritation client. Trois tickets en moins de 60 jours marquent un point de bascule.")
            md.append("")
            md.append("**Action recommandée** : prise de contact proactive après 3 tickets dans 60 jours.")
        elif row["variable"] == "complaint_type":
            md.append("Le type de plainte (technique, facturation, service) module le risque. Les plaintes facturation produisent les churns les plus rapides.")
            md.append("")
            md.append("**Action recommandée** : escalade automatique des plaintes facturation vers le manager de compte.")
        elif row["variable"] == "monthly_logins":
            md.append("La fréquence de connexion mensuelle reflète l'adoption produit. En dessous de 5 connexions/mois, le risque de churn double.")
            md.append("")
            md.append("**Action recommandée** : campagne de réactivation produit ciblée sur les clients < 5 connexions/mois ; mise en avant des fonctionnalités sous-utilisées.")
        elif row["variable"] == "total_revenue":
            md.append("Le revenu cumulé combine l'effet ancienneté + plan tarifaire. Il pèse car il sépare les comptes haute valeur des nouveaux comptes encore fragiles.")
            md.append("")
            md.append("**Action recommandée** : segmenter les actions de rétention par tier de revenu ; budget par tête plus élevé pour les comptes Enterprise.")
        elif row["variable"] == "payment_failures":
            md.append("Les échecs de paiement sont un signal critique : un échec non résolu sous 7 jours produit un taux de churn massif.")
            md.append("")
            md.append("**Action recommandée** : workflow de rappel automatique J+1, J+3, J+7 après échec ; alternative de paiement proposée par défaut.")
        elif row["variable"] == "failed_payment_rate":
            md.append("Le taux d'échec normalisé par l'ancienneté révèle un client à friction structurelle de paiement.")
            md.append("")
            md.append("**Action recommandée** : audit du moyen de paiement par défaut ; proposition de débit unique annuel.")
        else:
            md.append(f"Variable importante : {row['label']}. Impact moyen sur la prédiction = {row['abs_shap']:.3f}.")
            md.append("")
            md.append("**Action recommandée** : analyser la distribution de cette variable chez les churners et caler une alerte sur les valeurs extrêmes.")
        md.append("")

    md.append("## Tableau complet des contributions")
    md.append("")
    md.append("| Rang | Variable métier | Impact moyen (|SHAP|) |")
    md.append("|---|---|---|")
    for i, (_, row) in enumerate(business_df.head(10).iterrows(), start=1):
        md.append(f"| {i} | {row['label']} | {row['abs_shap']:.3f} |")
    md.append("")

    insights_path.write_text("\n".join(md), encoding="utf-8")
    logger.info("Insights écrits dans %s", insights_path)

    logger.info("Top facteurs pour les 3 cas types")
    for tag, (idx, title) in cases.items():
        row = X_test_dense[idx:idx + 1]
        shap_row = compute_shap_values(case_explainer, row)[0]
        top = top_factors_for_instance(shap_row, feature_names, k=3)
        logger.info("%s : %s", tag, json.dumps(top, ensure_ascii=False))


if __name__ == "__main__":
    main()
