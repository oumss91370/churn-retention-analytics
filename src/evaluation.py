"""Métriques d'évaluation et graphes comparatifs."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Calcule les métriques principales sur une prédiction binaire.

    Args:
        y_true: vraies étiquettes.
        y_pred: prédictions binaires.
        y_proba: probabilités de la classe positive (optionnel).

    Returns:
        Dictionnaire de métriques.
    """
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))

    return metrics


def confusion_to_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Retourne la matrice de confusion sous forme de dictionnaire TN/FP/FN/TP."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Construit un tableau précision / rappel / F1 par seuil.

    Args:
        y_true: vraies étiquettes.
        y_proba: probabilités de la classe positive.
        thresholds: seuils à tester ; par défaut 0.05 à 0.95 pas de 0.05.

    Returns:
        DataFrame avec colonnes threshold, precision, recall, f1.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)

    rows = []
    for thr in thresholds:
        y_hat = (y_proba >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "precision": float(precision_score(y_true, y_hat, zero_division=0)),
                "recall": float(recall_score(y_true, y_hat, zero_division=0)),
                "f1": float(f1_score(y_true, y_hat, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    min_precision: float | None = None,
) -> float:
    """Cherche le seuil maximisant F1 (par défaut) ou le rappel sous contrainte.

    Args:
        y_true: vraies étiquettes.
        y_proba: probabilités classe positive.
        metric: `f1` ou `recall`.
        min_precision: contrainte minimum sur la précision (pour `recall`).

    Returns:
        Seuil optimal.
    """
    table = threshold_analysis(y_true, y_proba)

    if metric == "recall" and min_precision is not None:
        eligible = table[table["precision"] >= min_precision]
        if not eligible.empty:
            return float(eligible.sort_values("recall", ascending=False).iloc[0]["threshold"])
        return 0.5

    return float(table.sort_values(metric, ascending=False).iloc[0]["threshold"])


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path | str,
    title: str = "Matrice de confusion",
) -> Path:
    """Sauvegarde la matrice de confusion en PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Reste", "Churn"])
    ax.set_yticklabels(["Reste", "Churn"])
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def save_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    path: Path | str,
    title: str = "Courbe ROC",
) -> Path:
    """Sauvegarde la courbe ROC en PNG."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def save_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    path: Path | str,
    title: str = "Courbe précision-rappel",
) -> Path:
    """Sauvegarde la courbe précision-rappel en PNG."""
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, label=f"PR-AUC = {ap:.3f}")
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def save_threshold_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    path: Path | str,
    title: str = "Précision / Rappel / F1 par seuil",
) -> Path:
    """Sauvegarde la courbe métriques vs seuil."""
    table = threshold_analysis(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["threshold"], table["precision"], label="Précision")
    ax.plot(table["threshold"], table["recall"], label="Rappel")
    ax.plot(table["threshold"], table["f1"], label="F1", linewidth=2)
    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
