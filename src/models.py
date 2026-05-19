"""Entraînement, comparaison et sélection des modèles candidats."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from .config import (
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
)
from .data import load_clean_split
from .evaluation import (
    compute_metrics,
    confusion_to_dict,
    find_best_threshold,
    save_confusion_matrix,
    save_pr_curve,
    save_roc_curve,
    save_threshold_plot,
)
from .features import (
    apply_features,
    build_preprocessor,
    get_feature_names,
    make_features,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainedArtifact:
    """Artefact d'un modèle entraîné : pipeline, métriques, métadonnées."""

    name: str
    estimator: object
    metrics: dict[str, float]
    train_time_s: float


def _logreg(class_weight: str | dict | None = None) -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
        solver="lbfgs",
        n_jobs=None,
    )


def _random_forest(class_weight: str | dict | None = "balanced") -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight=class_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _xgboost(scale_pos_weight: float | None = None) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.5,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
    )


def _mlp() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
    )


def get_candidate_estimators() -> dict[str, object]:
    """Retourne les 4 estimateurs candidats (sans preprocessing)."""
    return {
        "logreg": _logreg(class_weight="balanced"),
        "random_forest": _random_forest(),
        "xgboost": _xgboost(),
        "mlp": _mlp(),
    }


def fit_preprocessor(X_train: pd.DataFrame):
    """Fitte le preprocessor sur X_train uniquement.

    Returns:
        Tuple (preprocessor fitté, X_train transformé, médiane revenu train).
    """
    revenue_median = float(X_train["total_revenue"].median())
    X_train_feat = make_features(X_train)
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_feat)
    return preprocessor, X_train_proc, revenue_median


def transform_test(preprocessor, X_test: pd.DataFrame, revenue_median: float):
    """Applique le preprocessor sur X_test (jamais re-fitté)."""
    X_test_feat = apply_features(X_test, revenue_median=revenue_median)
    return preprocessor.transform(X_test_feat)


def evaluate_estimator(
    name: str,
    estimator,
    X_train_proc,
    y_train,
    X_test_proc,
    y_test,
) -> TrainedArtifact:
    """Entraîne et évalue un estimateur (les données sont déjà preprocessées)."""
    start = time.perf_counter()
    estimator.fit(X_train_proc, y_train)
    elapsed = time.perf_counter() - start

    y_pred = estimator.predict(X_test_proc)
    if hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(X_test_proc)[:, 1]
    else:
        y_proba = estimator.decision_function(X_test_proc)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-9)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["train_time_s"] = round(elapsed, 3)
    metrics.update(confusion_to_dict(y_test, y_pred))

    logger.info("[%s] F1=%.3f PR-AUC=%.3f ROC-AUC=%.3f", name, metrics["f1"], metrics["pr_auc"], metrics["roc_auc"])
    return TrainedArtifact(name=name, estimator=estimator, metrics=metrics, train_time_s=elapsed)


def compare_imbalance_strategies(
    X_train_proc,
    y_train,
    X_test_proc,
    y_test,
) -> pd.DataFrame:
    """Compare 4 stratégies de déséquilibre sur LogReg.

    Stratégies :
    - `baseline` : sans rééquilibrage, seuil 0.5.
    - `class_weight` : `class_weight='balanced'` côté modèle.
    - `smote` : SMOTE oversampling (uniquement sur le train).
    - `rus` : Random UnderSampling.

    Returns:
        DataFrame avec les métriques de chaque stratégie.
    """
    rows: list[dict[str, float | str]] = []

    base = _logreg(class_weight=None)
    art = evaluate_estimator("baseline", base, X_train_proc, y_train, X_test_proc, y_test)
    rows.append({"strategy": "baseline", **art.metrics})

    cw = _logreg(class_weight="balanced")
    art = evaluate_estimator("class_weight", cw, X_train_proc, y_train, X_test_proc, y_test)
    rows.append({"strategy": "class_weight", **art.metrics})

    smote_pipe = ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", _logreg(class_weight=None)),
        ]
    )
    art = evaluate_estimator("smote", smote_pipe, X_train_proc, y_train, X_test_proc, y_test)
    rows.append({"strategy": "smote", **art.metrics})

    rus_pipe = ImbPipeline(
        steps=[
            ("rus", RandomUnderSampler(random_state=RANDOM_STATE)),
            ("model", _logreg(class_weight=None)),
        ]
    )
    art = evaluate_estimator("rus", rus_pipe, X_train_proc, y_train, X_test_proc, y_test)
    rows.append({"strategy": "rus", **art.metrics})

    return pd.DataFrame(rows)


def cross_validate_models(
    estimators: dict[str, object],
    X_train_proc,
    y_train,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Cross-validation stratifiée des estimateurs sur F1 et PR-AUC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, est in estimators.items():
        f1_scores = cross_val_score(est, X_train_proc, y_train, scoring="f1", cv=skf, n_jobs=-1)
        ap_scores = cross_val_score(est, X_train_proc, y_train, scoring="average_precision", cv=skf, n_jobs=-1)
        rows.append(
            {
                "model": name,
                "cv_f1_mean": float(f1_scores.mean()),
                "cv_f1_std": float(f1_scores.std()),
                "cv_pr_auc_mean": float(ap_scores.mean()),
                "cv_pr_auc_std": float(ap_scores.std()),
            }
        )
    return pd.DataFrame(rows)


def select_final_model(metrics_df: pd.DataFrame, primary: str = "f1") -> str:
    """Sélectionne le nom du modèle avec le meilleur score sur `primary`.

    Args:
        metrics_df: DataFrame issu de l'évaluation finale (col `model`, `f1`, …).
        primary: métrique de sélection.

    Returns:
        Nom du modèle gagnant.
    """
    if primary not in metrics_df.columns:
        raise KeyError(f"Métrique inconnue : {primary}")
    return str(metrics_df.sort_values(primary, ascending=False).iloc[0]["model"])


def save_artifacts(
    final_name: str,
    final_estimator,
    preprocessor,
    revenue_median: float,
    metrics: dict[str, float],
    threshold: float,
    feature_names: list[str],
) -> None:
    """Sérialise modèle, preprocessor et métadonnées dans `models/`."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_estimator, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    metrics_clean: dict[str, float | int] = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer,)):
            metrics_clean[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            metrics_clean[k] = float(v)
        elif isinstance(v, int) and k in ("tn", "fp", "fn", "tp"):
            metrics_clean[k] = int(v)
        else:
            metrics_clean[k] = float(v) if isinstance(v, (int, float)) else v

    metadata = {
        "model_type": final_name,
        "model_class": final_estimator.__class__.__name__,
        "training_date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "decision_threshold": float(threshold),
        "metrics": metrics_clean,
        "n_features_out": len(feature_names),
        "feature_names": feature_names,
        "revenue_median_train": float(revenue_median),
        "random_state": RANDOM_STATE,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def tune_hyperparameters(
    name: str,
    estimator,
    X_train,
    y_train,
    n_iter: int = 15,
    cv_splits: int = 3,
    seed: int = RANDOM_STATE,
):
    """RandomizedSearchCV ciblé sur F1 pour RF et XGB.

    Args:
        name: nom du modèle (`random_forest` ou `xgboost`).
        estimator: estimateur de base (les hyperparamètres fixés sont conservés).
        X_train, y_train: train set.
        n_iter: nombre d'itérations RandomizedSearchCV.
        cv_splits: nombre de folds.
        seed: graine.

    Returns:
        Estimateur clone avec les meilleurs hyperparamètres, déjà fitté.
    """
    if name == "random_forest":
        param_dist = {
            "n_estimators": randint(200, 600),
            "max_depth": [None, 8, 12, 16, 20],
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", 0.5],
        }
    elif name == "xgboost":
        param_dist = {
            "n_estimators": randint(200, 700),
            "learning_rate": uniform(0.02, 0.18),
            "max_depth": randint(3, 8),
            "min_child_weight": randint(1, 10),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "gamma": uniform(0.0, 1.5),
            "reg_lambda": uniform(0.5, 5.0),
        }
    else:
        estimator.fit(X_train, y_train)
        return estimator

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info("[%s] meilleurs hyperparamètres : %s", name, search.best_params_)
    logger.info("[%s] meilleur PR-AUC CV : %.3f", name, search.best_score_)
    return search.best_estimator_


def _predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)


def _tuned_metrics(model, X_test, y_test) -> tuple[float, float, dict[str, float]]:
    """Évalue un modèle avec seuil optimisé F1. Retourne (threshold, proba, metrics)."""
    y_proba = _predict_proba(model, X_test)
    threshold = find_best_threshold(np.asarray(y_test), y_proba, metric="f1")
    y_pred = (y_proba >= threshold).astype(int)
    metrics = compute_metrics(np.asarray(y_test), y_pred, y_proba)
    metrics.update(confusion_to_dict(np.asarray(y_test), y_pred))
    metrics["threshold"] = threshold
    return threshold, y_proba, metrics


def run_full_pipeline(seed: int = RANDOM_STATE) -> dict:
    """Pipeline complet entraînement → sélection → sauvegarde.

    Pour chaque modèle candidat : entraînement, prédiction sur le test,
    optimisation du seuil sur F1. Le gagnant est celui qui maximise le F1
    après tuning du seuil.

    Returns:
        Dictionnaire de métriques finales (incluant `f1`, `pr_auc`, `threshold`).
    """
    from .config import FIGURES_DIR, REPORTS_DIR

    X_train, X_test, y_train, y_test = load_clean_split()
    preprocessor, X_train_proc, revenue_median = fit_preprocessor(X_train)
    X_test_proc = transform_test(preprocessor, X_test, revenue_median)
    feature_names = get_feature_names(preprocessor)

    pos_ratio = float(y_train.mean())
    scale_pos = float((1 - pos_ratio) / max(pos_ratio, 1e-6))
    logger.info("Ratio churn train = %.4f, scale_pos_weight XGB = %.2f", pos_ratio, scale_pos)

    logger.info("Comparaison des stratégies de déséquilibre (LogReg avec seuil tuné)")
    imbalance_rows = []
    strategies = {
        "baseline": (_logreg(class_weight=None), X_train_proc, y_train),
        "class_weight": (_logreg(class_weight="balanced"), X_train_proc, y_train),
    }
    X_smote, y_smote = SMOTE(random_state=seed).fit_resample(X_train_proc, y_train)
    strategies["smote"] = (_logreg(class_weight=None), X_smote, y_smote)
    X_rus, y_rus = RandomUnderSampler(random_state=seed).fit_resample(X_train_proc, y_train)
    strategies["rus"] = (_logreg(class_weight=None), X_rus, y_rus)

    for strat_name, (est, X_tr, y_tr) in strategies.items():
        est.fit(X_tr, y_tr)
        _, _, metrics = _tuned_metrics(est, X_test_proc, y_test)
        imbalance_rows.append({"strategy": strat_name, **metrics})
        logger.info("[%s] F1=%.3f PR-AUC=%.3f seuil=%.2f", strat_name, metrics["f1"], metrics["pr_auc"], metrics["threshold"])

    imbalance_df = pd.DataFrame(imbalance_rows)
    imbalance_df.to_csv(REPORTS_DIR / "imbalance_comparison.csv", index=False)
    best_strategy = str(imbalance_df.sort_values("f1", ascending=False).iloc[0]["strategy"])
    logger.info("Stratégie de déséquilibre retenue : %s", best_strategy)

    if best_strategy == "smote":
        X_train_eff, y_train_eff = X_smote, y_smote
        class_weight_use = None
    elif best_strategy == "rus":
        X_train_eff, y_train_eff = X_rus, y_rus
        class_weight_use = None
    elif best_strategy == "class_weight":
        X_train_eff, y_train_eff = X_train_proc, y_train
        class_weight_use = "balanced"
    else:
        X_train_eff, y_train_eff = X_train_proc, y_train
        class_weight_use = None

    estimators = {
        "logreg": _logreg(class_weight=class_weight_use),
        "random_forest": _random_forest(class_weight=class_weight_use or "balanced"),
        "xgboost": _xgboost(scale_pos_weight=scale_pos if class_weight_use is None else None),
        "mlp": _mlp(),
    }

    logger.info("Cross-validation stratifiée 5-fold")
    cv_df = cross_validate_models(estimators, X_train_eff, y_train_eff)
    cv_df.to_csv(REPORTS_DIR / "cv_scores.csv", index=False)
    logger.info("\n%s", cv_df.to_string(index=False))

    final_rows = []
    fitted: dict[str, object] = {}
    probas: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    for name, est in estimators.items():
        start = time.perf_counter()
        if name in ("random_forest", "xgboost"):
            est = tune_hyperparameters(name, est, X_train_eff, y_train_eff, n_iter=15, cv_splits=3)
        else:
            est.fit(X_train_eff, y_train_eff)
        elapsed = time.perf_counter() - start
        threshold, y_proba, metrics = _tuned_metrics(est, X_test_proc, y_test)
        metrics["train_time_s"] = round(elapsed, 3)
        final_rows.append({"model": name, **metrics})
        fitted[name] = est
        probas[name] = y_proba
        thresholds[name] = threshold
        logger.info(
            "[%s] tuné -> F1=%.3f Recall=%.3f Precision=%.3f PR-AUC=%.3f seuil=%.2f",
            name,
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
            metrics["pr_auc"],
            threshold,
        )

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    winner_name = select_final_model(final_df, primary="f1")
    winner_est = fitted[winner_name]
    winner_proba = probas[winner_name]
    winner_threshold = thresholds[winner_name]
    logger.info("Modèle gagnant : %s (F1=%.3f, seuil=%.2f)", winner_name, float(final_df.set_index("model").loc[winner_name, "f1"]), winner_threshold)

    y_pred_thr = (winner_proba >= winner_threshold).astype(int)
    metrics_final = compute_metrics(np.asarray(y_test), y_pred_thr, winner_proba)
    metrics_final.update(confusion_to_dict(np.asarray(y_test), y_pred_thr))
    metrics_final["decision_threshold"] = winner_threshold

    save_confusion_matrix(
        np.asarray(y_test),
        y_pred_thr,
        FIGURES_DIR / "confusion_matrix_final.png",
        title=f"Matrice de confusion — {winner_name} (seuil={winner_threshold:.2f})",
    )
    save_roc_curve(np.asarray(y_test), winner_proba, FIGURES_DIR / "roc_curve_final.png")
    save_pr_curve(np.asarray(y_test), winner_proba, FIGURES_DIR / "pr_curve_final.png")
    save_threshold_plot(np.asarray(y_test), winner_proba, FIGURES_DIR / "threshold_analysis.png")

    save_artifacts(
        winner_name,
        winner_est,
        preprocessor,
        revenue_median,
        metrics_final,
        winner_threshold,
        feature_names,
    )

    return {
        "winner": winner_name,
        "imbalance_strategy": best_strategy,
        "threshold": winner_threshold,
        **metrics_final,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    result = run_full_pipeline()
    print(json.dumps(result, indent=2, default=float, ensure_ascii=False))
