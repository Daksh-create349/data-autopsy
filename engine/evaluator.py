"""Lightweight, reproducible scikit-learn pipeline evaluation for target leakage impact."""

from typing import Any, Dict, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from engine.utils import to_python_scalar


def build_lightweight_pipeline(
    feature_names: List[str],
    df: pd.DataFrame,
    is_classification: bool,
    random_state: int = 42,
) -> Tuple[Pipeline, List[str], List[str]]:
    """Build a deterministic scikit-learn pipeline ensuring preprocessing fits strictly on training data."""
    # Exclude non-predictive surrogate ID strings (uniqueness >= 85% with >= 15 rows)
    n_rows = len(df)
    clean_features = [
        c for c in feature_names
        if not (df[c].dtype == "object" and n_rows >= 15 and (df[c].nunique() / n_rows) >= 0.85)
    ]
    if not clean_features:
        clean_features = feature_names

    num_cols = [c for c in clean_features if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    cat_cols = [c for c in clean_features if c not in num_cols]

    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if is_classification:
        model = LogisticRegression(max_iter=500, solver="liblinear", random_state=random_state)
    else:
        model = Ridge(random_state=random_state)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline, num_cols, cat_cols


def evaluate_leakage_impact(
    df: pd.DataFrame,
    target_column: str,
    suspicious_features: List[str],
    test_size: float = 0.3,
    random_state: int = 42,
) -> Optional[Dict[str, Any]]:
    """Compare baseline model performance with suspicious features included vs excluded.
    
    Uses scikit-learn Pipelines so all preprocessing (imputation, scaling, one-hot encoding)
    is strictly fitted ONLY on training data, avoiding any data leakage during validation.
    Preserves original row indices for evidence tracking.
    """
    if df.empty or target_column not in df.columns:
        return None

    # Filter valid rows where target is observed
    valid_mask = df[target_column].notna()
    clean_df = df[valid_mask]

    if len(clean_df) < 15:
        # Insufficient sample size for reliable train/test split
        return {
            "status": "skipped",
            "reason": "Insufficient sample size (N < 15) for reproducible train/test pipeline evaluation.",
            "suspicious_features": suspicious_features,
        }

    target = clean_df[target_column]
    all_features = [c for c in clean_df.columns if c != target_column]
    valid_suspicious = [c for c in suspicious_features if c in all_features]

    if not valid_suspicious:
        return None

    features_without = [c for c in all_features if c not in valid_suspicious]
    if not features_without:
        # Cannot compare if no remaining features exist
        return {
            "status": "skipped",
            "reason": "No non-suspicious features remaining to train ablated comparison baseline.",
            "suspicious_features": valid_suspicious,
        }

    # Determine task type
    is_classification = False
    is_binary = False
    u_targets = target.unique()
    n_unique_target = len(u_targets)

    if (
        not pd.api.types.is_numeric_dtype(target)
        or pd.api.types.is_bool_dtype(target)
        or n_unique_target <= 10
    ):
        is_classification = True
        is_binary = (n_unique_target == 2)

    # Train/test split with original index preservation
    stratify = None
    if is_classification and n_unique_target >= 2:
        class_counts = target.value_counts()
        if (class_counts >= 2).all() and len(clean_df) * test_size >= n_unique_target:
            stratify = target

    try:
        train_df, test_df = train_test_split(
            clean_df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except Exception:
        # Fallback to non-stratified split if stratification fails due to rare classes
        train_df, test_df = train_test_split(
            clean_df,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    train_indices = [int(i) for i in train_df.index[:25].tolist()]
    test_indices = [int(i) for i in test_df.index[:25].tolist()]

    y_train = train_df[target_column]
    y_test = test_df[target_column]

    # Convert binary/multiclass target if object
    if is_classification and not pd.api.types.is_numeric_dtype(y_train):
        y_train = y_train.astype(str)
        y_test = y_test.astype(str)

    # Pipeline A: With suspicious features
    pipe_with, _, _ = build_lightweight_pipeline(all_features, clean_df, is_classification, random_state)
    # Pipeline B: Without suspicious features
    pipe_without, _, _ = build_lightweight_pipeline(features_without, clean_df, is_classification, random_state)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe_with.fit(train_df[all_features], y_train)
            preds_with = pipe_with.predict(test_df[all_features])

            pipe_without.fit(train_df[features_without], y_train)
            preds_without = pipe_without.predict(test_df[features_without])
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Pipeline execution failed: {str(e)}",
            "suspicious_features": valid_suspicious,
        }

    # Evaluate metrics
    metrics_summary: Dict[str, Any] = {}

    if is_classification:
        if is_binary:
            task_type = "binary_classification"
            primary_metric = "ROC-AUC"
            try:
                # Attempt proba
                pos_label = u_targets[1]
                proba_with = pipe_with.predict_proba(test_df[all_features])[:, 1]
                proba_without = pipe_without.predict_proba(test_df[features_without])[:, 1]
                score_with = float(roc_auc_score(y_test == pos_label, proba_with))
                score_without = float(roc_auc_score(y_test == pos_label, proba_without))
            except Exception:
                primary_metric = "Balanced Accuracy"
                score_with = float(balanced_accuracy_score(y_test, preds_with))
                score_without = float(balanced_accuracy_score(y_test, preds_without))
        else:
            task_type = "multiclass_classification"
            primary_metric = "Balanced Accuracy"
            score_with = float(balanced_accuracy_score(y_test, preds_with))
            score_without = float(balanced_accuracy_score(y_test, preds_without))

        acc_with = float(accuracy_score(y_test, preds_with))
        acc_without = float(accuracy_score(y_test, preds_without))
        metrics_summary["accuracy_with"] = round(acc_with, 4)
        metrics_summary["accuracy_without"] = round(acc_without, 4)
    else:
        task_type = "regression"
        primary_metric = "R-Squared"
        score_with = float(r2_score(y_test, preds_with))
        score_without = float(r2_score(y_test, preds_without))

        rmse_with = float(root_mean_squared_error(y_test, preds_with))
        rmse_without = float(root_mean_squared_error(y_test, preds_without))
        metrics_summary["rmse_with"] = round(rmse_with, 4)
        metrics_summary["rmse_without"] = round(rmse_without, 4)

    delta = round(score_with - score_without, 4)
    score_with_r = round(score_with, 4)
    score_without_r = round(score_without, 4)

    return {
        "status": "success",
        "task_type": task_type,
        "primary_metric": primary_metric,
        "score_with_suspicious": score_with_r,
        "score_without_suspicious": score_without_r,
        "delta": delta,
        "suspicious_features": valid_suspicious,
        "non_suspicious_features": features_without,
        "train_rows_count": len(train_df),
        "test_rows_count": len(test_df),
        "sample_train_indices": train_indices,
        "sample_test_indices": test_indices,
        "additional_metrics": metrics_summary,
        "interpretation": (
            f"Model {primary_metric} shifted by {delta:+.4f} (from {score_without_r} to {score_with_r}) "
            f"when '{', '.join(valid_suspicious)}' was included."
        ),
        "alternative_explanations": (
            "A substantial performance drop upon removing a feature does NOT prove target leakage. "
            "Legitimate, highly predictive domain features (e.g., prior diagnosis predicting mortality) "
            "naturally cause significant performance degradation when ablated. "
            "Leakage is confirmed by operational timeline verification (was feature recorded after outcome?), "
            "not solely by metric differences."
        ),
    }
