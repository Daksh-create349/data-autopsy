"""Unit tests for advanced target data leakage detection and baseline pipeline evaluation."""

import numpy as np
import pandas as pd
import pytest

from config import AppConfig
from engine.detectors import detect_target_leakage
from engine.evaluator import evaluate_leakage_impact
from engine.models import DatasetMetadata, Severity
from engine.profiler import profile_dataset


@pytest.fixture
def mock_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        filename="leakage_test.csv",
        file_size_bytes=2048,
        sha256_hash="deadbeef1234",
        encoding="utf-8",
        delimiter=",",
        row_count=100,
        column_count=5,
        loaded_at="2026-10-19T00:00:00Z",
    )


# 1. Exact Target Copies
def test_exact_target_copies(mock_metadata):
    """Verify exact target copies in numeric, string, and casing variants are detected with LEAK-001."""
    df = pd.DataFrame({
        "target": ["Approved", "Denied", "Approved", "Denied", "Approved"] * 20,
        "exact_copy": ["Approved", "Denied", "Approved", "Denied", "Approved"] * 20,
        "case_copy": [" approved ", "denied", "APPROVED", "denied ", "Approved"] * 20,
        "clean_feature": np.random.randn(100),
    })
    prof = profile_dataset(df, mock_metadata)
    findings = detect_target_leakage(df, target_column="target", profile=prof)

    leak_001_cols = [f.evidence.column_names[0] for f in findings if f.rule_id == "LEAK-001"]
    assert "exact_copy" in leak_001_cols
    assert "case_copy" in leak_001_cols
    assert "clean_feature" not in leak_001_cols

    finding = next(f for f in findings if f.id == "LEAK-001-exact_copy")
    assert finding.severity == Severity.CRITICAL
    assert finding.evidence.metrics["match_rate"] == 1.0
    assert len(finding.evidence.row_indices) > 0


# 2. Inverted Binary Targets
def test_inverted_binary_targets(mock_metadata):
    """Verify inverted binary targets (0/1 negation or inverse categorical mapping) trigger LEAK-005."""
    binary_target = np.array([1, 0, 1, 1, 0] * 20)
    inverted_binary = 1 - binary_target

    df = pd.DataFrame({
        "label": binary_target,
        "inverted_label": inverted_binary,
        "inverted_str": ["No" if x == 1 else "Yes" for x in binary_target],
        "target_str": ["Yes" if x == 1 else "No" for x in binary_target],
        "feature_noise": np.random.randn(100),
    })
    prof = profile_dataset(df, mock_metadata)

    # Test numeric inversion
    findings_num = detect_target_leakage(df, target_column="label", profile=prof)
    inverted_num = next((f for f in findings_num if f.rule_id == "LEAK-005" and "inverted_label" in f.evidence.column_names), None)
    assert inverted_num is not None
    assert inverted_num.severity == Severity.CRITICAL

    # Test string categorical inversion
    findings_str = detect_target_leakage(df, target_column="target_str", profile=prof)
    inverted_cat = next((f for f in findings_str if f.rule_id == "LEAK-005" and "inverted_str" in f.evidence.column_names), None)
    assert inverted_cat is not None
    assert inverted_cat.severity == Severity.CRITICAL


# 3. Categorical Target Encoding
def test_categorical_target_encoding(mock_metadata):
    """Verify deterministic partition / 100% conditional purity triggers LEAK-004."""
    target = ["Cat_A", "Cat_B", "Cat_C"] * 30
    # Each sub_code deterministically maps to exactly one target class
    sub_codes = ["Sub_A1", "Sub_B1", "Sub_C1"] * 30

    df = pd.DataFrame({
        "target": target,
        "sub_code": sub_codes,
        "clean_cat": ["X", "Y"] * 45,
    })
    prof = profile_dataset(df, mock_metadata)
    findings = detect_target_leakage(df, target_column="target", profile=prof)

    purity_leak = next((f for f in findings if f.rule_id == "LEAK-004" and "sub_code" in f.evidence.column_names), None)
    assert purity_leak is not None
    assert purity_leak.evidence.metrics["conditional_purity"] == 1.0


def test_categorical_feature_continuous_target_encoding(mock_metadata):
    """Verify categorical feature explaining >= 95% of continuous target variance triggers LEAK-004-CAT-CONT."""
    np.random.seed(42)
    # Target values tightly clustered around category means
    categories = ["Tier_1", "Tier_2", "Tier_3"] * 30
    mean_map = {"Tier_1": 100.0, "Tier_2": 500.0, "Tier_3": 1000.0}
    continuous_target = [mean_map[c] + np.random.normal(0, 1.0) for c in categories]

    df = pd.DataFrame({
        "income": continuous_target,
        "income_tier": categories,
        "age": np.random.randint(20, 70, size=90),
    })
    prof = profile_dataset(df, mock_metadata)
    findings = detect_target_leakage(df, target_column="income", profile=prof)

    cat_cont_leak = next((f for f in findings if "LEAK-004" in f.rule_id and "income_tier" in f.evidence.column_names), None)
    assert cat_cont_leak is not None
    assert cat_cont_leak.evidence.metrics["eta_squared"] >= 0.95


# 4. Legitimate Strong Predictors (Calibrated Severity & False-Positive Documentation)
def test_legitimate_strong_predictors(mock_metadata):
    """Verify legitimate strong predictors (r ~ 0.88-0.92) receive calibrated severity and clear warnings."""
    np.random.seed(42)
    x = np.linspace(10, 100, 100)
    # Moderate-strong correlation (r ~ 0.88-0.92, below critical 0.95 threshold)
    y = x + np.random.normal(0, 15, 100)

    df = pd.DataFrame({
        "target_metric": y,
        "biomarker": x,
        "noise": np.random.randn(100),
    })
    prof = profile_dataset(df, mock_metadata)
    findings = detect_target_leakage(df, target_column="target_metric", profile=prof)

    strong_finding = next((f for f in findings if "STRONG" in f.id), None)
    assert strong_finding is not None
    # Severity should be LOW for legitimate strong predictors
    assert strong_finding.severity == Severity.LOW
    assert "does not prove data leakage" in strong_finding.limitations.lower()


# 5. Missing Values and Multiclass Targets
def test_informative_missingness_leakage(mock_metadata):
    """Verify feature with selective missingness correlated with target triggers LEAK-006."""
    target = ["Admitted"] * 50 + ["Discharged"] * 50
    # ICU severity is only recorded for Admitted patients (100% missing for Discharged)
    icu_score = [85.0 + i for i in range(50)] + [None] * 50

    df = pd.DataFrame({
        "admission_status": target,
        "icu_score": icu_score,
        "age": np.random.randint(20, 80, size=100),
    })
    prof = profile_dataset(df, mock_metadata)
    findings = detect_target_leakage(df, target_column="admission_status", profile=prof)

    missing_leak = next((f for f in findings if f.rule_id == "LEAK-006"), None)
    assert missing_leak is not None
    assert "icu_score" in missing_leak.evidence.column_names
    assert len(missing_leak.evidence.row_indices) > 0


def test_multiclass_target_separation(mock_metadata):
    """Verify numeric feature partitioning 3+ target classes triggers multiclass separation finding."""
    np.random.seed(42)
    classes = ["Stage_1", "Stage_2", "Stage_3"] * 30
    stage_values = {"Stage_1": 10.0, "Stage_2": 50.0, "Stage_3": 100.0}
    feat_values = [stage_values[c] + np.random.normal(0, 0.5) for c in classes]

    df = pd.DataFrame({
        "disease_stage": classes,
        "cell_density": feat_values,
        "patient_age": np.random.randint(30, 80, size=90),
    })
    prof = profile_dataset(df, mock_metadata)
    findings = detect_target_leakage(df, target_column="disease_stage", profile=prof)

    multi_leak = next((f for f in findings if "MULTI" in f.id), None)
    assert multi_leak is not None
    assert multi_leak.evidence.metrics["eta_squared"] >= 0.95
    assert multi_leak.evidence.metrics["classes_count"] == 3


# 6. Small Datasets and Duplicate Records
def test_small_dataset_sample_size_guard(mock_metadata):
    """Verify small datasets (N < 30) have documented sample size limitations and non-critical severity."""
    # N = 12
    target = np.array([0, 1] * 6)
    df = pd.DataFrame({
        "target": target,
        "numeric_correlated": target * 5.0 + np.random.normal(0, 0.05, 12),
        "clean": np.random.randn(12),
    })
    prof = profile_dataset(df, mock_metadata)
    findings = detect_target_leakage(df, target_column="target", profile=prof)

    leak_finding = next((f for f in findings if f.rule_id in ["LEAK-002", "LEAK-003", "LEAK-004", "LEAK-005"]), None)
    if leak_finding:
        # Severity downgraded on small dataset
        assert leak_finding.severity in [Severity.MEDIUM, Severity.LOW]
        assert "small dataset sample size" in leak_finding.limitations.lower()


def test_duplicate_records_handled_in_leakage(mock_metadata):
    """Verify datasets with duplicate records do not crash leakage detection and document duplicate presence."""
    base_rows = [
        {"y": 1, "x_leak": 10.0, "x_clean": 5.0},
        {"y": 0, "x_leak": 2.0, "x_clean": 1.0},
    ]
    # Repeat base rows 25 times (50 rows total, 48 duplicates)
    df = pd.DataFrame(base_rows * 25)
    prof = profile_dataset(df, mock_metadata)
    findings = detect_target_leakage(df, target_column="y", profile=prof)

    leak_finding = next((f for f in findings if "x_leak" in f.evidence.column_names), None)
    assert leak_finding is not None
    assert "duplicate rows" in leak_finding.limitations.lower()


# 7. Scikit-Learn Pipeline Baseline Comparison
def test_lightweight_pipeline_baseline_comparison():
    """Verify evaluate_leakage_impact trains pipelines with preprocessing strictly on training split."""
    rng = np.random.RandomState(42)
    n = 100
    y = np.array([0, 1] * 50)
    # Direct leak feature
    leak_feature = y * 10.0 + rng.normal(0, 0.01, n)
    noisy_feature = rng.randn(n)
    categorical_noise = rng.choice(["Type_A", "Type_B"], size=n)

    df = pd.DataFrame({
        "target": y,
        "leak_feature": leak_feature,
        "noisy_feature": noisy_feature,
        "categorical_noise": categorical_noise,
    })

    result = evaluate_leakage_impact(
        df=df,
        target_column="target",
        suspicious_features=["leak_feature"],
        test_size=0.3,
        random_state=42,
    )

    assert result is not None
    assert result["status"] == "success"
    assert result["task_type"] == "binary_classification"
    assert "ROC-AUC" in result["primary_metric"]
    assert result["score_with_suspicious"] > result["score_without_suspicious"]
    assert result["delta"] > 0
    assert len(result["sample_train_indices"]) > 0
    assert len(result["sample_test_indices"]) > 0
    assert "does not prove target leakage" in result["alternative_explanations"].lower()


def test_high_cardinality_string_target_no_error(mock_metadata):
    """Verify high-cardinality string identifier targets do not crash ANOVA or leakage detectors."""
    n = 60
    ids = [f"U{2000 + i}" for i in range(n)]
    df = pd.DataFrame({
        "customer_id": ids,
        "tenure": np.random.randint(1, 72, size=n),
        "contract": np.random.choice(["monthly", "annual"], size=n),
    })
    prof = profile_dataset(df, mock_metadata)
    # Must not raise TypeError: Could not convert string to numeric
    findings = detect_target_leakage(df, target_column="customer_id", profile=prof)
    assert isinstance(findings, list)

