"""Comprehensive deterministic unit tests for all 10 forensic analysis checks."""

import numpy as np
import pandas as pd
import pytest

from config import AppConfig
from engine.detectors import (
    detect_missing_values,
    detect_duplicate_records,
    detect_constant_and_near_constant,
    detect_potential_identifiers,
    detect_numeric_outliers,
    detect_high_cardinality,
    detect_inconsistent_categoricals,
    detect_target_imbalance,
    detect_numeric_correlations,
    detect_target_leakage,
    run_all_detectors,
)
from engine.models import DatasetMetadata, Severity
from engine.profiler import profile_dataset


@pytest.fixture
def mock_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        filename="test.csv",
        file_size_bytes=1024,
        sha256_hash="abc123def456",
        encoding="utf-8",
        delimiter=",",
        row_count=100,
        column_count=5,
        loaded_at="2026-10-19T00:00:00Z",
    )


# 1. Missing Values
def test_missing_values_detection(mock_metadata):
    df = pd.DataFrame({
        "all_null": [None] * 50,
        "critical_null": [1.0] * 10 + [None] * 40,  # 80% null > 50%
        "elevated_null": [1.0] * 35 + [None] * 15,  # 30% null > 20%
        "sparse_null": [1.0] * 48 + [None] * 2,     # 4% null
        "healthy": [1.0] * 50,
    })
    prof = profile_dataset(df, mock_metadata)
    findings = detect_missing_values(df, prof)

    rules = {f.id: f.rule_id for f in findings}
    assert "INT-001-all_null" in rules
    assert "INT-002-critical_null" in rules
    assert "INT-003-elevated_null" in rules
    assert "INT-004-sparse_null" in rules
    assert not any("healthy" in f.id for f in findings)


# 2. Duplicate Rows and Candidates
def test_duplicate_records_detection(mock_metadata):
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "feature_a": ["Alpha", "Beta", "Gamma", "Beta", "Alpha"],
        "feature_b": [10, 20, 30, 20, 10],
    })
    # Add an exact duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # Row 5 is identical to Row 0
    prof = profile_dataset(df, mock_metadata)
    findings = detect_duplicate_records(df, prof)

    exact_dup = next((f for f in findings if f.rule_id == "DUP-001"), None)
    assert exact_dup is not None
    assert exact_dup.evidence.metrics["duplicate_rows_count"] == 1
    assert 5 in exact_dup.evidence.row_indices


def test_candidate_duplicate_records_detection(mock_metadata):
    # Distinct IDs but identical non-ID features
    df = pd.DataFrame({
        "id": [101, 102, 103, 104],
        "city": ["Berlin", "Tokyo", "Berlin", "Paris"],
        "metric": [42.0, 99.0, 42.0, 15.0],
    })
    prof = profile_dataset(df, mock_metadata)
    findings = detect_duplicate_records(df, prof)

    cand_dup = next((f for f in findings if f.rule_id == "DUP-002"), None)
    assert cand_dup is not None
    assert cand_dup.evidence.metrics["candidate_count"] == 1
    assert 2 in cand_dup.evidence.row_indices


# 3. Constant and Near-Constant Columns
def test_constant_and_near_constant_columns(mock_metadata):
    n = 100
    df = pd.DataFrame({
        "constant": [42] * n,
        "near_constant": [1] * 99 + [0],  # 99% single value
        "balanced": [1] * 50 + [0] * 50,
    })
    prof = profile_dataset(df, mock_metadata)
    findings = detect_constant_and_near_constant(df, prof)

    rules = {f.id: f.rule_id for f in findings}
    assert "STAT-001-constant" in rules
    assert "STAT-002-near_constant" in rules
    assert not any("balanced" in f.id for f in findings)


# 4. Potential Identifiers
def test_potential_identifiers_detection(mock_metadata):
    n = 50
    df = pd.DataFrame({
        "surrogate_id": [f"ID_{i:04d}" for i in range(n)],
        "counter": list(range(n)),
        "category": ["A", "B"] * (n // 2),
    })
    prof = profile_dataset(df, mock_metadata)
    findings = detect_potential_identifiers(df, prof)

    flagged_cols = [f.evidence.column_names[0] for f in findings if f.rule_id == "STAT-003"]
    assert "surrogate_id" in flagged_cols
    assert "counter" in flagged_cols
    assert "category" not in flagged_cols


# 5. Numeric Outliers (IQR and Modified Z-Score)
def test_numeric_outliers_detection(mock_metadata):
    np.random.seed(42)
    # 95 normal values around 50, 5 extreme outliers
    vals = np.random.normal(50, 2, 95).tolist()
    outliers = [250.0, 300.0, -100.0]
    df = pd.DataFrame({"sensor_reading": vals + outliers})

    prof = profile_dataset(df, mock_metadata)
    findings = detect_numeric_outliers(df, prof)

    out_finding = next((f for f in findings if f.rule_id == "OUT-001"), None)
    assert out_finding is not None
    assert out_finding.evidence.metrics["outlier_count"] >= 3
    # Check that indices of extreme values are captured
    assert any(idx in out_finding.evidence.row_indices for idx in [95, 96, 97])


# 6. High Cardinality Categoricals
def test_high_cardinality_detection(mock_metadata):
    n = 100
    df = pd.DataFrame({
        "low_card": ["A", "B", "C", "D"] * (n // 4),
        "high_card": [f"Category_{i}" for i in range(60)] + ["Common"] * 40,
    })
    prof = profile_dataset(df, mock_metadata)
    findings = detect_high_cardinality(df, prof)

    card_finding = next((f for f in findings if f.rule_id == "CARD-001"), None)
    assert card_finding is not None
    assert card_finding.evidence.column_names == ["high_card"]
    assert card_finding.evidence.metrics["unique_count"] == 61


# 7. Inconsistent Categorical Representations
def test_inconsistent_categoricals_detection(mock_metadata):
    df = pd.DataFrame({
        "gender": ["Male", "male", " Male ", "Female", "female", "Other"] * 5,
        "clean_cat": ["A", "B", "C", "A", "B", "C"] * 5,
    })
    prof = profile_dataset(df, mock_metadata)
    findings = detect_inconsistent_categoricals(df, prof)

    inconsist_finding = next((f for f in findings if f.rule_id == "CAT-001"), None)
    assert inconsist_finding is not None
    assert inconsist_finding.evidence.column_names == ["gender"]
    assert inconsist_finding.evidence.metrics["conflicting_clusters_count"] >= 2


# 8. Target Class Imbalance
def test_target_class_imbalance(mock_metadata):
    # Imbalance ratio 95:5 = 19:1
    df = pd.DataFrame({
        "feature": list(range(100)),
        "imbalanced_target": [0] * 95 + [1] * 5,
        "balanced_target": [0] * 50 + [1] * 50,
    })

    findings_imb = detect_target_imbalance(df, "imbalanced_target")
    assert len(findings_imb) == 1
    assert findings_imb[0].rule_id in ["IMB-001", "IMB-002"]
    assert findings_imb[0].evidence.metrics["imbalance_ratio"] == 19.0

    findings_bal = detect_target_imbalance(df, "balanced_target")
    assert len(findings_bal) == 0


# 9. Strong Numeric Correlation & Redundancy
def test_numeric_correlations_detection(mock_metadata):
    x = np.linspace(1, 100, 100)
    df = pd.DataFrame({
        "x": x,
        "redundant_x": x * 2.0 + 0.001 * np.random.randn(100),  # r ~ 1.0
        "correlated_x": x + np.random.normal(0, 10, 100),      # high r
        "uncorrelated": np.random.permutation(x),
    })
    prof = profile_dataset(df, mock_metadata)
    findings = detect_numeric_correlations(df, prof)

    extreme_finding = next((f for f in findings if f.rule_id == "CORR-001"), None)
    assert extreme_finding is not None
    assert "x" in extreme_finding.evidence.column_names
    assert "redundant_x" in extreme_finding.evidence.column_names
    assert extreme_finding.evidence.metrics["pearson_r"] >= 0.98


# 10. Target Leakage Candidates
def test_target_leakage_detection(mock_metadata):
    target = np.array([0, 1] * 50)
    df = pd.DataFrame({
        "outcome": target,
        "exact_proxy": target,                               # Direct 100% duplicate
        "numeric_leak": target * 10.0 + np.random.normal(0, 0.01, 100), # r > 0.99
        "clean_feature": np.random.randn(100),
    })
    prof = profile_dataset(df, mock_metadata)
    findings = detect_target_leakage(df, target_column="outcome", profile=prof)

    leak_rules = [f.rule_id for f in findings]
    assert "LEAK-001" in leak_rules  # Exact proxy
    assert "LEAK-002" in leak_rules or "LEAK-003" in leak_rules  # Numeric correlation / separation


# Edge Cases: Empty, Single Row, All Null, Booleans, Datetimes
def test_detectors_edge_cases(mock_metadata):
    # Empty DataFrame
    empty_df = pd.DataFrame()
    empty_prof = profile_dataset(empty_df, mock_metadata)
    findings_empty = run_all_detectors(empty_df, empty_prof)
    assert isinstance(findings_empty, list)
    assert len(findings_empty) == 0

    # Single Row DataFrame
    single_df = pd.DataFrame({"a": [1], "b": ["x"]})
    single_prof = profile_dataset(single_df, mock_metadata)
    findings_single = run_all_detectors(single_df, single_prof)
    assert isinstance(findings_single, list)

    # DataFrame with Mixed Datetimes and Booleans
    mixed_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=30, freq="D"),
        "is_active": [True, False] * 15,
        "notes": [f"Note {i}" for i in range(30)],
    })
    mixed_prof = profile_dataset(mixed_df, mock_metadata)
    findings_mixed = run_all_detectors(mixed_df, mixed_prof)
    assert isinstance(findings_mixed, list)


# Immutability Check: Source DataFrame Never Silently Modified
def test_source_dataframe_never_modified(mock_metadata):
    df_original = pd.DataFrame({
        "col1": [1.0, 2.0, None, 4.0, 100.0],
        "col2": [" Male ", "male", "Female", "female", "Other"],
    })
    df_copy = df_original.copy(deep=True)

    prof = profile_dataset(df_original, mock_metadata)
    run_all_detectors(df_original, prof, target_column="col1")

    # Verify identical values, dtypes, columns, and index
    pd.testing.assert_frame_equal(df_original, df_copy)
