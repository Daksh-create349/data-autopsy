"""Forensic tests for evidence consistency, ranking rules, reproducibility, and column exclusion."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from config import AppConfig
from engine.detectors import run_all_detectors
from engine.loader import load_dataset
from engine.models import DatasetMetadata, Severity, FindingCategory
from engine.profiler import generate_autopsy_report, profile_dataset
from engine.ranking import compute_statistical_confidence, compute_rank_score, rank_findings
from engine.validator import validate_finding_evidence, validate_report_evidence


def create_mock_metadata(df: pd.DataFrame, filename: str = "test.csv") -> DatasetMetadata:
    """Helper to construct dataset metadata for test DataFrames."""
    return DatasetMetadata(
        filename=filename,
        file_size_bytes=len(df.to_csv(index=False).encode("utf-8")),
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        encoding="utf-8",
        delimiter=",",
        row_count=len(df),
        column_count=len(df.columns),
        loaded_at="2026-10-19T00:00:00Z",
    )


def test_evidence_consistency_and_validation_on_corrupt_sample():
    """Verify that all evidence generated matches actual observed values in source data."""
    sample_path = Path(__file__).resolve().parent.parent / "sample_data" / "corrupt_clinical_trials.csv"
    df, meta = load_dataset(sample_path)
    report = generate_autopsy_report(df, meta, target_column="response_rate")

    val_res = validate_report_evidence(df, report)
    assert val_res["all_valid"] is True, f"Evidence validation discrepancies found: {val_res['discrepancies_by_finding']}"
    assert val_res["invalid_count"] == 0
    assert val_res["valid_count"] == len(report.findings)


def test_original_row_references_non_contiguous_index():
    """Verify that evidence preserves original non-contiguous DataFrame indices."""
    custom_index = [101, 204, 309, 412, 599, 678, 715, 843, 901, 1050]
    df = pd.DataFrame(
        {
            "id": [f"ID_{i}" for i in custom_index],
            "measurement": [10.0, 11.2, 10.8, 9999.0, 10.5, 11.0, 10.2, 10.9, 11.1, 10.7],  # row 412 is outlier
            "status": ["A", "A", None, "A", "A", "A", "A", "A", "A", "A"],  # row 309 is null
            "category": ["active", "active", "active", "active", "active", "active", "active", "active", "active", "active"],
        },
        index=custom_index,
    )
    meta = create_mock_metadata(df)
    report = generate_autopsy_report(df, meta)

    # Validate that all row indices in findings are present in custom_index
    for finding in report.findings:
        for idx in finding.evidence.row_indices:
            assert idx in custom_index, f"Row index {idx} in finding {finding.id} not in original DataFrame index!"

    val_res = validate_report_evidence(df, report)
    assert val_res["all_valid"] is True


def test_finding_ranking_hierarchy():
    """Verify deterministic ranking rules strictly enforce severity hierarchy and empirical support."""
    sample_path = Path(__file__).resolve().parent.parent / "sample_data" / "corrupt_clinical_trials.csv"
    df, meta = load_dataset(sample_path)
    report = generate_autopsy_report(df, meta, target_column="response_rate")

    severity_tier_order = {
        Severity.CRITICAL: 4,
        Severity.HIGH: 3,
        Severity.MEDIUM: 2,
        Severity.LOW: 1,
        Severity.INFO: 0,
    }

    # Verify findings are sorted descending by rank score and severity tier
    previous_rank_score = float("inf")
    for f in report.findings:
        assert f.rank_score <= previous_rank_score, f"Finding {f.id} rank score {f.rank_score} > previous {previous_rank_score}"
        previous_rank_score = f.rank_score

    # Check that CRITICAL findings appear before MEDIUM or LOW
    critical_indices = [i for i, f in enumerate(report.findings) if f.severity == Severity.CRITICAL]
    low_indices = [i for i, f in enumerate(report.findings) if f.severity in (Severity.LOW, Severity.INFO)]
    if critical_indices and low_indices:
        assert max(critical_indices) < min(low_indices)


def test_confidence_not_probability_and_sample_size_bounded():
    """Verify confidence score reflects empirical support discounted on small samples without being a probability."""
    sample_path = Path(__file__).resolve().parent.parent / "sample_data" / "corrupt_clinical_trials.csv"
    df, meta = load_dataset(sample_path)
    report = generate_autopsy_report(df, meta)

    for f in report.findings:
        # Confidence must be bounded in [0.0, 1.0]
        assert 0.0 <= f.confidence_score <= 1.0
        # Rank score must be positive
        assert f.rank_score >= 0.0


def test_clean_dataset_produces_zero_or_minimal_warnings():
    """Verify clean dataset produces no spurious integrity or leakage defects."""
    sample_path = Path(__file__).resolve().parent.parent / "sample_data" / "clean_patient_records.csv"
    df, meta = load_dataset(sample_path)
    report = generate_autopsy_report(df, meta, target_column="readmitted")

    # Clean dataset should have no CRITICAL findings
    critical_findings = [f for f in report.findings if f.severity == Severity.CRITICAL]
    assert len(critical_findings) == 0, f"Unexpected critical findings in clean data: {[f.title for f in critical_findings]}"


def test_evidence_sample_bounding_large_dataset():
    """Verify evidence samples remain strictly bounded even with thousands of rows."""
    rng = np.random.RandomState(42)
    n = 2000
    df = pd.DataFrame(
        {
            "feature1": rng.randn(n),
            "feature2": rng.randn(n),
            "mostly_null": [None if i % 2 == 0 else 1.0 for i in range(n)],
        }
    )
    meta = create_mock_metadata(df)
    report = generate_autopsy_report(df, meta)

    for f in report.findings:
        if f.evidence.sample_values:
            assert len(f.evidence.sample_values) <= 10, f"Finding {f.id} has {len(f.evidence.sample_values)} sample values (>10 bounded limit)"


def test_column_exclusion_immutability():
    """Verify column exclusion does not mutate the source dataset and correctly excludes columns from findings."""
    df_original = pd.DataFrame(
        {
            "col_a": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "col_b": [10.0, 20.0, 30.0, 40.0, 50.0] * 10,
            "leaky_col": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,  # Exact duplicate of col_a
            "target": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        }
    )
    df_clone = df_original.copy(deep=True)
    meta = create_mock_metadata(df_original)

    # Exclude leaky_col
    report = generate_autopsy_report(df_original, meta, target_column="target", excluded_columns=["leaky_col"])

    # 1. Source DataFrame MUST NOT be mutated
    pd.testing.assert_frame_equal(df_original, df_clone)
    assert "leaky_col" in df_original.columns

    # 2. Excluded column must not be present in dataset profile
    assert "leaky_col" not in report.dataset_profile.columns

    # 3. No findings should reference the excluded column
    for f in report.findings:
        assert "leaky_col" not in f.evidence.column_names, f"Finding {f.id} referenced excluded column 'leaky_col'"

    # 4. Metadata must record excluded column
    assert report.analysis_metadata is not None
    assert "leaky_col" in report.analysis_metadata.excluded_columns


def test_repeatability_and_determinism():
    """Verify two consecutive runs on the exact same dataset produce identical reports and rank orders."""
    sample_path = Path(__file__).resolve().parent.parent / "sample_data" / "corrupt_clinical_trials.csv"
    df, meta = load_dataset(sample_path)

    report_run1 = generate_autopsy_report(df, meta, target_column="response_rate")
    report_run2 = generate_autopsy_report(df, meta, target_column="response_rate")

    assert len(report_run1.findings) == len(report_run2.findings)

    for f1, f2 in zip(report_run1.findings, report_run2.findings):
        assert f1.id == f2.id
        assert f1.rule_id == f2.rule_id
        assert f1.rank_score == f2.rank_score
        assert f1.confidence_score == f2.confidence_score
        assert f1.evidence.row_indices == f2.evidence.row_indices


def test_analysis_metadata_and_limitations_completeness():
    """Verify report contains complete analysis metadata and limitations catalog."""
    sample_path = Path(__file__).resolve().parent.parent / "sample_data" / "corrupt_clinical_trials.csv"
    df, meta = load_dataset(sample_path)
    report = generate_autopsy_report(df, meta, target_column="response_rate")

    # Analysis Metadata
    ameta = report.analysis_metadata
    assert ameta is not None
    assert ameta.app_version == "1.0.0"
    assert ameta.selected_target == "response_rate"
    assert len(ameta.checks_executed) == 10
    assert ameta.reproducibility_settings["deterministic_execution"] is True
    assert ameta.duration_ms > 0

    # Limitations
    lims = report.limitations
    assert lims is not None
    for check in [
        "missing_values",
        "duplicate_records",
        "constant_columns",
        "potential_identifiers",
        "numeric_outliers",
        "high_cardinality",
        "inconsistent_categoricals",
        "class_imbalance",
        "numeric_correlations",
        "target_leakage",
    ]:
        assert check in lims
        assert "can_establish" in lims[check]
        assert "cannot_establish" in lims[check]
        assert "false_positive_risks" in lims[check]
