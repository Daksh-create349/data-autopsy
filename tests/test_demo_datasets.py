"""Automated tests validating real forensic engine behavior on reproducible demo datasets."""

from pathlib import Path
import pandas as pd
import pytest

from engine.loader import load_dataset
from engine.models import Severity
from engine.profiler import generate_autopsy_report
from engine.validator import validate_report_evidence

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def test_demo_1_clean_classification():
    """Verify clean dataset produces zero critical or high alerts."""
    csv_path = SAMPLE_DIR / "demo_1_clean_classification.csv"
    df, meta = load_dataset(csv_path)
    report = generate_autopsy_report(df, meta, target_column="credit_approved")

    # Clean data must not trigger unsupported critical or high alerts
    severe_findings = [f for f in report.findings if f.severity in (Severity.CRITICAL, Severity.HIGH)]
    assert len(severe_findings) == 0, f"Unexpected severe findings in clean data: {[f.title for f in severe_findings]}"

    # Health statistics must be immaculate
    assert report.dataset_profile.total_null_cells == 0
    assert report.dataset_profile.duplicate_rows_count == 0

    # Evidence matches verbatim
    val_res = validate_report_evidence(df, report)
    assert val_res["all_valid"] is True


def test_demo_2_missing_and_duplicates():
    """Verify exact detection of 100% null, partial missingness, and duplicate rows."""
    csv_path = SAMPLE_DIR / "demo_2_missing_and_duplicates.csv"
    df, meta = load_dataset(csv_path)
    report = generate_autopsy_report(df, meta, target_column="churned")

    rule_ids = [f.rule_id for f in report.findings]
    assert "INT-001" in rule_ids, "100% null column 'deprecated_telemetry' was not flagged"
    assert "INT-003" in rule_ids, "High null ratio in 'optional_survey' was not flagged"
    assert "DUP-001" in rule_ids, "Exact duplicate records were not flagged"

    # Verify duplicate count matches generated ground truth (15 duplicates)
    assert report.dataset_profile.duplicate_rows_count == 15

    # Evidence consistency
    val_res = validate_report_evidence(df, report)
    assert val_res["all_valid"] is True


def test_demo_3_imbalanced_classification():
    """Verify detection of extreme class imbalance (ratio > 20:1) without treating dataset as corrupt."""
    csv_path = SAMPLE_DIR / "demo_3_imbalanced_classification.csv"
    df, meta = load_dataset(csv_path)
    report = generate_autopsy_report(df, meta, target_column="is_fraud")

    rule_ids = [f.rule_id for f in report.findings]
    assert "IMB-001" in rule_ids, "Severe class imbalance (29:1) was not flagged"

    imbalance_finding = next(f for f in report.findings if f.rule_id == "IMB-001")
    assert imbalance_finding.severity == Severity.HIGH
    assert imbalance_finding.evidence.metrics["imbalance_ratio"] >= 20.0

    # Dataset must not be flagged as corrupt
    null_findings = [f for f in report.findings if f.rule_id.startswith("INT-")]
    assert len(null_findings) == 0

    val_res = validate_report_evidence(df, report)
    assert val_res["all_valid"] is True


def test_demo_4_target_copy_leakage():
    """Verify direct target duplication (LEAK-001) and categorical encoding (LEAK-004)."""
    csv_path = SAMPLE_DIR / "demo_4_target_copy_leakage.csv"
    df, meta = load_dataset(csv_path)
    report = generate_autopsy_report(df, meta, target_column="readmitted_30d")

    leak_rules = [f.rule_id for f in report.findings if f.rule_id.startswith("LEAK-")]
    assert "LEAK-001" in leak_rules, "Exact copy of target 'readmit_status_code' was not flagged"
    assert "LEAK-004" in leak_rules, "100% pure target encoding 'discharge_disposition' was not flagged"

    critical_leak = next(f for f in report.findings if f.rule_id == "LEAK-001")
    assert critical_leak.severity == Severity.CRITICAL
    assert "readmit_status_code" in critical_leak.evidence.column_names

    val_res = validate_report_evidence(df, report)
    assert val_res["all_valid"] is True


def test_demo_5_statistical_outliers():
    """Verify statistical outlier detection on heavy-tailed latency telemetry without false corruption alerts."""
    csv_path = SAMPLE_DIR / "demo_5_statistical_outliers.csv"
    df, meta = load_dataset(csv_path)
    report = generate_autopsy_report(df, meta, target_column="server_failure")

    outlier_findings = [f for f in report.findings if f.rule_id == "OUT-001"]
    assert len(outlier_findings) >= 1, "Expected OUT-001 for heavy-tailed 'latency_ms'"

    out_f = next(f for f in outlier_findings if "latency_ms" in f.evidence.column_names)
    assert out_f.evidence.metrics["outlier_count"] >= 5

    # Outliers must not be labeled as structural corruption (severity is Medium/Low, not Critical)
    assert out_f.severity in (Severity.MEDIUM, Severity.LOW)

    val_res = validate_report_evidence(df, report)
    assert val_res["all_valid"] is True


def test_demo_6_legitimate_strong_predictors():
    """Verify genuine strong causal predictors (r ~ 0.89) trigger advisory note but NO critical leakage alerts."""
    csv_path = SAMPLE_DIR / "demo_6_legitimate_strong_predictors.csv"
    df, meta = load_dataset(csv_path)
    report = generate_autopsy_report(df, meta, target_column="monthly_loan_payment")

    # Critical leakage checks must NOT trigger on legitimate physical relationships
    critical_leak = [f for f in report.findings if f.rule_id in ("LEAK-001", "LEAK-002", "LEAK-004", "LEAK-005")]
    assert len(critical_leak) == 0, f"False critical leakage alert raised on legitimate predictor: {[f.title for f in critical_leak]}"

    # Strong predictor should trigger an informational/low advisory warning (LEAK-003)
    strong_advisories = [f for f in report.findings if f.rule_id == "LEAK-003"]
    assert len(strong_advisories) >= 1
    assert any("loan_principal" in f.evidence.column_names for f in strong_advisories)

    val_res = validate_report_evidence(df, report)
    assert val_res["all_valid"] is True
