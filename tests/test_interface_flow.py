"""Interface flow tests verifying complete Data Autopsy pipeline and section data contracts."""

import io
import json
from pathlib import Path
import pandas as pd
import pytest

from ai.explainer import ForensicExplainer
from config import AppConfig
from engine.loader import load_dataset
from engine.profiler import generate_autopsy_report
from engine.validator import validate_finding_evidence, validate_report_evidence
from reports.exporter import ReportExporter


@pytest.fixture
def sample_dataset_path() -> Path:
    return Path(__file__).resolve().parent.parent / "sample_data" / "corrupt_clinical_trials.csv"


def test_interface_ingestion_and_target_selection(sample_dataset_path):
    """Test ingestion parsing, metadata extraction, and target selection validation."""
    df, meta = load_dataset(sample_dataset_path)

    assert meta.row_count == 10
    assert meta.column_count == 6
    assert "response_rate" in df.columns
    assert meta.sha256_hash is not None


def test_interface_analysis_and_all_sections_data(sample_dataset_path):
    """Test full analysis generation supporting all 7 workstation sections."""
    df, meta = load_dataset(sample_dataset_path)

    # 1. Overview Section Data
    report = generate_autopsy_report(df, meta, target_column="response_rate")
    assert report.metadata.filename == "corrupt_clinical_trials.csv"
    assert report.execution_duration_ms > 0
    assert len(report.findings) > 0

    # 2. Dataset Profile Section Data
    prof = report.dataset_profile
    assert prof.total_cells == 60
    assert prof.total_null_cells > 0
    assert len(prof.columns) == 6
    assert "dosage" in prof.columns
    assert prof.columns["dosage"].inferred_type == "numeric"

    # 3. Forensic Findings Section Data
    severities = [f.severity.value for f in report.findings]
    assert "CRITICAL" in severities or "HIGH" in severities
    for f in report.findings:
        assert f.rank_score >= 0.0
        assert 0.0 <= f.confidence_score <= 1.0

    # 4. Leakage Investigation Section Data
    leakage_findings = [f for f in report.findings if f.rule_id.startswith("LEAK-")]
    assert len(leakage_findings) >= 1
    # Check baseline comparison or reason
    if report.baseline_comparison:
        assert "status" in report.baseline_comparison

    # 5. Evidence Inspector Section Data
    val_res = validate_report_evidence(df, report)
    assert val_res["all_valid"] is True
    for f in report.findings:
        is_val, discs = validate_finding_evidence(df, f)
        assert is_val is True, f"Finding {f.id} evidence discrepancies: {discs}"

    # 6. AI Investigator Section Data
    explainer = ForensicExplainer()
    # Offline fallback works unconditionally
    first_finding = report.findings[0]
    fallback_res = explainer._fallback_result(first_finding, "test_status")
    assert fallback_res.source == "deterministic_fallback"
    assert "[Deterministic Forensic Analysis]" in fallback_res.formatted_text

    # 7. Report Export Section Data
    json_export = ReportExporter.to_json(report)
    parsed_json = json.loads(json_export)
    assert parsed_json["metadata"]["filename"] == "corrupt_clinical_trials.csv"
    assert len(parsed_json["findings"]) == len(report.findings)

    md_export = ReportExporter.to_markdown(report)
    assert "# DATA AUTOPSY: Forensic Examination Dossier" in md_export
    assert "## 1. Executive Summary & Findings" in md_export
    assert "## 2. Dataset Profiling Summary" in md_export
    assert "## 3. Analysis Metadata & Reproducibility Context" in md_export


def test_interface_non_mutating_column_exclusion(sample_dataset_path):
    """Test column exclusion without mutating source dataset."""
    df, meta = load_dataset(sample_dataset_path)
    df_clone = df.copy(deep=True)

    report = generate_autopsy_report(
        df,
        meta,
        target_column="response_rate",
        excluded_columns=["dosage"],
    )

    # Source dataset must be completely unmodified
    pd.testing.assert_frame_equal(df, df_clone)
    assert "dosage" in df.columns

    # Excluded column must be removed from profile and findings
    assert "dosage" not in report.dataset_profile.columns
    for f in report.findings:
        assert "dosage" not in f.evidence.column_names
