"""Unit tests for forensic profiler and integrity detection engine."""

import json
from pathlib import Path
import pytest
import pandas as pd

from config import AppConfig
from engine.loader import load_dataset
from engine.models import DatasetMetadata, FindingCategory, Severity
from engine.profiler import (
    generate_autopsy_report,
    infer_column_type,
    profile_column,
    profile_dataset,
    run_basic_integrity_checks,
)
from engine.utils import to_python_scalar


def test_infer_column_types():
    """Verify logical type inference across diverse series."""
    assert infer_column_type(pd.Series([1, 2, 3, 4, 5])) == "numeric"
    assert infer_column_type(pd.Series([True, False, True])) == "boolean"
    assert infer_column_type(pd.Series(["2026-01-01", "2026-01-02", "2026-01-03"])) == "datetime"
    assert infer_column_type(pd.Series(["Yes", "No", "Yes", "Yes"])) == "categorical"
    assert infer_column_type(pd.Series([None, None])) == "empty"


def test_profile_column_numerical():
    """Verify numerical column statistics calculation."""
    s = pd.Series([10.0, 20.0, 30.0, None], name="score")
    col_prof = profile_column(s)

    assert col_prof.name == "score"
    assert col_prof.total_count == 4
    assert col_prof.null_count == 1
    assert col_prof.null_percentage == 25.0
    assert col_prof.unique_count == 3
    assert col_prof.summary_stats["min"] == 10.0
    assert col_prof.summary_stats["max"] == 30.0
    assert col_prof.summary_stats["mean"] == 20.0
    assert col_prof.summary_stats["median"] == 20.0


def test_profile_column_categorical():
    """Verify categorical column profiling and top counts."""
    s = pd.Series(["A", "B", "A", "C", "A"], name="category")
    col_prof = profile_column(s)

    assert col_prof.name == "category"
    assert col_prof.inferred_type == "categorical"
    assert col_prof.null_count == 0
    assert col_prof.summary_stats["top_values"]["A"] == 3


def test_profile_dataset_duplicates():
    """Verify duplicate row detection preserving raw row indices."""
    df = pd.DataFrame({
        "id": [1, 2, 1, 3],
        "name": ["Alpha", "Beta", "Alpha", "Gamma"],
    })
    from datetime import datetime, timezone
    from engine.models import DatasetMetadata
    meta = DatasetMetadata(
        filename="test.csv",
        file_size_bytes=100,
        sha256_hash="abc",
        encoding="utf-8",
        delimiter=",",
        row_count=4,
        column_count=2,
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )

    prof = profile_dataset(df, meta)
    assert prof.duplicate_rows_count == 1
    assert prof.duplicate_rows_percentage == 25.0
    assert prof.exact_duplicate_row_indices == [2]


def test_run_basic_integrity_checks():
    """Verify detection of complete nulls, critical nulls, and constant columns."""
    df = pd.DataFrame({
        "all_null": [None, None, None, None],
        "mostly_null": [10.0, None, None, None],
        "constant": ["FIXED", "FIXED", "FIXED", "FIXED"],
        "healthy": [1, 2, 3, 4],
    })
    meta = DatasetMetadata(
        filename="edge_cases.csv",
        file_size_bytes=200,
        sha256_hash="123456",
        encoding="utf-8",
        delimiter=",",
        row_count=4,
        column_count=4,
        loaded_at="2026-10-19T00:00:00Z",
    )
    prof = profile_dataset(df, meta)
    findings = run_basic_integrity_checks(df, prof)

    rule_ids = [f.rule_id for f in findings]
    assert "INT-001" in rule_ids  # 100% null
    assert "INT-002" in rule_ids  # 75% null > 50% critical threshold
    assert "STAT-001" in rule_ids  # Zero variance / constant column


def test_generate_autopsy_report_serialization():
    """Verify end-to-end report creation and JSON serialization."""
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    meta = DatasetMetadata(
        filename="sample.csv",
        file_size_bytes=50,
        sha256_hash="def456",
        encoding="utf-8",
        delimiter=",",
        row_count=3,
        column_count=2,
        loaded_at="2026-10-19T00:00:00Z",
    )

    report = generate_autopsy_report(df, meta)
    assert report.metadata.filename == "sample.csv"
    assert report.execution_duration_ms >= 0.0

    # Ensure JSON serializable
    report_json = report.to_json()
    parsed = json.loads(report_json)
    assert parsed["metadata"]["sha256_hash"] == "def456"
    assert "col_a" in parsed["dataset_profile"]["columns"]


def test_corrupt_clinical_trials_sample():
    """Verify forensic checks on pre-packaged corrupt clinical trial data."""
    sample_path = Path(__file__).resolve().parent.parent / "sample_data" / "corrupt_clinical_trials.csv"
    df, meta = load_dataset(sample_path)
    report = generate_autopsy_report(df, meta)

    # Must detect unobserved_metric 100% null and duplicate row T1001
    finding_rule_ids = {f.rule_id for f in report.findings}
    assert "INT-001" in finding_rule_ids
    assert "DUP-001" in finding_rule_ids


def test_column_profile_properties():
    """Verify ColumnProfile properties return clean values for numerical and categorical columns."""
    num_s = pd.Series([10.0, 0.0, 20.0, 0.0, None], name="test_num")
    prof = profile_column(num_s)
    assert prof.zeros_count == 2
    assert prof.zeros_percentage == 40.0
    assert prof.mean == 7.5
    assert prof.median == 5.0
    assert prof.std_dev == 9.5743

    cat_s = pd.Series(["A", "B", "A", None], name="test_cat")
    prof_cat = profile_column(cat_s)
    assert prof_cat.zeros_count == 0
    assert prof_cat.zeros_percentage == 0.0
    assert prof_cat.mean is None
    assert prof_cat.median is None
    assert prof_cat.std_dev is None
    assert prof_cat.mode == "A"


