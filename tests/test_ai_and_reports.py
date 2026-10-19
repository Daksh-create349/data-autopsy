"""Unit tests for AI explanation fallback and dossier reporting modules."""

import json
from pathlib import Path
import pytest
import pandas as pd

from ai.explainer import ForensicExplainer
from engine.models import (
    AutopsyReport,
    DatasetMetadata,
    Evidence,
    Finding,
    FindingCategory,
    Severity,
)
from engine.profiler import generate_autopsy_report
from reports.exporter import ReportExporter


@pytest.fixture
def sample_finding() -> Finding:
    return Finding(
        id="INT-001-test_col",
        rule_id="INT-001",
        category=FindingCategory.INTEGRITY,
        severity=Severity.CRITICAL,
        title="Column 'test_col' Contains 100% Null Values",
        summary="Entire column lacks any observable values.",
        evidence=Evidence(
            description="All 100 rows null.",
            column_names=["test_col"],
            metrics={"null_percentage": 100.0},
        ),
        recommendation="Drop column or inspect extraction pipeline.",
    )


def test_explainer_offline_deterministic_fallback(sample_finding):
    """Verify explainer falls back gracefully to deterministic rule explanation when Ollama is offline."""
    from config import AppConfig
    offline_cfg = AppConfig(ollama_host="http://localhost:59999")
    explainer = ForensicExplainer(cfg=offline_cfg)
    assert not explainer.is_available()

    explanation = explainer.generate_explanation(sample_finding)
    assert isinstance(explanation, str)
    assert "[Deterministic Forensic Analysis]" in explanation
    assert "INTEGRITY" in explanation
    assert "Drop column" in explanation


def test_explainer_check_status_offline(monkeypatch, sample_finding):
    """Verify check_ollama_status handles completely unavailable Ollama daemon."""
    import requests

    def mock_get(*args, **kwargs):
        raise requests.exceptions.ConnectionError("Connection refused")

    monkeypatch.setattr(requests, "get", mock_get)
    explainer = ForensicExplainer()
    status = explainer.check_ollama_status()

    assert status["available"] is False
    assert status["model_installed"] is False
    assert status["status"] == "ollama_offline"
    assert "ollama pull qwen2.5:1.5b" in status["setup_instructions"]
    assert "ollama run qwen2.5:1.5b" in status["setup_instructions"]

    res = explainer.explain_finding_structured(sample_finding)
    assert res.source == "deterministic_fallback"
    assert res.status == "fallback_offline"
    assert "[Deterministic Forensic Analysis]" in res.formatted_text


def test_explainer_check_status_missing_model(monkeypatch, sample_finding):
    """Verify check_ollama_status detects when Ollama is up but qwen model is missing."""
    import requests

    class MockResponse:
        status_code = 200

        def json(self):
            return {"models": [{"name": "llama3:8b"}, {"name": "mistral:latest"}]}

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse())
    explainer = ForensicExplainer()
    status = explainer.check_ollama_status()

    assert status["available"] is True
    assert status["model_installed"] is False
    assert status["status"] == "model_missing"
    assert "ollama pull qwen2.5:1.5b" in status["setup_instructions"]

    res = explainer.explain_finding_structured(sample_finding)
    assert res.source == "deterministic_fallback"
    assert res.status == "fallback_model_missing"


def test_explainer_check_status_ready(monkeypatch):
    """Verify check_ollama_status reports ready when configured lightweight model is installed."""
    import requests

    class MockResponse:
        status_code = 200

        def json(self):
            return {"models": [{"name": "qwen2.5:1.5b:latest"}]}

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse())
    explainer = ForensicExplainer()
    status = explainer.check_ollama_status()

    assert status["available"] is True
    assert status["model_installed"] is True
    assert status["status"] == "ready"
    assert explainer.is_available() is True


def test_explainer_valid_mocked_ai_response(monkeypatch, sample_finding):
    """Verify explainer parses valid structured JSON response from local Qwen model."""
    import requests

    monkeypatch.setattr(
        ForensicExplainer,
        "check_ollama_status",
        lambda self: {"available": True, "model_installed": True, "status": "ready", "installed_models": [self.model]},
    )

    valid_json_payload = {
        "interpretation": "100% missing values indicate extraction failure.",
        "root_causes": "Upstream ETL column mapping omitted.",
        "investigation_steps": "Check SQL extraction query and data warehouse schema.",
        "limitations": "Optional clinical fields may be conditionally missing.",
    }

    class MockPostResponse:
        status_code = 200

        def json(self):
            return {"response": json.dumps(valid_json_payload)}

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockPostResponse())

    explainer = ForensicExplainer()
    result = explainer.explain_finding_structured(sample_finding)

    assert result.source == "ai_qwen"
    assert result.status == "success"
    assert result.model == "qwen2.5:1.5b"
    assert result.interpretation == valid_json_payload["interpretation"]
    assert result.root_causes == valid_json_payload["root_causes"]
    assert "[AI Forensic Interpretation (qwen2.5:1.5b)]" in result.formatted_text
    assert "Advisory Notice: AI-generated interpretation" in result.formatted_text

    text = explainer.generate_explanation(sample_finding)
    assert "[AI Forensic Interpretation (qwen2.5:1.5b)]" in text


def test_explainer_malformed_response_fallback(monkeypatch, sample_finding):
    """Verify explainer falls back to deterministic rules when model returns malformed JSON."""
    import requests

    monkeypatch.setattr(
        ForensicExplainer,
        "check_ollama_status",
        lambda self: {"available": True, "model_installed": True, "status": "ready"},
    )

    class MockMalformedResponse:
        status_code = 200

        def json(self):
            return {"response": "This is non-JSON rambling text without required keys."}

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockMalformedResponse())

    explainer = ForensicExplainer()
    result = explainer.explain_finding_structured(sample_finding)

    assert result.source == "deterministic_fallback"
    assert result.status == "fallback_malformed"
    assert "[Deterministic Forensic Analysis]" in result.formatted_text


def test_explainer_timeout_fallback(monkeypatch, sample_finding):
    """Verify explainer handles local generation timeouts with deterministic fallback."""
    import requests

    monkeypatch.setattr(
        ForensicExplainer,
        "check_ollama_status",
        lambda self: {"available": True, "model_installed": True, "status": "ready"},
    )

    def mock_post_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("Local generation timed out")

    monkeypatch.setattr(requests, "post", mock_post_timeout)

    explainer = ForensicExplainer()
    result = explainer.explain_finding_structured(sample_finding)

    assert result.source == "deterministic_fallback"
    assert result.status == "fallback_timeout"
    assert "[Deterministic Forensic Analysis]" in result.formatted_text


def test_explainer_connection_error_fallback(monkeypatch, sample_finding):
    """Verify explainer handles connection drop during generation gracefully."""
    import requests

    monkeypatch.setattr(
        ForensicExplainer,
        "check_ollama_status",
        lambda self: {"available": True, "model_installed": True, "status": "ready"},
    )

    def mock_post_conn_err(*args, **kwargs):
        raise requests.exceptions.ConnectionError("Ollama daemon crashed")

    monkeypatch.setattr(requests, "post", mock_post_conn_err)

    explainer = ForensicExplainer()
    result = explainer.explain_finding_structured(sample_finding)

    assert result.source == "deterministic_fallback"
    assert result.status == "fallback_offline"
    assert "[Deterministic Forensic Analysis]" in result.formatted_text


def test_explainer_guard_against_heavy_models():
    """Verify explainer and config strictly reject 7B, 14B or larger models, defaulting to 1.5b."""
    from config import AppConfig, _get_qwen_model

    # Direct config helper tests
    import os
    os.environ["QWEN_MODEL"] = "qwen2.5:7b"
    assert _get_qwen_model() == "qwen2.5:1.5b"

    os.environ["QWEN_MODEL"] = "qwen2.5:14b"
    assert _get_qwen_model() == "qwen2.5:1.5b"

    os.environ["QWEN_MODEL"] = "qwen2.5:0.5b"
    assert _get_qwen_model() == "qwen2.5:0.5b"

    del os.environ["QWEN_MODEL"]

    # Explainer sanitizer tests
    assert ForensicExplainer._sanitize_model_name("qwen2.5:7b") == "qwen2.5:1.5b"
    assert ForensicExplainer._sanitize_model_name("llama3:8b") == "qwen2.5:1.5b"
    assert ForensicExplainer._sanitize_model_name("qwen2.5:14b") == "qwen2.5:1.5b"
    assert ForensicExplainer._sanitize_model_name("qwen2.5:0.5b") == "qwen2.5:0.5b"


def test_explainer_minimal_evidence_payload(sample_finding):
    """Verify minimum necessary evidence extraction does not leak raw rows or large payloads."""
    # Add fake row indices and samples to finding
    sample_finding.evidence.row_indices = list(range(5000))
    sample_finding.evidence.sample_values = [{"row": i, "val": i * 10} for i in range(100)]

    explainer = ForensicExplainer()
    payload = explainer._extract_minimal_evidence(sample_finding)

    assert "row_indices" not in payload
    assert "sample_values" not in payload
    assert payload["rule_id"] == "INT-001"
    assert payload["metrics"] == {"null_percentage": 100.0}
    assert payload["affected_columns"] == ["test_col"]
    assert len(json.dumps(payload)) < 500


def test_report_exporter_json_and_markdown(sample_finding, tmp_path):
    """Verify dossier export to both JSON and Markdown formats."""
    df = pd.DataFrame({"test_col": [None, None, None]})
    meta = DatasetMetadata(
        filename="test_dataset.csv",
        file_size_bytes=42,
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        encoding="utf-8",
        delimiter=",",
        row_count=3,
        column_count=1,
        loaded_at="2026-10-19T00:00:00Z",
    )
    report = generate_autopsy_report(df, meta)

    # JSON export
    json_output = ReportExporter.to_json(report)
    parsed = json.loads(json_output)
    assert parsed["metadata"]["filename"] == "test_dataset.csv"
    assert "test_col" in parsed["dataset_profile"]["columns"]

    # Markdown export
    md_output = ReportExporter.to_markdown(report)
    assert "# DATA AUTOPSY: Forensic Examination Dossier" in md_output
    assert "test_dataset.csv" in md_output
    assert "SHA256 Digest" in md_output

    # File saving
    file_path = tmp_path / "dossier.json"
    saved = ReportExporter.save_to_file(report, file_path, "json")
    assert saved.exists()
    assert saved.read_text(encoding="utf-8").startswith("{")


def test_report_exporter_with_baseline_comparison(tmp_path):
    """Verify dossier export includes baseline comparison section when target leakage is evaluated."""
    import numpy as np
    rng = np.random.RandomState(42)
    n = 60
    y = np.array([0, 1] * 30)
    df = pd.DataFrame({
        "target": y,
        "leak_col": y.astype(float),
        "clean_feature": rng.randn(n),
    })
    meta = DatasetMetadata(
        filename="leakage_dataset.csv",
        file_size_bytes=1024,
        sha256_hash="abcdef1234567890",
        encoding="utf-8",
        delimiter=",",
        row_count=n,
        column_count=3,
        loaded_at="2026-10-19T00:00:00Z",
    )
    report = generate_autopsy_report(df, meta, target_column="target")
    assert report.baseline_comparison is not None

    md_output = ReportExporter.to_markdown(report)
    assert "## 1.1. Empirical Pipeline Baseline Comparison" in md_output
    assert "leak_col" in md_output
    assert "Performance Shift (Delta)" in md_output

    json_output = ReportExporter.to_json(report)
    parsed = json.loads(json_output)
    assert "baseline_comparison" in parsed
    assert parsed["baseline_comparison"]["status"] == "success"
