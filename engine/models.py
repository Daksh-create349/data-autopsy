"""Shared domain models and data structures for forensic investigation."""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class Severity(str, Enum):
    """Investigation finding severity levels."""
    INFORMATIONAL = "INFORMATIONAL"
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FindingCategory(str, Enum):
    """Forensic investigation finding categories."""
    INTEGRITY = "INTEGRITY"
    STATISTICAL = "STATISTICAL"
    LEAKAGE = "LEAKAGE"
    IMBALANCE = "IMBALANCE"
    DUPLICATION = "DUPLICATION"
    CORRELATION = "CORRELATION"
    OUTLIER = "OUTLIER"
    CARDINALITY = "CARDINALITY"


@dataclass
class Evidence:
    """Concrete, reproducible forensic evidence backing a finding."""
    description: str
    column_names: List[str] = field(default_factory=list)
    row_indices: List[int] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    sample_values: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence to a serializable dictionary."""
        return asdict(self)


@dataclass
class Finding:
    """A deterministic forensic finding with optional AI explanation."""
    id: str
    rule_id: str
    category: FindingCategory
    severity: Severity
    title: str
    summary: str
    evidence: Evidence
    check_name: str = ""
    why_it_matters: str = ""
    recommendation: Optional[str] = None
    limitations: str = ""
    ai_explanation: Optional[str] = None
    confidence_score: float = 1.0  # Statistical confidence in [0.0, 1.0] (deterministic empirical support; not a probability)
    rank_score: float = 0.0  # Composite deterministic ranking score

    @property
    def affected_columns(self) -> List[str]:
        return self.evidence.column_names

    @property
    def row_indices(self) -> List[int]:
        return self.evidence.row_indices

    @property
    def measured_values(self) -> Dict[str, Any]:
        return self.evidence.metrics

    @property
    def detection_thresholds(self) -> Dict[str, Any]:
        return self.evidence.thresholds

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary representation."""
        data = asdict(self)
        data["category"] = self.category.value
        data["severity"] = self.severity.value
        data["affected_columns"] = self.affected_columns
        data["row_indices"] = self.row_indices
        data["measured_values"] = self.measured_values
        data["detection_thresholds"] = self.detection_thresholds
        data["confidence_score"] = self.confidence_score
        data["rank_score"] = self.rank_score
        return data


@dataclass
class ColumnProfile:
    """Forensic profile of an individual column."""
    name: str
    original_dtype: str
    inferred_type: str
    total_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    memory_bytes: int
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    sample_values: List[Any] = field(default_factory=list)

    @property
    def zeros_count(self) -> int:
        return int(self.summary_stats.get("zeros_count", 0))

    @property
    def zeros_percentage(self) -> float:
        return float(self.summary_stats.get("zeros_percentage", 0.0))

    @property
    def mean(self) -> Optional[float]:
        val = self.summary_stats.get("mean")
        return float(val) if val is not None else None

    @property
    def median(self) -> Optional[float]:
        val = self.summary_stats.get("median")
        return float(val) if val is not None else None

    @property
    def std_dev(self) -> Optional[float]:
        val = self.summary_stats.get("std")
        return float(val) if val is not None else None

    @property
    def mode(self) -> Any:
        return self.summary_stats.get("mode")

    def to_dict(self) -> Dict[str, Any]:
        """Convert column profile to dictionary."""
        return asdict(self)


@dataclass
class DatasetMetadata:
    """File and structural metadata for an uploaded dataset."""
    filename: str
    file_size_bytes: int
    sha256_hash: str
    encoding: str
    delimiter: str
    row_count: int
    column_count: int
    loaded_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)


@dataclass
class DatasetProfile:
    """Comprehensive forensic profile of a tabular dataset."""
    metadata: DatasetMetadata
    columns: Dict[str, ColumnProfile]
    total_memory_bytes: int
    duplicate_rows_count: int
    duplicate_rows_percentage: float
    exact_duplicate_row_indices: List[int] = field(default_factory=list)
    total_cells: int = 0
    total_null_cells: int = 0
    null_cells_percentage: float = 0.0
    column_type_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset profile to serializable dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "columns": {col: prof.to_dict() for col, prof in self.columns.items()},
            "total_memory_bytes": self.total_memory_bytes,
            "duplicate_rows_count": self.duplicate_rows_count,
            "duplicate_rows_percentage": self.duplicate_rows_percentage,
            "exact_duplicate_row_indices": self.exact_duplicate_row_indices,
            "total_cells": self.total_cells,
            "total_null_cells": self.total_null_cells,
            "null_cells_percentage": self.null_cells_percentage,
            "column_type_counts": self.column_type_counts,
        }


@dataclass
class AnalysisMetadata:
    """Forensic metadata detailing analysis parameters, settings, and reproducibility context."""
    app_version: str
    selected_target: Optional[str]
    excluded_columns: List[str]
    analysis_settings: Dict[str, Any]
    checks_executed: List[str]
    reproducibility_settings: Dict[str, Any]
    started_at: str
    completed_at: str
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis metadata to dictionary."""
        return asdict(self)


@dataclass
class AutopsyReport:
    """Complete forensic autopsy report combining profiling and findings."""
    metadata: DatasetMetadata
    dataset_profile: DatasetProfile
    findings: List[Finding]
    execution_duration_ms: float
    completed_at: str
    analysis_metadata: Optional[AnalysisMetadata] = None
    baseline_comparison: Optional[Dict[str, Any]] = None
    limitations: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert autopsy report to JSON-serializable dictionary."""
        data = {
            "metadata": self.metadata.to_dict(),
            "dataset_profile": self.dataset_profile.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "execution_duration_ms": self.execution_duration_ms,
            "completed_at": self.completed_at,
        }
        if self.analysis_metadata is not None:
            data["analysis_metadata"] = self.analysis_metadata.to_dict()
        if self.baseline_comparison is not None:
            data["baseline_comparison"] = self.baseline_comparison
        if self.limitations is not None:
            data["limitations"] = self.limitations
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to indented JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

