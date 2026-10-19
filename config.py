"""Configuration settings for Data Autopsy forensic investigation engine."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List

# Base paths
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
REPORTS_DIR = BASE_DIR / "reports_output"


def _get_env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_qwen_model() -> str:
    """Resolve local lightweight Qwen model. Defaults to qwen2.5:1.5b.
    
    Guards strictly against 7B, 14B or larger models to keep workstation responsive.
    """
    raw = os.getenv("QWEN_MODEL") or os.getenv("OLLAMA_MODEL") or "qwen2.5:1.5b"
    raw_cleaned = raw.strip()
    raw_lower = raw_cleaned.lower()
    disallowed_sizes = ["7b", "8b", "14b", "32b", "70b", "72b"]
    if any(sz in raw_lower for sz in disallowed_sizes):
        return "qwen2.5:1.5b"
    return raw_cleaned


@dataclass(frozen=True)
class AppConfig:
    """Application-wide configuration parameters."""

    # File loading limits & validation
    max_file_size_mb: int = field(
        default_factory=lambda: _get_env_int("DATA_AUTOPSY_MAX_FILE_SIZE_MB", 100)
    )
    supported_encodings: List[str] = field(
        default_factory=lambda: ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    )
    supported_delimiters: List[str] = field(
        default_factory=lambda: [",", ";", "\t", "|"]
    )

    # Profiling & Investigation limits
    max_rows_deep_profile: int = field(
        default_factory=lambda: _get_env_int("DATA_AUTOPSY_MAX_ROWS_DEEP_PROFILE", 500_000)
    )
    sample_preview_rows: int = 10
    duplicate_check_sample_max_rows: int = 100_000

    # Detection thresholds
    null_ratio_warning_threshold: float = field(
        default_factory=lambda: _get_env_float("NULL_RATIO_WARNING_THRESHOLD", 0.20)
    )
    null_ratio_critical_threshold: float = field(
        default_factory=lambda: _get_env_float("NULL_RATIO_CRITICAL_THRESHOLD", 0.50)
    )
    imbalance_ratio_threshold: float = field(
        default_factory=lambda: _get_env_float("IMBALANCE_RATIO_THRESHOLD", 4.0)
    )
    imbalance_ratio_critical: float = field(
        default_factory=lambda: _get_env_float("IMBALANCE_RATIO_CRITICAL", 20.0)
    )
    near_constant_threshold: float = field(
        default_factory=lambda: _get_env_float("NEAR_CONSTANT_THRESHOLD", 0.99)
    )
    identifier_unique_ratio_threshold: float = field(
        default_factory=lambda: _get_env_float("IDENTIFIER_UNIQUE_RATIO_THRESHOLD", 0.98)
    )
    high_cardinality_unique_count: int = field(
        default_factory=lambda: _get_env_int("HIGH_CARDINALITY_UNIQUE_COUNT", 50)
    )
    high_cardinality_ratio_threshold: float = field(
        default_factory=lambda: _get_env_float("HIGH_CARDINALITY_RATIO_THRESHOLD", 0.30)
    )
    outlier_iqr_multiplier: float = field(
        default_factory=lambda: _get_env_float("OUTLIER_IQR_MULTIPLIER", 1.5)
    )
    outlier_modified_z_threshold: float = field(
        default_factory=lambda: _get_env_float("OUTLIER_MODIFIED_Z_THRESHOLD", 3.5)
    )
    correlation_high_threshold: float = field(
        default_factory=lambda: _get_env_float("CORRELATION_HIGH_THRESHOLD", 0.90)
    )
    correlation_extreme_threshold: float = field(
        default_factory=lambda: _get_env_float("CORRELATION_EXTREME_THRESHOLD", 0.98)
    )
    leakage_threshold: float = field(
        default_factory=lambda: _get_env_float("LEAKAGE_THRESHOLD", 0.95)
    )
    max_correlation_columns: int = field(
        default_factory=lambda: _get_env_int("MAX_CORRELATION_COLUMNS", 50)
    )
    max_correlation_sample_rows: int = field(
        default_factory=lambda: _get_env_int("MAX_CORRELATION_SAMPLE_ROWS", 50_000)
    )

    # Ollama Local LLM Configuration (strictly local - no data leaks)
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=_get_qwen_model
    )
    ollama_timeout_seconds: int = field(
        default_factory=lambda: _get_env_int("OLLAMA_TIMEOUT_SECONDS", 30)
    )

    # Logging
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    @property
    def max_file_size_bytes(self) -> int:
        """Return maximum file size converted to bytes."""
        return self.max_file_size_mb * 1024 * 1024


# Default global instance
config = AppConfig()
