"""Forensic dataset and column profiler for tabular data."""

from datetime import datetime, timezone
import platform
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config import AppConfig, config as default_config
from engine.limitations import get_analysis_limitations
from engine.models import (
    AnalysisMetadata,
    AutopsyReport,
    ColumnProfile,
    DatasetMetadata,
    DatasetProfile,
    Evidence,
    Finding,
    FindingCategory,
    Severity,
)
from engine.ranking import rank_findings
from engine.utils import execution_timer, to_python_scalar


def infer_column_type(series: pd.Series) -> str:
    """Infer the logical forensic type of a pandas Series.
    
    Categories: 'numeric', 'categorical', 'boolean', 'datetime', 'id_like', 'text', 'empty'
    """
    valid = series.dropna()
    if valid.empty:
        return "empty"

    # Boolean check
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if valid.isin([0, 1, "0", "1", True, False, "true", "false", "True", "False"]).all() and len(valid.unique()) <= 2:
        return "boolean"

    # Numeric check
    if pd.api.types.is_numeric_dtype(series):
        # Check if it acts like a sequential ID column
        if len(valid) > 20 and valid.nunique() == len(valid):
            if np.issubdtype(valid.dtype, np.integer):
                sorted_vals = np.sort(valid.values)
                diffs = np.diff(sorted_vals)
                if np.all(diffs == 1):
                    return "id_like"
        return "numeric"

    # Datetime check
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Object / String checks
    sample = valid.astype(str).head(100)
    # Check if looks like dates
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.85:
            return "datetime"
    except Exception:
        pass

    # Categorical vs Text vs ID
    unique_count = valid.nunique()
    total_count = len(valid)
    unique_ratio = unique_count / total_count if total_count > 0 else 0.0

    if unique_ratio > 0.95 and total_count > 30:
        return "id_like"
    if unique_count <= 50 or unique_ratio < 0.20:
        return "categorical"

    return "text"


def profile_column(series: pd.Series, max_sample_values: int = 5) -> ColumnProfile:
    """Compute detailed forensic profile for a single column without altering data."""
    total_count = int(len(series))
    null_count = int(series.isna().sum())
    null_percentage = round((null_count / total_count * 100), 2) if total_count > 0 else 0.0

    valid = series.dropna()
    unique_count = int(valid.nunique())
    unique_percentage = round((unique_count / total_count * 100), 2) if total_count > 0 else 0.0
    memory_bytes = int(series.memory_usage(deep=True))
    inferred = infer_column_type(series)

    summary_stats: Dict[str, Any] = {}
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) and not valid.empty:
        zeros_cnt = int((valid == 0).sum())
        summary_stats = {
            "min": to_python_scalar(valid.min()),
            "max": to_python_scalar(valid.max()),
            "mean": to_python_scalar(round(float(valid.mean()), 4)),
            "std": to_python_scalar(round(float(valid.std()), 4)) if len(valid) > 1 else 0.0,
            "median": to_python_scalar(valid.median()),
            "q25": to_python_scalar(valid.quantile(0.25)),
            "q75": to_python_scalar(valid.quantile(0.75)),
            "zeros_count": zeros_cnt,
            "zeros_percentage": round(zeros_cnt / total_count * 100, 2) if total_count > 0 else 0.0,
        }
    elif not valid.empty:
        top_counts = valid.value_counts().head(5)
        mode_val = valid.mode().iloc[0] if not valid.mode().empty else None
        summary_stats = {
            "mode": to_python_scalar(mode_val),
            "top_values": {str(k): int(v) for k, v in top_counts.items()},
            "cardinality_ratio": round(float(unique_count / total_count), 4) if total_count > 0 else 0.0,
        }

    sample_vals = [to_python_scalar(x) for x in valid.head(max_sample_values).tolist()]

    return ColumnProfile(
        name=str(series.name),
        original_dtype=str(series.dtype),
        inferred_type=inferred,
        total_count=total_count,
        null_count=null_count,
        null_percentage=null_percentage,
        unique_count=unique_count,
        unique_percentage=unique_percentage,
        memory_bytes=memory_bytes,
        summary_stats=summary_stats,
        sample_values=sample_vals,
    )


def profile_dataset(
    df: pd.DataFrame,
    metadata: DatasetMetadata,
    cfg: AppConfig = default_config,
) -> DatasetProfile:
    """Perform structural and statistical profiling on the entire dataset."""
    col_profiles: Dict[str, ColumnProfile] = {}
    total_memory = int(df.memory_usage(deep=True).sum()) if not df.empty else 0

    for col in df.columns:
        col_profiles[str(col)] = profile_column(df[col], cfg.sample_preview_rows)

    # Detect duplicate rows preserving original 0-indexed row locations
    duplicate_count = 0
    duplicate_pct = 0.0
    duplicate_indices: List[int] = []

    if not df.empty and len(df) > 0:
        duplicate_mask = df.duplicated(keep="first")
        duplicate_count = int(duplicate_mask.sum())
        duplicate_pct = round((duplicate_count / len(df) * 100), 2)
        if duplicate_count > 0:
            dup_series = df.index[duplicate_mask]
            duplicate_indices = [int(i) for i in dup_series[:1000].tolist()]

    # Overall cells and nulls summary
    total_cells = int(len(df) * len(df.columns)) if not df.empty else 0
    total_nulls = int(df.isna().sum().sum()) if not df.empty else 0
    null_cells_pct = round(total_nulls / total_cells * 100, 2) if total_cells > 0 else 0.0

    # Inferred column type frequency distribution
    type_counts: Dict[str, int] = {}
    for col_prof in col_profiles.values():
        type_counts[col_prof.inferred_type] = type_counts.get(col_prof.inferred_type, 0) + 1

    return DatasetProfile(
        metadata=metadata,
        columns=col_profiles,
        total_memory_bytes=total_memory,
        duplicate_rows_count=duplicate_count,
        duplicate_rows_percentage=duplicate_pct,
        exact_duplicate_row_indices=duplicate_indices,
        total_cells=total_cells,
        total_null_cells=total_nulls,
        null_cells_percentage=null_cells_pct,
        column_type_counts=type_counts,
    )


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


def run_basic_integrity_checks(
    df: pd.DataFrame,
    profile: DatasetProfile,
    cfg: AppConfig = default_config,
) -> List[Finding]:
    """Execute baseline deterministic integrity checks on tabular data.
    
    Preserved for backwards compatibility.
    """
    findings: List[Finding] = []
    findings.extend(detect_missing_values(df, profile, cfg))
    findings.extend(detect_duplicate_records(df, profile, cfg))
    findings.extend(detect_constant_and_near_constant(df, profile, cfg))
    return findings


def generate_autopsy_report(
    df: pd.DataFrame,
    metadata: DatasetMetadata,
    target_column: Optional[str] = None,
    excluded_columns: Optional[List[str]] = None,
    cfg: AppConfig = default_config,
) -> AutopsyReport:
    """Generate complete forensic autopsy report with all 10 deterministic checks.
    
    Supports non-mutating column exclusion for subsequent forensic iterations.
    """
    # Defensive column exclusion: strictly non-mutating
    excluded = list(excluded_columns) if excluded_columns else []
    valid_excluded = [c for c in excluded if c in df.columns and c != target_column]

    if valid_excluded:
        active_df = df.drop(columns=valid_excluded)
    else:
        active_df = df

    started_at = datetime.now(timezone.utc).isoformat()
    with execution_timer() as timer:
        profile = profile_dataset(active_df, metadata, cfg)
        raw_findings = run_all_detectors(active_df, profile, target_column=target_column, cfg=cfg)
        ranked_findings = rank_findings(raw_findings, len(active_df))

    completed_at = datetime.now(timezone.utc).isoformat()

    baseline_res = None
    for f in ranked_findings:
        if "baseline_comparison" in f.evidence.metrics:
            baseline_res = f.evidence.metrics["baseline_comparison"]
            break

    analysis_meta = AnalysisMetadata(
        app_version="1.0.0",
        selected_target=target_column,
        excluded_columns=valid_excluded,
        analysis_settings={
            "null_ratio_critical": cfg.null_ratio_critical_threshold,
            "null_ratio_warning": cfg.null_ratio_warning_threshold,
            "correlation_threshold": cfg.correlation_high_threshold,
            "correlation_extreme": cfg.correlation_extreme_threshold,
            "leakage_threshold": cfg.leakage_threshold,
            "outlier_modified_z_threshold": cfg.outlier_modified_z_threshold,
            "outlier_iqr_multiplier": cfg.outlier_iqr_multiplier,
            "imbalance_ratio": cfg.imbalance_ratio_threshold,
        },
        checks_executed=[
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
        ],
        reproducibility_settings={
            "random_seed": 42,
            "deterministic_execution": True,
            "python_version": sys.version.split()[0],
            "os_platform": platform.system(),
        },
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=timer["duration_ms"],
    )

    return AutopsyReport(
        metadata=metadata,
        dataset_profile=profile,
        findings=ranked_findings,
        execution_duration_ms=timer["duration_ms"],
        completed_at=completed_at,
        analysis_metadata=analysis_meta,
        baseline_comparison=baseline_res,
        limitations=get_analysis_limitations(),
    )

