"""Deterministic forensic anomaly detectors for tabular datasets."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from config import AppConfig, config as default_config
from engine.models import (
    DatasetProfile,
    Evidence,
    Finding,
    FindingCategory,
    Severity,
)
from engine.utils import to_python_scalar


def detect_missing_values(
    df: pd.DataFrame,
    profile: DatasetProfile,
    cfg: AppConfig = default_config,
) -> List[Finding]:
    """Check 1: Missing values (counts, percentages, exact row indices)."""
    findings: List[Finding] = []

    for col_name, col_prof in profile.columns.items():
        if col_prof.null_count == 0:
            continue

        null_indices = [int(i) for i in df.index[df[col_name].isna()][:50].tolist()]
        sample_nulls = [{"row_index": int(i), "column": col_name, "value": None} for i in null_indices[:5]]

        if col_prof.null_percentage == 100.0:
            findings.append(
                Finding(
                    id=f"INT-001-{col_name}",
                    rule_id="INT-001",
                    check_name="missing_values",
                    category=FindingCategory.INTEGRITY,
                    severity=Severity.CRITICAL,
                    title=f"Complete Null Column: '{col_name}' (100.0% Missing)",
                    summary=f"Column '{col_name}' has {col_prof.total_count:,} records, all of which are unpopulated (NaN / Null).",
                    evidence=Evidence(
                        description="Entire column lacks any observable values.",
                        column_names=[col_name],
                        row_indices=null_indices,
                        metrics={
                            "null_count": col_prof.null_count,
                            "null_percentage": 100.0,
                            "total_rows": col_prof.total_count,
                        },
                        thresholds={"critical_threshold": 100.0},
                        sample_values=sample_nulls,
                    ),
                    why_it_matters="Completely unpopulated features provide zero predictive or analytical utility and cause downstream modeling pipelines to fail.",
                    recommendation=f"Drop column '{col_name}' and inspect data pipeline extraction queries.",
                    limitations="None. 100% missing columns contain no recoverable information.",
                )
            )
        elif col_prof.null_percentage >= (cfg.null_ratio_critical_threshold * 100):
            findings.append(
                Finding(
                    id=f"INT-002-{col_name}",
                    rule_id="INT-002",
                    check_name="missing_values",
                    category=FindingCategory.INTEGRITY,
                    severity=Severity.HIGH,
                    title=f"Critical Missing Rate in '{col_name}' ({col_prof.null_percentage:.1f}%)",
                    summary=f"Column '{col_name}' is missing {col_prof.null_count:,} of {col_prof.total_count:,} values.",
                    evidence=Evidence(
                        description=f"{col_prof.null_count:,} records lack values in column '{col_name}'.",
                        column_names=[col_name],
                        row_indices=null_indices,
                        metrics={
                            "null_count": col_prof.null_count,
                            "null_percentage": col_prof.null_percentage,
                            "total_rows": col_prof.total_count,
                        },
                        thresholds={"critical_threshold": cfg.null_ratio_critical_threshold * 100},
                        sample_values=sample_nulls,
                    ),
                    why_it_matters="Severe missingness causes large sample loss under listwise deletion or severe distortion if naively imputed.",
                    recommendation="Investigate root cause (MCAR, MAR, MNAR); consider dropping or specialized indicator encoding.",
                    limitations="Optional survey fields or sparse event indicators legitimately exhibit elevated missing rates.",
                )
            )
        elif col_prof.null_percentage >= (cfg.null_ratio_warning_threshold * 100):
            findings.append(
                Finding(
                    id=f"INT-003-{col_name}",
                    rule_id="INT-003",
                    check_name="missing_values",
                    category=FindingCategory.INTEGRITY,
                    severity=Severity.MEDIUM,
                    title=f"Elevated Missing Rate in '{col_name}' ({col_prof.null_percentage:.1f}%)",
                    summary=f"Column '{col_name}' missing rate exceeds warning threshold of {cfg.null_ratio_warning_threshold * 100:.0f}%.",
                    evidence=Evidence(
                        description=f"{col_prof.null_count:,} missing records detected in '{col_name}'.",
                        column_names=[col_name],
                        row_indices=null_indices,
                        metrics={
                            "null_count": col_prof.null_count,
                            "null_percentage": col_prof.null_percentage,
                            "total_rows": col_prof.total_count,
                        },
                        thresholds={"warning_threshold": cfg.null_ratio_warning_threshold * 100},
                        sample_values=sample_nulls,
                    ),
                    why_it_matters="Moderate missingness requires explicit imputation strategy (KNN, IterativeImputer, median) before statistical analysis.",
                    recommendation="Evaluate missingness mechanism and select non-biasing imputation method.",
                    limitations="Moderate missing rates are common in real-world tabular data.",
                )
            )
        else:
            findings.append(
                Finding(
                    id=f"INT-004-{col_name}",
                    rule_id="INT-004",
                    check_name="missing_values",
                    category=FindingCategory.INTEGRITY,
                    severity=Severity.LOW,
                    title=f"Sparse Missing Values in '{col_name}' ({col_prof.null_percentage:.2f}%)",
                    summary=f"Column '{col_name}' contains {col_prof.null_count:,} missing values.",
                    evidence=Evidence(
                        description=f"{col_prof.null_count:,} isolated missing values observed.",
                        column_names=[col_name],
                        row_indices=null_indices,
                        metrics={
                            "null_count": col_prof.null_count,
                            "null_percentage": col_prof.null_percentage,
                            "total_rows": col_prof.total_count,
                        },
                        thresholds={"info_threshold": 0.0},
                        sample_values=sample_nulls,
                    ),
                    why_it_matters="Isolated missing values can still crash strict algorithms without automatic null handling.",
                    recommendation="Apply standard univariate imputation (median/mode) or drop affected rows if negligible.",
                    limitations="Routine missing values occurring in operational pipelines.",
                )
            )

    return findings


def detect_duplicate_records(
    df: pd.DataFrame,
    profile: DatasetProfile,
    cfg: AppConfig = default_config,
) -> List[Finding]:
    """Check 2: Exact duplicate rows and candidate duplicate records."""
    findings: List[Finding] = []

    if df.empty or len(df) < 2:
        return findings

    # Part A: Exact duplicate rows across all columns
    exact_mask = df.duplicated(keep="first")
    exact_count = int(exact_mask.sum())

    if exact_count > 0:
        exact_pct = round(exact_count / len(df) * 100, 2)
        exact_indices = [int(i) for i in df.index[exact_mask][:50].tolist()]
        first_col = list(df.columns)[0]
        exact_samples = [
            {"row_index": int(i), "column": first_col, "value": to_python_scalar(df.loc[i, first_col])}
            for i in exact_indices[:5]
        ]

        findings.append(
            Finding(
                id="DUP-001",
                rule_id="DUP-001",
                check_name="exact_duplicate_rows",
                category=FindingCategory.DUPLICATION,
                severity=Severity.HIGH if exact_pct > 5.0 else Severity.MEDIUM,
                title=f"{exact_count:,} Exact Duplicate Rows Detected ({exact_pct:.2f}%)",
                summary=f"Identical values found across all columns for {exact_count:,} rows.",
                evidence=Evidence(
                    description=f"Full row vector matches another preceding row in the dataset.",
                    row_indices=exact_indices,
                    metrics={
                        "duplicate_rows_count": exact_count,
                        "duplicate_rows_percentage": exact_pct,
                        "total_rows": len(df),
                    },
                    thresholds={"duplicate_count_threshold": 0},
                    sample_values=exact_samples,
                ),
                why_it_matters="Exact duplicates falsely inflate statistical significance, bias gradient updates, and cause severe train/test data leakage.",
                recommendation="Deduplicate records prior to splitting or modeling unless identical entries represent independent valid events.",
                limitations="Low-dimensional discrete tables (e.g. 2 binary features) naturally produce duplicate observations by chance.",
            )
        )

    # Part B: Candidate near-duplicates (identical across non-ID features)
    candidate_id_cols = {
        c for c, p in profile.columns.items()
        if p.inferred_type == "id_like"
        or p.unique_percentage == 100.0
        or c.lower() in ["id", "uuid", "guid", "key", "index"]
        or c.lower().endswith(("_id", "_key", "id"))
    }
    non_id_cols = [c for c in df.columns if c not in candidate_id_cols]

    if 1 <= len(non_id_cols) < len(df.columns):
        subset_mask = df.duplicated(subset=non_id_cols, keep="first") & (~exact_mask)
        cand_count = int(subset_mask.sum())
        if cand_count > 0:
            cand_pct = round(cand_count / len(df) * 100, 2)
            cand_indices = [int(i) for i in df.index[subset_mask][:50].tolist()]
            cand_col = non_id_cols[0]
            cand_samples = [
                {"row_index": int(i), "column": cand_col, "value": to_python_scalar(df.loc[i, cand_col])}
                for i in cand_indices[:5]
            ]
            findings.append(
                Finding(
                    id="DUP-002",
                    rule_id="DUP-002",
                    check_name="duplicate_candidates",
                    category=FindingCategory.DUPLICATION,
                    severity=Severity.MEDIUM if cand_pct > 2.0 else Severity.LOW,
                    title=f"{cand_count:,} Duplicate Candidate Records (Near-Duplicates Detected)",
                    summary=f"Found {cand_count:,} records that share identical values across non-identifier columns ({', '.join(non_id_cols[:5])}).",
                    evidence=Evidence(
                        description="Records are identical except for surrogate key / identifier columns.",
                        column_names=non_id_cols,
                        row_indices=cand_indices,
                        metrics={
                            "candidate_count": cand_count,
                            "candidate_percentage": cand_pct,
                            "matching_columns_count": len(non_id_cols),
                        },
                        thresholds={"candidate_count_threshold": 1},
                        sample_values=cand_samples,
                    ),
                    why_it_matters="Near-duplicates often indicate re-submissions, data entry duplicates with new auto-generated IDs, or imperfect sync merges.",
                    recommendation="Investigate if flagged records represent duplicate business entities that should be merged.",
                    limitations="Datasets with limited feature resolution can produce identical feature vectors for distinct individuals.",
                )
            )

    return findings


def detect_constant_and_near_constant(
    df: pd.DataFrame,
    profile: DatasetProfile,
    cfg: AppConfig = default_config,
) -> List[Finding]:
    """Check 3: Constant and near-constant columns."""
    findings: List[Finding] = []

    for col_name, col_prof in profile.columns.items():
        if col_prof.null_percentage == 100.0:
            continue

        valid = df[col_name].dropna()
        if valid.empty:
            continue

        # Zero variance / Constant
        if col_prof.unique_count == 1:
            val = col_prof.sample_values[0] if col_prof.sample_values else None
            const_samples = [
                {"row_index": int(i), "column": col_name, "value": to_python_scalar(df.loc[i, col_name])}
                for i in valid.index[:5]
            ]
            findings.append(
                Finding(
                    id=f"STAT-001-{col_name}",
                    rule_id="STAT-001",
                    check_name="constant_columns",
                    category=FindingCategory.STATISTICAL,
                    severity=Severity.MEDIUM,
                    title=f"Zero Variance / Constant Column: '{col_name}'",
                    summary=f"Column '{col_name}' holds single constant value {repr(val)} across all {col_prof.total_count:,} records.",
                    evidence=Evidence(
                        description="Unique non-null count is exactly 1.",
                        column_names=[col_name],
                        row_indices=[int(i) for i in valid.index[:50].tolist()],
                        metrics={
                            "unique_count": 1,
                            "constant_value": to_python_scalar(val),
                            "total_rows": col_prof.total_count,
                        },
                        thresholds={"unique_count": 1},
                        sample_values=const_samples,
                    ),
                    why_it_matters="Provides zero statistical variance or information gain; can cause division-by-zero or non-invertible covariance matrices in linear models.",
                    recommendation="Drop constant column prior to feature scaling and model training.",
                    limitations="None. Constant columns provide zero mutual information.",
                )
            )
        # Near-constant
        elif len(valid) >= 20:
            counts = valid.value_counts()
            if not counts.empty:
                dom_count = int(counts.iloc[0])
                dom_val = counts.index[0]
                dom_pct = round(dom_count / len(valid) * 100, 2)
                threshold_pct = cfg.near_constant_threshold * 100

                if dom_pct >= threshold_pct:
                    rare_indices = [int(i) for i in valid.index[valid != dom_val][:20].tolist()]
                    near_const_samples = [
                        {"row_index": int(i), "column": col_name, "value": to_python_scalar(df.loc[i, col_name])}
                        for i in rare_indices[:5]
                    ]
                    findings.append(
                        Finding(
                            id=f"STAT-002-{col_name}",
                            rule_id="STAT-002",
                            check_name="near_constant_columns",
                            category=FindingCategory.STATISTICAL,
                            severity=Severity.LOW,
                            title=f"Near-Constant Column: '{col_name}' ({dom_pct:.2f}% Single Value)",
                            summary=f"Dominant value {repr(dom_val)} accounts for {dom_count:,} of {len(valid):,} valid observations ({dom_pct:.2f}%).",
                            evidence=Evidence(
                                description=f"Single value dominates >= {threshold_pct:.0f}% of valid rows.",
                                column_names=[col_name],
                                row_indices=rare_indices,
                                metrics={
                                    "dominant_value": to_python_scalar(dom_val),
                                    "dominant_count": dom_count,
                                    "dominant_percentage": dom_pct,
                                    "rare_variants_count": len(valid) - dom_count,
                                },
                                thresholds={"near_constant_threshold": threshold_pct},
                                sample_values=near_const_samples,
                            ),
                            why_it_matters="Near-zero variance may cause tree splits to ignore the rare variant or cause numerical instability in regularized linear models.",
                            recommendation="Verify whether the rare value represents a critical anomaly signal (e.g. rare failure) or noise before dropping.",
                            limitations="In fraud or safety-critical monitoring, near-constant flags with rare 1s represent the primary target signal.",
                        )
                    )

    return findings


def detect_potential_identifiers(
    df: pd.DataFrame,
    profile: DatasetProfile,
    cfg: AppConfig = default_config,
) -> List[Finding]:
    """Check 4: Unique-value counts and potential identifier columns."""
    findings: List[Finding] = []

    if df.empty or len(df) < 20:
        return findings

    for col_name, col_prof in profile.columns.items():
        if col_prof.null_percentage == 100.0:
            continue

        valid = df[col_name].dropna()
        if len(valid) < 20:
            continue

        unique_ratio = valid.nunique() / len(valid)
        threshold = cfg.identifier_unique_ratio_threshold

        if unique_ratio >= threshold and col_prof.inferred_type != "text":
            id_indices = [int(i) for i in valid.index[:50].tolist()]
            id_samples = [
                {"row_index": int(i), "column": col_name, "value": to_python_scalar(df.loc[i, col_name])}
                for i in valid.index[:5]
            ]
            findings.append(
                Finding(
                    id=f"STAT-003-{col_name}",
                    rule_id="STAT-003",
                    check_name="potential_identifiers",
                    category=FindingCategory.STATISTICAL,
                    severity=Severity.INFORMATIONAL,
                    title=f"Potential Identifier Column: '{col_name}' ({col_prof.unique_percentage:.1f}% Unique)",
                    summary=f"Column '{col_name}' has {col_prof.unique_count:,} unique values across {len(valid):,} records (ratio {unique_ratio:.3f}).",
                    evidence=Evidence(
                        description="Column exhibits uniqueness characteristic of primary or surrogate keys.",
                        column_names=[col_name],
                        row_indices=id_indices,
                        metrics={
                            "unique_count": col_prof.unique_count,
                            "unique_ratio": round(unique_ratio, 4),
                            "unique_percentage": col_prof.unique_percentage,
                            "inferred_type": col_prof.inferred_type,
                        },
                        thresholds={"unique_ratio_threshold": threshold},
                        sample_values=id_samples,
                    ),
                    why_it_matters="High-uniqueness keys (IDs, UUIDs, counters) cause severe memorization and overfitting if passed as raw features to ML models.",
                    recommendation="Exclude identifier column from model training; use exclusively for tracking or joins.",
                    limitations="Continuous floating-point measurements with high precision can legitimately have 100% uniqueness without being synthetic keys.",
                )
            )

    return findings


def detect_numeric_outliers(
    df: pd.DataFrame,
    profile: DatasetProfile,
    cfg: AppConfig = default_config,
) -> List[Finding]:
    """Check 5: Numeric outliers using IQR and Boris Iglewicz/David Hoaglin modified Z-scores."""
    findings: List[Finding] = []

    if df.empty or len(df) < 10:
        return findings

    for col_name, col_prof in profile.columns.items():
        if col_prof.inferred_type not in ["numeric"]:
            continue

        valid = pd.to_numeric(df[col_name], errors="coerce").dropna()
        if len(valid) < 10 or valid.nunique() <= 2:
            continue

        # Method A: Interquartile Range (IQR)
        q25 = float(valid.quantile(0.25))
        q75 = float(valid.quantile(0.75))
        iqr = q75 - q25

        if iqr <= 0:
            continue

        lower_bound = q25 - (cfg.outlier_iqr_multiplier * iqr)
        upper_bound = q75 + (cfg.outlier_iqr_multiplier * iqr)
        extreme_lower = q25 - (3.0 * iqr)
        extreme_upper = q75 + (3.0 * iqr)

        iqr_mask = (valid < lower_bound) | (valid > upper_bound)
        extreme_iqr_mask = (valid < extreme_lower) | (valid > extreme_upper)

        # Method B: Modified Z-score using median and MAD
        med = float(valid.median())
        abs_dev = np.abs(valid - med)
        mad = float(np.median(abs_dev))

        mod_z_mask = pd.Series(False, index=valid.index)
        if mad > 1e-9:
            mod_z = 0.6745 * abs_dev / mad
            mod_z_mask = mod_z > cfg.outlier_modified_z_threshold

        combined_mask = iqr_mask | mod_z_mask
        outlier_count = int(combined_mask.sum())

        if outlier_count > 0:
            outlier_pct = round(outlier_count / len(valid) * 100, 2)
            outlier_indices = [int(i) for i in valid.index[combined_mask][:50].tolist()]

            # Sample top extreme values with standardized column and row_index keys
            outlier_series = valid[combined_mask]
            extreme_samples = [
                {"row_index": int(idx), "column": col_name, "value": to_python_scalar(val)}
                for idx, val in outlier_series.head(5).items()
            ]

            has_extreme = bool(extreme_iqr_mask.any())

            findings.append(
                Finding(
                    id=f"OUT-001-{col_name}",
                    rule_id="OUT-001",
                    check_name="numeric_outliers",
                    category=FindingCategory.OUTLIER,
                    severity=Severity.MEDIUM if (has_extreme or outlier_pct > 3.0) else Severity.LOW,
                    title=f"Numeric Outliers in '{col_name}' ({outlier_count:,} flagged, {outlier_pct:.1f}%)",
                    summary=f"Detected {outlier_count:,} statistical outliers using IQR ([{lower_bound:.2f}, {upper_bound:.2f}]) and Modified Z-Score (> {cfg.outlier_modified_z_threshold}).",
                    evidence=Evidence(
                        description="Values deviate significantly from the central distribution.",
                        column_names=[col_name],
                        row_indices=outlier_indices,
                        metrics={
                            "outlier_count": outlier_count,
                            "outlier_percentage": outlier_pct,
                            "iqr_lower_bound": round(lower_bound, 4),
                            "iqr_upper_bound": round(upper_bound, 4),
                            "median": round(med, 4),
                            "mad": round(mad, 4) if mad > 1e-9 else None,
                        },
                        thresholds={
                            "iqr_multiplier": cfg.outlier_iqr_multiplier,
                            "modified_z_threshold": cfg.outlier_modified_z_threshold,
                        },
                        sample_values=extreme_samples,
                    ),
                    why_it_matters="Outliers distort statistical moments (mean, variance), leverage linear regression fits, and can signal sensor anomalies or missing-value sentinel codes (-999, 9999).",
                    recommendation="Inspect flagged rows against domain limits. Verify whether entries are valid physical extremes or data capture errors before clipping or winsorizing.",
                    limitations="Heavy-tailed or power-law distributions (e.g. income, network packet sizes) naturally produce extreme observations; outliers do not inherently imply data errors.",
                )
            )

    return findings


def detect_high_cardinality(
    df: pd.DataFrame,
    profile: DatasetProfile,
    cfg: AppConfig = default_config,
) -> List[Finding]:
    """Check 6: High-cardinality categorical columns."""
    findings: List[Finding] = []

    if df.empty or len(df) < 20:
        return findings

    for col_name, col_prof in profile.columns.items():
        if col_prof.inferred_type not in ["categorical", "text"]:
            continue

        # Skip columns already identified as 100% unique identifiers
        if col_prof.unique_percentage >= 98.0:
            continue

        valid = df[col_name].dropna()
        if len(valid) < 20:
            continue

        u_count = col_prof.unique_count
        u_pct = col_prof.unique_percentage

        exceeds_count = u_count >= cfg.high_cardinality_unique_count
        exceeds_ratio = (u_pct >= cfg.high_cardinality_ratio_threshold * 100) and (u_count >= 15)

        if exceeds_count or exceeds_ratio:
            card_indices = [int(i) for i in valid.index[:50].tolist()]
            card_samples = [
                {"row_index": int(i), "column": col_name, "value": to_python_scalar(df.loc[i, col_name])}
                for i in valid.index[:5]
            ]
            findings.append(
                Finding(
                    id=f"CARD-001-{col_name}",
                    rule_id="CARD-001",
                    check_name="high_cardinality",
                    category=FindingCategory.CARDINALITY,
                    severity=Severity.LOW,
                    title=f"High Cardinality in '{col_name}' ({u_count:,} Distinct Categories)",
                    summary=f"Categorical column '{col_name}' has {u_count:,} distinct categories ({u_pct:.1f}% unique).",
                    evidence=Evidence(
                        description="Category count exceeds cardinality threshold.",
                        column_names=[col_name],
                        row_indices=card_indices,
                        metrics={
                            "unique_count": u_count,
                            "unique_percentage": u_pct,
                            "total_valid_rows": len(valid),
                            "sample_categories": col_prof.sample_values[:5],
                        },
                        thresholds={
                            "max_cardinality_threshold": cfg.high_cardinality_unique_count,
                            "ratio_threshold": cfg.high_cardinality_ratio_threshold * 100,
                        },
                        sample_values=card_samples,
                    ),
                    why_it_matters="One-hot encoding high-cardinality features causes sparse matrix memory explosion; tree-based models split preferentially on high-cardinality features leading to severe overfitting.",
                    recommendation="Group infrequent categories into an 'Other' bucket, apply target encoding with out-of-fold regularization, or use entity embeddings.",
                    limitations="Structured domain codes (e.g. zip codes, airport codes, diagnostic ICD codes) legitimately feature high cardinality.",
                )
            )

    return findings


def detect_inconsistent_categoricals(
    df: pd.DataFrame,
    profile: DatasetProfile,
    cfg: AppConfig = default_config,
) -> List[Finding]:
    """Check 7: Inconsistent categorical representations (whitespace and casing variations)."""
    findings: List[Finding] = []

    if df.empty or len(df) < 2:
        return findings

    for col_name, col_prof in profile.columns.items():
        if col_prof.inferred_type not in ["categorical", "text"]:
            continue

        valid = df[col_name].dropna().astype(str)
        raw_uniques = valid.unique()

        # Limit to reasonable unique sets to prevent high computational overhead
        if len(raw_uniques) < 2 or len(raw_uniques) > 500:
            continue

        # Group raw values by normalized representation
        normalized_clusters: Dict[str, List[str]] = {}
        for raw_val in raw_uniques:
            norm_val = " ".join(raw_val.strip().lower().split())
            normalized_clusters.setdefault(norm_val, []).append(raw_val)

        conflicts = {norm: vars_ for norm, vars_ in normalized_clusters.items() if len(vars_) > 1}

        if conflicts:
            # Find affected rows
            all_conflicting_raw_vals = [v for variants in conflicts.values() for v in variants]
            affected_rows = [int(i) for i in valid.index[valid.isin(all_conflicting_raw_vals)][:50].tolist()]

            cluster_examples = [
                {"canonical": norm, "variants": vars_}
                for norm, vars_ in list(conflicts.items())[:5]
            ]
            inconsistent_samples = [
                {"row_index": int(i), "column": col_name, "value": to_python_scalar(df.loc[i, col_name])}
                for i in affected_rows[:5]
            ]

            findings.append(
                Finding(
                    id=f"CAT-001-{col_name}",
                    rule_id="CAT-001",
                    check_name="inconsistent_categoricals",
                    category=FindingCategory.INTEGRITY,
                    severity=Severity.MEDIUM,
                    title=f"Inconsistent String Variants in '{col_name}' ({len(conflicts)} Conflicting Clusters)",
                    summary=f"Detected {len(conflicts)} category clusters in '{col_name}' that differ only by leading/trailing whitespace or casing.",
                    evidence=Evidence(
                        description="Conflicting representations map to identical normalized text.",
                        column_names=[col_name],
                        row_indices=affected_rows,
                        metrics={
                            "conflicting_clusters_count": len(conflicts),
                            "raw_unique_count": len(raw_uniques),
                            "normalized_unique_count": len(normalized_clusters),
                            "affected_rows_sample_count": len(affected_rows),
                            "cluster_examples": cluster_examples,
                        },
                        thresholds={"inconsistency_detected": True},
                        sample_values=inconsistent_samples,
                    ),
                    why_it_matters="Casing and whitespace inconsistencies artificially fragment single categories into multiple levels (e.g. 'Yes' vs 'yes'), corrupting frequency statistics and one-hot encodings.",
                    recommendation="Apply .str.strip().str.lower() normalization to categorical features during dataset cleaning.",
                    limitations="Case-sensitive domain codes (e.g. gene notations, password hashes, or specialized identifiers) may intentionally distinguish case.",
                )
            )

    return findings


def detect_target_imbalance(
    df: pd.DataFrame,
    target_column: Optional[str],
    cfg: AppConfig = default_config,
) -> List[Finding]:
    """Check 8: Target class imbalance when user selects a target column."""
    findings: List[Finding] = []

    if not target_column or target_column not in df.columns or df.empty:
        return findings

    valid = df[target_column].dropna()
    if len(valid) < 10:
        return findings

    unique_count = valid.nunique()
    # Imbalance applies to discrete / categorical / binary targets
    if unique_count < 2 or unique_count > 20:
        return findings

    counts = valid.value_counts()
    majority_class = counts.index[0]
    minority_class = counts.index[-1]
    majority_count = int(counts.iloc[0])
    minority_count = int(counts.iloc[-1])

    if minority_count <= 0:
        return findings

    imbalance_ratio = round(majority_count / minority_count, 2)
    minority_pct = round(minority_count / len(valid) * 100, 2)
    majority_pct = round(majority_count / len(valid) * 100, 2)

    class_dist = {str(k): int(v) for k, v in counts.items()}
    minority_indices = [int(i) for i in valid.index[valid == minority_class][:50].tolist()]
    minority_samples = [
        {"row_index": int(i), "column": target_column, "value": to_python_scalar(df.loc[i, target_column])}
        for i in minority_indices[:5]
    ]

    if imbalance_ratio >= cfg.imbalance_ratio_critical:
        findings.append(
            Finding(
                id="IMB-001",
                rule_id="IMB-001",
                check_name="class_imbalance",
                category=FindingCategory.IMBALANCE,
                severity=Severity.HIGH,
                title=f"Severe Class Imbalance in Target '{target_column}' ({imbalance_ratio:.1f}:1 Ratio)",
                summary=f"Minority class {repr(minority_class)} accounts for only {minority_pct:.2f}% ({minority_count:,} rows) vs majority class {repr(majority_class)} at {majority_pct:.2f}%.",
                evidence=Evidence(
                    description=f"Severe target imbalance ratio exceeds critical threshold of {cfg.imbalance_ratio_critical:.1f}:1.",
                    column_names=[target_column],
                    row_indices=minority_indices,
                    metrics={
                        "imbalance_ratio": imbalance_ratio,
                        "majority_class": to_python_scalar(majority_class),
                        "majority_percentage": majority_pct,
                        "minority_class": to_python_scalar(minority_class),
                        "minority_percentage": minority_pct,
                        "class_distribution": class_dist,
                    },
                    thresholds={"critical_ratio": cfg.imbalance_ratio_critical},
                    sample_values=minority_samples,
                ),
                why_it_matters="Models trained on severely imbalanced targets can achieve deceptively high accuracy by trivially predicting the majority class while completely failing on the minority class.",
                recommendation="Evaluate performance using PR-AUC, Balanced Accuracy, or Macro F1; use StratifiedKFold cross-validation and consider SMOTE or class-weighted losses.",
                limitations="Class imbalance is intrinsic to real-world domains such as fraud detection, disease diagnosis, and fault monitoring. Imbalance does not imply corrupted data.",
            )
        )
    elif imbalance_ratio >= cfg.imbalance_ratio_threshold:
        findings.append(
            Finding(
                id="IMB-002",
                rule_id="IMB-002",
                check_name="class_imbalance",
                category=FindingCategory.IMBALANCE,
                severity=Severity.MEDIUM,
                title=f"Moderate Class Imbalance in Target '{target_column}' ({imbalance_ratio:.1f}:1 Ratio)",
                summary=f"Majority class {repr(majority_class)} holds {majority_pct:.1f}% vs minority {repr(minority_class)} at {minority_pct:.1f}%.",
                evidence=Evidence(
                    description=f"Target imbalance exceeds warning threshold of {cfg.imbalance_ratio_threshold:.1f}:1.",
                    column_names=[target_column],
                    row_indices=minority_indices,
                    metrics={
                        "imbalance_ratio": imbalance_ratio,
                        "majority_class": to_python_scalar(majority_class),
                        "majority_percentage": majority_pct,
                        "minority_class": to_python_scalar(minority_class),
                        "minority_percentage": minority_pct,
                        "class_distribution": class_dist,
                    },
                    thresholds={"warning_ratio": cfg.imbalance_ratio_threshold},
                    sample_values=minority_samples,
                ),
                why_it_matters="Moderate imbalance requires stratification and threshold tuning to ensure minority class sensitivity.",
                recommendation="Use stratified train/test splits and inspect precision-recall trade-offs.",
                limitations="Routine distribution in multi-class or binary classification settings.",
            )
        )

    return findings


def detect_numeric_correlations(
    df: pd.DataFrame,
    profile: DatasetProfile,
    cfg: AppConfig = default_config,
) -> List[Finding]:
    """Check 9: Strong numeric correlations and potentially redundant features."""
    findings: List[Finding] = []

    if df.empty or len(df) < 5:
        return findings

    # Collect numeric columns
    num_cols = [c for c, p in profile.columns.items() if p.inferred_type in ["numeric", "boolean"]]
    if len(num_cols) < 2:
        return findings

    # Avoid expensive pairwise computation on very wide datasets
    if len(num_cols) > cfg.max_correlation_columns:
        # Prioritize top columns by variance
        variances = df[num_cols].var(numeric_only=True).dropna()
        num_cols = variances.nlargest(cfg.max_correlation_columns).index.tolist()

    # Sample rows if dataset is extremely large
    corr_df = df[num_cols]
    if len(corr_df) > cfg.max_correlation_sample_rows:
        corr_df = corr_df.sample(n=cfg.max_correlation_sample_rows, random_state=42)

    corr_matrix = corr_df.corr(method="pearson")

    # Inspect upper triangle of correlation matrix
    visited_pairs = set()
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            col1 = num_cols[i]
            col2 = num_cols[j]
            r = corr_matrix.loc[col1, col2]

            if pd.isna(r):
                continue

            abs_r = abs(float(r))
            corr_indices = [int(idx) for idx in corr_df.index[:50].tolist()]
            corr_samples = [
                {
                    "row_index": int(idx),
                    "column": col1,
                    "value": to_python_scalar(df.loc[idx, col1]),
                    "col2": col2,
                    "val2": to_python_scalar(df.loc[idx, col2]),
                }
                for idx in corr_df.index[:5]
            ]

            if abs_r >= cfg.correlation_extreme_threshold:
                findings.append(
                    Finding(
                        id=f"CORR-001-{col1}-{col2}",
                        rule_id="CORR-001",
                        check_name="feature_redundancy",
                        category=FindingCategory.CORRELATION,
                        severity=Severity.HIGH,
                        title=f"Extreme Correlation / Redundancy Between '{col1}' and '{col2}' (r = {r:.4f})",
                        summary=f"Features '{col1}' and '{col2}' exhibit near-perfect linear dependence (|r| = {abs_r:.4f} >= {cfg.correlation_extreme_threshold}).",
                        evidence=Evidence(
                            description="Pairwise Pearson correlation coefficient indicates redundant features.",
                            column_names=[col1, col2],
                            row_indices=corr_indices,
                            metrics={
                                "pearson_r": round(float(r), 4),
                                "r_squared": round(float(r**2), 4),
                                "sample_rows_analyzed": len(corr_df),
                            },
                            thresholds={"extreme_correlation_threshold": cfg.correlation_extreme_threshold},
                            sample_values=corr_samples,
                        ),
                        why_it_matters="Severe multicollinearity destabilizes estimated regression coefficients, inflates standard errors, and adds redundant dimensionality without adding information.",
                        recommendation=f"Drop one of the collinear features ('{col1}' or '{col2}') or apply principal component analysis (PCA).",
                        limitations="Near-perfect correlation can occur naturally for physical measurements in different units or mathematical transformations.",
                    )
                )
            elif abs_r >= cfg.correlation_high_threshold:
                findings.append(
                    Finding(
                        id=f"CORR-002-{col1}-{col2}",
                        rule_id="CORR-002",
                        check_name="feature_correlation",
                        category=FindingCategory.CORRELATION,
                        severity=Severity.MEDIUM,
                        title=f"Strong Correlation Between '{col1}' and '{col2}' (r = {r:.4f})",
                        summary=f"Features '{col1}' and '{col2}' are strongly correlated (|r| = {abs_r:.4f} >= {cfg.correlation_high_threshold}).",
                        evidence=Evidence(
                            description="Pairwise correlation exceeds warning threshold.",
                            column_names=[col1, col2],
                            row_indices=corr_indices,
                            metrics={
                                "pearson_r": round(float(r), 4),
                                "r_squared": round(float(r**2), 4),
                                "sample_rows_analyzed": len(corr_df),
                            },
                            thresholds={"high_correlation_threshold": cfg.correlation_high_threshold},
                            sample_values=corr_samples,
                        ),
                        why_it_matters="Collinear features split feature importance in tree models and inflate coefficient variance in linear models.",
                        recommendation="Review feature set; consider feature selection or regularization (Lasso / Ridge / ElasticNet).",
                        limitations="Strong correlation does not prove causation or exact linear substitution in non-linear models.",
                    )
                )

    return findings


def compute_eta_squared(categories: pd.Series, values: pd.Series) -> float:
    """Calculate ANOVA eta-squared (proportion of continuous variance explained by categories)."""
    if not pd.api.types.is_numeric_dtype(values):
        try:
            values = pd.to_numeric(values, errors="coerce")
        except Exception:
            return 0.0

    df_temp = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if len(df_temp) < 5 or df_temp["cat"].nunique() < 2 or not pd.api.types.is_numeric_dtype(df_temp["val"]):
        return 0.0
    try:
        total_mean = float(df_temp["val"].mean())
        ss_total = float(((df_temp["val"] - total_mean) ** 2).sum())
        if ss_total <= 1e-12:
            return 0.0
        group_means = df_temp.groupby("cat")["val"].transform("mean")
        ss_between = float(((group_means - total_mean) ** 2).sum())
        return min(1.0, max(0.0, float(ss_between / ss_total)))
    except Exception:
        return 0.0



def check_inverted_binary(feat: pd.Series, target: pd.Series) -> bool:
    """Check if feature is an exact binary inversion / negation of target."""
    df_temp = pd.DataFrame({"f": feat, "t": target}).dropna()
    if len(df_temp) < 5 or df_temp["t"].nunique() != 2 or df_temp["f"].nunique() != 2:
        return False

    # Check numeric 0/1 or min/max inversion
    if pd.api.types.is_numeric_dtype(df_temp["f"]) and pd.api.types.is_numeric_dtype(df_temp["t"]):
        t_min, t_max = df_temp["t"].min(), df_temp["t"].max()
        f_min, f_max = df_temp["f"].min(), df_temp["f"].max()
        inv_f = df_temp["f"].map({f_min: t_max, f_max: t_min})
        if bool((inv_f == df_temp["t"]).all()):
            return True

    # Check categorical / string domain with inverted mapping
    norm_f = df_temp["f"].astype(str).str.strip().str.lower()
    norm_t = df_temp["t"].astype(str).str.strip().str.lower()
    u_f = set(norm_f.unique())
    u_t = set(norm_t.unique())
    if u_f == u_t and len(u_f) == 2:
        val_a, val_b = list(u_f)
        inv_f = norm_f.map({val_a: val_b, val_b: val_a})
        if bool((inv_f == norm_t).all()):
            return True

    # Check crosstab off-diagonal inversion for matching labels
    crosstab = pd.crosstab(norm_f, norm_t)
    if crosstab.shape == (2, 2) and set(crosstab.index) == set(crosstab.columns):
        row0, row1 = crosstab.index[0], crosstab.index[1]
        if bool(crosstab.loc[row0, row0] == 0 and crosstab.loc[row1, row1] == 0 and crosstab.loc[row0, row1] > 0 and crosstab.loc[row1, row0] > 0):
            return True

    return False


def check_informative_missingness(feat: pd.Series, target: pd.Series) -> Optional[Dict[str, Any]]:
    """Check if feature missingness perfectly or strongly predicts target outcome."""
    missing_mask = feat.isna()
    if missing_mask.sum() == 0 or missing_mask.sum() == len(feat):
        return None

    valid_t = target.dropna()
    common_idx = missing_mask.index.intersection(valid_t.index)
    if len(common_idx) < 10 or valid_t.loc[common_idx].nunique() < 2 or valid_t.loc[common_idx].nunique() > 10:
        return None

    m_series = missing_mask.loc[common_idx]
    t_series = valid_t.loc[common_idx]

    contingency = pd.crosstab(m_series, t_series)
    if (contingency > 0).sum(axis=1).max() == 1 and len(contingency) >= 2:
        return {"type": "deterministic_purity", "purity": 1.0, "missing_count": int(m_series.sum())}

    if t_series.nunique() == 2:
        u_t = t_series.unique()
        bin_t = (t_series == u_t[0]).astype(int)
        r = float(m_series.corr(bin_t))
        if pd.notna(r) and abs(r) >= 0.95:
            return {"type": "correlation", "correlation": round(r, 4), "missing_count": int(m_series.sum())}

    return None


def detect_target_leakage(
    df: pd.DataFrame,
    target_column: Optional[str],
    profile: DatasetProfile,
    cfg: AppConfig = default_config,
    run_baseline_comparison: bool = True,
) -> List[Finding]:
    """Check 10: Deep forensic target leakage investigation across all feature types."""
    findings: List[Finding] = []

    if not target_column or target_column not in df.columns or df.empty or len(df) < 5:
        return findings

    target_series = df[target_column]
    valid_target_count = int(target_series.notna().sum())
    if valid_target_count < 5:
        return findings

    n_total = len(df)
    is_small_dataset = (n_total < 30)
    has_duplicates = (profile.duplicate_rows_count > 0 and (profile.duplicate_rows_count / n_total) > 0.15)
    suspicious_cols: List[str] = []

    for col_name in df.columns:
        if col_name == target_column:
            continue

        feature_series = df[col_name]
        valid_df = pd.DataFrame({"feat": feature_series, "target": target_series}).dropna()

        if len(valid_df) < 5:
            continue

        feat_valid = valid_df["feat"]
        target_valid = valid_df["target"]
        matching_indices = [int(i) for i in valid_df.index[:25].tolist()]
        leak_samples = [
            {
                "row_index": int(i),
                "column": col_name,
                "value": to_python_scalar(df.loc[i, col_name]),
            }
            for i in matching_indices[:5]
        ]

        # 1. Exact Copy / Identity Check (LEAK-001)
        is_exact = (feat_valid == target_valid).all()
        if not is_exact and (feat_valid.dtype == "object" or target_valid.dtype == "object"):
            # Check string case/whitespace trimmed equivalence
            norm_feat = feat_valid.astype(str).str.strip().str.lower()
            norm_target = target_valid.astype(str).str.strip().str.lower()
            is_exact = (norm_feat == norm_target).all()

        if is_exact:
            suspicious_cols.append(col_name)
            findings.append(
                Finding(
                    id=f"LEAK-001-{col_name}",
                    rule_id="LEAK-001",
                    check_name="target_leakage",
                    category=FindingCategory.LEAKAGE,
                    severity=Severity.CRITICAL,
                    title=f"Critical Target Leakage: '{col_name}' is Exact Copy of Target '{target_column}'",
                    summary=f"Feature '{col_name}' matches target '{target_column}' identically across 100% of non-null observations ({len(valid_df):,} rows).",
                    evidence=Evidence(
                        description="Feature is an exact duplicate or renamed proxy of the target variable.",
                        column_names=[col_name, target_column],
                        row_indices=matching_indices,
                        metrics={"match_rate": 1.0, "analyzed_rows": len(valid_df), "is_small_dataset": is_small_dataset},
                        thresholds={"match_rate": 1.0},
                        sample_values=leak_samples,
                    ),
                    why_it_matters="Direct target leakage yields artificial 100% accuracy during training and validation, which completely fails in production when the target is unobserved.",
                    recommendation=f"Drop '{col_name}' immediately from all training and validation pipelines.",
                    limitations="None. Exact copies are unambiguous structural data leakage.",
                )
            )
            continue

        # 2. Inverted Binary Target Check (LEAK-005)
        if check_inverted_binary(feat_valid, target_valid):
            suspicious_cols.append(col_name)
            findings.append(
                Finding(
                    id=f"LEAK-005-{col_name}",
                    rule_id="LEAK-005",
                    check_name="target_leakage",
                    category=FindingCategory.LEAKAGE,
                    severity=Severity.CRITICAL,
                    title=f"Critical Target Leakage: '{col_name}' is Inverted Binary Copy of Target '{target_column}'",
                    summary=f"Feature '{col_name}' is an exact negative / inverted proxy of binary target '{target_column}' across 100% of rows.",
                    evidence=Evidence(
                        description="Feature values deterministically equal the exact inverse of the target outcome.",
                        column_names=[col_name, target_column],
                        row_indices=matching_indices,
                        metrics={"inverted_match_rate": 1.0, "analyzed_rows": len(valid_df)},
                        thresholds={"match_rate": 1.0},
                        sample_values=leak_samples,
                    ),
                    why_it_matters="Inverted binary features provide identical 100% mutual information as the target itself, causing complete model failure in production.",
                    recommendation=f"Drop inverted target proxy '{col_name}' prior to modeling.",
                    limitations="None. An exact binary inversion of the target constitutes direct label leakage.",
                )
            )
            continue

        # 3. Informative Missingness Leakage (LEAK-006)
        inf_missing = check_informative_missingness(df[col_name], df[target_column])
        if inf_missing:
            suspicious_cols.append(col_name)
            missing_row_indices = [int(i) for i in df.index[df[col_name].isna()][:25].tolist()]
            missing_samples = [
                {
                    "row_index": int(i),
                    "column": col_name,
                    "value": to_python_scalar(df.loc[i, col_name]),
                }
                for i in missing_row_indices[:5]
            ]
            findings.append(
                Finding(
                    id=f"LEAK-006-{col_name}",
                    rule_id="LEAK-006",
                    check_name="target_leakage",
                    category=FindingCategory.LEAKAGE,
                    severity=Severity.HIGH if not is_small_dataset else Severity.MEDIUM,
                    title=f"Informative Missingness Leakage: Missingness in '{col_name}' Predicts Target '{target_column}'",
                    summary=f"Missing value pattern in '{col_name}' ({inf_missing.get('missing_count')} nulls) strongly predicts the target outcome.",
                    evidence=Evidence(
                        description="Missingness mask exhibits deterministic or near-perfect association with target classes.",
                        column_names=[col_name, target_column],
                        row_indices=missing_row_indices,
                        metrics=inf_missing,
                        thresholds={"missingness_leakage": True},
                        sample_values=missing_samples,
                    ),
                    why_it_matters="Selective missingness occurs when a field is only recorded when a specific outcome happens (e.g., post-admission test recorded only for positive cases).",
                    recommendation="Investigate data collection protocol; do not use null indicator flags for this column in model training.",
                    limitations="Conditional surveys or domain routing rules legitimately skip sections for certain subgroups.",
                )
            )

        # 4. Numeric Feature vs Continuous Target (Pearson correlation: LEAK-002 / LEAK-003)
        if pd.api.types.is_numeric_dtype(feat_valid) and pd.api.types.is_numeric_dtype(target_valid) and target_valid.nunique() > 10:
            r = feat_valid.corr(target_valid, method="pearson")
            if pd.notna(r):
                abs_r = abs(float(r))
                if abs_r >= cfg.leakage_threshold:
                    suspicious_cols.append(col_name)
                    findings.append(
                        Finding(
                            id=f"LEAK-002-{col_name}",
                            rule_id="LEAK-002",
                            check_name="target_leakage",
                            category=FindingCategory.LEAKAGE,
                            severity=Severity.HIGH if not is_small_dataset else Severity.MEDIUM,
                            title=f"Suspected Target Leakage: '{col_name}' Correlates Strongly with Target (r = {r:.4f})",
                            summary=f"Numeric feature '{col_name}' Pearson correlation with target '{target_column}' is {r:.4f} (threshold {cfg.leakage_threshold}).",
                            evidence=Evidence(
                                description="Extreme linear correlation with continuous prediction target.",
                                column_names=[col_name, target_column],
                                row_indices=matching_indices,
                                metrics={"pearson_r": round(float(r), 4), "r_squared": round(float(r**2), 4), "sample_size": len(valid_df)},
                                thresholds={"leakage_correlation_threshold": cfg.leakage_threshold},
                                sample_values=leak_samples,
                            ),
                            why_it_matters="Extreme correlation with the target often signals that the feature is a downstream consequence or alternative measurement of the target.",
                            recommendation=f"Verify timeline of data collection for '{col_name}'; ensure feature is available strictly prior to the target event.",
                            limitations="Legitimate strong predictors (e.g., highly calibrated sensor metrics) can display high correlation without being leakage.",
                        )
                    )
                elif abs_r >= 0.85:
                    findings.append(
                        Finding(
                            id=f"LEAK-003-STRONG-{col_name}",
                            rule_id="LEAK-003",
                            check_name="target_leakage",
                            category=FindingCategory.STATISTICAL,
                            severity=Severity.LOW,
                            title=f"Strong Predictor / Potential Leakage Candidate: '{col_name}' (r = {r:.4f})",
                            summary=f"Feature '{col_name}' exhibits strong predictive association with '{target_column}' (|r| = {abs_r:.4f}).",
                            evidence=Evidence(
                                description="Strong linear association observed below critical leakage threshold.",
                                column_names=[col_name, target_column],
                                row_indices=matching_indices,
                                metrics={"pearson_r": round(float(r), 4), "r_squared": round(float(r**2), 4)},
                                thresholds={"high_correlation_threshold": 0.85},
                                sample_values=leak_samples,
                            ),
                            why_it_matters="Strong predictive features should be verified for temporal validity before inclusion.",
                            recommendation="Confirm that feature collection precedes target outcome.",
                            limitations="High correlation is expected for genuine causal drivers and does not prove data leakage.",
                        )
                    )
            continue

        # 5. Numeric Feature vs Discrete/Categorical Target (Separation / AUC & Multiclass ANOVA: LEAK-002 / LEAK-003)
        if target_valid.nunique() >= 2 and pd.api.types.is_numeric_dtype(feat_valid):
            # Binary target AUC
            if target_valid.nunique() == 2:
                try:
                    from sklearn.metrics import roc_auc_score
                    u_targets = target_valid.unique()
                    bin_target = (target_valid == u_targets[0]).astype(int)
                    auc = float(roc_auc_score(bin_target, feat_valid))
                    eff_auc = auc if auc >= 0.5 else 1.0 - auc
                    if eff_auc >= 0.98:
                        suspicious_cols.append(col_name)
                        findings.append(
                            Finding(
                                id=f"LEAK-002-{col_name}",
                                rule_id="LEAK-002",
                                check_name="target_leakage",
                                category=FindingCategory.LEAKAGE,
                                severity=Severity.HIGH if not is_small_dataset else Severity.MEDIUM,
                                title=f"Suspected Target Leakage: '{col_name}' Almost Perfectly Separates Target (AUC = {eff_auc:.4f})",
                                summary=f"Feature '{col_name}' achieves ROC-AUC of {eff_auc:.4f} against binary target '{target_column}'.",
                                evidence=Evidence(
                                    description="Numeric feature separates target classes with near-perfect boundary.",
                                    column_names=[col_name, target_column],
                                    row_indices=matching_indices,
                                    metrics={"roc_auc": round(eff_auc, 4), "sample_size": len(valid_df)},
                                    thresholds={"auc_threshold": 0.98},
                                    sample_values=leak_samples,
                                ),
                                why_it_matters="Near-perfect target separation in tabular data frequently indicates the feature was measured post-outcome.",
                                recommendation="Confirm the chronological timestamp of feature recording relative to the target outcome.",
                                limitations="Gold-standard diagnostic tests can yield near-perfect separation legitimately.",
                            )
                        )
                    elif eff_auc >= 0.88:
                        findings.append(
                            Finding(
                                id=f"LEAK-003-STRONG-{col_name}",
                                rule_id="LEAK-003",
                                check_name="target_leakage",
                                category=FindingCategory.STATISTICAL,
                                severity=Severity.LOW,
                                title=f"Strong Predictor / Potential Leakage Candidate: '{col_name}' (AUC = {eff_auc:.4f})",
                                summary=f"Feature '{col_name}' separates target classes strongly (AUC = {eff_auc:.4f}) without reaching definitive leakage threshold.",
                                evidence=Evidence(
                                    description="Strong classification separation observed.",
                                    column_names=[col_name, target_column],
                                    row_indices=matching_indices,
                                    metrics={"roc_auc": round(eff_auc, 4)},
                                    thresholds={"auc_threshold": 0.88},
                                    sample_values=leak_samples,
                                ),
                                why_it_matters="Strong predictive features must be verified for temporal precedence.",
                                recommendation="Verify timestamp of feature recording.",
                                limitations="Legitimate strong predictors naturally produce elevated AUC scores.",
                            )
                        )
                except Exception:
                    pass
            # Multiclass target (3 to 10 classes)
            elif 3 <= target_valid.nunique() <= 10:
                eta_sq = compute_eta_squared(target_valid, feat_valid)
                if eta_sq >= 0.95:
                    suspicious_cols.append(col_name)
                    findings.append(
                        Finding(
                            id=f"LEAK-002-MULTI-{col_name}",
                            rule_id="LEAK-002",
                            check_name="target_leakage",
                            category=FindingCategory.LEAKAGE,
                            severity=Severity.HIGH if not is_small_dataset else Severity.MEDIUM,
                            title=f"Suspected Target Leakage: '{col_name}' Strongly Partitions Multiclass Target (eta^2 = {eta_sq:.4f})",
                            summary=f"Numeric feature '{col_name}' explains {eta_sq*100:.1f}% of between-group variance for multiclass target '{target_column}' ({target_valid.nunique()} classes).",
                            evidence=Evidence(
                                description="ANOVA effect size indicates near-deterministic class separation.",
                                column_names=[col_name, target_column],
                                row_indices=matching_indices,
                                metrics={"eta_squared": round(eta_sq, 4), "classes_count": int(target_valid.nunique())},
                                thresholds={"eta_squared_threshold": 0.95},
                                sample_values=leak_samples,
                            ),
                            why_it_matters="A continuous feature that separates multiple target classes with >=95% variance explained is often an encoded proxy.",
                            recommendation="Examine if feature was derived from target labels.",
                            limitations="Strong physical biomarkers across distinct clinical stages can display high effect sizes.",
                        )
                    )

        # 6. Categorical Target Encoding / Deterministic Partition (LEAK-004)
        if target_valid.nunique() <= 10 and feat_valid.nunique() <= 30:
            contingency = pd.crosstab(feat_valid, target_valid)
            # Check 100% class purity per category
            if (contingency > 0).sum(axis=1).max() == 1 and len(contingency) >= 2:
                suspicious_cols.append(col_name)
                findings.append(
                    Finding(
                        id=f"LEAK-004-{col_name}",
                        rule_id="LEAK-004",
                        check_name="target_leakage",
                        category=FindingCategory.LEAKAGE,
                        severity=Severity.HIGH if not is_small_dataset else Severity.MEDIUM,
                        title=f"Target Leakage / Deterministic Target Encoding: '{col_name}' Perfectly Partitions Target '{target_column}'",
                        summary=f"Every distinct category of '{col_name}' maps to exactly one target class with 100% purity ({len(contingency)} distinct categories).",
                        evidence=Evidence(
                            description="Contingency table shows deterministic partitioning of target classes.",
                            column_names=[col_name, target_column],
                            row_indices=matching_indices,
                            metrics={"conditional_purity": 1.0, "category_levels": int(len(contingency))},
                            thresholds={"purity_threshold": 1.0},
                            sample_values=leak_samples,
                        ),
                        why_it_matters="Deterministic 1-to-1 category mapping indicates the feature is a coded representation or downstream sub-type of the label itself.",
                        recommendation="Check if feature is an administrative post-label classification code.",
                        limitations="Deterministic physical relationships (e.g. State -> Country) legitimately partition classes without data error.",
                    )
                )
        elif pd.api.types.is_numeric_dtype(target_valid) and target_valid.nunique() > 10 and feat_valid.nunique() <= 20:
            # Continuous target vs Categorical feature
            eta_sq = compute_eta_squared(feat_valid, target_valid)
            if eta_sq >= 0.95:
                suspicious_cols.append(col_name)
                findings.append(
                    Finding(
                        id=f"LEAK-004-CAT-CONT-{col_name}",
                        rule_id="LEAK-004",
                        check_name="target_leakage",
                        category=FindingCategory.LEAKAGE,
                        severity=Severity.HIGH if not is_small_dataset else Severity.MEDIUM,
                        title=f"Suspected Target Encoding Leakage: '{col_name}' Explains {eta_sq*100:.1f}% of Target Variance",
                        summary=f"Categorical feature '{col_name}' partitions continuous target '{target_column}' with eta^2 = {eta_sq:.4f}.",
                        evidence=Evidence(
                            description="Feature categories account for >=95% of target variance.",
                            column_names=[col_name, target_column],
                            row_indices=matching_indices,
                            metrics={"eta_squared": round(eta_sq, 4), "category_count": int(feat_valid.nunique())},
                            thresholds={"eta_squared_threshold": 0.95},
                            sample_values=leak_samples,
                        ),
                        why_it_matters="Binned target encodings or target-derived brackets falsely inflate predictive accuracy.",
                        recommendation="Verify feature origin; avoid binned targets disguised as independent features.",
                        limitations="Genuine domain strata can explain large fractions of target variance.",
                    )
                )

    # 7. Optional Baseline Pipeline Comparison
    if run_baseline_comparison and suspicious_cols and len(df) >= 15:
        from engine.evaluator import evaluate_leakage_impact
        unique_suspicious = list(dict.fromkeys(suspicious_cols))
        baseline_res = evaluate_leakage_impact(df, target_column, unique_suspicious)
        if baseline_res and baseline_res.get("status") == "success":
            for f in findings:
                if f.rule_id.startswith("LEAK-"):
                    f.evidence.metrics["baseline_comparison"] = baseline_res

    # Document small dataset & duplicate limitations on all findings
    if is_small_dataset or has_duplicates:
        for f in findings:
            if f.rule_id.startswith("LEAK-"):
                addon = []
                if is_small_dataset:
                    addon.append(f"Small dataset sample size (N={n_total} < 30); empirical metrics have high variance.")
                if has_duplicates:
                    addon.append(f"Dataset has {profile.duplicate_rows_count:,} duplicate rows ({profile.duplicate_rows_percentage:.1f}%), which may artificially inflate metrics.")
                f.limitations = f.limitations + " " + " ".join(addon)

    return findings


def run_all_detectors(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target_column: Optional[str] = None,
    cfg: Optional[AppConfig] = None,
) -> List[Finding]:
    """Execute all 10 deterministic forensic detectors in disciplined sequence."""
    app_cfg = cfg or default_config
    findings: List[Finding] = []

    # 1. Missing values
    findings.extend(detect_missing_values(df, profile, app_cfg))

    # 2. Exact duplicate rows & candidate records
    findings.extend(detect_duplicate_records(df, profile, app_cfg))

    # 3. Constant and near-constant columns
    findings.extend(detect_constant_and_near_constant(df, profile, app_cfg))

    # 4. Unique-value counts & potential identifiers
    findings.extend(detect_potential_identifiers(df, profile, app_cfg))

    # 5. Numeric outliers (IQR & modified Z-score)
    findings.extend(detect_numeric_outliers(df, profile, app_cfg))

    # 6. High-cardinality categoricals
    findings.extend(detect_high_cardinality(df, profile, app_cfg))

    # 7. Inconsistent categoricals (casing & whitespace)
    findings.extend(detect_inconsistent_categoricals(df, profile, app_cfg))

    # 8. Target class imbalance (if target column specified)
    if target_column:
        findings.extend(detect_target_imbalance(df, target_column, app_cfg))

    # 9. Strong numeric correlations & feature redundancy
    findings.extend(detect_numeric_correlations(df, profile, app_cfg))

    # 10. Potential target leakage candidates (if target column specified)
    if target_column:
        findings.extend(detect_target_leakage(df, target_column, profile, app_cfg))

    return findings
