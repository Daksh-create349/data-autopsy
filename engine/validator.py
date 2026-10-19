"""Forensic validator ensuring finding evidence strictly corresponds to actual dataset contents.

Validates that referenced columns, row indices, and sample values mathematically match
the source DataFrame without fabrication.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from engine.models import AutopsyReport, Finding


def validate_finding_evidence(df: pd.DataFrame, finding: Finding) -> Tuple[bool, List[str]]:
    """Validate that finding's evidence, row indices, column names, and sample values match real data.
    
    Args:
        df: The source tabular dataset.
        finding: The finding under inspection.
        
    Returns:
        (is_valid, list_of_discrepancies)
    """
    discrepancies: List[str] = []

    # 1. Column names existence
    for col in finding.evidence.column_names:
        if col not in df.columns:
            discrepancies.append(f"Referenced column '{col}' does not exist in dataset.")

    # 2. Row indices existence in DataFrame index
    for idx in finding.evidence.row_indices:
        if idx not in df.index:
            discrepancies.append(f"Referenced row index {idx} does not exist in dataset index.")

    # 3. Sample values matching actual values
    sample_values = finding.evidence.sample_values or []
    for sample in sample_values:
        if not isinstance(sample, dict):
            continue

        # Standard row-column format: {"row_index": idx, "column": col, "value": val}
        if "row_index" in sample and "column" in sample:
            r_idx = sample["row_index"]
            c_name = sample["column"]
            if r_idx in df.index and c_name in df.columns:
                actual_val = df.loc[r_idx, c_name]
                expected_val = sample.get("value")

                # Null value check
                if expected_val is None:
                    if not pd.isna(actual_val):
                        discrepancies.append(
                            f"Row {r_idx} column '{c_name}' expected null, but actual value is {actual_val}."
                        )
                elif pd.isna(actual_val):
                    discrepancies.append(
                        f"Row {r_idx} column '{c_name}' expected {expected_val}, but actual value is NaN."
                    )
                else:
                    try:
                        if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float, np.number)):
                            if not np.isclose(float(actual_val), float(expected_val), atol=1e-4):
                                discrepancies.append(
                                    f"Row {r_idx} column '{c_name}' numeric value {actual_val} differs from sample {expected_val}."
                                )
                        else:
                            if str(actual_val).strip() != str(expected_val).strip():
                                discrepancies.append(
                                    f"Row {r_idx} column '{c_name}' string value '{actual_val}' differs from sample '{expected_val}'."
                                )
                    except Exception:
                        pass

        # Target leakage format: {"row_index": idx, "feature_column": f_col, "feature_value": f_val, "target_column": t_col, "target_value": t_val}
        if "row_index" in sample and "feature_column" in sample and "target_column" in sample:
            r_idx = sample["row_index"]
            f_col = sample["feature_column"]
            t_col = sample["target_column"]
            if r_idx in df.index:
                if f_col in df.columns:
                    actual_f = df.loc[r_idx, f_col]
                    expected_f = sample.get("feature_value")
                    if not (pd.isna(actual_f) and expected_f is None) and str(actual_f).strip() != str(expected_f).strip():
                        discrepancies.append(
                            f"Row {r_idx} feature '{f_col}' actual '{actual_f}' differs from sample '{expected_f}'."
                        )
                if t_col in df.columns:
                    actual_t = df.loc[r_idx, t_col]
                    expected_t = sample.get("target_value")
                    if not (pd.isna(actual_t) and expected_t is None) and str(actual_t).strip() != str(expected_t).strip():
                        discrepancies.append(
                            f"Row {r_idx} target '{t_col}' actual '{actual_t}' differs from sample '{expected_t}'."
                        )

    return (len(discrepancies) == 0, discrepancies)


def validate_report_evidence(df: pd.DataFrame, report: AutopsyReport) -> Dict[str, Any]:
    """Validate all findings in an autopsy report against the source DataFrame."""
    results: Dict[str, Any] = {
        "all_valid": True,
        "total_findings": len(report.findings),
        "valid_count": 0,
        "invalid_count": 0,
        "discrepancies_by_finding": {},
    }

    for finding in report.findings:
        is_valid, discs = validate_finding_evidence(df, finding)
        if is_valid:
            results["valid_count"] += 1
        else:
            results["all_valid"] = False
            results["invalid_count"] += 1
            results["discrepancies_by_finding"][finding.id] = discs

    return results
