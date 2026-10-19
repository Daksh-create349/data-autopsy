"""Deterministic ranking engine separating forensic severity from statistical confidence.

Guarantees reproducible, mathematically calibrated finding prioritization:
1. Severity denotes potential impact or risk (CRITICAL, HIGH, MEDIUM, LOW, INFO).
2. Statistical Confidence denotes empirical support calibrated by sample size and effect size.
3. Ranking score combines both via documented deterministic weighting.
"""

from typing import List
import numpy as np

from engine.models import Finding, Severity

# Documented severity weights
SEVERITY_WEIGHTS = {
    Severity.CRITICAL: 500.0,
    Severity.HIGH: 400.0,
    Severity.MEDIUM: 300.0,
    Severity.LOW: 200.0,
    Severity.INFO: 100.0,
    Severity.INFORMATIONAL: 100.0,
}


def compute_statistical_confidence(finding: Finding, n_rows: int) -> float:
    """Compute deterministic empirical confidence score in [0.0, 1.0].
    
    IMPORTANT: This metric is a deterministic empirical support score reflecting statistical
    precision and sample size calibration, NOT an uncalibrated probability.
    """
    metrics = finding.evidence.metrics or {}

    # 1. Exact structural findings have 100% deterministic certainty
    if finding.rule_id in ["INT-001", "LEAK-001", "LEAK-005", "STAT-001", "DUP-001"]:
        return 1.0000

    # 2. Sample size scaling factor: small samples (N < 30) discount statistical confidence
    sample_size = float(metrics.get("sample_size") or metrics.get("analyzed_rows") or metrics.get("total_rows") or n_rows)
    n_factor = float(np.sqrt(min(1.0, max(5, sample_size) / 50.0)))

    # 3. Correlation / Linear association
    if "pearson_r" in metrics:
        r = abs(float(metrics["pearson_r"]))
        return round(float(min(1.0, max(0.1, r * n_factor))), 4)

    # 4. Binary classification separation / ROC-AUC
    if "roc_auc" in metrics:
        auc = float(metrics["roc_auc"])
        separation = abs(2.0 * (auc - 0.5))  # 0.0 at random 0.5, 1.0 at perfect 1.0
        return round(float(min(1.0, max(0.1, separation * n_factor))), 4)

    # 5. ANOVA eta-squared
    if "eta_squared" in metrics:
        eta2 = float(metrics["eta_squared"])
        return round(float(min(1.0, max(0.1, eta2 * n_factor))), 4)

    # 6. Missing values percentage
    if "null_percentage" in metrics:
        pct = float(metrics["null_percentage"]) / 100.0
        return round(float(min(1.0, max(0.2, pct))), 4)

    # 7. Candidate duplicates
    if "candidate_duplicates_count" in metrics:
        cand_pct = float(metrics.get("candidate_duplicates_percentage", 10.0)) / 100.0
        return round(float(min(1.0, max(0.3, cand_pct * n_factor))), 4)

    # 8. Outliers
    if "outlier_percentage" in metrics:
        out_pct = float(metrics["outlier_percentage"]) / 100.0
        return round(float(min(1.0, max(0.3, out_pct * n_factor + 0.3))), 4)

    # Default fallback calibrated by severity and sample factor
    base_weight = 0.85 if finding.severity in [Severity.CRITICAL, Severity.HIGH] else 0.70
    return round(float(min(1.0, base_weight * n_factor)), 4)


def compute_rank_score(finding: Finding, confidence: float, n_rows: int) -> float:
    """Compute composite deterministic rank score for finding sorting.
    
    Ranking Rules:
    1. Primary tier: Severity weight (CRITICAL=500, HIGH=400, MEDIUM=300, LOW=200, INFO=100).
    2. Secondary relevance factor (0 to 100):
       - 50% from statistical confidence score [0..1] * 50
       - 50% from affected row fraction [0..1] * 50
    """
    sev_weight = SEVERITY_WEIGHTS.get(finding.severity, 100.0)

    # Calculate affected row fraction
    metrics = finding.evidence.metrics or {}
    affected_count = len(finding.evidence.row_indices)
    if "null_count" in metrics:
        affected_count = int(metrics["null_count"])
    elif "outlier_count" in metrics:
        affected_count = int(metrics["outlier_count"])
    elif "exact_duplicate_rows_count" in metrics:
        affected_count = int(metrics["exact_duplicate_rows_count"])
    elif "candidate_duplicates_count" in metrics:
        affected_count = int(metrics["candidate_duplicates_count"])

    affected_ratio = min(1.0, float(affected_count) / max(1, n_rows))
    relevance = (confidence * 50.0) + (affected_ratio * 50.0)
    return round(sev_weight + relevance, 2)


def rank_findings(findings: List[Finding], n_rows: int) -> List[Finding]:
    """Annotate and rank findings deterministically.
    
    Ordering hierarchy:
    1. Rank score (descending)
    2. Severity weight (descending)
    3. Statistical confidence score (descending)
    4. Rule ID (ascending, for deterministic tie-breaking)
    5. Finding ID (ascending, for deterministic tie-breaking)
    """
    for f in findings:
        conf = compute_statistical_confidence(f, n_rows)
        rank = compute_rank_score(f, conf, n_rows)
        f.confidence_score = conf
        f.rank_score = rank

    # Deterministic multi-key sort
    return sorted(
        findings,
        key=lambda f: (
            -f.rank_score,
            -SEVERITY_WEIGHTS.get(f.severity, 0.0),
            -f.confidence_score,
            f.rule_id,
            f.id,
        ),
    )
