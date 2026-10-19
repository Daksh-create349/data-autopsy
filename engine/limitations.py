"""Documented forensic limitations explaining what each analysis can and cannot establish."""

from typing import Any, Dict


def get_analysis_limitations() -> Dict[str, Dict[str, str]]:
    """Return documented forensic limitations and false-positive risks for each check.
    
    Provides clear analytical boundaries separating empirical statistical observation
    from causal or real-world domain truth.
    """
    return {
        "missing_values": {
            "check_id": "INT-001..004",
            "can_establish": "Exact counts, percentages, and original row indices of unpopulated (NaN/Null) cells per column.",
            "cannot_establish": "Causal missingness mechanism (MCAR vs MAR vs MNAR); whether absence was intentional survey skip logic.",
            "false_positive_risks": "Optional demographic questions or sparse event counters legitimately exhibit high missingness without representing data corruption.",
        },
        "duplicate_records": {
            "check_id": "DUP-001..002",
            "can_establish": "Exact multi-attribute identical row copies and candidate record collisions across shared identifying subsets.",
            "cannot_establish": "Whether repeated rows represent clerical data entry error, multiple transactions by the same individual, or legitimate independent events with identical values.",
            "false_positive_risks": "Coarsely binned data (e.g. state, age bucket, gender) naturally produces identical rows for distinct individuals.",
        },
        "constant_columns": {
            "check_id": "STAT-001..002",
            "can_establish": "Zero variance (single distinct value) or near-constant dominance where >=99% of rows contain an identical value.",
            "cannot_establish": "Whether the feature varies in a broader population outside the sampled dataset or under different experimental conditions.",
            "false_positive_risks": "Rare event indicators in small samples legitimately appear constant (all 0) purely due to sampling limits.",
        },
        "potential_identifiers": {
            "check_id": "STAT-003",
            "can_establish": "High uniqueness ratios (>98%) and sequential integer progression characteristic of surrogate keys.",
            "cannot_establish": "True database primary key constraints or semantic real-world identity.",
            "false_positive_risks": "Fine-grained continuous measurements with high decimal precision can achieve near-100% uniqueness without being surrogate IDs.",
        },
        "numeric_outliers": {
            "check_id": "OUT-001",
            "can_establish": "Observations that fall outside standard Tukey IQR thresholds (1.5 * IQR) and modified Z-score bounds.",
            "cannot_establish": "Whether an extreme observation is data entry noise, sensor malfunction, or genuine heavy-tailed phenomenon.",
            "false_positive_risks": "Fat-tailed distributions naturally generate statistical outliers that are valid and critical for machine learning modeling.",
        },
        "high_cardinality": {
            "check_id": "CARD-001",
            "can_establish": "Excessive number of discrete categories relative to sample size.",
            "cannot_establish": "Whether categories share hierarchical structure or can be meaningfully grouped into latent categories.",
            "false_positive_risks": "Standardized domain taxonomies (e.g., ICD-10 medical diagnostic codes, postal zip codes) legitimately possess high cardinality.",
        },
        "inconsistent_categoricals": {
            "check_id": "CAT-001",
            "can_establish": "String clusters that differ exclusively by whitespace variations or casing differences.",
            "cannot_establish": "Semantic synonyms, spelling errors, phonetic duplicates, or intentional case-sensitive codes.",
            "false_positive_risks": "Case-sensitive domain codes (e.g., chemical formulas, gene markers, base64 strings) intentionally distinguish case.",
        },
        "class_imbalance": {
            "check_id": "IMB-001..002",
            "can_establish": "Empirical majority-to-minority class ratio in the sampled target variable.",
            "cannot_establish": "True population base rate if the dataset was gathered using case-control or stratified sampling.",
            "false_positive_risks": "Rare diseases or fraud detection problems are inherently imbalanced in nature.",
        },
        "numeric_correlations": {
            "check_id": "CORR-001..002",
            "can_establish": "Linear Pearson correlation coefficient and multicollinearity between numeric column pairs.",
            "cannot_establish": "Causal direction, unmeasured confounding, non-linear relationships, or mathematical substitutability in non-linear models.",
            "false_positive_risks": "Correlated features may both be independent causal drivers of an outcome (e.g. systolic and diastolic blood pressure).",
        },
        "target_leakage": {
            "check_id": "LEAK-001..006",
            "can_establish": "Exact target copies, inverted binary proxies, deterministic target encodings, informative missingness, and extreme class separation.",
            "cannot_establish": "Temporal directionality without operational metadata (whether feature was recorded before or after outcome); strong correlation or separation does NOT prove leakage.",
            "false_positive_risks": "Legitimate gold-standard diagnostic predictors (e.g. biopsy results) naturally separate outcomes near-perfectly without constituting leakage.",
        },
        "baseline_comparison": {
            "check_id": "BASELINE-EVAL",
            "can_establish": "Empirical out-of-fold performance shift when suspicious features are included versus excluded in a strictly split pipeline.",
            "cannot_establish": "That performance degradation upon feature removal constitutes proof of leakage; genuine strong predictors also cause large performance drops when ablated.",
            "false_positive_risks": "Small test sets (N < 50) have wide confidence intervals; score deltas can be driven by sample variance rather than true leakage.",
        },
    }
