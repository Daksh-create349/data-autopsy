# Forensic Test & Demonstration Datasets

Deterministic synthetic datasets generated with fixed random seeds (`seed=42`) using `scripts/generate_demo_datasets.py`.
Designed to validate real detector logic, verify evidence consistency, demonstrate local AI interpretations, and confirm that clean data triggers zero false critical alarms.

---

## Competition Demonstration Datasets

### 1. `demo_1_clean_classification.csv`
- **Domain:** Personal credit approval risk (`100 rows × 7 columns`).
- **Target Feature:** `credit_approved` (balanced binary class, ~50/50).
- **Intended Properties:** Realistic continuous variables (`annual_income_k`, `credit_score`), clean categories (`home_ownership`), unique IDs (`client_id`), 0% missingness, 0% duplicates.
- **Expected Engine Behavior:** Clean baseline. **Zero CRITICAL or HIGH findings**. Only informational potential identifier flag on `client_id`.

### 2. `demo_2_missing_and_duplicates.csv`
- **Domain:** Telecom customer churn (`90 rows × 7 columns`).
- **Target Feature:** `churned` (binary).
- **Intended Properties:** 
  - `deprecated_telemetry`: 100% unobserved null values (`INT-001`).
  - `optional_survey`: 45% missing values (`INT-003`).
  - 15 exact duplicate records (`DUP-001`).
- **Expected Engine Behavior:** Flags `INT-001` (Critical 100% null), `INT-003` (Medium missing ratio), and `DUP-001` (High duplicate row count). Zero false leakage alerts.

### 3. `demo_3_imbalanced_classification.csv`
- **Domain:** Financial transaction fraud detection (`120 rows × 6 columns`).
- **Target Feature:** `is_fraud` (binary).
- **Intended Properties:** Severe class imbalance: 116 legitimate transactions vs 4 fraud transactions (imbalance ratio = 29.0:1, exceeding the 20.0:1 critical threshold).
- **Expected Engine Behavior:** Flags `IMB-001` (High severity class imbalance). Correctly notes that minority class scarcity requires stratified evaluation or cost-sensitive learning without treating the dataset as corrupted.

### 4. `demo_4_target_copy_leakage.csv`
- **Domain:** Clinical hospital readmission (`80 rows × 7 columns`).
- **Target Feature:** `readmitted_30d` (binary).
- **Intended Properties:** 
  - `readmit_status_code`: Exact 1:1 identical copy of the target variable (`LEAK-001`).
  - `discharge_disposition`: Encoded categorical feature with 100% class purity mapping (`LEAK-004`).
- **Expected Engine Behavior:** Flags `LEAK-001` (Critical target leakage) and `LEAK-004` (Target encoding leakage). Pipeline baseline comparison shows near-100% accuracy drop when ablated.

### 5. `demo_5_statistical_outliers.csv`
- **Domain:** Cloud server telemetry & latency monitor (`100 rows × 6 columns`).
- **Target Feature:** `server_failure` (binary).
- **Intended Properties:** `latency_ms` follows heavy-tailed log-normal distribution with 95 normal measurements (15ms - 45ms) and 5 plausible network spike events (850ms - 1920ms).
- **Expected Engine Behavior:** Flags `OUT-001` (Statistical outliers via Tukey IQR and modified Z-score). Explainer and limitations document that these represent plausible extreme real-world events rather than pipeline corruptions.

### 6. `demo_6_legitimate_strong_predictors.csv`
- **Domain:** Residential mortgage monthly payment model (`100 rows × 7 columns`).
- **Target Feature:** `monthly_loan_payment` (continuous).
- **Intended Properties:** `loan_principal` is a genuine physical driver of payment amount with Pearson correlation $r \approx 0.89$ (above 0.85 strong predictor threshold, but strictly below 0.95 leakage threshold).
- **Expected Engine Behavior:** Flags `LEAK-003-STRONG` (Low severity advisory notice). **Does NOT raise a CRITICAL alert**. Explainer notes temporal precedence must be verified, distinguishing genuine causal drivers from label leakage.

---

## Baseline Test Fixtures

- `clean_patient_records.csv`: Minimal 10-row baseline test fixture.
- `corrupt_clinical_trials.csv`: 10-row multi-defect edge-case fixture.
