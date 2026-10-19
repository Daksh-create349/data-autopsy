"""Deterministic generator for competition-ready forensic demonstration datasets.

Generates 6 reproducible synthetic CSV datasets with fixed random seeds:
1. demo_1_clean_classification.csv: Benchmark clean data (zero critical/high integrity findings).
2. demo_2_missing_and_duplicates.csv: Extreme missingness (100% null) and duplicate record copies.
3. demo_3_imbalanced_classification.csv: Severe class imbalance (ratio > 20:1).
4. demo_4_target_copy_leakage.csv: Direct target duplicate and deterministic categorical proxy.
5. demo_5_statistical_outliers.csv: Plausible heavy-tailed network telemetry with Tukey/Z-score outliers.
6. demo_6_legitimate_strong_predictors.csv: Strong genuine causal predictor (r ~ 0.89) without direct leakage.
"""

from pathlib import Path
import numpy as np
import pandas as pd

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def generate_clean_classification(seed: int = 42) -> Path:
    """1. Clean classification dataset with valid distributions and zero integrity flaws."""
    rng = np.random.RandomState(seed)
    n = 100

    age = rng.randint(22, 65, size=n)
    income = np.round(rng.uniform(30.0, 140.0, size=n), 2)
    credit_score = rng.randint(580, 820, size=n)
    employment_years = rng.randint(1, 25, size=n)
    home_ownership = rng.choice(["OWN", "MORTGAGE", "RENT"], size=n, p=[0.3, 0.45, 0.25])

    # Balanced probabilistic approval based on realistic credit criteria
    logit = -6.0 + 0.006 * credit_score + 0.02 * income + 0.05 * employment_years + rng.normal(0, 0.5, size=n)
    prob = 1.0 / (1.0 + np.exp(-logit))
    credit_approved = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "client_id": [f"C{1000 + i}" for i in range(n)],
        "age": age,
        "annual_income_k": income,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "home_ownership": home_ownership,
        "credit_approved": credit_approved,
    })

    out_path = SAMPLE_DATA_DIR / "demo_1_clean_classification.csv"
    df.to_csv(out_path, index=False)
    return out_path


def generate_missing_and_duplicates(seed: int = 42) -> Path:
    """2. Dataset containing 100% null column, high missingness, and exact duplicate rows."""
    rng = np.random.RandomState(seed)
    n_unique = 75

    tenure = rng.randint(1, 72, size=n_unique)
    charges = np.round(rng.uniform(19.99, 119.99, size=n_unique), 2)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], size=n_unique)
    churn = rng.choice([0, 1], size=n_unique, p=[0.7, 0.3])

    # 45% missing survey responses
    survey_feedback = []
    for _ in range(n_unique):
        if rng.uniform() < 0.45:
            survey_feedback.append(None)
        else:
            survey_feedback.append(f"Score_{rng.randint(1, 10)}")

    # 100% missing legacy field
    deprecated_telemetry = [None] * n_unique

    df_base = pd.DataFrame({
        "customer_id": [f"U{2000 + i}" for i in range(n_unique)],
        "tenure_months": tenure,
        "monthly_charges": charges,
        "contract_type": contract,
        "optional_survey": survey_feedback,
        "deprecated_telemetry": deprecated_telemetry,
        "churned": churn,
    })

    # Append 15 exact duplicate rows
    duplicates = df_base.iloc[:15].copy()
    df = pd.concat([df_base, duplicates], ignore_index=True)

    out_path = SAMPLE_DATA_DIR / "demo_2_missing_and_duplicates.csv"
    df.to_csv(out_path, index=False)
    return out_path


def generate_imbalanced_classification(seed: int = 42) -> Path:
    """3. Financial fraud dataset with severe class imbalance (ratio > 20:1)."""
    rng = np.random.RandomState(seed)
    n_normal = 116
    n_fraud = 4
    n_total = n_normal + n_fraud  # 120 total rows (29:1 ratio)

    # Normal transactions
    amounts_normal = np.round(rng.uniform(5.0, 180.0, size=n_normal), 2)
    trust_normal = np.round(rng.uniform(0.70, 0.99, size=n_normal), 2)
    hours_normal = rng.randint(7, 23, size=n_normal)
    loc_normal = rng.choice(["DOMESTIC", "INTERNATIONAL"], size=n_normal, p=[0.92, 0.08])
    fraud_normal = np.zeros(n_normal, dtype=int)

    # Fraud transactions
    amounts_fraud = np.round(rng.uniform(1500.0, 4800.0, size=n_fraud), 2)
    trust_fraud = np.round(rng.uniform(0.05, 0.30, size=n_fraud), 2)
    hours_fraud = rng.choice([2, 3, 4], size=n_fraud)
    loc_fraud = ["INTERNATIONAL"] * n_fraud
    fraud_flag = np.ones(n_fraud, dtype=int)

    df = pd.DataFrame({
        "tx_id": [f"TX{10000 + i}" for i in range(n_total)],
        "amount_usd": np.concatenate([amounts_normal, amounts_fraud]),
        "device_trust_score": np.concatenate([trust_normal, trust_fraud]),
        "tx_hour": np.concatenate([hours_normal, hours_fraud]),
        "location_type": np.concatenate([loc_normal, loc_fraud]),
        "is_fraud": np.concatenate([fraud_normal, fraud_flag]),
    })

    out_path = SAMPLE_DATA_DIR / "demo_3_imbalanced_classification.csv"
    df.to_csv(out_path, index=False)
    return out_path


def generate_target_copy_leakage(seed: int = 42) -> Path:
    """4. Clinical trial dataset with explicit target leakage (exact target copy and deterministic proxy)."""
    rng = np.random.RandomState(seed)
    n = 80

    age = rng.randint(40, 85, size=n)
    systolic_bp = np.round(rng.normal(135, 15, size=n), 1)
    hba1c = np.round(rng.normal(6.8, 1.2, size=n), 1)
    readmitted = rng.choice([0, 1], size=n, p=[0.6, 0.4])

    # Direct target duplicate
    readmit_status_code = readmitted.copy()

    # Deterministic categorical encoding (100% purity)
    discharge_disposition = [f"CODE_POS_{i%3}" if r == 1 else f"CODE_NEG_{i%3}" for i, r in enumerate(readmitted)]

    df = pd.DataFrame({
        "patient_id": [f"P{1000 + i}" for i in range(n)],
        "age": age,
        "systolic_bp": systolic_bp,
        "hba1c": hba1c,
        "readmit_status_code": readmit_status_code,
        "discharge_disposition": discharge_disposition,
        "readmitted_30d": readmitted,
    })

    out_path = SAMPLE_DATA_DIR / "demo_4_target_copy_leakage.csv"
    df.to_csv(out_path, index=False)
    return out_path


def generate_statistical_outliers(seed: int = 42) -> Path:
    """5. Network server latency dataset containing plausible heavy-tailed statistical outliers."""
    rng = np.random.RandomState(seed)
    n = 100

    # 95 normal log-normal latency readings (15ms - 45ms)
    latencies_normal = np.round(rng.lognormal(mean=3.2, sigma=0.3, size=95), 1)
    # 5 extreme latency spike events (850ms - 1950ms)
    latencies_outliers = np.array([850.0, 920.5, 1240.0, 1550.0, 1920.0])
    latencies = np.concatenate([latencies_normal, latencies_outliers])
    rng.shuffle(latencies)

    cpu_util = np.round(rng.uniform(20.0, 92.0, size=n), 1)
    requests_sec = rng.randint(150, 2400, size=n)
    region = rng.choice(["us-east-1", "us-west-2", "eu-central-1"], size=n)
    failure_flag = (latencies > 500.0).astype(int)

    df = pd.DataFrame({
        "server_id": [f"SRV-{100 + i}" for i in range(n)],
        "latency_ms": latencies,
        "cpu_utilization_pct": cpu_util,
        "requests_per_sec": requests_sec,
        "region": region,
        "server_failure": failure_flag,
    })

    out_path = SAMPLE_DATA_DIR / "demo_5_statistical_outliers.csv"
    df.to_csv(out_path, index=False)
    return out_path


def generate_legitimate_strong_predictors(seed: int = 42) -> Path:
    """6. Loan dataset with genuine strong causal predictor (r ~ 0.89) without direct target leakage."""
    rng = np.random.RandomState(seed)
    n = 100

    loan_principal = np.round(rng.uniform(15000.0, 95000.0, size=n), 2)
    interest_rate = np.round(rng.uniform(4.5, 11.5, size=n), 2)
    credit_score = rng.randint(620, 810, size=n)
    home_value = np.round(rng.uniform(180000.0, 650000.0, size=n), 2)
    employment_type = rng.choice(["salaried", "self-employed"], size=n)

    # Realistic loan payment variations producing genuine causal correlation r ~ 0.88
    term_months = rng.choice([36, 48, 60], size=n)
    noise = rng.normal(0, 110.0, size=n)
    monthly_payment = np.round(loan_principal / term_months + 80.0 + noise, 2)

    df = pd.DataFrame({
        "account_id": [f"ACC-{5000 + i}" for i in range(n)],
        "loan_principal": loan_principal,
        "interest_rate": interest_rate,
        "credit_score": credit_score,
        "home_value": home_value,
        "employment_type": employment_type,
        "monthly_loan_payment": monthly_payment,
    })

    out_path = SAMPLE_DATA_DIR / "demo_6_legitimate_strong_predictors.csv"
    df.to_csv(out_path, index=False)
    return out_path


def generate_all_demo_datasets() -> None:
    """Generate all 6 synthetic demo datasets."""
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    p1 = generate_clean_classification()
    p2 = generate_missing_and_duplicates()
    p3 = generate_imbalanced_classification()
    p4 = generate_target_copy_leakage()
    p5 = generate_statistical_outliers()
    p6 = generate_legitimate_strong_predictors()
    print(f"Generated 6 demo datasets in {SAMPLE_DATA_DIR}:")
    print(f"  1. {p1.name}")
    print(f"  2. {p2.name}")
    print(f"  3. {p3.name}")
    print(f"  4. {p4.name}")
    print(f"  5. {p5.name}")
    print(f"  6. {p6.name}")


if __name__ == "__main__":
    generate_all_demo_datasets()
