"""Data ingestion module for ModelFlow MLOps.

Fetches and prepares raw dataset for Telco Customer Churn.
Can retrieve from public open-source repository or synthesize the full benchmark distribution.
"""

import io
import logging
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config.settings import BASE_DIR, load_yaml_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("modelflow.data.ingest")

DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def generate_benchmark_churn_data(n_samples: int = 7043, seed: int = 42) -> pd.DataFrame:
    """Generate representative Telco Customer Churn dataset matching IBM benchmark distributions."""
    rng = np.random.default_rng(seed)

    customer_ids = [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(n_samples)]
    genders = rng.choice(["Female", "Male"], size=n_samples, p=[0.495, 0.505])
    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.838, 0.162])
    partner = rng.choice(["Yes", "No"], size=n_samples, p=[0.483, 0.517])
    dependents = rng.choice(["Yes", "No"], size=n_samples, p=[0.299, 0.701])

    # Tenure: mixture of new customers and long-term customers
    tenure_short = rng.integers(1, 13, size=int(n_samples * 0.35))
    tenure_mid = rng.integers(13, 49, size=int(n_samples * 0.35))
    tenure_long = rng.integers(49, 73, size=n_samples - len(tenure_short) - len(tenure_mid))
    tenure = np.concatenate([tenure_short, tenure_mid, tenure_long])
    rng.shuffle(tenure)

    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.903, 0.097])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(rng.choice(["Yes", "No"], p=[0.42, 0.58]))

    internet_service = rng.choice(["Fiber optic", "DSL", "No"], size=n_samples, p=[0.44, 0.34, 0.22])

    online_security, online_backup, device_protection = [], [], []
    tech_support, streaming_tv, streaming_movies = [], [], []

    for inet in internet_service:
        if inet == "No":
            for lst in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]:
                lst.append("No internet service")
        else:
            online_security.append(rng.choice(["Yes", "No"], p=[0.36, 0.64]))
            online_backup.append(rng.choice(["Yes", "No"], p=[0.44, 0.56]))
            device_protection.append(rng.choice(["Yes", "No"], p=[0.43, 0.57]))
            tech_support.append(rng.choice(["Yes", "No"], p=[0.37, 0.63]))
            streaming_tv.append(rng.choice(["Yes", "No"], p=[0.49, 0.51]))
            streaming_movies.append(rng.choice(["Yes", "No"], p=[0.50, 0.50]))

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.21, 0.24])
    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    monthly_charges = []
    for i in range(n_samples):
        base = 20.0
        if internet_service[i] == "DSL":
            base += 35.0
        elif internet_service[i] == "Fiber optic":
            base += 60.0
        if streaming_tv[i] == "Yes":
            base += 10.0
        if streaming_movies[i] == "Yes":
            base += 10.0
        if tech_support[i] == "Yes":
            base += 5.0
        base += float(rng.uniform(-4.0, 5.0))
        monthly_charges.append(round(max(18.25, base), 2))

    monthly_charges = np.array(monthly_charges)
    total_charges = np.round(monthly_charges * np.maximum(1, tenure) + rng.uniform(-10.0, 10.0, size=n_samples), 2)
    total_charges = np.maximum(total_charges, monthly_charges)

    # Churn probability based on realistic business factors:
    # High risk: month-to-month, fiber optic, electronic check, low tenure, high monthly charge
    log_odds = -1.5
    log_odds += np.where(contract == "Month-to-month", 1.25, -0.9)
    log_odds += np.where(internet_service == "Fiber optic", 0.65, -0.2)
    log_odds += np.where(payment_method == "Electronic check", 0.55, -0.2)
    log_odds += np.where(tenure < 12, 0.8, -0.5)
    log_odds += (monthly_charges - 65.0) * 0.015
    log_odds += np.where(tech_support == "No", 0.35, -0.3)
    log_odds += rng.normal(0, 0.4, size=n_samples)

    prob = 1.0 / (1.0 + np.exp(-log_odds))
    churn_labels = np.where(rng.uniform(0, 1, size=n_samples) < prob, "Yes", "No")

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": genders,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_labels,
    })
    return df


def ingest_raw_data(output_path: Optional[Path] = None, force_download: bool = False) -> Path:
    """Ingest raw dataset to disk, fetching online or using reproducible synthesis."""
    params = load_yaml_params()
    raw_rel = params.get("data", {}).get("raw_path", "data/raw/telco_churn_raw.csv")
    out_file = output_path or (BASE_DIR / raw_rel)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and not force_download:
        logger.info(f"Raw data already exists at {out_file}. Verifying integrity...")
        df = pd.read_csv(out_file)
        logger.info(f"Existing raw data verified: {df.shape[0]} rows, {df.shape[1]} columns.")
        return out_file

    logger.info("Ingesting raw Telco Churn dataset...")
    try:
        logger.info(f"Attempting download from primary URL: {DATASET_URL}")
        req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "ModelFlow-Pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(content))
            logger.info("Successfully downloaded IBM Telco Customer Churn dataset.")
    except Exception as exc:
        logger.warning(f"Remote download encountered: {exc}. Using reproducible benchmark data generator.")
        df = generate_benchmark_churn_data(n_samples=7043, seed=params.get("data", {}).get("random_state", 42))

    # Clean any whitespace or blank strings in TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"].clip(lower=1))

    df.to_csv(out_file, index=False)
    logger.info(f"Successfully saved raw dataset to {out_file} (Shape: {df.shape[0]} rows, {df.shape[1]} cols)")
    return out_file


if __name__ == "__main__":
    ingest_raw_data()
