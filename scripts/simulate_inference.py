"""Simulates real-time inference streaming to test operational metrics and drift detection."""

import logging
import time

import pandas as pd
import requests

from src.config.settings import BASE_DIR, load_yaml_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("modelflow.simulation")


def run_simulation(endpoint_url: str = "http://127.0.0.1:8000/predict", n_requests: int = 50, delay_sec: float = 0.05):
    """Send simulated requests from test dataset to running FastAPI service."""
    params = load_yaml_params()
    test_path = BASE_DIR / params.get("data", {}).get("processed_test_path", "data/processed/test.csv")
    if not test_path.exists():
        logger.error(f"Test data not found at {test_path}. Run pipeline first.")
        return

    df = pd.read_csv(test_path)
    records = df.drop(columns=["Churn"], errors="ignore").to_dict(orient="records")

    logger.info(f"Simulating {n_requests} inference requests to {endpoint_url}...")
    successes = 0

    for i in range(min(n_requests, len(records))):
        record = records[i]
        # Clean types for JSON payload
        payload = {}
        for k, v in record.items():
            if pd.isna(v):
                payload[k] = None
            elif isinstance(v, (int, float, str)):
                payload[k] = v
            else:
                payload[k] = str(v)

        try:
            resp = requests.post(endpoint_url, json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                successes += 1
                logger.info(
                    f"Req {i+1}/{n_requests}: Pred={data['prediction']} | "
                    f"Prob={data['churn_probability']:.3f} | "
                    f"Tier={data['risk_tier']} | "
                    f"Latency={data['latency_ms']}ms"
                )
            else:
                logger.warning(f"Req {i+1} status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Req {i+1} failed: {e}")

        time.sleep(delay_sec)

    logger.info(f"Simulation completed. {successes}/{n_requests} successful.")


if __name__ == "__main__":
    run_simulation()
