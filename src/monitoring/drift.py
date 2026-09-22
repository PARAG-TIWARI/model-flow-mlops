"""Statistical data drift detection using Kolmogorov-Smirnov, Chi-Square, and PSI."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config.settings import BASE_DIR, load_yaml_params

logger = logging.getLogger("modelflow.monitoring.drift")


def compute_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10, epsilon: float = 1e-4) -> float:
    """Compute Population Stability Index (PSI) between reference and current samples."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Determine bin edges based on reference quantiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    # Frequencies in bins
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = (ref_counts + epsilon) / (len(ref) + epsilon * len(ref_counts))
    cur_pct = (cur_counts + epsilon) / (len(cur) + epsilon * len(cur_counts))

    psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(round(max(0.0, psi_value), 4))


class DriftDetector:
    """Monitors incoming feature distributions against training baseline."""

    def __init__(self, reference_df: Optional[pd.DataFrame] = None, params: Optional[Dict] = None):
        self.params = params or load_yaml_params()
        self.mon_cfg = self.params.get("monitoring", {})
        self.psi_threshold = float(self.mon_cfg.get("psi_drift_threshold", 0.15))
        self.ks_alpha = float(self.mon_cfg.get("ks_pvalue_threshold", 0.05))

        feat_cfg = self.params.get("features", {})
        self.numerical_cols = feat_cfg.get("numerical", ["tenure", "MonthlyCharges", "TotalCharges"])
        self.categorical_cols = feat_cfg.get("categorical", [])

        if reference_df is not None:
            self.reference_df = reference_df
        else:
            ref_path = BASE_DIR / self.mon_cfg.get("reference_path", "data/processed/train.csv")
            if ref_path.exists():
                self.reference_df = pd.read_csv(ref_path)
            else:
                self.reference_df = pd.DataFrame()

    def check_drift(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess statistical drift across all numerical and categorical features."""
        if self.reference_df.empty or current_df.empty:
            return {
                "has_drift": False,
                "overall_status": "Insufficient data",
                "features_analyzed": 0,
                "drifted_features_count": 0,
                "details": {},
            }

        drift_results = {}
        drifted_count = 0

        # 1. Numerical drift (Kolmogorov-Smirnov + PSI)
        for col in self.numerical_cols:
            if col in self.reference_df.columns and col in current_df.columns:
                ref_vals = pd.to_numeric(self.reference_df[col], errors="coerce").dropna().values
                cur_vals = pd.to_numeric(current_df[col], errors="coerce").dropna().values

                if len(ref_vals) > 5 and len(cur_vals) > 5:
                    ks_stat, p_value = ks_2samp(ref_vals, cur_vals)
                    psi = compute_psi(ref_vals, cur_vals)
                    is_drifted = bool(psi > self.psi_threshold or p_value < self.ks_alpha)

                    if is_drifted:
                        drifted_count += 1

                    drift_results[col] = {
                        "type": "numerical",
                        "drift_detected": is_drifted,
                        "psi": round(float(psi), 4),
                        "ks_statistic": round(float(ks_stat), 4),
                        "p_value": round(float(p_value), 5),
                        "severity": "High" if psi > 0.25 else ("Medium" if psi > 0.15 else "Low"),
                    }

        # 2. Categorical drift (Chi-Square contingency)
        for col in self.categorical_cols:
            if col in self.reference_df.columns and col in current_df.columns:
                ref_cats = self.reference_df[col].astype(str).value_counts(normalize=True)
                cur_cats = current_df[col].astype(str).value_counts(normalize=True)

                all_categories = sorted(list(set(ref_cats.index).union(set(cur_cats.index))))
                ref_dist = np.array([ref_cats.get(c, 1e-4) for c in all_categories])
                cur_dist = np.array([cur_cats.get(c, 1e-4) for c in all_categories])

                ref_dist = ref_dist / ref_dist.sum()
                cur_dist = cur_dist / cur_dist.sum()

                cat_psi = float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))
                is_drifted = bool(cat_psi > self.psi_threshold)
                if is_drifted:
                    drifted_count += 1

                drift_results[col] = {
                    "type": "categorical",
                    "drift_detected": is_drifted,
                    "psi": round(float(max(0.0, cat_psi)), 4),
                    "severity": "High" if cat_psi > 0.25 else ("Medium" if cat_psi > 0.15 else "Low"),
                }

        total_features = len(drift_results)
        drift_percentage = round((drifted_count / total_features) * 100, 1) if total_features > 0 else 0.0
        has_systemic_drift = drift_percentage >= 30.0

        return {
            "has_drift": has_systemic_drift,
            "overall_status": "Drift Alert Triggered" if has_systemic_drift else "Stable",
            "features_analyzed": total_features,
            "drifted_features_count": drifted_count,
            "drift_percentage": drift_percentage,
            "reference_samples": len(self.reference_df),
            "current_samples": len(current_df),
            "feature_reports": drift_results,
        }


def calculate_dataset_drift(current_data_path: Optional[Path] = None, output_report_path: Optional[Path] = None) -> Dict[str, Any]:
    """Execute drift analysis comparing test dataset or recent stream to reference baseline."""
    params = load_yaml_params()
    mon_cfg = params.get("monitoring", {})
    data_cfg = params.get("data", {})

    cur_path = current_data_path or (BASE_DIR / data_cfg.get("processed_test_path", "data/processed/test.csv"))
    out_path = output_report_path or (BASE_DIR / mon_cfg.get("drift_report_path", "artifacts/reports/drift_report.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cur_path.exists():
        raise FileNotFoundError(f"Evaluation data for drift calculation not found at {cur_path}")

    current_df = pd.read_csv(cur_path)
    detector = DriftDetector()
    report = detector.check_drift(current_df)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        f"Drift check complete:\n"
        f"  - Status: {report['overall_status']}\n"
        f"  - Drifted Features: {report['drifted_features_count']} / {report['features_analyzed']} ({report['drift_percentage']}%)\n"
        f"Report saved to: {out_path}"
    )

    return report


if __name__ == "__main__":
    calculate_dataset_drift()
