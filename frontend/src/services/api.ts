/**
 * API Client for ModelFlow MLOps platform.
 * Supports configurable VITE_API_BASE_URL for deployment environments.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export interface HealthData {
  status: string;
  uptime_seconds: number;
  model_loaded: boolean;
  model_version: string;
  environment: string;
}

export interface ModelInfoData {
  status: string;
  model_name: string;
  version: string;
  framework: string;
  pipeline_type: string;
  promoted_at: string;
  loaded_at: string;
  target: string;
  features: {
    numerical: string[];
    categorical: string[];
  };
  hyperparameters: Record<string, any>;
  validation_metrics: Record<string, any>;
  promotion_criteria: Record<string, any>;
}

export interface PredictionPayload {
  gender: string;
  SeniorCitizen: number | string;
  Partner: string;
  Dependents: string;
  tenure: number;
  PhoneService: string;
  MultipleLines: string;
  InternetService: string;
  OnlineSecurity: string;
  OnlineBackup: string;
  DeviceProtection: string;
  TechSupport: string;
  StreamingTV: string;
  StreamingMovies: string;
  Contract: string;
  PaperlessBilling: string;
  PaymentMethod: string;
  MonthlyCharges: number;
  TotalCharges?: number;
}

export interface PredictionResponse {
  prediction: string;
  churn_probability: number;
  confidence: number;
  risk_tier: 'Low Risk' | 'Medium Risk' | 'High Risk';
  recommendation: string;
  model_version: string;
  model_name: string;
  latency_ms: number;
}

export interface OperationalMetrics {
  uptime_seconds: number;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  error_rate_pct: number;
  latency_ms: {
    p50: number;
    p95: number;
    p99: number;
    mean: number;
    min: number;
    max: number;
  };
  predictions: {
    distribution: Record<string, number>;
    predicted_churn_rate_pct: number;
  };
  buffered_inferences_count: number;
}

export interface DriftReport {
  has_drift: boolean;
  overall_status: string;
  features_analyzed: number;
  drifted_features_count: number;
  drift_percentage: number;
  reference_samples: number;
  current_samples: number;
  feature_reports: Record<string, any>;
}

export interface ExperimentsData {
  experiment_name: string;
  champion_model: string;
  test_metrics: Record<string, any>;
  runs: Array<{
    run_id: string;
    run_name: string;
    status: string;
    duration_sec?: number;
    params: Record<string, any>;
    metrics: Record<string, number>;
  }>;
}

export async function fetchHealth(): Promise<HealthData> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfoData> {
  const res = await fetch(`${API_BASE}/model`);
  if (!res.ok) throw new Error(`Fetch model info failed (${res.status})`);
  return res.json();
}

export async function fetchOperationalMetrics(): Promise<OperationalMetrics> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error(`Fetch metrics failed (${res.status})`);
  return res.json();
}

export async function fetchDriftReport(): Promise<DriftReport> {
  const res = await fetch(`${API_BASE}/monitoring/drift`);
  if (!res.ok) throw new Error(`Fetch drift report failed (${res.status})`);
  return res.json();
}

export async function fetchExperiments(): Promise<ExperimentsData> {
  const res = await fetch(`${API_BASE}/experiments`);
  if (!res.ok) throw new Error(`Fetch experiments failed (${res.status})`);
  return res.json();
}

export async function executePrediction(payload: PredictionPayload): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Inference error (${res.status}): ${errText}`);
  }
  return res.json();
}
