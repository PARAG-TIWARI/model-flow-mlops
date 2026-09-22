import { useState, type FormEvent } from 'react';
import { PlayCircle, Clock } from 'lucide-react';
import { executePrediction } from '../services/api';
import type { PredictionPayload, PredictionResponse } from '../services/api';

export const Prediction = () => {
  const [form, setForm] = useState<PredictionPayload>({
    gender: 'Female',
    SeniorCitizen: 0,
    Partner: 'Yes',
    Dependents: 'No',
    tenure: 12,
    PhoneService: 'Yes',
    MultipleLines: 'No',
    InternetService: 'Fiber optic',
    OnlineSecurity: 'No',
    OnlineBackup: 'Yes',
    DeviceProtection: 'No',
    TechSupport: 'No',
    StreamingTV: 'Yes',
    StreamingMovies: 'Yes',
    Contract: 'Month-to-month',
    PaperlessBilling: 'Yes',
    PaymentMethod: 'Electronic check',
    MonthlyCharges: 89.85,
    TotalCharges: 1078.20,
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const applyPreset = (type: 'loyal' | 'moderate' | 'high_risk') => {
    if (type === 'loyal') {
      setForm({
        gender: 'Male',
        SeniorCitizen: 0,
        Partner: 'Yes',
        Dependents: 'Yes',
        tenure: 65,
        PhoneService: 'Yes',
        MultipleLines: 'Yes',
        InternetService: 'DSL',
        OnlineSecurity: 'Yes',
        OnlineBackup: 'Yes',
        DeviceProtection: 'Yes',
        TechSupport: 'Yes',
        StreamingTV: 'No',
        StreamingMovies: 'No',
        Contract: 'Two year',
        PaperlessBilling: 'No',
        PaymentMethod: 'Credit card (automatic)',
        MonthlyCharges: 49.50,
        TotalCharges: 3217.50,
      });
    } else if (type === 'moderate') {
      setForm({
        gender: 'Female',
        SeniorCitizen: 0,
        Partner: 'No',
        Dependents: 'No',
        tenure: 24,
        PhoneService: 'Yes',
        MultipleLines: 'No',
        InternetService: 'Fiber optic',
        OnlineSecurity: 'Yes',
        OnlineBackup: 'No',
        DeviceProtection: 'Yes',
        TechSupport: 'No',
        StreamingTV: 'Yes',
        StreamingMovies: 'No',
        Contract: 'One year',
        PaperlessBilling: 'Yes',
        PaymentMethod: 'Bank transfer (automatic)',
        MonthlyCharges: 79.20,
        TotalCharges: 1900.80,
      });
    } else {
      setForm({
        gender: 'Female',
        SeniorCitizen: 1,
        Partner: 'No',
        Dependents: 'No',
        tenure: 2,
        PhoneService: 'Yes',
        MultipleLines: 'No',
        InternetService: 'Fiber optic',
        OnlineSecurity: 'No',
        OnlineBackup: 'No',
        DeviceProtection: 'No',
        TechSupport: 'No',
        StreamingTV: 'Yes',
        StreamingMovies: 'Yes',
        Contract: 'Month-to-month',
        PaperlessBilling: 'Yes',
        PaymentMethod: 'Electronic check',
        MonthlyCharges: 95.50,
        TotalCharges: 191.00,
      });
    }
  };

  const handlePredict = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const resp = await executePrediction({
        ...form,
        TotalCharges: Number(form.MonthlyCharges) * Math.max(1, Number(form.tenure)),
      });
      setResult(resp);
    } catch (err: any) {
      setError(err.message || 'Inference request failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header and Preset Switcher */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
            <PlayCircle className="w-6 h-6 text-blue-400" />
            <span>Real-Time Customer Churn Simulator</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Feed customer telemetry into the live FastAPI serving endpoint for instant probability scoring.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-xs text-slate-400">Load Presets:</span>
          <button
            onClick={() => applyPreset('loyal')}
            className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/20 cursor-pointer"
          >
            Loyal
          </button>
          <button
            onClick={() => applyPreset('moderate')}
            className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/30 hover:bg-blue-500/20 cursor-pointer"
          >
            Moderate
          </button>
          <button
            onClick={() => applyPreset('high_risk')}
            className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/30 hover:bg-rose-500/20 cursor-pointer"
          >
            High Risk
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Form Inputs (2 columns) */}
        <form onSubmit={handlePredict} className="lg:col-span-2 glass-panel p-6 sm:p-8 rounded-2xl border border-slate-800 space-y-6">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400">Customer Subscription Parameters</h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
            {/* Tenure */}
            <div>
              <label className="block text-slate-300 font-semibold mb-1">Tenure (Months): {form.tenure}</label>
              <input
                type="range"
                min="0"
                max="72"
                value={form.tenure}
                onChange={(e) => setForm({ ...form, tenure: parseInt(e.target.value) || 0 })}
                className="w-full accent-blue-500 cursor-pointer"
              />
            </div>

            {/* Monthly Charges */}
            <div>
              <label className="block text-slate-300 font-semibold mb-1">Monthly Charges ($): {form.MonthlyCharges}</label>
              <input
                type="range"
                min="18"
                max="120"
                step="0.5"
                value={form.MonthlyCharges}
                onChange={(e) => setForm({ ...form, MonthlyCharges: parseFloat(e.target.value) || 0 })}
                className="w-full accent-cyan-500 cursor-pointer"
              />
            </div>

            {/* Contract */}
            <div>
              <label className="block text-slate-300 font-semibold mb-1">Contract Type</label>
              <select
                value={form.Contract}
                onChange={(e) => setForm({ ...form, Contract: e.target.value })}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-slate-200"
              >
                <option value="Month-to-month">Month-to-month</option>
                <option value="One year">One year</option>
                <option value="Two year">Two year</option>
              </select>
            </div>

            {/* Internet Service */}
            <div>
              <label className="block text-slate-300 font-semibold mb-1">Internet Service</label>
              <select
                value={form.InternetService}
                onChange={(e) => setForm({ ...form, InternetService: e.target.value })}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-slate-200"
              >
                <option value="Fiber optic">Fiber optic</option>
                <option value="DSL">DSL</option>
                <option value="No">No Internet Service</option>
              </select>
            </div>

            {/* Payment Method */}
            <div>
              <label className="block text-slate-300 font-semibold mb-1">Payment Method</label>
              <select
                value={form.PaymentMethod}
                onChange={(e) => setForm({ ...form, PaymentMethod: e.target.value })}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-slate-200"
              >
                <option value="Electronic check">Electronic check</option>
                <option value="Mailed check">Mailed check</option>
                <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                <option value="Credit card (automatic)">Credit card (automatic)</option>
              </select>
            </div>

            {/* Tech Support */}
            <div>
              <label className="block text-slate-300 font-semibold mb-1">Tech Support</label>
              <select
                value={form.TechSupport}
                onChange={(e) => setForm({ ...form, TechSupport: e.target.value })}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-slate-200"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>

            {/* Online Security */}
            <div>
              <label className="block text-slate-300 font-semibold mb-1">Online Security</label>
              <select
                value={form.OnlineSecurity}
                onChange={(e) => setForm({ ...form, OnlineSecurity: e.target.value })}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-slate-200"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>

            {/* Paperless Billing */}
            <div>
              <label className="block text-slate-300 font-semibold mb-1">Paperless Billing</label>
              <select
                value={form.PaperlessBilling}
                onChange={(e) => setForm({ ...form, PaperlessBilling: e.target.value })}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-slate-200"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-500 hover:from-blue-500 hover:to-cyan-400 text-white font-bold transition-all shadow-lg shadow-blue-500/25 cursor-pointer disabled:opacity-50"
          >
            {loading ? 'Evaluating Inference Pipeline...' : 'Run Real-Time Prediction'}
          </button>
        </form>

        {/* Prediction Results Gauge & Outcome Card */}
        <div className="glass-panel p-6 sm:p-8 rounded-2xl border border-slate-800 flex flex-col justify-between space-y-6">
          <div>
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-4">Inference Diagnostics</h2>

            {error && (
              <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-xs">
                {error}
              </div>
            )}

            {!result && !error && (
              <div className="text-center py-12 text-slate-400 text-xs">
                <PlayCircle className="w-12 h-12 mx-auto text-slate-600 mb-2" />
                Click "Run Real-Time Prediction" or choose a preset to evaluate.
              </div>
            )}

            {result && (
              <div className="space-y-6 animate-fade-in">
                {/* Risk Badge */}
                <div className="text-center">
                  <div
                    className={`inline-block px-4 py-1.5 rounded-full text-xs font-bold border ${
                      result.risk_tier === 'High Risk'
                        ? 'bg-rose-500/20 text-rose-400 border-rose-500/40'
                        : result.risk_tier === 'Medium Risk'
                        ? 'bg-amber-500/20 text-amber-400 border-amber-500/40'
                        : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40'
                    }`}
                  >
                    {result.risk_tier}
                  </div>
                  <div className="text-4xl font-extrabold text-white mt-2">
                    {(result.churn_probability * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-slate-400 mt-0.5">Calibrated Churn Likelihood</div>
                </div>

                {/* Progress bar gauge */}
                <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      result.risk_tier === 'High Risk'
                        ? 'bg-gradient-to-r from-amber-500 to-rose-500'
                        : result.risk_tier === 'Medium Risk'
                        ? 'bg-gradient-to-r from-cyan-500 to-amber-500'
                        : 'bg-gradient-to-r from-blue-500 to-emerald-500'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(5, result.churn_probability * 100))}%` }}
                  />
                </div>

                {/* Recommendation */}
                <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700/60 space-y-1">
                  <span className="text-xs text-cyan-400 font-semibold">Retention Strategy:</span>
                  <p className="text-xs text-slate-300 leading-relaxed">{result.recommendation}</p>
                </div>
              </div>
            )}
          </div>

          {result && (
            <div className="pt-4 border-t border-slate-800 flex items-center justify-between text-[11px] font-mono text-slate-400">
              <span className="flex items-center space-x-1">
                <Clock className="w-3.5 h-3.5 text-blue-400" />
                <span>Latency: {result.latency_ms} ms</span>
              </span>
              <span>Model: v{result.model_version}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
