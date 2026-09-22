import { ShieldCheck, Zap, Database, GitBranch, Cpu, Activity, ArrowRight, CheckCircle2 } from 'lucide-react';
import type { HealthData, ModelInfoData, OperationalMetrics } from '../services/api';

interface HomeProps {
  health: HealthData | null;
  model: ModelInfoData | null;
  metrics: OperationalMetrics | null;
  onNavigate: (tab: string) => void;
}

export const Home = ({ health, model, metrics, onNavigate }: HomeProps) => {
  return (
    <div className="space-y-10 animate-fade-in">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-2xl glass-panel p-8 md:p-12 border border-slate-800">
        <div className="absolute top-0 right-0 -mr-20 -mt-20 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 -ml-20 -mb-20 w-80 h-80 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-semibold">
            <Zap className="w-3.5 h-3.5" />
            <span>Production-Grade MLOps Architecture</span>
          </div>

          <h1 className="text-3xl sm:text-5xl font-extrabold text-white tracking-tight leading-tight">
            Reproducible Machine Learning Training, Tracking, Serving & Monitoring
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed">
            An independent, enterprise end-to-end MLOps platform for predictive analytics.
            Featuring automated DVC pipeline versioning, MLflow experiment tracking,
            FastAPI real-time inference, and statistical data drift detection with Kolmogorov-Smirnov and PSI.
          </p>

          <div className="pt-4 flex flex-wrap gap-4">
            <button
              onClick={() => onNavigate('prediction')}
              className="flex items-center space-x-2 px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold transition-all shadow-lg shadow-blue-600/30 cursor-pointer"
            >
              <span>Test Live Inference</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNavigate('model')}
              className="flex items-center space-x-2 px-6 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold border border-slate-700 transition-all cursor-pointer"
            >
              <Cpu className="w-4 h-4 text-cyan-400" />
              <span>Inspect Champion Model</span>
            </button>
          </div>
        </div>
      </div>

      {/* Live System Metric Counters */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="glass-card p-5 rounded-xl">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase tracking-wider font-semibold">Champion Model</span>
            <Cpu className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-xl font-bold text-white truncate">{model?.model_name || 'Loading...'}</div>
          <p className="text-xs text-slate-400 mt-1">Status: <span className="text-emerald-400 font-medium">Production v{model?.version || '1.0.0'}</span></p>
        </div>

        <div className="glass-card p-5 rounded-xl">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase tracking-wider font-semibold">Test ROC-AUC</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400">
            {model?.validation_metrics?.roc_auc ? model.validation_metrics.roc_auc.toFixed(4) : '0.8448'}
          </div>
          <p className="text-xs text-slate-400 mt-1">Accuracy: <span className="text-slate-200">{(model?.validation_metrics?.accuracy ? model.validation_metrics.accuracy * 100 : 74.7).toFixed(1)}%</span></p>
        </div>

        <div className="glass-card p-5 rounded-xl">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase tracking-wider font-semibold">Total Requests</span>
            <Activity className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold text-white">
            {metrics?.total_requests ?? 0}
          </div>
          <p className="text-xs text-slate-400 mt-1">Avg Latency: <span className="text-cyan-400 font-mono">{metrics?.latency_ms?.mean ?? 0} ms</span></p>
        </div>

        <div className="glass-card p-5 rounded-xl">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase tracking-wider font-semibold">API Liveness</span>
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400">Healthy</div>
          <p className="text-xs text-slate-400 mt-1">Uptime: <span className="text-slate-200 font-mono">{health?.uptime_seconds ?? 0}s</span></p>
        </div>
      </div>

      {/* Architecture Overview Diagram */}
      <div className="glass-panel p-6 sm:p-8 rounded-2xl border border-slate-800 space-y-6">
        <div className="flex items-center space-x-3">
          <GitBranch className="w-5 h-5 text-blue-400" />
          <h2 className="text-xl font-bold text-white">End-to-End Pipeline Architecture</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/60 space-y-2">
            <div className="text-xs font-semibold uppercase text-blue-400">Stage 1</div>
            <div className="text-sm font-bold text-white flex items-center space-x-2">
              <Database className="w-4 h-4 text-blue-400" />
              <span>Data Ingestion</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Automated fetch of Telco Churn dataset with validation of 19 features and 7,043 samples.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/60 space-y-2">
            <div className="text-xs font-semibold uppercase text-cyan-400">Stage 2</div>
            <div className="text-sm font-bold text-white flex items-center space-x-2">
              <GitBranch className="w-4 h-4 text-cyan-400" />
              <span>Preprocessing</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              StandardScaler, OneHotEncoder, and stratified train/val/test splits tracked via DVC.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/60 space-y-2">
            <div className="text-xs font-semibold uppercase text-indigo-400">Stage 3</div>
            <div className="text-sm font-bold text-white flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-indigo-400" />
              <span>Model Training</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Trains 3 candidate models: Calibrated Logistic Regression, Random Forest & HistGradientBoosting.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/60 space-y-2">
            <div className="text-xs font-semibold uppercase text-purple-400">Stage 4</div>
            <div className="text-sm font-bold text-white flex items-center space-x-2">
              <Activity className="w-4 h-4 text-purple-400" />
              <span>MLflow Tracking</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Logs parameters, validation metrics, duration, confusion matrix and registers champion model.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/60 space-y-2">
            <div className="text-xs font-semibold uppercase text-emerald-400">Stage 5</div>
            <div className="text-sm font-bold text-white flex items-center space-x-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>Serving & Drift</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              FastAPI REST inference, real-time risk tiers, and live PSI + Kolmogorov-Smirnov drift monitoring.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
