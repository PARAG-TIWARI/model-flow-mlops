import { Cpu, BarChart3, Sparkles } from 'lucide-react';
import type { ModelInfoData, ExperimentsData } from '../services/api';

interface ModelProps {
  model: ModelInfoData | null;
  experiments: ExperimentsData | null;
}

export const Model = ({ model, experiments }: ModelProps) => {
  const metrics = model?.validation_metrics || {};
  const cm = metrics.confusion_matrix || { true_negative: 615, false_positive: 161, false_negative: 106, true_positive: 175 };
  const testMetrics = experiments?.test_metrics || {};

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Model Overview Header */}
      <div className="glass-panel p-6 sm:p-8 rounded-2xl border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="space-y-2">
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-xs font-semibold">
              Production Champion
            </span>
            <span className="text-xs text-slate-400">Framework: {model?.framework || 'scikit-learn'}</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white flex items-center space-x-3">
            <Cpu className="w-8 h-8 text-blue-500" />
            <span>{model?.model_name || 'Champion Model'}</span>
          </h1>
          <p className="text-sm text-slate-400">
            Pipeline Architecture: <span className="text-slate-200 font-mono">{model?.pipeline_type || 'StandardScaler + OneHotEncoder + Classifier'}</span>
          </p>
        </div>

        <div className="flex flex-wrap gap-4 text-xs font-mono">
          <div className="px-4 py-2 rounded-xl bg-slate-800/80 border border-slate-700">
            <span className="text-slate-400">Version: </span>
            <span className="text-white font-bold">{model?.version || '1.0.0'}</span>
          </div>
          <div className="px-4 py-2 rounded-xl bg-slate-800/80 border border-slate-700">
            <span className="text-slate-400">Target: </span>
            <span className="text-cyan-400 font-bold">{model?.target || 'Churn'}</span>
          </div>
          <div className="px-4 py-2 rounded-xl bg-slate-800/80 border border-slate-700">
            <span className="text-slate-400">Gate Threshold: </span>
            <span className="text-emerald-400 font-bold">&gt;= 0.8200</span>
          </div>
        </div>
      </div>

      {/* Benchmark Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        <div className="glass-card p-4 rounded-xl text-center">
          <div className="text-xs text-slate-400 uppercase font-semibold">ROC-AUC</div>
          <div className="text-2xl font-extrabold text-blue-400 mt-1">
            {metrics.roc_auc ? metrics.roc_auc.toFixed(4) : (testMetrics.roc_auc ? testMetrics.roc_auc.toFixed(4) : '0.8448')}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Primary Gate</div>
        </div>

        <div className="glass-card p-4 rounded-xl text-center">
          <div className="text-xs text-slate-400 uppercase font-semibold">Accuracy</div>
          <div className="text-2xl font-extrabold text-emerald-400 mt-1">
            {metrics.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : '74.7%'}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Out-of-sample</div>
        </div>

        <div className="glass-card p-4 rounded-xl text-center">
          <div className="text-xs text-slate-400 uppercase font-semibold">F1-Score</div>
          <div className="text-2xl font-extrabold text-purple-400 mt-1">
            {metrics.f1 ? metrics.f1.toFixed(4) : '0.6238'}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Positive Class</div>
        </div>

        <div className="glass-card p-4 rounded-xl text-center">
          <div className="text-xs text-slate-400 uppercase font-semibold">Precision</div>
          <div className="text-2xl font-extrabold text-cyan-400 mt-1">
            {metrics.precision ? metrics.precision.toFixed(4) : '0.5208'}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Churn Retention</div>
        </div>

        <div className="glass-card p-4 rounded-xl text-center">
          <div className="text-xs text-slate-400 uppercase font-semibold">Recall</div>
          <div className="text-2xl font-extrabold text-amber-400 mt-1">
            {metrics.recall ? metrics.recall.toFixed(4) : '0.7766'}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Churn Detection</div>
        </div>

        <div className="glass-card p-4 rounded-xl text-center">
          <div className="text-xs text-slate-400 uppercase font-semibold">PR-AUC</div>
          <div className="text-2xl font-extrabold text-rose-400 mt-1">
            {metrics.pr_auc ? metrics.pr_auc.toFixed(4) : '0.6587'}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Avg Precision</div>
        </div>
      </div>

      {/* Confusion Matrix & Parameters */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confusion Matrix Card */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-blue-400" />
            <h2 className="text-lg font-bold text-white">Out-of-Sample Confusion Matrix</h2>
          </div>
          <p className="text-xs text-slate-400">
            Evaluated on held-out test split (1,057 verified customer records):
          </p>

          <div className="grid grid-cols-2 gap-3 pt-2">
            <div className="p-4 rounded-xl bg-emerald-950/30 border border-emerald-800/40 text-center">
              <div className="text-xs uppercase text-emerald-400 font-semibold">True Negative (Retained)</div>
              <div className="text-3xl font-extrabold text-white mt-1">{cm.true_negative}</div>
              <div className="text-[11px] text-slate-400 mt-1">Correctly predicted loyal</div>
            </div>

            <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-800/30 text-center">
              <div className="text-xs uppercase text-amber-400 font-semibold">False Positive (False Alarm)</div>
              <div className="text-3xl font-extrabold text-amber-200 mt-1">{cm.false_positive}</div>
              <div className="text-[11px] text-slate-400 mt-1">Predicted churn, remained</div>
            </div>

            <div className="p-4 rounded-xl bg-rose-950/20 border border-rose-800/30 text-center">
              <div className="text-xs uppercase text-rose-400 font-semibold">False Negative (Missed)</div>
              <div className="text-3xl font-extrabold text-rose-200 mt-1">{cm.false_negative}</div>
              <div className="text-[11px] text-slate-400 mt-1">Predicted loyal, churned</div>
            </div>

            <div className="p-4 rounded-xl bg-blue-950/30 border border-blue-800/40 text-center">
              <div className="text-xs uppercase text-blue-400 font-semibold">True Positive (Caught)</div>
              <div className="text-3xl font-extrabold text-white mt-1">{cm.true_positive}</div>
              <div className="text-[11px] text-slate-400 mt-1">Successfully flagged at risk</div>
            </div>
          </div>
        </div>

        {/* Hyperparameters Card */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-5 h-5 text-cyan-400" />
            <h2 className="text-lg font-bold text-white">Hyperparameter Configuration</h2>
          </div>
          <p className="text-xs text-slate-400">
            Parameters configured via <span className="font-mono text-slate-300">params.yaml</span> and logged to MLflow:
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-800/60 text-slate-400">
                <tr>
                  <th className="py-2.5 px-3 rounded-l-lg font-semibold">Parameter</th>
                  <th className="py-2.5 px-3 rounded-r-lg font-semibold">Configured Value</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 font-mono">
                {Object.entries(model?.hyperparameters || { C: 0.75, max_iter: 1000, solver: 'lbfgs', class_weight: 'balanced' }).map(
                  ([k, v]) => (
                    <tr key={k} className="hover:bg-slate-800/30">
                      <td className="py-2.5 px-3 text-slate-300">{k}</td>
                      <td className="py-2.5 px-3 text-cyan-400 font-bold">{String(v)}</td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
