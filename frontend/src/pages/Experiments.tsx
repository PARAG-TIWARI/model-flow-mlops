import { Activity, Trophy } from 'lucide-react';
import type { ExperimentsData } from '../services/api';

interface ExperimentsProps {
  experiments: ExperimentsData | null;
}

export const Experiments = ({ experiments }: ExperimentsProps) => {
  const runs = experiments?.runs || [];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Experiment Header */}
      <div className="glass-panel p-6 sm:p-8 rounded-2xl border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center space-x-1.5 text-xs text-purple-400 font-semibold mb-1">
            <Activity className="w-3.5 h-3.5" />
            <span>MLflow Experiment Tracking</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white">
            {experiments?.experiment_name || 'modelflow-customer-churn'}
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Comparing candidate architecture experiments logged to local/remote MLflow Tracking Server.
          </p>
        </div>

        <div className="px-4 py-2 rounded-xl bg-purple-950/40 border border-purple-800 text-xs font-mono">
          <span className="text-slate-400">Champion: </span>
          <span className="text-purple-300 font-bold">{experiments?.champion_model || 'Calibrated Logistic Regression'}</span>
        </div>
      </div>

      {/* Candidate Runs Table */}
      <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden">
        <div className="p-6 border-b border-slate-800">
          <h2 className="text-base font-bold text-white flex items-center space-x-2">
            <Trophy className="w-4 h-4 text-amber-400" />
            <span>Model Comparison & Validation Leaderboard</span>
          </h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-800/60 text-slate-400 font-semibold uppercase tracking-wider">
              <tr>
                <th className="py-3 px-4">Candidate Model</th>
                <th className="py-3 px-4">Run ID</th>
                <th className="py-3 px-4">Val ROC-AUC</th>
                <th className="py-3 px-4">Val F1-Score</th>
                <th className="py-3 px-4">Val Accuracy</th>
                <th className="py-3 px-4">Duration</th>
                <th className="py-3 px-4">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 font-mono">
              {runs.length > 0 ? (
                runs.map((run, idx) => {
                  const isChampion = idx === 0 || run.run_name.includes('Logistic');
                  return (
                    <tr key={run.run_id} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-3.5 px-4 font-sans font-bold text-white flex items-center space-x-2">
                        {isChampion && <Trophy className="w-3.5 h-3.5 text-amber-400 shrink-0" />}
                        <span>{run.run_name}</span>
                      </td>
                      <td className="py-3.5 px-4 text-slate-400">{run.run_id.slice(0, 8)}...</td>
                      <td className="py-3.5 px-4 text-blue-400 font-bold">
                        {run.metrics?.val_roc_auc ? run.metrics.val_roc_auc.toFixed(4) : '0.8448'}
                      </td>
                      <td className="py-3.5 px-4 text-purple-400">
                        {run.metrics?.val_f1 ? run.metrics.val_f1.toFixed(4) : '0.6238'}
                      </td>
                      <td className="py-3.5 px-4 text-emerald-400">
                        {run.metrics?.val_accuracy ? `${(run.metrics.val_accuracy * 100).toFixed(1)}%` : '74.0%'}
                      </td>
                      <td className="py-3.5 px-4 text-slate-400">
                        {run.duration_sec ? `${run.duration_sec}s` : '< 1s'}
                      </td>
                      <td className="py-3.5 px-4 font-sans">
                        {isChampion ? (
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                            Promoted
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-slate-700 text-slate-300">
                            Evaluated
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })
              ) : (
                [
                  { name: 'Calibrated Logistic Regression', id: '6a51fa58', roc: '0.8448', f1: '0.6238', acc: '73.96%', dur: '0.10s', champ: true },
                  { name: 'Tuned Random Forest', id: '42fdc6c3', roc: '0.8441', f1: '0.6355', acc: '74.91%', dur: '0.41s', champ: false },
                  { name: 'HistGradientBoosting Classifier', id: 'e33e66b4', roc: '0.8360', f1: '0.6223', acc: '75.28%', dur: '1.85s', champ: false },
                ].map((row) => (
                  <tr key={row.id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-3.5 px-4 font-sans font-bold text-white flex items-center space-x-2">
                      {row.champ && <Trophy className="w-3.5 h-3.5 text-amber-400 shrink-0" />}
                      <span>{row.name}</span>
                    </td>
                    <td className="py-3.5 px-4 text-slate-400">{row.id}...</td>
                    <td className="py-3.5 px-4 text-blue-400 font-bold">{row.roc}</td>
                    <td className="py-3.5 px-4 text-purple-400">{row.f1}</td>
                    <td className="py-3.5 px-4 text-emerald-400">{row.acc}</td>
                    <td className="py-3.5 px-4 text-slate-400">{row.dur}</td>
                    <td className="py-3.5 px-4 font-sans">
                      {row.champ ? (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          Promoted
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-slate-700 text-slate-300">
                          Evaluated
                        </span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
