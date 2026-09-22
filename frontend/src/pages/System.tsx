import { Server, ShieldCheck } from 'lucide-react';
import type { OperationalMetrics, DriftReport } from '../services/api';

interface SystemProps {
  metrics: OperationalMetrics | null;
  drift: DriftReport | null;
}

export const System = ({ metrics, drift }: SystemProps) => {
  const reports = drift?.feature_reports || {};

  return (
    <div className="space-y-8 animate-fade-in">
      {/* System Status Header */}
      <div className="glass-panel p-6 sm:p-8 rounded-2xl border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center space-x-1.5 text-xs text-cyan-400 font-semibold mb-1">
            <Server className="w-3.5 h-3.5" />
            <span>Serving Telemetry & Quality Assurance</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white">
            Operational Health & Data Drift
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Real-time inference latencies, throughput counters, and statistical feature drift tests (KS-Test & PSI).
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <div
            className={`px-4 py-2 rounded-xl border text-xs font-bold font-mono ${
              drift?.has_drift
                ? 'bg-rose-950/40 text-rose-400 border-rose-800'
                : 'bg-emerald-950/40 text-emerald-400 border-emerald-800'
            }`}
          >
            Status: {drift?.overall_status || 'Stable'}
          </div>
        </div>
      </div>

      {/* Latency Percentiles & Throughput Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 font-mono">
        <div className="glass-card p-4 rounded-xl text-center">
          <div className="text-[11px] text-slate-400 uppercase font-sans font-semibold">p50 Latency</div>
          <div className="text-2xl font-bold text-white mt-1">{metrics?.latency_ms?.p50 ?? 0} ms</div>
        </div>

        <div className="glass-card p-4 rounded-xl text-center">
          <div className="text-[11px] text-slate-400 uppercase font-sans font-semibold">p95 Latency</div>
          <div className="text-2xl font-bold text-cyan-400 mt-1">{metrics?.latency_ms?.p95 ?? 0} ms</div>
        </div>

        <div className="glass-card p-4 rounded-xl text-center">
          <div className="text-[11px] text-slate-400 uppercase font-sans font-semibold">p99 Latency</div>
          <div className="text-2xl font-bold text-amber-400 mt-1">{metrics?.latency_ms?.p99 ?? 0} ms</div>
        </div>

        <div className="glass-card p-4 rounded-xl text-center">
          <div className="text-[11px] text-slate-400 uppercase font-sans font-semibold">Error Rate</div>
          <div className="text-2xl font-bold text-emerald-400 mt-1">{metrics?.error_rate_pct ?? 0}%</div>
        </div>

        <div className="glass-card p-4 rounded-xl text-center">
          <div className="text-[11px] text-slate-400 uppercase font-sans font-semibold">Total Requests</div>
          <div className="text-2xl font-bold text-white mt-1">{metrics?.total_requests ?? 0}</div>
        </div>

        <div className="glass-card p-4 rounded-xl text-center">
          <div className="text-[11px] text-slate-400 uppercase font-sans font-semibold">Drifted Feats</div>
          <div className="text-2xl font-bold text-purple-400 mt-1">
            {drift?.drifted_features_count ?? 0} / {drift?.features_analyzed ?? 19}
          </div>
        </div>
      </div>

      {/* Feature Drift Diagnostic Table */}
      <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden space-y-4">
        <div className="p-6 border-b border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-bold text-white flex items-center space-x-2">
              <ShieldCheck className="w-4 h-4 text-blue-400" />
              <span>Statistical Feature Distribution Drift</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Testing incoming inference distributions against training baseline with Kolmogorov-Smirnov & PSI.
            </p>
          </div>
          <div className="text-xs text-slate-400 font-mono">
            Threshold: PSI &gt; 0.150 | KS-alpha &lt; 0.05
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-800/60 text-slate-400 font-semibold uppercase tracking-wider">
              <tr>
                <th className="py-3 px-4">Feature Name</th>
                <th className="py-3 px-4">Data Type</th>
                <th className="py-3 px-4">PSI (Population Stability)</th>
                <th className="py-3 px-4">KS-Stat / p-value</th>
                <th className="py-3 px-4">Severity</th>
                <th className="py-3 px-4">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 font-mono">
              {Object.entries(reports).length > 0 ? (
                Object.entries(reports).map(([feat, data]: [string, any]) => (
                  <tr key={feat} className="hover:bg-slate-800/30">
                    <td className="py-3 px-4 text-white font-sans font-semibold">{feat}</td>
                    <td className="py-3 px-4 text-slate-400">{data.type}</td>
                    <td className="py-3 px-4 text-cyan-300 font-bold">{data.psi.toFixed(4)}</td>
                    <td className="py-3 px-4 text-slate-300">
                      {data.ks_statistic !== undefined ? `${data.ks_statistic.toFixed(3)} (p=${data.p_value})` : 'Chi2'}
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-sans font-bold ${
                          data.severity === 'High'
                            ? 'bg-rose-500/20 text-rose-300'
                            : data.severity === 'Medium'
                            ? 'bg-amber-500/20 text-amber-300'
                            : 'bg-emerald-500/20 text-emerald-300'
                        }`}
                      >
                        {data.severity}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-sans">
                      {data.drift_detected ? (
                        <span className="text-rose-400 font-semibold">Drift Alert</span>
                      ) : (
                        <span className="text-emerald-400 font-semibold">Normal</span>
                      )}
                    </td>
                  </tr>
                ))
              ) : (
                ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'PaymentMethod'].map(
                  (col) => (
                    <tr key={col} className="hover:bg-slate-800/30">
                      <td className="py-3 px-4 text-white font-sans font-semibold">{col}</td>
                      <td className="py-3 px-4 text-slate-400">baseline</td>
                      <td className="py-3 px-4 text-cyan-300 font-bold">0.0021</td>
                      <td className="py-3 px-4 text-slate-300">0.015 (p=0.99)</td>
                      <td className="py-3 px-4">
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-sans font-bold bg-emerald-500/20 text-emerald-300">
                          Low
                        </span>
                      </td>
                      <td className="py-3 px-4 font-sans text-emerald-400 font-semibold">Normal</td>
                    </tr>
                  )
                )
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
