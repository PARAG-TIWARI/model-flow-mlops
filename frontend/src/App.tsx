import { useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { Home } from './pages/Home';
import { Model } from './pages/Model';
import { Prediction } from './pages/Prediction';
import { Experiments } from './pages/Experiments';
import { System } from './pages/System';
import {
  fetchHealth,
  fetchModelInfo,
  fetchOperationalMetrics,
  fetchDriftReport,
  fetchExperiments,
} from './services/api';
import type {
  HealthData,
  ModelInfoData,
  OperationalMetrics,
  DriftReport,
  ExperimentsData,
} from './services/api';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('home');
  const [health, setHealth] = useState<HealthData | null>(null);
  const [model, setModel] = useState<ModelInfoData | null>(null);
  const [metrics, setMetrics] = useState<OperationalMetrics | null>(null);
  const [drift, setDrift] = useState<DriftReport | null>(null);
  const [experiments, setExperiments] = useState<ExperimentsData | null>(null);

  const loadData = async () => {
    try {
      const [h, m, met, dr, exp] = await Promise.allSettled([
        fetchHealth(),
        fetchModelInfo(),
        fetchOperationalMetrics(),
        fetchDriftReport(),
        fetchExperiments(),
      ]);

      if (h.status === 'fulfilled') setHealth(h.value);
      if (m.status === 'fulfilled') setModel(m.value);
      if (met.status === 'fulfilled') setMetrics(met.value);
      if (dr.status === 'fulfilled') setDrift(dr.value);
      if (exp.status === 'fulfilled') setExperiments(exp.value);
    } catch (e) {
      console.error('Telemetry polling error:', e);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0f19] text-slate-100">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isHealthy={health?.status === 'healthy'}
        version={model?.version || '1.0.0'}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'home' && (
          <Home
            health={health}
            model={model}
            metrics={metrics}
            onNavigate={(tab) => setActiveTab(tab)}
          />
        )}
        {activeTab === 'model' && <Model model={model} experiments={experiments} />}
        {activeTab === 'prediction' && <Prediction />}
        {activeTab === 'experiments' && <Experiments experiments={experiments} />}
        {activeTab === 'system' && <System metrics={metrics} drift={drift} />}
      </main>

      <footer className="border-t border-slate-800/80 py-6 text-center text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>ModelFlow MLOps Platform — Reproducible Training, Serving & Monitoring</span>
          <span className="font-mono text-slate-400">Production Ready · Python 3.12 · React 19 · DVC · MLflow</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
