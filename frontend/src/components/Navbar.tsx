import { Activity, Cpu, PlayCircle, Layers, Server } from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isHealthy: boolean;
  version: string;
}

export const Navbar = ({ activeTab, setActiveTab, isHealthy, version }: NavbarProps) => {
  const navItems = [
    { id: 'home', label: 'Home', icon: Layers },
    { id: 'model', label: 'Model', icon: Cpu },
    { id: 'prediction', label: 'Live Prediction', icon: PlayCircle },
    { id: 'experiments', label: 'Experiments', icon: Activity },
    { id: 'system', label: 'System & Drift', icon: Server },
  ];

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800 px-6 py-3">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('home')}>
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-cyan-400 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-lg text-white tracking-tight">ModelFlow</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30 font-mono">
                v{version}
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">MLOps Serving & Monitoring</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-1 sm:space-x-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-600/25'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span className="hidden md:inline">{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Health status badge */}
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1.5 px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                isHealthy ? 'bg-emerald-400 shadow-sm shadow-emerald-400/50 animate-pulse' : 'bg-amber-400'
              }`}
            />
            <span className="text-slate-300 font-mono">{isHealthy ? 'API Active' : 'Connecting'}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
