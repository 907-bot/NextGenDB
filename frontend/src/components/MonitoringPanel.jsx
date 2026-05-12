import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Cpu, Database, Radio, Server, Zap, AlertCircle, CheckCircle } from 'lucide-react';
import axios from 'axios';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const StatusDot = ({ status }) => {
  const ok = status && (status.includes('ACTIVE') || status.includes('OK'));
  return (
    <span className={`inline-block w-2 h-2 rounded-full mr-2 ${ok ? 'bg-green-400 shadow-[0_0_6px_#4ade80]' : 'bg-yellow-400 shadow-[0_0_6px_#facc15]'}`} />
  );
};

const MetricCard = ({ icon: Icon, label, value, unit = '', color = 'cyan' }) => (
  <div className={`glass-card p-4 border-${color}-500/10 flex items-center gap-4`}>
    <div className={`w-10 h-10 rounded-lg bg-${color}-500/10 flex items-center justify-center flex-shrink-0`}>
      <Icon className={`w-5 h-5 text-${color}-400`} />
    </div>
    <div className="min-w-0">
      <div className="text-[10px] text-white/40 uppercase tracking-widest font-mono truncate">{label}</div>
      <div className={`text-xl font-black text-${color}-300`}>
        {value}<span className="text-xs text-white/30 ml-1 font-normal">{unit}</span>
      </div>
    </div>
  </div>
);

const LayerStatusRow = ({ layer, status }) => (
  <motion.div
    initial={{ opacity: 0, x: -8 }}
    animate={{ opacity: 1, x: 0 }}
    className="flex items-center justify-between py-2 border-b border-white/5 last:border-0"
  >
    <div className="flex items-center gap-2">
      <StatusDot status={status} />
      <span className="text-xs font-mono text-white/70">{layer}</span>
    </div>
    <span className="text-[10px] font-mono text-white/40 truncate max-w-[140px]">{status}</span>
  </motion.div>
);

const NodeCard = ({ node }) => (
  <div className="flex-shrink-0 w-48 p-3 rounded-xl bg-white/5 border border-cyan-500/10 hover:border-cyan-500/30 transition-all">
    <div className="flex items-center gap-2 mb-2">
      <Server className="w-3 h-3 text-cyan-400" />
      <span className="text-[10px] font-mono text-cyan-400">{node.node_id}</span>
    </div>
    <div className="text-[10px] text-white/50 space-y-1">
      <div>Role: <span className="text-white/80">{node.role}</span></div>
      <div>Host: <span className="text-white/80">{node.host}:{node.port}</span></div>
      <div>Uptime: <span className="text-green-400">{node.uptime_s}s</span></div>
    </div>
  </div>
);

export default function MonitoringPanel() {
  const [data, setData]       = useState(null);
  const [error, setError]     = useState(false);
  const [lastPoll, setLastPoll] = useState(null);

  const poll = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/metrics/dashboard`, { timeout: 4000 });
      setData(res.data);
      setError(false);
      setLastPoll(new Date().toLocaleTimeString());
    } catch {
      setError(true);
    }
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, [poll]);

  const m = data?.metrics || {};
  const layers = data?.layer_status || {};
  const nodes  = data?.cluster_nodes || [];

  return (
    <div className="glass-card border-white/5 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/5 pb-4">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-green-400" />
          <h2 className="text-sm font-bold tracking-widest text-white/80 uppercase">Observability — Layer 10</h2>
        </div>
        <div className="flex items-center gap-2">
          {error
            ? <AlertCircle className="w-4 h-4 text-red-400" />
            : <CheckCircle className="w-4 h-4 text-green-400" />
          }
          <span className="text-[10px] font-mono text-white/30">
            {error ? 'OFFLINE' : `polled ${lastPoll}`}
          </span>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 gap-3">
        <MetricCard icon={Zap}      label="Queries Total"   value={(m.query_total?.success ?? 0) + (m.query_total?.error ?? 0)} color="cyan" />
        <MetricCard icon={Activity}  label="Avg Latency"     value={m.avg_latency_ms ?? 0} unit="ms" color="magenta" />
        <MetricCard icon={Database}  label="Graph Nodes"     value={m.graph_nodes ?? 0} color="cyan" />
        <MetricCard icon={Database}  label="Graph Edges"     value={m.graph_edges ?? 0} color="cyan" />
        <MetricCard icon={Cpu}       label="GNN Steps"       value={m.gnn_steps ?? 0} color="green" />
        <MetricCard icon={Radio}     label="Stream Events"   value={Object.values(m.stream_events ?? {}).reduce((a, b) => a + b, 0)} color="yellow" />
      </div>

      {/* Confidence */}
      {m.avg_confidence > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-[10px] font-mono text-white/40">
            <span>AVG_CONFIDENCE</span><span>{(m.avg_confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${m.avg_confidence * 100}%` }}
              transition={{ duration: 0.8 }}
              className="h-full bg-gradient-to-r from-cyan-500 to-green-400 rounded-full"
            />
          </div>
        </div>
      )}

      {/* 10-Layer Status */}
      <div>
        <div className="text-[10px] font-mono text-white/30 uppercase tracking-widest mb-3">
          Layer Status — All 10
        </div>
        <div className="space-y-0">
          {Object.entries(layers).map(([layer, status]) => (
            <LayerStatusRow key={layer} layer={layer.replace(/_/g, ' ').toUpperCase()} status={status} />
          ))}
          {Object.keys(layers).length === 0 && (
            <div className="text-[11px] text-white/20 font-mono animate-pulse py-2">
              Polling engine status...
            </div>
          )}
        </div>
      </div>

      {/* Cluster Nodes — Layer 9 */}
      {nodes.length > 0 && (
        <div>
          <div className="text-[10px] font-mono text-white/30 uppercase tracking-widest mb-3">
            Distributed Nodes — Layer 9
          </div>
          <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-hide">
            {nodes.map(n => <NodeCard key={n.node_id} node={n} />)}
          </div>
        </div>
      )}

      {/* External Links */}
      <div className="flex gap-3 pt-2 border-t border-white/5">
        <a href="http://localhost:9090" target="_blank" rel="noreferrer"
           className="text-[10px] font-mono text-orange-400/70 hover:text-orange-400 transition-colors">
          ⬡ Prometheus →
        </a>
        <a href="http://localhost:3000" target="_blank" rel="noreferrer"
           className="text-[10px] font-mono text-yellow-400/70 hover:text-yellow-400 transition-colors">
          ⬡ Grafana →
        </a>
        <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer"
           className="text-[10px] font-mono text-cyan-400/70 hover:text-cyan-400 transition-colors">
          ⬡ API Docs →
        </a>
      </div>
    </div>
  );
}
