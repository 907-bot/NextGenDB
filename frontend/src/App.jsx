import React, { useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import QueryInput from './components/QueryInput';
import GraphViewer from './components/GraphViewer';
import ReasoningPanel from './components/ReasoningPanel';
import ConfidenceMeter from './components/ConfidenceMeter';
import TimelineView from './components/TimelineView';
import MonitoringPanel from './components/MonitoringPanel';

const API = import.meta.env.VITE_API_URL || 'https://nextgendb.onrender.com/api/v1';

function App() {
  const [loading, setLoading] = useState(false);
  const [data, setData]       = useState(null);
  const [activeTab, setActiveTab] = useState('query'); // 'query' | 'monitor'

  const handleSearch = async (query) => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/query`, { query });
      setData(response.data);
      setActiveTab('query');
    } catch (error) {
      console.error('Backend connection failed, using mock data', error);
      setData({
        answer: 'Simulated response for: ' + query,
        confidence: 0.88 + Math.random() * 0.1,
        steps: [
          { id: 1, action: 'RETRIEVE_CONTEXT', description: 'Decomposing query nodes' },
          { id: 2, action: 'CAUSAL_ANALYSIS',  description: 'Calculating causal weights' },
          { id: 3, action: 'TEMPORAL_REASONING', description: 'Sequencing events' },
          { id: 4, action: 'SYNTHESIZE',        description: 'Generating answer' },
        ],
        graph_snapshot: { nodes: [], links: [] },
        timeline: [
          { event: 'Pattern Recognised', timestamp: new Date().toISOString() },
          { event: 'Logic Converged',    timestamp: new Date().toISOString() },
        ],
      });
    } finally {
      setLoading(false);
    }
  };

  const nodes = data?.graph_snapshot?.nodes || [];

  return (
    <div className="min-h-screen p-8 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <header className="flex justify-between items-end border-b border-white/5 pb-6">
        <div>
          <h1 className="text-4xl font-black bg-gradient-to-r from-cyan-400 to-violet-400 bg-clip-text text-transparent">
            NextGenDB
          </h1>
          <p className="text-white/40 text-xs font-mono mt-1 tracking-widest uppercase">
            Neural Graph Intelligence Engine — 10 Layers Active
          </p>
        </div>

        {/* Tab switcher */}
        <div className="flex items-center gap-2 bg-white/5 rounded-xl p-1 border border-white/10">
          {['query', 'monitor'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-lg text-xs font-mono uppercase tracking-widest transition-all ${
                activeTab === tab
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                  : 'text-white/30 hover:text-white/60'
              }`}
            >
              {tab === 'query' ? '⬡ Query Engine' : '⬡ Observability'}
            </button>
          ))}
        </div>
      </header>

      {/* Query Engine Tab */}
      <AnimatePresence mode="wait">
        {activeTab === 'query' && (
          <motion.main
            key="query"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-8"
          >
            <QueryInput onSearch={handleSearch} loading={loading} />

            <AnimatePresence>
              {data && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="grid grid-cols-1 lg:grid-cols-3 gap-8"
                >
                  {/* Left — Graph + Answer + Timeline */}
                  <div className="lg:col-span-2 space-y-8">
                    <GraphViewer nodes={nodes} />

                    <div className="glass-card border-white/5 relative overflow-hidden">
                      <div className="absolute top-0 right-0 p-4 opacity-5 text-9xl font-black select-none">AI</div>
                      <h3 className="text-sm font-mono text-cyan-500 mb-2">SYNTHESIZED_RESPONSE</h3>
                      <p className="text-xl text-white leading-relaxed font-light">{data.answer}</p>
                    </div>

                    <TimelineView events={data.timeline} />
                  </div>

                  {/* Right — Confidence + Reasoning */}
                  <div className="space-y-8">
                    <ConfidenceMeter confidence={data.confidence} />
                    <ReasoningPanel steps={data.steps} />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.main>
        )}

        {/* Observability Tab — Layer 8-10 */}
        {activeTab === 'monitor' && (
          <motion.div
            key="monitor"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="grid grid-cols-1 lg:grid-cols-3 gap-8"
          >
            {/* Full-width monitoring panel */}
            <div className="lg:col-span-2">
              <MonitoringPanel />
            </div>

            {/* Right column — quick tips */}
            <div className="space-y-4">
              <div className="glass-card border-cyan-500/10 space-y-4">
                <h3 className="text-sm font-mono text-cyan-500 uppercase tracking-widest">Stack Links</h3>
                {[
                  { label: 'Prometheus UI',   url: 'http://localhost:9090', color: 'text-orange-400' },
                  { label: 'Grafana Boards',  url: 'http://localhost:3000', color: 'text-yellow-400' },
                  { label: 'FastAPI Docs',    url: 'http://localhost:8000/docs', color: 'text-cyan-400' },
                  { label: 'Raw Metrics',     url: 'http://localhost:8000/api/v1/metrics/raw', color: 'text-green-400' },
                  { label: 'Health Check',    url: 'http://localhost:8000/api/v1/health', color: 'text-violet-400' },
                  { label: 'Node Registry',   url: 'http://localhost:8000/api/v1/nodes', color: 'text-pink-400' },
                ].map(({ label, url, color }) => (
                  <a key={url} href={url} target="_blank" rel="noreferrer"
                    className={`flex items-center justify-between p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-all group ${color}`}>
                    <span className="text-xs font-mono">{label}</span>
                    <span className="text-white/20 group-hover:text-white/60 text-xs">→</span>
                  </a>
                ))}
              </div>

              <div className="glass-card border-white/5 text-[10px] font-mono text-white/30 space-y-2 leading-relaxed">
                <div className="text-white/50 mb-2 uppercase tracking-widest">Layer Map</div>
                <div>L8 → Kafka / In-Process Queue</div>
                <div>L9 → Node Registry + K8s HPA</div>
                <div>L10 → Prometheus + Grafana + OTel</div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Background glows */}
      <div className="fixed inset-0 -z-10 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-cyan-500/5 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-violet-500/5 blur-[120px] rounded-full" />
      </div>
    </div>
  );
}

export default App;
