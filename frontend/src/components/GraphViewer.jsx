import React from 'react';
import { motion } from 'framer-motion';

const GraphViewer = ({ nodes = [] }) => {
  // Simple circular layout for nodes
  const width = 800;
  const height = 400;
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) / 3;

  const getCoords = (index, total) => {
    if (total === 1) return { x: centerX, y: centerY };
    const angle = (index / total) * 2 * Math.PI;
    return {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle)
    };
  };

  return (
    <div className="glass-card h-[450px] relative overflow-hidden flex flex-col border-cyan-500/20">
      <div className="p-4 border-b border-white/5 flex justify-between items-center bg-white/5">
        <div className="text-[10px] font-mono text-cyan-500 uppercase tracking-widest">
          Neural_Topology_Visualizer v2.0
        </div>
        <div className="text-[10px] font-mono text-white/40">
          NODES_ACTIVE: {nodes.length}
        </div>
      </div>
      
      <div className="flex-1 relative">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,242,255,0.03),transparent)] pointer-events-none" />
        
        <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`}>
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>

          {/* Render Connections (simple all-to-center or sequential for now) */}
          {nodes.map((node, i) => {
             const from = getCoords(i, nodes.length);
             const to = getCoords((i + 1) % nodes.length, nodes.length);
             return (
               <line 
                 key={`edge-${i}`}
                 x1={from.x} y1={from.y} 
                 x2={to.x} y2={to.y} 
                 stroke="rgba(0,242,255,0.1)" 
                 strokeWidth="1" 
               />
             );
          })}
          
          {/* Render Nodes */}
          {nodes.map((node, i) => {
            const { x, y } = getCoords(i, nodes.length);
            const isEvent = node.type === 'EVENT';
            
            return (
              <g key={node.id || i}>
                <motion.circle
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  cx={x} cy={y} r={isEvent ? 8 : 6}
                  fill={isEvent ? "#ff00e5" : "#00f2ff"}
                  filter="url(#glow)"
                />
                <text 
                  x={x + 12} y={y + 4} 
                  fill="white" 
                  fontSize="10" 
                  fontFamily="monospace"
                  className="pointer-events-none select-none opacity-60"
                >
                  {node.label || node.id}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
};

export default GraphViewer;
