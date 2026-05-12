import React from 'react';
import { motion } from 'framer-motion';
import { Terminal, Activity } from 'lucide-react';

const ReasoningPanel = ({ steps = [] }) => {
  return (
    <div className="glass-card flex-1 min-w-[300px] border-magenta-500/10">
      <div className="flex items-center gap-2 mb-6 border-b border-white/5 pb-4">
        <Terminal className="text-magenta-500 w-5 h-5" />
        <h2 className="text-lg font-bold tracking-tight text-magenta-50">REASONING ENGINE</h2>
      </div>
      
      <div className="space-y-4">
        {steps.map((step, idx) => (
          <motion.div 
            key={idx}
            initial={{ x: -10, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: idx * 0.1 }}
            className="flex gap-4 group"
          >
            <div className="flex flex-col items-center">
              <div className="w-2 h-2 rounded-full bg-magenta-500 group-hover:scale-150 transition-transform" />
              <div className="w-[1px] flex-1 bg-white/10 my-1" />
            </div>
            <div className="pb-4">
              <div className="text-[10px] text-magenta-400/60 font-mono mb-1">
                STEP_0{idx + 1} | {step.action}
              </div>
              <p className="text-sm text-white/80 leading-relaxed italic">
                {step.description}
              </p>
            </div>
          </motion.div>
        ))}
        {steps.length === 0 && (
          <div className="text-white/20 text-sm font-mono animate-pulse">
            Waiting for neural input...
          </div>
        )}
      </div>
    </div>
  );
};

export default ReasoningPanel;
