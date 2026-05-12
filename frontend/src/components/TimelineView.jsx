import React from 'react';
import { motion } from 'framer-motion';
import { Clock } from 'lucide-react';

const TimelineView = ({ events = [] }) => {
  return (
    <div className="glass-card w-full border-white/5">
      <div className="flex items-center gap-2 mb-6">
        <Clock className="text-cyan-500 w-4 h-4" />
        <h3 className="text-sm font-bold tracking-widest text-white/60">TEMPORAL FLUX</h3>
      </div>
      
      <div className="flex gap-4 overflow-x-auto pb-4 scrollbar-hide">
        {events.map((event, idx) => (
          <motion.div 
            key={idx}
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: idx * 0.1 }}
            className="flex-shrink-0 w-64 p-4 bg-white/5 rounded-xl border border-white/10 hover:border-cyan-500/30 transition-all hover:bg-white/[0.07]"
          >
            <div className="text-[10px] text-cyan-500 font-mono mb-2">T_{idx * 100}ms</div>
            <div className="text-sm font-medium text-white mb-1">{event.event}</div>
            <div className="text-[11px] text-white/40">{new Date(event.timestamp).toLocaleTimeString()}</div>
          </motion.div>
        ))}
        {events.length === 0 && (
          <div className="w-full text-center py-4 text-white/10 text-xs font-mono uppercase tracking-widest">
            History dormant
          </div>
        )}
      </div>
    </div>
  );
};

export default TimelineView;
