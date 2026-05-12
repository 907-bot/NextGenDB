import React from 'react';
import { motion } from 'framer-motion';

const ConfidenceMeter = ({ confidence = 0 }) => {
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (confidence * circumference);

  return (
    <div className="glass-card flex flex-col items-center justify-center gap-4 w-full md:w-64 border-cyan-500/10">
      <div className="relative w-32 h-32 flex items-center justify-center">
        <svg className="w-full h-full -rotate-90">
          <circle
            cx="64" cy="64" r={radius}
            fill="transparent"
            stroke="rgba(255, 255, 255, 0.05)"
            strokeWidth="8"
          />
          <motion.circle
            cx="64" cy="64" r={radius}
            fill="transparent"
            stroke="#00f2ff"
            strokeWidth="8"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.5, ease: "easeOut" }}
            strokeLinecap="round"
            className="filter drop-shadow-[0_0_8px_rgba(0,242,255,0.5)]"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-black text-white">
            {Math.round(confidence * 100)}%
          </span>
          <span className="text-[10px] text-cyan-500/60 font-mono -mt-1">CONFIDENCE</span>
        </div>
      </div>
      <div className="text-center">
        <div className="text-[10px] text-white/40 uppercase tracking-[0.2em] mb-1">Status</div>
        <div className="text-xs font-bold text-cyan-400">NEURAL_STABLE_FIX</div>
      </div>
    </div>
  );
};

export default ConfidenceMeter;
