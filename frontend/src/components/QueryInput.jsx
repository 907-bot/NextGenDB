import React, { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

const QueryInput = ({ onSearch, loading }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) onSearch(query);
  };

  return (
    <motion.div 
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="w-full max-w-4xl mx-auto mt-8 flex flex-col gap-4"
    >
      <form onSubmit={handleSubmit} className="relative">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Neural Query: ask anything about the graph topology..."
          className="w-full bg-black/40 border border-cyan-500/30 rounded-2xl py-4 px-12 text-lg text-cyan-50 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-500/20 transition-all backdrop-blur-xl"
        />
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-cyan-500/60 w-5 h-5" />
        <button 
          type="submit"
          className="absolute right-3 top-1/2 -translate-y-1/2 bg-cyan-500 hover:bg-cyan-400 text-black px-6 py-2 rounded-xl font-bold transition-colors disabled:opacity-50"
          disabled={loading}
        >
          {loading ? <Loader2 className="animate-spin w-5 h-5" /> : 'EXECUTE'}
        </button>
      </form>
    </motion.div>
  );
};

export default QueryInput;
