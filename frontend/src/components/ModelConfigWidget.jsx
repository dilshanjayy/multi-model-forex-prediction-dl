import React, { useState } from 'react';
import { Cpu, Database, ChevronDown, ChevronRight, Settings2, Info } from 'lucide-react';

export default function ModelConfigWidget({ config }) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [openSections, setOpenSections] = useState({ core: true, advanced: false });

  if (!config) return <div className="text-[#8b949e] text-xs">No configuration available.</div>;

  // Strict Whitelist for "CLEAN" mode
  const whitelist = [
    'type', 'target', 'learning_rate', 'epochs', 'lookback', 'input_dim', 'dropout', 'batch_size'
  ];

  // Primary Specs for the very top (High-Signal)
  const primarySpecs = [
    { label: 'ARCH', value: config.model?.type?.split('-')[0] || 'DL' },
    { label: 'TARGET', value: config.model?.target?.split('_')[1] || 'N/A' },
    { label: 'WIN', value: config.model?.params?.lookback || config.data?.lookback || 'N/A' },
  ];

  const toggleSection = (id) => {
    setOpenSections(prev => ({ ...prev, [id]: !prev[id] }));
  };

  // Helper to extract flattened data
  const getAllData = () => {
    const flattened = {
      ...(config.model?.params || {}),
      ...(config.model || {}),
      ...(config.data || {}),
      ...(config.training || {}),
    };
    // Remove objects to keep display clean
    return Object.entries(flattened).filter(([_, v]) => typeof v !== 'object');
  };

  const allData = getAllData();
  const cleanData = allData.filter(([k]) => whitelist.includes(k));
  const advancedData = allData.filter(([k]) => !whitelist.includes(k));

  const renderDataGrid = (data) => (
    <div className="grid grid-cols-2 gap-x-6 gap-y-4">
      {data.map(([key, value]) => (
        <div key={key} className="flex flex-col space-y-1">
          <span className="text-[10px] text-[#8b949e] uppercase font-black tracking-widest opacity-60">
            {key.replace(/_/g, ' ')}
          </span>
          <span className="text-xs text-white value-mono font-bold">
            {String(value).toUpperCase()}
          </span>
        </div>
      ))}
    </div>
  );

  return (
    <div className="glass-panel flex-1 flex flex-col h-full overflow-hidden">
      <div className="px-4 py-3 border-b border-[#30363d] bg-[#161b2250] flex justify-between items-center">
          <h3 className="text-xs font-black text-[#8b949e] uppercase tracking-widest">SYSTEMS SPEC-SHEET</h3>
          <button 
            onClick={() => setShowAdvanced(!showAdvanced)}
            className={`flex items-center space-x-1 px-2 py-1 rounded-[2px] transition-all ${showAdvanced ? 'bg-[#58a6ff20] border border-[#58a6ff40] text-[#58a6ff]' : 'bg-[#30363d40] border border-[#30363d] text-[#8b949e]'}`}
          >
            <Settings2 size={12} />
            <span className="text-[9px] font-black uppercase">{showAdvanced ? 'FULL' : 'SIMPLE'}</span>
          </button>
      </div>
      
      {/* High-Level Summary Bar */}
      <div className="flex divide-x divide-[#30363d50] border-b border-[#30363d50] bg-[#0d111750]">
          {primarySpecs.map((spec, idx) => (
            <div key={idx} className="flex-1 px-4 py-4 text-center">
                <p className="text-[9px] font-black text-[#8b949e] mb-1 uppercase tracking-[0.2em]">{spec.label}</p>
                <p className="text-sm font-bold text-[#58a6ff] value-mono">{spec.value}</p>
            </div>
          ))}
      </div>

      <div className="flex-1 p-6 overflow-y-auto custom-scrollbar space-y-8">
        {/* Core Architecture Section */}
        <div>
            <div className="flex items-center space-x-2 mb-5">
                <Cpu size={14} className="text-[#3fb950] opacity-70" />
                <span className="text-xs font-black text-white uppercase tracking-[0.1em]">Core Parameters</span>
            </div>
            {renderDataGrid(cleanData)}
        </div>

        {/* Advanced Section (Conditional) */}
        {showAdvanced && (
          <div className="pt-6 border-t border-[#30363d] animate-in fade-in slide-in-from-top-2 duration-300">
             <div className="flex items-center space-x-2 mb-5">
                <Database size={14} className="text-[#8b949e]" />
                <span className="text-xs font-black text-[#8b949e] uppercase tracking-[0.1em]">Environment & Meta</span>
            </div>
            {renderDataGrid(advancedData)}
          </div>
        )}
      </div>

      <div className="px-4 py-3 border-t border-[#30363d] bg-[#0d111750] flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <div className="w-1.5 h-1.5 bg-[#3fb950] rounded-full animate-pulse"></div>
            <p className="text-[9px] text-[#8b949e] font-black uppercase tracking-widest">Weights Verified</p>
          </div>
          <span className="text-[9px] font-mono text-[#8b949e] opacity-40">{config.project?.name?.toUpperCase() || 'V1.0'}</span>
      </div>
    </div>
  );
}
