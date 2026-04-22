import React from 'react';

export default function ModelConfigWidget({ config }) {
  if (!config) return <div className="text-[#8b949e] text-xs">No configuration available.</div>;

  // Helper to safely format values
  const formatValue = (val) => {
    if (typeof val === 'boolean') return val ? 'True' : 'False';
    if (typeof val === 'object' && val !== null) return JSON.stringify(val);
    return val;
  };

  // Extract key sections if they exist, otherwise just list top-level keys
  const modelParams = config.model || config.hyperparameters || {};
  const dataParams = config.data || {};
  const trainParams = config.training || {};

  const renderSection = (title, data) => {
    if (!data || Object.keys(data).length === 0) return null;
    return (
      <div className="mb-4">
        <h4 className="text-xs font-bold text-white uppercase mb-2 border-b border-[#30363d] pb-1">{title}</h4>
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          {Object.entries(data).map(([key, value]) => (
            <div key={key} className="flex justify-between items-center text-xs">
              <span className="text-[#8b949e] capitalize truncate max-w-[100px]" title={key}>{key.replace(/_/g, ' ')}</span>
              <span className="text-white text-right truncate max-w-[120px]" title={formatValue(value)}>{formatValue(value)}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="metric-card flex-1 flex flex-col h-full overflow-hidden">
      <h3 className="metric-label mb-4">Model Architecture</h3>
      <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
        {Object.keys(modelParams).length > 0 || Object.keys(dataParams).length > 0 ? (
          <>
            {renderSection("Algorithm Parameters", modelParams)}
            {renderSection("Data Processing", dataParams)}
            {renderSection("Training Setup", trainParams)}
          </>
        ) : (
          // Fallback if structure is flat
          renderSection("General Parameters", config)
        )}
      </div>
    </div>
  );
}
