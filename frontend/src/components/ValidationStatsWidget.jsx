import React from 'react';

export default function ValidationStatsWidget({ stats }) {
  if (!stats) return <div className="text-[#8b949e] text-xs">No metrics available.</div>;

  const extractMetric = (key, defaultVal = '-') => {
    return stats[key] !== undefined ? stats[key] : defaultVal;
  };

  const safeNum = (val) => {
    const num = parseFloat(val);
    return isNaN(num) ? null : num;
  };

  const getMetricColor = (val, isInverse = false) => {
    const num = safeNum(val);
    if (num === null) return 'text-white';
    if (num > 0) return isInverse ? 'text-[#f85149]' : 'text-[#3fb950]';
    if (num < 0) return isInverse ? 'text-[#3fb950]' : 'text-[#f85149]';
    return 'text-white';
  };

  return (
    <div className="metric-card flex-1 flex flex-col h-full">
      <h3 className="metric-label mb-4">Static Validation Metrics</h3>
      <p className="text-xs text-[#8b949e] mb-4 border-b border-[#30363d] pb-2">
        Performance on out-of-sample data during training phase.
      </p>

      <div className="grid grid-cols-2 gap-4 flex-1">
        {/* Metric Block */}
        <div className="bg-[#1c2128] p-3 rounded">
          <p className="text-[10px] text-[#8b949e] uppercase tracking-wider">Net Return</p>
          <p className={`text-lg font-bold ${getMetricColor(extractMetric('Return [%]'))}`}>
            {safeNum(extractMetric('Return [%]'))?.toFixed(2)}%
          </p>
        </div>

        <div className="bg-[#1c2128] p-3 rounded">
          <p className="text-[10px] text-[#8b949e] uppercase tracking-wider">Sharpe Ratio</p>
          <p className={`text-lg font-bold text-white`}>
            {safeNum(extractMetric('Sharpe Ratio'))?.toFixed(2)}
          </p>
        </div>

        <div className="bg-[#1c2128] p-3 rounded">
          <p className="text-[10px] text-[#8b949e] uppercase tracking-wider">Max Drawdown</p>
          <p className={`text-lg font-bold ${getMetricColor(extractMetric('Max. Drawdown [%]'), true)}`}>
            {safeNum(extractMetric('Max. Drawdown [%]'))?.toFixed(2)}%
          </p>
        </div>

        <div className="bg-[#1c2128] p-3 rounded">
          <p className="text-[10px] text-[#8b949e] uppercase tracking-wider">Win Rate</p>
          <p className={`text-lg font-bold text-white`}>
            {safeNum(extractMetric('Win Rate [%]'))?.toFixed(1)}%
          </p>
        </div>

        <div className="bg-[#1c2128] p-3 rounded">
          <p className="text-[10px] text-[#8b949e] uppercase tracking-wider">Profit Factor</p>
          <p className={`text-lg font-bold ${getMetricColor(safeNum(extractMetric('Profit Factor')) - 1)}`}>
            {safeNum(extractMetric('Profit Factor'))?.toFixed(2)}
          </p>
        </div>

        <div className="bg-[#1c2128] p-3 rounded">
          <p className="text-[10px] text-[#8b949e] uppercase tracking-wider">Total Trades</p>
          <p className={`text-lg font-bold text-[#58a6ff]`}>
            {extractMetric('# Trades')}
          </p>
        </div>
      </div>
    </div>
  );
}
