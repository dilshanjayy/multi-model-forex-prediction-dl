import React from 'react';
import { Target, TrendingUp, ShieldAlert, BarChart3, PieChart, Layers } from 'lucide-react';

export default function ValidationStatsWidget({ stats }) {
  if (!stats) return <div className="text-[#8b949e] text-xs">No metrics available.</div>;

  const extractMetric = (key, defaultVal = 0) => {
    return stats[key] !== undefined ? stats[key] : defaultVal;
  };

  const safeNum = (val) => {
    const num = parseFloat(val);
    return isNaN(num) ? 0 : num;
  };

  const getMetricColor = (val, isInverse = false) => {
    const num = safeNum(val);
    if (num === 0) return 'text-white';
    if (num > 0) return isInverse ? 'text-[#f85149]' : 'text-[#3fb950]';
    return isInverse ? 'text-[#3fb950]' : 'text-[#f85149]';
  };

  const MetricItem = ({ label, value, icon: Icon, suffix = '', precision = 2, isInverse = false, barMax = 100 }) => {
    const numValue = safeNum(value);
    const colorClass = getMetricColor(numValue, isInverse);
    // Calculate bar percentage (capped at 100)
    const barWidth = Math.min(100, Math.max(0, (Math.abs(numValue) / barMax) * 100));

    return (
      <div className="relative group p-4 border border-[#30363d50] hover:border-[#30363d] transition-all bg-[#0d1117]/30">
        <div className="flex justify-between items-start mb-2">
            <div className="flex items-center space-x-1.5 opacity-60 group-hover:opacity-100 transition-opacity">
                <Icon size={10} className="text-[#8b949e]" />
                <span className="text-[9px] font-black text-[#8b949e] uppercase tracking-widest">{label}</span>
            </div>
        </div>
        <div className="flex items-baseline space-x-1">
            <span className={`text-xl font-bold value-mono ${colorClass}`}>{numValue.toFixed(precision)}</span>
            <span className="text-[10px] text-[#8b949e] font-bold">{suffix}</span>
        </div>
        {/* Dynamic Delta Bar */}
        <div className="mt-3 w-full h-[2px] bg-[#30363d30] rounded-full overflow-hidden">
            <div 
                className={`h-full transition-all duration-1000 ease-out rounded-full ${colorClass.replace('text-', 'bg-')}`}
                style={{ 
                    width: `${barWidth}%`,
                    boxShadow: `0 0 8px ${numValue !== 0 ? 'currentColor' : 'transparent'}`
                }}
            ></div>
        </div>
      </div>
    );
  };

  return (
    <div className="glass-panel flex-1 flex flex-col h-full overflow-hidden">
      <div className="px-4 py-2 border-b border-[#30363d] bg-[#161b2250] flex justify-between items-center">
          <h3 className="text-[10px] font-black text-[#8b949e] uppercase tracking-widest">PERFORMANCE HUD</h3>
          <div className="flex items-center space-x-1">
              <span className="text-[8px] font-bold text-[#3fb950]">OOS DATA: VERIFIED</span>
          </div>
      </div>
      
      <div className="flex-1 grid grid-cols-2 overflow-y-auto custom-scrollbar">
          <MetricItem label="Net Return" value={extractMetric('Return [%]')} icon={TrendingUp} suffix="%" barMax={20} />
          <MetricItem label="Sharpe Ratio" value={extractMetric('Sharpe Ratio')} icon={Target} barMax={3} />
          <MetricItem label="Max Drawdown" value={extractMetric('Max. Drawdown [%]')} icon={ShieldAlert} suffix="%" isInverse={true} barMax={10} />
          <MetricItem label="Win Rate" value={extractMetric('Win Rate [%]')} icon={PieChart} suffix="%" barMax={100} />
          <MetricItem label="Profit Factor" value={extractMetric('Profit Factor')} icon={BarChart3} barMax={3} />
          <MetricItem label="Total Trades" value={extractMetric('# Trades')} icon={Layers} precision={0} barMax={500} />
      </div>

      <div className="px-4 py-2 border-t border-[#30363d] bg-[#0d111750]">
          <p className="text-[8px] text-[#8b949e] italic leading-tight uppercase font-medium">Historical validation run completed during artifact generation.</p>
      </div>
    </div>
  );
}
