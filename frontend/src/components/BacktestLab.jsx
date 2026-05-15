import { useState, useRef, useEffect } from 'react';
import { useStore } from '../store';
import { createChart } from 'lightweight-charts';
import toast from 'react-hot-toast';
import { Calendar, Sliders, Target, ShieldAlert, TrendingUp, BarChart3, PieChart, Layers, FlaskConical, Play } from 'lucide-react';

export default function BacktestLab() {
  const selectedModel = useStore(state => state.selectedModel);
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState('2025-01-01');
  const [strategy, setStrategy] = useState('ContinuousSignalExecution');
  const [atrMult, setAtrMult] = useState(3.0);
  const [confThreshold, setConfThreshold] = useState(0.5);
  
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  
  const chartContainerRef = useRef();
  const chartRef = useRef(null);
  const seriesRef = useRef(null);

  const runBacktest = async () => {
    if (!selectedModel) return;
    
    const toastId = toast.loading("Processing historical data...");
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:8000/api/v1/backtest/${selectedModel}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_date: startDate,
          end_date: endDate,
          strategy: strategy,
          exit_atr_multiplier: parseFloat(atrMult),
          conf_threshold: parseFloat(confThreshold)
        })
      });
      
      const data = await res.json();
      setResults(data);
      toast.success("Simulation Complete", { id: toastId });
    } catch (err) {
      console.error(err);
      toast.error("Backtest failed. Check logs.", { id: toastId });
    }
    setLoading(false);
  };

  useEffect(() => {
    if (results && results.equity_curve && chartContainerRef.current) {
      if (!chartRef.current) {
        chartRef.current = createChart(chartContainerRef.current, {
          layout: { background: { type: 'solid', color: '#06090f' }, textColor: '#8b949e' },
          grid: { vertLines: { visible: false }, horzLines: { color: '#1c2128' } },
          timeScale: { borderColor: '#30363d', timeVisible: true, rightOffset: 12 },
          rightPriceScale: { borderColor: '#30363d' },
          autoSize: true,
        });
        seriesRef.current = chartRef.current.addLineSeries({
          color: '#58a6ff', lineWidth: 2,
        });
      }
      
      const formattedData = results.equity_curve.map(p => ({
        time: new Date(p.time).getTime() / 1000,
        value: p.value
      }));
      
      seriesRef.current.setData(formattedData);
      chartRef.current.timeScale().fitContent();
    }
  }, [results]);

  const extractMetric = (key, defaultVal = 0) => {
    return results?.metrics[key] !== undefined ? results.metrics[key] : defaultVal;
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
    const barWidth = Math.min(100, Math.max(0, (Math.abs(numValue) / barMax) * 100));

    return (
      <div className="relative group p-4 border border-[#30363d50] hover:border-[#30363d] transition-all bg-[#0d1117]/30 rounded">
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

  const inputClass = "w-full bg-[#0d1117] border border-[#30363d] text-white font-mono text-sm py-2 px-3 rounded focus:outline-none focus:border-[#58a6ff] transition-colors";

  return (
    <div className="grid grid-cols-12 gap-6 h-full">
      {/* COMMAND CENTER (LEFT) */}
      <div className="glass-panel col-span-4 flex flex-col overflow-hidden">
        <div className="px-4 py-3 border-b border-[#30363d] bg-[#161b2250] flex justify-between items-center">
            <h3 className="text-xs font-black text-[#8b949e] uppercase tracking-widest">SIMULATION CONTROLS</h3>
            <FlaskConical size={14} className="text-[#58a6ff]" />
        </div>
        
        <div className="p-6 flex-1 overflow-y-auto custom-scrollbar space-y-6">
          {/* Timeframe Module */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2 border-b border-[#30363d] pb-2">
                <Calendar size={14} className="text-[#3fb950] opacity-70" />
                <span className="text-[10px] font-black text-white uppercase tracking-[0.1em]">Timeframe</span>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label-muted mb-1 block">START DATE</label>
                <input type="date" className={inputClass} value={startDate} onChange={e => setStartDate(e.target.value)} />
              </div>
              <div>
                <label className="label-muted mb-1 block">END DATE</label>
                <input type="date" className={inputClass} value={endDate} onChange={e => setEndDate(e.target.value)} />
              </div>
            </div>
          </div>
          
          {/* Execution Logic Module */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2 border-b border-[#30363d] pb-2">
                <Sliders size={14} className="text-[#58a6ff] opacity-70" />
                <span className="text-[10px] font-black text-white uppercase tracking-[0.1em]">Execution Logic</span>
            </div>
            <div>
              <label className="label-muted mb-1 block">STRATEGY ENGINE</label>
              <select className={inputClass} value={strategy} onChange={e => setStrategy(e.target.value)}>
                <option value="TripleBarrier">TripleBarrier (ATR TP/SL)</option>
                <option value="MajorityVote">MajorityVote (Hold Trend)</option>
                <option value="ContinuousSignalExecution">ContinuousSignalExecution (Instant React)</option>
              </select>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label-muted mb-1 block">ATR EXIT MULTIPLIER</label>
                <input type="number" step="0.5" className={inputClass} value={atrMult} onChange={e => setAtrMult(e.target.value)} />
              </div>
              <div>
                <label className="label-muted mb-1 block">CONF THRESHOLD</label>
                <input type="number" step="0.05" className={inputClass} value={confThreshold} onChange={e => setConfThreshold(e.target.value)} />
              </div>
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-[#30363d] bg-[#0d111750]">
          <button 
            onClick={runBacktest}
            disabled={loading || !selectedModel}
            className="w-full bg-[#58a6ff] text-[#06090f] font-bold py-3 rounded flex items-center justify-center space-x-2 hover:bg-[#79b8ff] transition-all disabled:opacity-50 disabled:grayscale active:scale-[0.98]"
          >
            {loading ? (
              <span className="tracking-[0.2em] uppercase animate-pulse">Calculating...</span>
            ) : (
              <>
                <Play size={16} fill="currentColor" />
                <span className="tracking-widest uppercase">Run Simulation</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* RESULTS HUD (RIGHT) */}
      <div className="glass-panel col-span-8 flex flex-col overflow-hidden">
        <div className="px-4 py-3 border-b border-[#30363d] bg-[#161b2250] flex justify-between items-center">
            <h3 className="text-xs font-black text-[#8b949e] uppercase tracking-widest">PERFORMANCE HUD</h3>
            <Target size={14} className="text-[#3fb950]" />
        </div>
        
        <div className="flex-1 flex flex-col p-6 overflow-y-auto custom-scrollbar">
          {/* Equity Curve Chart */}
          <div className="h-[300px] bg-[#06090f] border border-[#30363d] rounded mb-6 relative overflow-hidden flex-shrink-0">
              {!results && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-[#8b949e] space-y-2">
                  <BarChart3 size={32} className="opacity-20" />
                  <p className="text-xs uppercase tracking-widest opacity-50">Equity Curve Awaiting Data</p>
                </div>
              )}
              <div ref={chartContainerRef} className="absolute inset-0" />
          </div>
          
          {/* Metrics Grid */}
          <div className="grid grid-cols-3 gap-4">
              <MetricItem label="Net Return" value={extractMetric('Return [%]')} icon={TrendingUp} suffix="%" barMax={20} />
              <MetricItem label="Sharpe Ratio" value={extractMetric('Sharpe Ratio')} icon={Target} barMax={3} />
              <MetricItem label="Max Drawdown" value={extractMetric('Max. Drawdown [%]')} icon={ShieldAlert} suffix="%" isInverse={true} barMax={10} />
              <MetricItem label="Win Rate" value={extractMetric('Win Rate [%]')} icon={PieChart} suffix="%" barMax={100} />
              <MetricItem label="Profit Factor" value={extractMetric('Profit Factor')} icon={BarChart3} barMax={3} />
              <MetricItem label="Total Trades" value={extractMetric('# Trades')} icon={Layers} precision={0} barMax={500} />
          </div>
        </div>
      </div>
    </div>
  );
}
