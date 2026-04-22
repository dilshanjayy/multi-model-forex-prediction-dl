import { useState, useRef, useEffect } from 'react';
import { useStore } from '../store';
import { createChart } from 'lightweight-charts';

export default function BacktestLab() {
  const selectedModel = useStore(state => state.selectedModel);
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState('2025-01-01');
  const [strategy, setStrategy] = useState('TripleBarrier');
  const [atrMult, setAtrMult] = useState(3.0);
  const [confThreshold, setConfThreshold] = useState(0.40);
  
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  
  const chartContainerRef = useRef();
  const chartRef = useRef(null);
  const seriesRef = useRef(null);

  const runBacktest = async () => {
    if (!selectedModel) return;
    
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
    } catch (err) {
      console.error(err);
      alert("Backtest failed.");
    }
    setLoading(false);
  };

  useEffect(() => {
    if (results && results.equity_curve && chartContainerRef.current) {
      if (!chartRef.current) {
        chartRef.current = createChart(chartContainerRef.current, {
          layout: { background: { type: 'solid', color: '#06090f' }, textColor: '#8b949e' },
          grid: { vertLines: { visible: false }, horzLines: { color: '#1c2128' } },
          timeScale: { borderColor: '#30363d' },
          rightPriceScale: { borderColor: '#30363d' },
          height: 250,
        });
        seriesRef.current = chartRef.current.addLineSeries({
          color: '#58a6ff', lineWidth: 2,
        });
      }
      
      // Format data for TV lightweight charts
      // The backend returns time as string 'YYYY-MM-DD HH:MM:SS'
      // Lightweight charts needs 'YYYY-MM-DD' or timestamp
      const formattedData = results.equity_curve.map(p => ({
        time: new Date(p.time).getTime() / 1000,
        value: p.value
      }));
      
      seriesRef.current.setData(formattedData);
      chartRef.current.timeScale().fitContent();
    }
  }, [results]);

  return (
    <div className="grid grid-cols-3 gap-6 h-full">
      <div className="metric-card col-span-1 flex flex-col">
        <h3 className="metric-label mb-4">Dynamic Backtest Laboratory</h3>
        <p className="text-sm text-[#8b949e] mb-4">Run custom simulations on historical data instantly.</p>
        
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="text-xs text-white">Start Date</label>
            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
          </div>
          <div>
            <label className="text-xs text-white">End Date</label>
            <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} />
          </div>
        </div>
        
        <div className="mb-4">
          <label className="text-xs text-white">Strategy</label>
          <select value={strategy} onChange={e => setStrategy(e.target.value)}>
            <option value="TripleBarrier">TripleBarrier</option>
            <option value="MajorityVote">MajorityVote</option>
            <option value="NaiveFlip">NaiveFlip</option>
          </select>
        </div>
        
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div>
            <label className="text-xs text-white">Exit ATR Multiplier</label>
            <input type="number" step="0.5" value={atrMult} onChange={e => setAtrMult(e.target.value)} />
          </div>
          <div>
            <label className="text-xs text-white">Confidence Threshold</label>
            <input type="number" step="0.05" value={confThreshold} onChange={e => setConfThreshold(e.target.value)} />
          </div>
        </div>
        
        <button 
          onClick={runBacktest}
          disabled={loading || !selectedModel}
          className="w-full bg-[#58a6ff] text-[#06090f] font-bold py-3 rounded mt-auto hover:bg-blue-400 transition-colors disabled:opacity-50"
        >
          {loading ? "CALCULATING..." : "RUN BACKTEST"}
        </button>
      </div>

      <div className="metric-card col-span-2 flex flex-col">
        <h3 className="metric-label mb-4">Simulation Results</h3>
        
        <div className="flex-1 min-h-[250px] bg-[#06090f] rounded border border-[#30363d] mb-4 relative">
            {!results && <div className="absolute inset-0 flex items-center justify-center text-[#8b949e]">Equity Curve Chart Will Render Here</div>}
            <div ref={chartContainerRef} className="w-full h-full" />
        </div>
        
        {results && results.metrics && (
          <div className="grid grid-cols-5 gap-4 border-t border-[#30363d] pt-4">
            <div>
              <p className="text-xs text-[#8b949e]">Profit Factor</p>
              <p className="text-lg font-bold text-white">{results.metrics["Profit Factor"].toFixed(2)}</p>
            </div>
            <div>
              <p className="text-xs text-[#8b949e]">Win Rate</p>
              <p className="text-lg font-bold text-white">{results.metrics["Win Rate [%]"].toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-xs text-[#8b949e]">Max Drawdown</p>
              <p className="text-lg font-bold text-[#f85149]">{results.metrics["Max Drawdown [%]"].toFixed(2)}%</p>
            </div>
            <div>
              <p className="text-xs text-[#8b949e]">Sharpe Ratio</p>
              <p className="text-lg font-bold text-white">{results.metrics["Sharpe Ratio"].toFixed(2)}</p>
            </div>
            <div>
              <p className="text-xs text-[#8b949e]">Total Trades</p>
              <p className="text-lg font-bold text-white">{results.metrics["Total Trades"]}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
