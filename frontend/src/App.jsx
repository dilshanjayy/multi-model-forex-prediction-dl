import { useEffect, useState } from 'react';
import { useStore } from './store';
import ChartWidget from './components/ChartWidget';
import ExplainabilityWidget from './components/ExplainabilityWidget';
import BacktestLab from './components/BacktestLab';
import ModelConfigWidget from './components/ModelConfigWidget';
import ValidationStatsWidget from './components/ValidationStatsWidget';
import SignalRadar from './components/SignalRadar';
import OrderTicket from './components/OrderTicket';

export default function App() {
  const { models, setModels, selectedModel, setSelectedModel, modelDetails, setModelDetails, liveData, setLiveData, activeTab, setActiveTab } = useStore();
  const [loading, setLoading] = useState(false);

  // Fetch models on mount
  useEffect(() => {
    fetch('http://localhost:8000/api/v1/models')
      .then(res => res.json())
      .then(data => {
        setModels(data.models || []);
        if (data.models && data.models.length > 0) {
          setSelectedModel(data.models[0]);
        }
      })
      .catch(err => console.error("Error fetching models:", err));
  }, []);

  // Fetch details and live prediction when selectedModel changes
  useEffect(() => {
    if (!selectedModel) return;
    
    // Fetch details
    fetch(`http://localhost:8000/api/v1/models/${selectedModel}`)
      .then(res => res.json())
      .then(data => setModelDetails(data))
      .catch(err => console.error("Error fetching details:", err));

    // Initial fetch for live data
    fetchPrediction();
    
    // Set up polling (to simulate WebSocket until Phase 2)
    const interval = setInterval(fetchPrediction, 2000);
    return () => clearInterval(interval);

  }, [selectedModel]);

  const fetchPrediction = async () => {
    if (!selectedModel) return;
    try {
      const res = await fetch(`http://localhost:8000/api/v1/predict/${selectedModel}`, { method: 'POST' });
      const data = await res.json();
      setLiveData(data);
    } catch (err) {
      console.error("Error fetching live prediction:", err);
    }
  };

  const handleTrade = async (direction) => {
    if (!liveData) return;
    setLoading(true);
    try {
       const res = await fetch(`http://localhost:8000/api/v1/trade`, {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({
            symbol: "EURUSD",
            lot_size: useStore.getState().lotSize,
            direction: direction,
            atr: liveData.atr,
            multiplier: liveData.atr_multiplier
         })
       });
       const data = await res.json();
       if (data.status === 'success') {
         alert(`Filled ${direction} @ ${data.price}`);
       } else {
         alert(`Error: ${data.message}`);
       }
    } catch(err) {
      console.error(err);
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#06090f]">
      {/* GLOBAL HUD / HEADER */}
      <header className="h-12 border-b border-[#30363d] bg-[#0d1117] flex items-center px-4 justify-between shrink-0">
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2">
             <div className="w-2 h-2 bg-[#3fb950] rounded-full animate-pulse shadow-[0_0_8px_#3fb950]"></div>
             <h1 className="text-sm font-black tracking-tighter text-white">FALCON.OS</h1>
          </div>
          
          <div className="h-6 w-px bg-[#30363d]"></div>
          
          <div className="flex items-center space-x-3">
             <span className="label-muted !text-[9px]">ACTIVE INSTANCE</span>
             <select 
               className="!bg-transparent !border-none !p-0 !text-xs !font-bold !text-[#58a6ff] !w-auto cursor-pointer" 
               value={selectedModel || ''} 
               onChange={e => setSelectedModel(e.target.value)}
            >
               {models.map(m => <option key={m} value={m} className="bg-[#0d1117]">{m}</option>)}
            </select>
          </div>
        </div>

        <nav className="flex space-x-1">
            {[
              { id: 'terminal', label: 'LIVE TERMINAL' },
              { id: 'explain', label: 'MODEL X-RAY' },
              { id: 'lab', label: 'BACKTEST LAB' }
            ].map(tab => (
              <button 
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-1.5 text-[10px] font-bold tracking-widest transition-all rounded ${activeTab === tab.id ? 'bg-[#58a6ff] text-[#06090f]' : 'text-[#8b949e] hover:text-white'}`}
              >
                {tab.label}
              </button>
            ))}
        </nav>

        <div className="flex items-center space-x-4">
             <div className="text-right">
                <p className="label-muted !text-[8px] leading-none mb-0.5">EST. LATENCY</p>
                <p className="value-mono text-[10px] text-[#3fb950] leading-none">12ms</p>
             </div>
             <div className="h-6 w-px bg-[#30363d]"></div>
             <button className="p-1.5 text-[#8b949e] hover:text-white transition-colors">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
             </button>
        </div>
      </header>

      {/* MAIN VIEWPORT */}
      <main className="flex-1 min-h-0">
        {activeTab === 'terminal' && (
           <div className="h-full grid grid-cols-12">
              {/* Left Analytics Pane */}
              <div className="col-span-2 border-r border-[#30363d] flex flex-col min-h-0 overflow-hidden">
                 <SignalRadar liveData={liveData} />
                 <div className="pane-header !border-t-0">TOP ALPHA DRIVERS</div>
                 <div className="flex-1 min-h-0 bg-[#0d1117] p-3">
                    <ExplainabilityWidget runId={selectedModel} />
                 </div>
              </div>

              {/* Center Chart Pane */}
              <div className="col-span-7 flex flex-col min-h-0 overflow-hidden bg-black">
                 <div className="h-8 border-b border-[#30363d] bg-[#0d1117] flex items-center px-4 space-x-6">
                    <div className="flex items-center space-x-2">
                       <span className="text-[10px] font-bold text-white uppercase tracking-widest">EURUSD</span>
                       <span className="text-[9px] text-[#8b949e]">H1</span>
                    </div>
                    <div className="h-3 w-px bg-[#30363d]"></div>
                    <div className="flex space-x-4">
                       <div className="flex space-x-1.5">
                          <span className="label-muted !text-[9px]">L:</span>
                          <span className="value-mono text-[9px] text-[#f85149]">{liveData?.chart?.candles?.at(-1)?.low?.toFixed(5)}</span>
                       </div>
                       <div className="flex space-x-1.5">
                          <span className="label-muted !text-[9px]">H:</span>
                          <span className="value-mono text-[9px] text-[#3fb950]">{liveData?.chart?.candles?.at(-1)?.high?.toFixed(5)}</span>
                       </div>
                    </div>
                 </div>
                 <div className="flex-1 relative">
                    <ChartWidget data={liveData?.chart} atr={liveData?.atr} atrMultiplier={liveData?.atr_multiplier} />
                 </div>
              </div>

              {/* Right Execution Pane */}
              <div className="col-span-3 border-l border-[#30363d] flex flex-col min-h-0 overflow-hidden">
                 <OrderTicket liveData={liveData} handleTrade={handleTrade} />
              </div>
           </div>
        )}

        {activeTab === 'explain' && (
           <div className="h-full p-4 overflow-y-auto">
              <div className="max-w-6xl mx-auto grid grid-cols-2 gap-6 h-[calc(100vh-100px)]">
                  <div className="flex flex-col h-full">
                     <ExplainabilityWidget runId={selectedModel} />
                  </div>
                  <div className="flex flex-col gap-6">
                     <ModelConfigWidget config={modelDetails?.config} />
                     <ValidationStatsWidget stats={modelDetails?.stats} />
                  </div>
              </div>
           </div>
        )}

        {activeTab === 'lab' && (
           <div className="h-full p-4 overflow-y-auto">
              <div className="max-w-7xl mx-auto h-full">
                 <BacktestLab />
              </div>
           </div>
        )}
      </main>

      {/* FOOTER STATUS BAR */}
      <footer className="h-6 border-t border-[#30363d] bg-[#0d1117] flex items-center px-4 justify-between shrink-0">
         <div className="flex space-x-4 items-center">
            <div className="flex items-center space-x-2 text-[9px] font-bold text-[#8b949e]">
               <span>MT5 SERVER:</span>
               <span className="text-[#3fb950]">RUNNING</span>
            </div>
            <div className="w-px h-3 bg-[#30363d]"></div>
            <div className="flex items-center space-x-2 text-[9px] font-bold text-[#8b949e]">
               <span>MODEL TYPE:</span>
               <span className="text-white uppercase">{modelDetails?.config?.model?.type || 'TRANSFORMER'}</span>
            </div>
         </div>
         
         <div className="text-[9px] font-mono text-[#8b949e]">
            {new Date().toISOString().replace('T', ' ').split('.')[0]} UTC
         </div>
      </footer>
    </div>
  );
}
