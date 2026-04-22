import React, { useState } from 'react';
import { useStore } from '../store';

export default function OrderTicket({ liveData, handleTrade }) {
  const lotSize = useStore(state => state.lotSize);
  const setLotSize = useStore(state => state.setLotSize);
  const [isPending, setIsPending] = useState(false);

  const onTrade = async (dir) => {
    setIsPending(true);
    await handleTrade(dir);
    setIsPending(false);
  };

  const currentPrice = liveData?.price || 0;
  const dist = (liveData?.atr || 0) * (liveData?.atr_multiplier || 1);
  const tp = currentPrice + dist;
  const sl = currentPrice - dist;

  return (
    <div className="pane h-full">
      <div className="pane-header">EXECUTION GATEWAY</div>
      <div className="pane-content flex flex-col space-y-4">
        <div>
          <label className="label-muted mb-1 block">VOLUME (LOTS)</label>
          <input 
            type="number" 
            step="0.01" 
            className="w-full bg-[#1c2128] border-[#30363d] text-white font-mono text-sm py-2 rounded focus:ring-1 focus:ring-[#58a6ff]"
            value={lotSize}
            onChange={(e) => setLotSize(parseFloat(e.target.value))}
          />
        </div>

        <div className="bg-[#1c2128]/50 border border-[#30363d] p-3 rounded space-y-2">
            <p className="label-muted text-[9px] border-b border-[#30363d] pb-1 mb-2">RISK PREVIEW (ATR BASED)</p>
            <div className="flex justify-between items-center text-xs">
                <span className="text-[#3fb950]">TAKE PROFIT</span>
                <span className="value-mono font-bold text-white">{tp.toFixed(5)}</span>
            </div>
            <div className="flex justify-between items-center text-xs">
                <span className="text-[#f85149]">STOP LOSS</span>
                <span className="value-mono font-bold text-white">{sl.toFixed(5)}</span>
            </div>
            <div className="flex justify-between items-center text-[10px] pt-1 opacity-60 italic">
                <span>Distance:</span>
                <span>{(dist * 10000).toFixed(1)} Pips</span>
            </div>
        </div>

        <div className="grid grid-cols-2 gap-2 pt-4">
            <button 
                onClick={() => onTrade('BUY')}
                disabled={isPending || !liveData}
                className="btn-action bg-[#238636] hover:bg-[#2ea043] text-white rounded border border-[#3fb950]/30 shadow-lg shadow-[#238636]/10"
            >
                BUY
            </button>
            <button 
                onClick={() => onTrade('SELL')}
                disabled={isPending || !liveData}
                className="btn-action bg-[#da3633] hover:bg-[#f85149] text-white rounded border border-[#f85149]/30 shadow-lg shadow-[#da3633]/10"
            >
                SELL
            </button>
        </div>

        <div className="flex-1 mt-6 flex flex-col min-h-0 overflow-hidden">
             <div className="label-muted mb-2 flex justify-between">
                 <span>ACTIVITY LOG</span>
                 <span className="text-[9px] opacity-50">REAL-TIME</span>
             </div>
             <div className="flex-1 bg-black/40 border border-[#30363d] rounded p-2 text-[9px] font-mono text-[#8b949e] overflow-y-auto space-y-1 custom-scrollbar">
                <p><span className="text-[#58a6ff]">[SYS]</span> API Session Initialized</p>
                <p><span className="text-[#58a6ff]">[SYS]</span> MT5 Stream: OK</p>
                {liveData && <p><span className="text-[#3fb950]">[DATA]</span> Tick: {liveData.price.toFixed(5)}</p>}
             </div>
        </div>
      </div>
    </div>
  );
}
