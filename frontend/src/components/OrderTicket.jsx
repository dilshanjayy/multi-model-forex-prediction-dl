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

  // Early return if data is not yet loaded or if it's an error object without price
  if (!liveData || liveData.price === undefined) {
    return (
      <div className="pane shrink-0 border-b border-[#30363d]">
        <div className="pane-header">EXECUTION GATEWAY</div>
        <div className="pane-content flex flex-col items-center justify-center space-y-2">
            <p className="text-[10px] text-[#8b949e] animate-pulse uppercase tracking-widest">Awaiting MT5 Stream...</p>
            {liveData?.detail && <p className="text-[9px] text-[#f85149]">{liveData.detail}</p>}
        </div>
      </div>
    );
  }

  const currentPrice = liveData.price || 0;
  const dist = (liveData.atr || 0) * (liveData.atr_multiplier || 1);
  const tp = currentPrice + dist;
  const sl = currentPrice - dist;

  return (
    <div className="pane shrink-0 border-b border-[#30363d]">
      <div className="pane-header">EXECUTION GATEWAY</div>
      <div className="pane-content flex flex-col space-y-4">
        <div>
          <label className="label-muted mb-1 block">VOLUME (LOTS)</label>
          <input 
            type="text" 
            inputMode="decimal"
            className="w-full bg-[#0d1117] border border-[#30363d] text-white font-mono text-sm py-2 px-3 rounded focus:outline-none focus:border-[#58a6ff] transition-colors"
            value={lotSize}
            onChange={(e) => {
              const val = e.target.value.replace(',', '.');
              if (/^\d*\.?\d*$/.test(val)) {
                setLotSize(val);
              }
            }}
            onBlur={() => {
              if (!lotSize || isNaN(parseFloat(lotSize))) setLotSize(0.01);
            }}
          />
        </div>

        <div className="bg-[#1c2128]/50 border border-[#30363d] p-3 rounded space-y-2">
            <p className="label-muted text-[9px] border-b border-[#30363d] pb-1 mb-2">RISK PREVIEW (ATR BASED)</p>
            <div className="flex justify-between items-center text-xs">
                <span className="text-[#3fb950]">TAKE PROFIT</span>
                <span className="value-mono font-bold text-white">{tp ? tp.toFixed(5) : '0.00000'}</span>
            </div>
            <div className="flex justify-between items-center text-xs">
                <span className="text-[#f85149]">STOP LOSS</span>
                <span className="value-mono font-bold text-white">{sl ? sl.toFixed(5) : '0.00000'}</span>
            </div>
            <div className="flex justify-between items-center text-[10px] pt-1 opacity-60 italic">
                <span>Distance:</span>
                <span>{dist ? (dist * 10000).toFixed(1) : '0.0'} Pips</span>
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
      </div>
    </div>
  );
}

