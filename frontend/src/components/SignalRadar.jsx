import React from 'react';

export default function SignalRadar({ liveData }) {
  const signal = liveData?.signal || 'HOLD';
  const confidence = liveData?.confidence || 0;
  const predictionClass = liveData?.prediction_class;

  const getStatusColor = () => {
    if (predictionClass === 0) return '#3fb950'; // BUY
    if (predictionClass === 1) return '#f85149'; // SELL
    return '#8b949e'; // HOLD
  };

  const color = getStatusColor();

  return (
    <div className="pane h-full">
      <div className="pane-header">ALGO STANCE</div>
      <div className="pane-content flex flex-col items-center justify-center space-y-6">
        {/* Signal HUD */}
        <div className="relative w-32 h-32 flex items-center justify-center">
            <div 
                className="absolute inset-0 rounded-full border-4 border-[#30363d] opacity-20"
                style={{ borderColor: color }}
            ></div>
            <div 
                className="absolute inset-0 rounded-full border-4 transition-all duration-1000 ease-out"
                style={{ 
                    borderColor: color, 
                    clipPath: `inset(${100 - (confidence * 100)}% 0 0 0)`,
                    filter: `drop-shadow(0 0 8px ${color})`
                }}
            ></div>
            <div className="text-center z-10">
                <p className="text-[10px] text-[#8b949e] uppercase font-bold tracking-tighter">CONFIDENCE</p>
                <p className="text-xl font-bold value-mono" style={{ color }}>{(confidence * 100).toFixed(1)}%</p>
            </div>
        </div>

        <div className="w-full text-center py-2 px-4 rounded" style={{ backgroundColor: `${color}15`, border: `1px solid ${color}40` }}>
            <p className="text-xs font-black tracking-widest" style={{ color }}>{signal}</p>
        </div>

        <div className="w-full space-y-3">
            <div className="flex justify-between items-center">
                <span className="label-muted">ATR RANGE</span>
                <span className="value-mono text-xs text-white">{liveData?.atr?.toFixed(5) || '0.00000'}</span>
            </div>
            <div className="flex justify-between items-center">
                <span className="label-muted">TARGET MULT</span>
                <span className="value-mono text-xs text-white">x{liveData?.atr_multiplier || '1.0'}</span>
            </div>
        </div>
      </div>
    </div>
  );
}
