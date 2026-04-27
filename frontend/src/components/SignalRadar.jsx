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
    <div className="pane border-b border-[#30363d] flex-1">
      <div className="pane-header">ALGO STANCE</div>
      <div className="pane-content flex flex-col items-center justify-center space-y-6">
        {/* Signal HUD */}
        <div className="relative w-32 h-32 flex items-center justify-center">
            {/* Background Track */}
            <svg className="absolute inset-0 w-full h-full -rotate-90">
                <circle cx="64" cy="64" r="54" fill="none" stroke="#30363d" strokeWidth="6" strokeDasharray="339.3" strokeDashoffset="0" strokeLinecap="round" opacity="0.3" />
            </svg>
            
            {/* Precision Progress Line */}
            <svg className="absolute inset-0 w-full h-full -rotate-90 transition-all duration-1000 ease-out">
                <circle 
                    cx="64" cy="64" r="54" fill="none" 
                    stroke={color} 
                    strokeWidth="6" 
                    strokeDasharray="339.3" 
                    strokeDashoffset={339.3 - (confidence * 339.3)} 
                    strokeLinecap="round" 
                    style={{ 
                        filter: `drop-shadow(0 0 3px ${color})`,
                    }}
                />
            </svg>

            {/* Inner Depth Shadow */}
            <div className="absolute w-24 h-24 rounded-full bg-[#0d1117] shadow-[inset_0_0_20px_rgba(0,0,0,0.5)] flex items-center justify-center">
                <div className="text-center z-10">
                    <p className="text-[10px] text-[#8b949e] uppercase font-black tracking-widest opacity-80">CONFIDENCE</p>
                    <p className="text-xl font-bold value-mono leading-none mt-1" style={{ color }}>{(confidence * 100).toFixed(1)}%</p>
                </div>
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
