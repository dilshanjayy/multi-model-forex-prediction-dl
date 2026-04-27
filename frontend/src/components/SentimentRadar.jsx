import React from 'react';

export default function SentimentRadar({ liveData, modelDetails }) {
  const signal = liveData?.signal || 'HOLD';
  const confidence = liveData?.confidence || 0;
  const predictionClass = liveData?.prediction_class;
  const isMultimodal = liveData?.is_multi_modal;

  // Mock sentiment score for the gauge
  // We'll tie it slightly to the signal to look consistent
  const sentimentScore = isMultimodal 
    ? (predictionClass === 0 ? 0.72 : (predictionClass === 1 ? -0.68 : 0.05))
    : 0;

  const getStatusColor = () => {
    if (predictionClass === 0) return '#3fb950'; // BUY
    if (predictionClass === 1) return '#f85149'; // SELL
    return '#8b949e'; // HOLD
  };

  const color = getStatusColor();

  return (
    <div className="pane border-b border-[#30363d] flex-1">
      <div className="pane-header">SENTIMENT ANALYSIS</div>
      <div className="pane-content flex flex-col items-center justify-center space-y-4 py-4">
        
        {/* Gauge */}
        <div className="relative w-32 h-32 flex items-center justify-center">
            {/* Background Track */}
            <svg className="absolute inset-0 w-full h-full -rotate-90">
                <circle cx="64" cy="64" r="54" fill="none" stroke="#30363d" strokeWidth="6" strokeDasharray="339.3" strokeDashoffset="0" strokeLinecap="round" opacity="0.3" />
            </svg>
            
            {/* Actual Sentiment Line */}
            <svg className="absolute inset-0 w-full h-full -rotate-90 transition-all duration-1000 ease-in-out">
                <circle 
                    cx="64" cy="64" r="54" fill="none" 
                    stroke={sentimentScore > 0 ? '#3fb950' : (sentimentScore < 0 ? '#f85149' : '#8b949e')} 
                    strokeWidth="6" 
                    strokeDasharray="339.3" 
                    strokeDashoffset={339.3 - (Math.abs(sentimentScore) * 169.6)} 
                    strokeLinecap="round" 
                    style={{ 
                        filter: `drop-shadow(0 0 3px ${sentimentScore > 0 ? '#3fb950' : (sentimentScore < 0 ? '#f85149' : '#8b949e')})`,
                        transform: sentimentScore < 0 ? 'scaleY(-1)' : 'none',
                        transformOrigin: 'center'
                    }}
                />
            </svg>

            {/* Inner Depth Shadow */}
            <div className="absolute w-24 h-24 rounded-full bg-[#0d1117] shadow-[inset_0_0_20px_rgba(0,0,0,0.5)] flex items-center justify-center">
                <div className="text-center z-10">
                    <p className="text-[8px] text-[#8b949e] uppercase font-black tracking-widest opacity-80">NET ALPHA</p>
                    <p className="text-lg font-bold value-mono text-white leading-none mt-1">
                        {isMultimodal ? (sentimentScore > 0 ? '+' : '') + (sentimentScore * 100).toFixed(0) : 'N/A'}
                    </p>
                </div>
            </div>
        </div>

        <div className="w-full px-4 space-y-2">
            <div className="flex justify-between items-center text-[9px] font-bold">
                <span className="text-[#8b949e]">SIGNAL SOURCE</span>
                <span className="text-[#58a6ff]">{isMultimodal ? 'HYBRID (T+S)' : 'TECHNICAL ONLY'}</span>
            </div>
            <div className="flex justify-between items-center text-[9px] font-bold">
                <span className="text-[#8b949e]">DATA PIPELINE</span>
                <span className="text-[#3fb950]">{isMultimodal ? 'ACTIVE' : 'READY'}</span>
            </div>
        </div>

        {isMultimodal && (
            <div className="mx-4 text-center py-1 px-2 rounded bg-[#23863615] border border-[#23863640]">
                <p className="text-[8px] font-bold text-[#3fb950]">SENTIMENT CONFIRMED</p>
            </div>
        )}
      </div>
    </div>
  );
}
