import { useState, useEffect } from 'react';

export default function ExplainabilityWidget({ runId }) {
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!runId) return;
    
    setLoading(true);
    setError(null);
    
    fetch(`http://localhost:8000/api/v1/explain/${runId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })
    .then(res => {
      if (!res.ok) throw new Error("Server Error or Model Not Supported");
      return res.json();
    })
    .then(data => {
      setExplanation(data.top_features || []);
    })
    .catch(err => {
      console.error("Error fetching explanation:", err);
      setError("Failed to generate explainability data.");
    })
    .finally(() => setLoading(false));
  }, [runId]);

  if (!runId) return <div className="metric-card h-full flex items-center justify-center text-[#8b949e]">SELECT INSTANCE TO INITIALIZE X-RAY</div>;

  const maxImpact = explanation && explanation.length > 0 
    ? Math.max(...explanation.map(f => Math.abs(f.impact))) 
    : 1;

  return (
    <div className="glass-panel h-full flex flex-col overflow-hidden">
      <div className="px-4 py-2 border-b border-[#30363d] bg-[#161b2250] flex justify-between items-center">
          <h3 className="text-[10px] font-black text-[#8b949e] uppercase tracking-widest">NEURAL IMPACT MATRIX (SHAP)</h3>
          <div className="flex items-center space-x-2">
              <div className="w-1.5 h-1.5 bg-[#58a6ff] rounded-full animate-pulse"></div>
              <span className="text-[8px] font-bold text-[#58a6ff]">LIVE SCANNING</span>
          </div>
      </div>
      
      <div className="flex-1 p-6 relative overflow-hidden flex flex-col justify-start">
        {/* Static Grid Guide */}
        <div className="absolute inset-0 pointer-events-none z-0">
            <div className="w-px h-full bg-[#30363d50] absolute left-1/2 -translate-x-1/2"></div>
        </div>

        {loading ? (
            <div className="flex flex-col items-center justify-center h-full space-y-4 animate-pulse z-10">
                <p className="text-[10px] font-black text-[#58a6ff] tracking-[0.3em] uppercase">Calculating Gradients...</p>
                <div className="w-48 h-1 bg-[#30363d] rounded-full overflow-hidden">
                    <div className="h-full bg-[#58a6ff] w-1/2 animate-[progress_1s_ease-in-out_infinite]"></div>
                </div>
            </div>
        ) : error ? (
            <div className="flex flex-col items-center justify-center h-full text-center space-y-2 z-10">
                <p className="text-[10px] font-bold text-[#f85149] uppercase">{error}</p>
                <p className="text-[8px] text-[#8b949e]">The current model architecture may not support deep explainability yet.</p>
            </div>
        ) : (
            <div className="space-y-4 z-10 overflow-y-auto custom-scrollbar pr-2">
                {explanation?.map((item, idx) => {
                    const isPositive = item.impact > 0;
                    const width = (Math.abs(item.impact) / maxImpact) * 45; // Max 45% width per side

                    return (
                        <div key={idx} className="group flex flex-col">
                            <div className="flex justify-between items-center mb-1">
                                <span className="text-[10px] font-bold text-[#c9d1d9] truncate max-w-[150px] uppercase tracking-tighter group-hover:text-[#58a6ff] transition-colors">{item.feature}</span>
                                <span className={`text-[10px] font-mono font-bold ${isPositive ? 'text-[#3fb950]' : 'text-[#f85149]'}`}>
                                    {isPositive ? '+' : ''}{(item.impact * 100).toFixed(2)}%
                                </span>
                            </div>
                            
                            <div className="relative h-1.5 bg-[#30363d30] rounded-full overflow-hidden flex justify-center">
                                {/* The Zero Center Line */}
                                <div className="absolute inset-y-0 left-1/2 w-px bg-[#30363d] z-10"></div>
                                
                                {/* The Impact Bar */}
                                <div 
                                    className={`absolute inset-y-0 ${isPositive ? 'left-1/2' : 'right-1/2'} transition-all duration-1000 ease-out rounded-full`}
                                    style={{ 
                                        width: `${width}%`,
                                        background: isPositive 
                                            ? 'linear-gradient(90deg, #3fb95040 0%, #3fb950 100%)' 
                                            : 'linear-gradient(270deg, #f8514940 0%, #f85149 100%)',
                                        boxShadow: `0 0 10px ${isPositive ? '#3fb95040' : '#f8514940'}`
                                    }}
                                ></div>
                            </div>
                        </div>
                    );
                })}
            </div>
        )}
      </div>

      <div className="px-4 py-3 border-t border-[#30363d] bg-[#0d1117] flex justify-between items-center">
          <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1.5">
                  <div className="w-2 h-2 rounded-sm bg-[#3fb950]"></div>
                  <span className="text-[8px] font-bold text-[#8b949e] uppercase">Supports Decision</span>
              </div>
              <div className="flex items-center space-x-1.5">
                  <div className="w-2 h-2 rounded-sm bg-[#f85149]"></div>
                  <span className="text-[8px] font-bold text-[#8b949e] uppercase">Opposes Decision</span>
              </div>
          </div>
          <span className="text-[8px] font-mono text-[#8b949e] italic opacity-60">SHAP KERNEL: ACTIVE</span>
      </div>

      <style>{`
        @keyframes scan {
          0% { top: -10%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { top: 110%; opacity: 0; }
        }
        @keyframes progress {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(200%); }
        }
      `}</style>
    </div>
  );
}
