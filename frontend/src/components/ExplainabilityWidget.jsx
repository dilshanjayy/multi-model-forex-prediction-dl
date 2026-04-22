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
      setError("Failed to generate explainability data. Please ensure the backend is running with the latest updates.");
    })
    .finally(() => setLoading(false));
  }, [runId]);

  if (!runId) return <div className="metric-card h-full">Select a model to view X-Ray.</div>;

  const maxImpact = explanation && explanation.length > 0 
    ? Math.max(...explanation.map(f => Math.abs(f.impact))) 
    : 1;

  return (
    <div className="metric-card h-full flex flex-col">
      <h3 className="metric-label mb-4">Model X-Ray (SHAP Explainability)</h3>
      <p className="text-sm text-[#8b949e] mb-4">Top Feature Contributions for Current Signal:</p>
      
      {loading && (
        <div className="flex flex-col items-center justify-center flex-1 text-[#8b949e] opacity-70">
           <svg className="w-8 h-8 mb-4 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
           </svg>
           <p className="text-xs tracking-widest uppercase">Calculating SHAP Gradients...</p>
        </div>
      )}
      
      {error && !loading && (
        <div className="flex-1 flex items-center justify-center text-[#f85149] text-xs px-4 text-center border border-[#f85149]/20 bg-[#f85149]/5 rounded">
           {error}
        </div>
      )}
      
      {!loading && !error && explanation && explanation.length > 0 && (
        <div className="flex flex-col gap-3 overflow-y-auto flex-1 custom-scrollbar pr-2">
          {explanation.map((feature, idx) => {
             if (feature.feature.includes("failed") || feature.feature.includes("Error")) {
                 return <div key={idx} className="text-xs text-[#f85149]">{feature.feature}</div>;
             }
             
             const isPositive = feature.impact > 0;
             const absImpact = Math.abs(feature.impact);
             // Dynamic scaling relative to the max impact
             const width = Math.max(2, (absImpact / maxImpact) * 100); 
             const colorClass = isPositive ? 'bg-[#3fb950]' : 'bg-[#f85149]';
             
             return (
               <div key={idx} className="w-full">
                 <div className="flex justify-between text-xs mb-1">
                   <span className="text-white font-mono">{feature.feature.replace(/_/g, ' ')}</span>
                   <span className={isPositive ? 'text-[#3fb950]' : 'text-[#f85149]'}>
                     {feature.impact > 0 ? '+' : ''}{feature.impact.toFixed(4)}
                   </span>
                 </div>
                 <div className="w-full bg-[#1c2128] rounded-full h-1.5 overflow-hidden">
                   <div className={`${colorClass} h-full transition-all duration-1000 ease-out`} style={{ width: `${width}%` }}></div>
                 </div>
               </div>
             )
          })}
          
          <div className="mt-auto text-[10px] text-[#8b949e] border-t border-[#30363d] pt-3 mt-4">
             <p className="flex items-center gap-2"><span className="w-2 h-2 bg-[#3fb950] inline-block rounded-full"></span> Positive pushes toward BUY (Class 0)</p>
             <p className="flex items-center gap-2 mt-1"><span className="w-2 h-2 bg-[#f85149] inline-block rounded-full"></span> Negative pushes toward SELL (Class 1)</p>
          </div>
        </div>
      )}
    </div>
  );
}
