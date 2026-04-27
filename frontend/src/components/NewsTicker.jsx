import React, { useState, useEffect } from 'react';

export default function NewsTicker() {
  const [news, setNews] = useState([]);

  useEffect(() => {
    const fetchNews = () => {
      fetch('http://localhost:8000/api/v1/news?limit=15')
        .then(res => res.json())
        .then(data => setNews(data.news || []))
        .catch(err => console.error("Error fetching news:", err));
    };

    fetchNews();
    const interval = setInterval(fetchNews, 30000); // Update every 30s
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col flex-1 min-h-0 bg-[#0d1117]">
      <div className="pane-header !border-t-0">LIVE SENTIMENT FEED</div>
      <div className="flex-1 overflow-y-auto p-2 space-y-2 custom-scrollbar">
        {news.length === 0 ? (
          <div className="text-[10px] text-[#8b949e] italic p-4 text-center">Waiting for news stream...</div>
        ) : (
          news.map((item, i) => (
            <div key={i} className="p-2 border border-[#30363d] rounded bg-[#161b22] hover:border-[#58a6ff] transition-colors group">
              <div className="flex justify-between items-start mb-1">
                <span className={`text-[8px] px-1 rounded font-bold ${
                  item.sentiment_label === 'Positive' ? 'bg-[#238636] text-white' : 
                  item.sentiment_label === 'Negative' ? 'bg-[#da3633] text-white' : 'bg-[#30363d] text-[#8b949e]'
                }`}>
                  {item.sentiment_label?.toUpperCase()}
                </span>
                <span className="text-[8px] text-[#8b949e] font-mono">
                  {new Date(item.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
              <p className="text-[10px] font-bold text-white leading-tight group-hover:text-[#58a6ff] transition-colors line-clamp-2">
                {item.title}
              </p>
              <div className="mt-1 flex justify-between items-center">
                <span className="text-[8px] text-[#8b949e] italic">{item.source}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
