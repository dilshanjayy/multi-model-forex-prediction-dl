import { useEffect, useState } from "react";
import { useStore } from "../store";

export default function Portfolio() {
    const token = useStore((state) => state.token);
    const [portfolioData, setPortfolioData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [syncing, setSyncing] = useState(false);

    const fetchPortfolio = async () => {
        try {
            const res = await fetch("http://localhost:8000/api/v1/portfolio", {
                headers: {
                    Authorization: `Bearer ${token}`,
                },
            });
            if (res.ok) {
                const data = await res.json();
                setPortfolioData(data);
            } else {
                console.error("Failed to fetch portfolio");
            }
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const syncAndFetch = async () => {
        setSyncing(true);
        try {
            await fetch("http://localhost:8000/api/v1/portfolio/sync", {
                method: "POST",
                headers: { Authorization: `Bearer ${token}` },
            });
            await fetchPortfolio();
        } catch (err) {
            console.error(err);
        } finally {
            setSyncing(false);
        }
    };

    useEffect(() => {
        // Initial sync and fetch
        syncAndFetch();

        // Set up 2-second polling for live PnL updates
        const interval = setInterval(() => {
            syncAndFetch();
        }, 2000);

        return () => clearInterval(interval);
    }, [token]);

    if (loading && !portfolioData) {
        return <div className="text-white p-4">Loading portfolio data...</div>;
    }

    if (!portfolioData) {
        return <div className="text-red-500 p-4">Failed to load portfolio.</div>;
    }

    return (
        <div className="h-full p-4 overflow-y-auto relative">
            <div className="max-w-6xl mx-auto flex flex-col gap-6">
                
                {/* Stats Row */}
                <div className="grid grid-cols-3 gap-4">
                    <div className="bg-[#0d1117] border border-[#30363d] rounded p-4 flex flex-col items-center justify-center">
                        <div className="text-[10px] text-[#8b949e] font-bold tracking-widest mb-1">TOTAL TRADES</div>
                        <div className="text-2xl font-mono text-white">{portfolioData.total_trades}</div>
                    </div>
                    <div className="bg-[#0d1117] border border-[#30363d] rounded p-4 flex flex-col items-center justify-center">
                        <div className="text-[10px] text-[#8b949e] font-bold tracking-widest mb-1">WIN RATE</div>
                        <div className="text-2xl font-mono text-[#58a6ff]">{portfolioData.win_rate}%</div>
                    </div>
                    <div className="bg-[#0d1117] border border-[#30363d] rounded p-4 flex flex-col items-center justify-center">
                        <div className="text-[10px] text-[#8b949e] font-bold tracking-widest mb-1">TOTAL PnL</div>
                        <div className={`text-2xl font-mono ${portfolioData.total_pnl >= 0 ? 'text-[#3fb950]' : 'text-[#f85149]'}`}>
                            {portfolioData.total_pnl >= 0 ? '+' : ''}{portfolioData.total_pnl.toFixed(2)}
                        </div>
                    </div>
                </div>

                {/* Trades Table */}
                <div className="bg-[#0d1117] border border-[#30363d] rounded flex flex-col">
                    <div className="p-3 border-b border-[#30363d] flex justify-between items-center">
                        <h2 className="text-[11px] font-bold tracking-widest text-white">TRADE HISTORY</h2>
                    </div>
                    <div className="p-0 overflow-x-auto">
                        <table className="w-full text-left border-collapse">
                            <thead>
                                <tr className="border-b border-[#30363d] bg-[#06090f]">
                                    <th className="p-3 text-[10px] font-bold text-[#8b949e] tracking-wider">DATE</th>
                                    <th className="p-3 text-[10px] font-bold text-[#8b949e] tracking-wider">SYMBOL</th>
                                    <th className="p-3 text-[10px] font-bold text-[#8b949e] tracking-wider">TYPE</th>
                                    <th className="p-3 text-[10px] font-bold text-[#8b949e] tracking-wider">LOTS</th>
                                    <th className="p-3 text-[10px] font-bold text-[#8b949e] tracking-wider">PRICE</th>
                                    <th className="p-3 text-[10px] font-bold text-[#8b949e] tracking-wider">STATUS</th>
                                    <th className="p-3 text-[10px] font-bold text-[#8b949e] tracking-wider text-right">PnL</th>
                                </tr>
                            </thead>
                            <tbody>
                                {portfolioData.trades.map((trade) => (
                                    <tr key={trade.id} className="border-b border-[#30363d]/50 hover:bg-[#161b22] transition-colors">
                                        <td className="p-3 text-[11px] font-mono text-[#c9d1d9]">
                                            {new Date(trade.timestamp).toLocaleString()}
                                        </td>
                                        <td className="p-3 text-[11px] font-bold text-white">{trade.symbol}</td>
                                        <td className={`p-3 text-[11px] font-bold tracking-wider ${trade.direction === 'BUY' ? 'text-[#3fb950]' : 'text-[#f85149]'}`}>
                                            {trade.direction}
                                        </td>
                                        <td className="p-3 text-[11px] font-mono text-[#c9d1d9]">{trade.lot_size}</td>
                                        <td className="p-3 text-[11px] font-mono text-[#c9d1d9]">{trade.price.toFixed(5)}</td>
                                        <td className="p-3 text-[10px] font-bold">
                                            <span className={`px-2 py-0.5 rounded ${trade.status === 'OPEN' ? 'bg-[#58a6ff]/20 text-[#58a6ff]' : 'bg-[#8b949e]/20 text-[#8b949e]'}`}>
                                                {trade.status || "OPEN"}
                                            </span>
                                        </td>
                                        <td className={`p-3 text-[11px] font-mono text-right ${trade.pnl >= 0 ? 'text-[#3fb950]' : 'text-[#f85149]'}`}>
                                            {trade.pnl >= 0 ? '+' : ''}{(trade.pnl || 0).toFixed(2)}
                                        </td>
                                    </tr>
                                ))}
                                {portfolioData.trades.length === 0 && (
                                    <tr>
                                        <td colSpan="7" className="p-6 text-center text-[#8b949e] text-xs italic">
                                            No trades executed yet.
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>

            </div>
        </div>
    );
}
