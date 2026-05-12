import React, { useEffect, useState, useRef } from "react";
import { createChart } from "lightweight-charts";
import { useStore } from "../store";

export default function Portfolio() {
    const token = useStore((state) => state.token);
    const [portfolioData, setPortfolioData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [syncing, setSyncing] = useState(false);
    const [expandedTrade, setExpandedTrade] = useState(null);
    const [chartError, setChartError] = useState(null);

    const chartContainerRef = useRef();
    const chartRef = useRef();
    const seriesRef = useRef();

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
            }
        } catch (err) {
            console.error("Failed to fetch portfolio:", err);
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

    // Auto-sync polling
    useEffect(() => {
        syncAndFetch();
        const interval = setInterval(syncAndFetch, 2000);
        return () => clearInterval(interval);
    }, [token]);

    // Chart Initialization
    useEffect(() => {
        // If container isn't ready or chart is already built, do nothing
        if (!chartContainerRef.current || chartRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { color: "transparent" },
                textColor: "#8b949e",
            },
            grid: {
                vertLines: { color: "#30363d20" },
                horzLines: { color: "#30363d20" },
            },
            width: chartContainerRef.current.clientWidth,
            height: 300,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
                borderVisible: false,
            },
            rightPriceScale: {
                borderVisible: false,
            },
        });

        const series = chart.addAreaSeries({
            lineColor: "#58a6ff",
            topColor: "#58a6ff40",
            bottomColor: "#58a6ff00",
            lineWidth: 2,
        });

        chartRef.current = chart;
        seriesRef.current = series;

        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };
        window.addEventListener("resize", handleResize);

        // If data arrived before chart was initialized, set it now
        if (portfolioData?.equity_curve?.length > 0) {
            try {
                const uniqueData = [];
                let lastTime = 0;
                for (const pt of portfolioData.equity_curve) {
                    let t = pt.time;
                    if (t <= lastTime) t = lastTime + 1;
                    uniqueData.push({ time: t, value: pt.value });
                    lastTime = t;
                }
                series.setData(uniqueData);
                chart.timeScale().fitContent();
            } catch(e) {}
        }

        return () => {
            window.removeEventListener("resize", handleResize);
            chart.remove();
            chartRef.current = null;
        };
    }, [portfolioData !== null]); // Only run when portfolioData changes from null to populated

    // Update Chart Data if it arrives AFTER init
    useEffect(() => {
        if (seriesRef.current && portfolioData?.equity_curve) {
            try {
                if (portfolioData.equity_curve.length > 0) {
                    const uniqueData = [];
                    let lastTime = 0;
                    for (const pt of portfolioData.equity_curve) {
                        let t = pt.time;
                        if (t <= lastTime) t = lastTime + 1;
                        uniqueData.push({ time: t, value: pt.value });
                        lastTime = t;
                    }
                    seriesRef.current.setData(uniqueData);
                    chartRef.current.timeScale().fitContent();
                }
            } catch (err) {
                console.error("Chart Error:", err);
                setChartError(err.message);
            }
        }
    }, [portfolioData]);

    if (loading && !portfolioData) {
        return <div className="text-white p-4 font-mono">INITIALIZING ANALYTICS...</div>;
    }

    if (!portfolioData) {
        return <div className="text-red-500 p-4">SYSTEM ERROR: UNABLE TO LOAD PORTFOLIO</div>;
    }

    return (
        <div className="h-full p-6 overflow-y-auto bg-[#06090f]">
            <div className="max-w-7xl mx-auto flex flex-col gap-6">
                
                {/* ADVANCED METRICS GRID */}
                <div className="grid grid-cols-6 gap-4">
                    <div className="bg-[#0d1117] border border-[#30363d] rounded p-4">
                        <div className="text-[9px] text-[#8b949e] font-bold tracking-widest mb-1 uppercase">Total PnL</div>
                        <div className={`text-xl font-mono ${portfolioData.total_pnl >= 0 ? 'text-[#3fb950]' : 'text-[#f85149]'}`}>
                            {portfolioData.total_pnl >= 0 ? '+' : ''}{portfolioData.total_pnl.toFixed(2)}
                        </div>
                    </div>
                    <div className="bg-[#0d1117] border border-[#30363d] rounded p-4">
                        <div className="text-[9px] text-[#8b949e] font-bold tracking-widest mb-1 uppercase">Win Rate</div>
                        <div className="text-xl font-mono text-white">{portfolioData.win_rate}%</div>
                    </div>
                    <div className="bg-[#0d1117] border border-[#30363d] rounded p-4">
                        <div className="text-[9px] text-[#8b949e] font-bold tracking-widest mb-1 uppercase">Profit Factor</div>
                        <div className="text-xl font-mono text-[#58a6ff]">{portfolioData.profit_factor}</div>
                    </div>
                    <div className="bg-[#0d1117] border border-[#30363d] rounded p-4">
                        <div className="text-[9px] text-[#8b949e] font-bold tracking-widest mb-1 uppercase">Sharpe Ratio</div>
                        <div className="text-xl font-mono text-[#d29922]">{portfolioData.sharpe_ratio}</div>
                    </div>
                    <div className="bg-[#0d1117] border border-[#30363d] rounded p-4">
                        <div className="text-[9px] text-[#8b949e] font-bold tracking-widest mb-1 uppercase">Max Drawdown</div>
                        <div className="text-xl font-mono text-[#f85149]">{portfolioData.max_drawdown.toFixed(2)}</div>
                    </div>
                    <div className="bg-[#0d1117] border border-[#30363d] rounded p-4">
                        <div className="text-[9px] text-[#8b949e] font-bold tracking-widest mb-1 uppercase">Executions</div>
                        <div className="text-xl font-mono text-white">{portfolioData.total_trades}</div>
                    </div>
                </div>

                {/* EQUITY CURVE & MODEL PERFORMANCE */}
                <div className="grid grid-cols-12 gap-6">
                    <div className="col-span-8 bg-[#0d1117] border border-[#30363d] rounded flex flex-col p-4">
                        <div className="flex justify-between items-center mb-4">
                            <h2 className="text-[11px] font-bold tracking-widest text-white uppercase">Cumulative Equity Curve</h2>
                            <div className="text-[10px] text-[#8b949e] font-mono">PNL/TIME (USD)</div>
                        </div>
                        {chartError ? (
                            <div className="w-full h-[300px] flex items-center justify-center text-red-500 font-mono text-xs text-center p-4 border border-red-500/20 bg-red-500/5 rounded">
                                CHART RENDER ERROR:<br/>{chartError}
                            </div>
                        ) : (
                            <div ref={chartContainerRef} className="w-full h-[300px]" />
                        )}
                    </div>

                    <div className="col-span-4 bg-[#0d1117] border border-[#30363d] rounded flex flex-col p-4">
                        <h2 className="text-[11px] font-bold tracking-widest text-white uppercase mb-4">Model Performance Distribution</h2>
                        <div className="flex flex-col gap-3">
                            {Object.entries(portfolioData.model_performance).map(([name, stats]) => (
                                <div key={name} className="border-b border-[#30363d]/50 pb-2">
                                    <div className="flex justify-between text-[10px] mb-1">
                                        <span className="font-bold text-[#c9d1d9]">{name}</span>
                                        <span className={stats.pnl >= 0 ? 'text-[#3fb950]' : 'text-[#f85149]'}>
                                            {stats.pnl >= 0 ? '+' : ''}{stats.pnl.toFixed(2)}
                                        </span>
                                    </div>
                                    <div className="w-full bg-[#30363d] h-1.5 rounded-full overflow-hidden">
                                        <div 
                                            className="bg-[#58a6ff] h-full" 
                                            style={{ width: `${(stats.wins / stats.total * 100) || 0}%` }}
                                        />
                                    </div>
                                    <div className="flex justify-between text-[8px] text-[#8b949e] mt-1 uppercase tracking-tighter">
                                        <span>Win Rate: {((stats.wins / stats.total * 100) || 0).toFixed(1)}%</span>
                                        <span>Total: {stats.total}</span>
                                    </div>
                                </div>
                            ))}
                            {Object.keys(portfolioData.model_performance).length === 0 && (
                                <div className="text-[#8b949e] text-xs italic text-center py-10">NO MODEL DATA AVAILABLE</div>
                            )}
                        </div>
                    </div>
                </div>

                {/* TRADE HISTORY TABLE */}
                <div className="bg-[#0d1117] border border-[#30363d] rounded flex flex-col">
                    <div className="p-3 border-b border-[#30363d]">
                        <h2 className="text-[11px] font-bold tracking-widest text-white uppercase text-center">Execution History Logs</h2>
                    </div>
                    <div className="p-0">
                        <table className="w-full text-left border-collapse">
                            <thead>
                                <tr className="border-b border-[#30363d] bg-[#06090f]">
                                    <th className="p-3 text-[9px] font-bold text-[#8b949e] tracking-widest uppercase">Timestamp</th>
                                    <th className="p-3 text-[9px] font-bold text-[#8b949e] tracking-widest uppercase">Symbol</th>
                                    <th className="p-3 text-[9px] font-bold text-[#8b949e] tracking-widest uppercase">Type</th>
                                    <th className="p-3 text-[9px] font-bold text-[#8b949e] tracking-widest uppercase text-right">Price</th>
                                    <th className="p-3 text-[9px] font-bold text-[#8b949e] tracking-widest uppercase text-center">Status</th>
                                    <th className="p-3 text-[9px] font-bold text-[#8b949e] tracking-widest uppercase text-right">PnL (USD)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {portfolioData.trades.map((trade) => (
                                    <React.Fragment key={trade.id}>
                                        <tr 
                                            onClick={() => setExpandedTrade(expandedTrade === trade.id ? null : trade.id)}
                                            className={`border-b border-[#30363d]/50 hover:bg-[#161b22] transition-colors cursor-pointer ${expandedTrade === trade.id ? 'bg-[#161b22]' : ''}`}
                                        >
                                            <td className="p-3 text-[10px] font-mono text-[#c9d1d9]">
                                                {new Date(trade.timestamp).toLocaleString()}
                                            </td>
                                            <td className="p-3 text-[10px] font-bold text-white tracking-widest">{trade.symbol}</td>
                                            <td className={`p-3 text-[10px] font-bold tracking-widest ${trade.direction === 'BUY' ? 'text-[#3fb950]' : 'text-[#f85149]'}`}>
                                                {trade.direction}
                                            </td>
                                            <td className="p-3 text-[10px] font-mono text-[#c9d1d9] text-right">{trade.price.toFixed(5)}</td>
                                            <td className="p-3 text-[9px] font-bold text-center">
                                                <span className={`px-2 py-0.5 rounded ${trade.status === 'OPEN' ? 'bg-[#58a6ff]/20 text-[#58a6ff]' : 'bg-[#8b949e]/20 text-[#8b949e]'}`}>
                                                    {trade.status || "OPEN"}
                                                </span>
                                            </td>
                                            <td className={`p-3 text-[10px] font-mono text-right ${trade.pnl >= 0 ? 'text-[#3fb950]' : 'text-[#f85149]'}`}>
                                                {trade.pnl >= 0 ? '+' : ''}{(trade.pnl || 0).toFixed(2)}
                                            </td>
                                        </tr>
                                        {expandedTrade === trade.id && (
                                            <tr className="bg-[#06090f]/50 border-b border-[#30363d]">
                                                <td colSpan="6" className="p-4">
                                                    <div className="grid grid-cols-4 gap-6">
                                                        <div className="flex flex-col">
                                                            <span className="text-[8px] text-[#8b949e] uppercase font-bold tracking-widest mb-1">Execution Model</span>
                                                            <span className="text-xs text-white font-mono">{trade.model_used}</span>
                                                        </div>
                                                        <div className="flex flex-col">
                                                            <span className="text-[8px] text-[#8b949e] uppercase font-bold tracking-widest mb-1">Lot Size</span>
                                                            <span className="text-xs text-white font-mono">{trade.lot_size}</span>
                                                        </div>
                                                        <div className="flex flex-col">
                                                            <span className="text-[8px] text-[#8b949e] uppercase font-bold tracking-widest mb-1">MT5 Ticket</span>
                                                            <span className="text-xs text-white font-mono">#{trade.mt5_order_ticket || "N/A"}</span>
                                                        </div>
                                                        <div className="flex flex-col">
                                                            <span className="text-[8px] text-[#8b949e] uppercase font-bold tracking-widest mb-1">Strategy Type</span>
                                                            <span className="text-xs text-[#58a6ff] font-bold tracking-widest uppercase">Triple Barrier (ATR)</span>
                                                        </div>
                                                    </div>
                                                </td>
                                            </tr>
                                        )}
                                    </React.Fragment>
                                ))}
                                {portfolioData.trades.length === 0 && (
                                    <tr>
                                        <td colSpan="6" className="p-8 text-center text-[#8b949e] text-[10px] tracking-widest italic uppercase">
                                            NO LIVE EXECUTIONS DETECTED IN LOGS
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
