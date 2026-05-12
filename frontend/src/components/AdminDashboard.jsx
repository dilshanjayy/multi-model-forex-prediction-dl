import React, { useState, useEffect } from "react";
import { useStore } from "../store";
import toast from "react-hot-toast";

export default function AdminDashboard() {
    const { token } = useStore();
    const [stats, setStats] = useState(null);
    const [file, setFile] = useState(null);
    const [isUploading, setIsUploading] = useState(false);

    useEffect(() => {
        if (!token) return;
        fetch("http://localhost:8000/api/v1/admin/stats", {
            headers: {
                Authorization: `Bearer ${token}`
            }
        })
        .then(res => {
            if (!res.ok) throw new Error("Failed to fetch stats");
            return res.json();
        })
        .then(data => setStats(data))
        .catch(err => console.error("Error fetching admin stats:", err));
    }, [token]);

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            setFile(e.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!file) return;
        const toastId = toast.loading("Uploading and extracting model...");
        setIsUploading(true);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("http://localhost:8000/api/v1/admin/upload_model", {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${token}`
                },
                body: formData
            });
            const data = await res.json();
            
            if (res.ok) {
                toast.success(data.message || "Model uploaded successfully!", { id: toastId });
                setFile(null);
                // Optionally reset the file input
                document.getElementById('model-upload-input').value = "";
            } else {
                throw new Error(data.detail || "Upload failed");
            }
        } catch (err) {
            console.error(err);
            toast.error(err.message || "Network error", { id: toastId });
        } finally {
            setIsUploading(false);
        }
    };

    const handleReload = async () => {
        const toastId = toast.loading("Hot-reloading model cache...");
        try {
            const res = await fetch("http://localhost:8000/api/v1/admin/reload_models", {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${token}`
                }
            });
            const data = await res.json();
            if (res.ok) {
                toast.success(data.message || "Cache reloaded successfully!", { id: toastId });
            } else {
                throw new Error(data.detail || "Reload failed");
            }
        } catch (err) {
            console.error(err);
            toast.error(err.message || "Network error", { id: toastId });
        }
    };

    return (
        <div className="max-w-6xl mx-auto space-y-6">
            <div className="flex items-center justify-between border-b border-[#30363d] pb-4">
                <div>
                    <h2 className="text-xl font-bold text-white tracking-tight">Admin Control Panel</h2>
                    <p className="text-xs text-[#8b949e]">System-wide metrics and dynamic model registry.</p>
                </div>
            </div>

            {/* Global Stats Grid */}
            <div className="grid grid-cols-4 gap-4">
                <div className="pane p-4">
                    <h3 className="text-[10px] font-bold text-[#8b949e] uppercase tracking-widest mb-1">Total Users</h3>
                    <p className="text-2xl font-bold value-mono text-white">{stats?.total_users ?? "---"}</p>
                </div>
                <div className="pane p-4">
                    <h3 className="text-[10px] font-bold text-[#8b949e] uppercase tracking-widest mb-1">Global Trades</h3>
                    <p className="text-2xl font-bold value-mono text-white">{stats?.total_trades ?? "---"}</p>
                </div>
                <div className="pane p-4">
                    <h3 className="text-[10px] font-bold text-[#8b949e] uppercase tracking-widest mb-1">Global Win Rate</h3>
                    <p className="text-2xl font-bold value-mono text-[#3fb950]">{stats?.global_win_rate ? `${stats.global_win_rate}%` : "---"}</p>
                </div>
                <div className="pane p-4">
                    <h3 className="text-[10px] font-bold text-[#8b949e] uppercase tracking-widest mb-1">Total Network PnL</h3>
                    <p className={`text-2xl font-bold value-mono ${stats?.global_pnl >= 0 ? 'text-[#3fb950]' : 'text-[#f85149]'}`}>
                        {stats?.global_pnl != null ? `$${stats.global_pnl}` : "---"}
                    </p>
                </div>
            </div>

            {/* Model Management */}
            <div className="grid grid-cols-2 gap-6">
                <div className="pane flex flex-col">
                    <div className="pane-header flex justify-between items-center">
                        <span>MODEL REGISTRY UPLOAD</span>
                    </div>
                    <div className="p-6 flex flex-col items-center justify-center border-2 border-dashed border-[#30363d] m-4 rounded-lg bg-[#0d1117]/50 hover:border-[#58a6ff] transition-colors">
                        <svg className="w-12 h-12 text-[#8b949e] mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                        <p className="text-sm font-bold text-white mb-1">Upload New Model (.zip)</p>
                        <p className="text-[10px] text-[#8b949e] text-center mb-4 max-w-xs">
                            Zip file must contain <code>config.yaml</code> and <code>model.joblib</code>. The folder name will be used as the Model ID.
                        </p>
                        <input 
                            id="model-upload-input"
                            type="file" 
                            accept=".zip" 
                            onChange={handleFileChange}
                            className="text-xs text-[#8b949e] file:mr-4 file:py-1.5 file:px-3 file:rounded file:border-0 file:text-[10px] file:font-bold file:bg-[#238636] file:text-white hover:file:bg-[#2ea043] cursor-pointer w-full max-w-xs"
                        />
                        <button 
                            onClick={handleUpload}
                            disabled={!file || isUploading}
                            className={`mt-4 w-full max-w-xs py-2 rounded text-xs font-bold transition-all ${!file || isUploading ? 'bg-[#21262d] text-[#8b949e] cursor-not-allowed' : 'bg-[#58a6ff] text-black hover:bg-[#79c0ff] shadow-[0_0_10px_rgba(88,166,255,0.3)]'}`}
                        >
                            {isUploading ? "UPLOADING..." : "DEPLOY TO NETWORK"}
                        </button>
                    </div>
                </div>

                <div className="flex flex-col gap-6">
                    <div className="pane p-6">
                        <h3 className="text-sm font-bold text-white mb-2">Most Popular Model</h3>
                        <p className="text-[10px] text-[#8b949e] mb-4">Based on historical execution volume across all users.</p>
                        <div className="flex items-center space-x-3 p-3 bg-[#161b22] border border-[#30363d] rounded">
                            <div className="w-8 h-8 rounded bg-[#3fb950]/20 flex items-center justify-center">
                                <span className="text-[#3fb950] font-bold text-sm">M</span>
                            </div>
                            <div>
                                <p className="text-xs font-bold text-white">{stats?.top_model ?? "---"}</p>
                                <p className="text-[10px] text-[#8b949e]">System Favorite</p>
                            </div>
                        </div>
                    </div>

                    <div className="pane p-6 border-[#d29922]/30 bg-[#d29922]/5">
                        <div className="flex items-center space-x-2 mb-2">
                            <svg className="w-5 h-5 text-[#d29922]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                            <h3 className="text-sm font-bold text-[#d29922]">Zero-Downtime Reload</h3>
                        </div>
                        <p className="text-[10px] text-[#8b949e] mb-4">
                            Hot-reload the inference cache. This forces the PyTorch engine to scan the <code>deployed_models</code> directory and load any newly uploaded models into RAM without dropping live user connections.
                        </p>
                        <button 
                            onClick={handleReload}
                            className="w-full py-2 bg-[#d29922] hover:bg-[#e3b341] text-black rounded text-xs font-bold transition-all shadow-[0_0_10px_rgba(210,153,34,0.3)]"
                        >
                            HOT-RELOAD INFERENCE CACHE
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
