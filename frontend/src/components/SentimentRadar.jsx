import React from "react";

export default function SentimentRadar({ liveData, modelDetails }) {
    const signal = liveData?.signal || "HOLD";
    const confidence = liveData?.confidence || 0;
    const predictionClass = liveData?.prediction_class;
    const isMultimodal = liveData?.is_multi_modal;

    // Use the real sentiment score from the backend
    const sentimentScore = liveData?.sentiment_score || 0;

    const getStatusColor = () => {
        if (predictionClass === 0) return "#3fb950"; // BUY
        if (predictionClass === 1) return "#f85149"; // SELL
        return "#8b949e"; // HOLD
    };

    const color = getStatusColor();

    return (
        <div className="pane border-b border-[#30363d] flex-1">
            <div className="pane-header">SENTIMENT ANALYSIS</div>
            <div className="pane-content flex flex-col items-center justify-center space-y-4 py-4 h-full">
                {!isMultimodal ? (
                    <div className="flex flex-col items-center justify-center space-y-3 opacity-50 my-auto">
                        <svg
                            className="w-12 h-12 text-[#8b949e]"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={1}
                                d="M18.364 5.636l-3.536 3.536m0 5.656l3.536 3.536M9.172 9.172L5.636 5.636m3.536 9.192l-3.536 3.536M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-5 0a4 4 0 11-8 0 4 4 0 018 0z"
                            />
                        </svg>
                        <p className="text-[10px] font-bold text-[#8b949e] tracking-widest text-center">
                            RADAR OFFLINE
                            <br />
                            <span className="text-[8px] font-normal">
                                TECHNICAL MODEL SELECTED
                            </span>
                        </p>
                    </div>
                ) : (
                    <>
                        {/* Gauge */}
                        <div className="relative w-32 h-32 flex items-center justify-center">
                            {/* Background Track */}
                            <svg className="absolute inset-0 w-full h-full -rotate-90">
                                <circle
                                    cx="64"
                                    cy="64"
                                    r="54"
                                    fill="none"
                                    stroke="#30363d"
                                    strokeWidth="6"
                                    strokeDasharray="339.3"
                                    strokeDashoffset="0"
                                    strokeLinecap="round"
                                    opacity="0.3"
                                />
                            </svg>

                            {/* Actual Sentiment Line */}
                            <svg className="absolute inset-0 w-full h-full -rotate-90 transition-all duration-1000 ease-in-out">
                                <circle
                                    cx="64"
                                    cy="64"
                                    r="54"
                                    fill="none"
                                    stroke={
                                        sentimentScore > 0
                                            ? "#3fb950"
                                            : sentimentScore < 0
                                              ? "#f85149"
                                              : "#8b949e"
                                    }
                                    strokeWidth="6"
                                    strokeDasharray="339.3"
                                    strokeDashoffset={
                                        339.3 - Math.abs(sentimentScore) * 169.6
                                    }
                                    strokeLinecap="round"
                                    style={{
                                        filter: `drop-shadow(0 0 3px ${sentimentScore > 0 ? "#3fb950" : sentimentScore < 0 ? "#f85149" : "#8b949e"})`,
                                        transform:
                                            sentimentScore < 0
                                                ? "scaleY(-1)"
                                                : "none",
                                        transformOrigin: "center",
                                    }}
                                />
                            </svg>

                            {/* Inner Depth Shadow */}
                            <div className="absolute w-24 h-24 rounded-full bg-[#0d1117] shadow-[inset_0_0_20px_rgba(0,0,0,0.5)] flex items-center justify-center">
                                <div className="text-center z-10">
                                    <p className="text-[8px] text-[#8b949e] uppercase font-black tracking-widest opacity-80">
                                        NET ALPHA
                                    </p>
                                    <p className="text-lg font-bold value-mono text-white leading-none mt-1">
                                        {(sentimentScore > 0 ? "+" : "") +
                                            (sentimentScore * 100).toFixed(0)}
                                    </p>
                                </div>
                            </div>
                        </div>

                        <div className="w-full px-4 space-y-2">
                            <div className="flex justify-between items-center text-[9px] font-bold">
                                <span className="text-[#8b949e]">
                                    SIGNAL SOURCE
                                </span>
                                <span className="text-[#58a6ff]">
                                    HYBRID (T+S)
                                </span>
                            </div>
                            <div className="flex justify-between items-center text-[9px] font-bold">
                                <span className="text-[#8b949e]">
                                    DATA PIPELINE
                                </span>
                                <span className="text-[#3fb950]">ACTIVE</span>
                            </div>
                        </div>

                        <div className="mx-4 text-center py-1 px-2 rounded bg-[#23863615] border border-[#23863640]">
                            <p className="text-[8px] font-bold text-[#3fb950]">
                                SENTIMENT CONFIRMED
                            </p>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}
