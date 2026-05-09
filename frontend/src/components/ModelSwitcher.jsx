import { useState, useRef, useEffect } from "react";
import { useStore } from "../store";

export default function ModelSwitcher() {
    const { models, selectedModel, setSelectedModel, modelDetails } = useStore();
    const [isOpen, setIsOpen] = useState(false);
    const dropdownRef = useRef(null);

    // Close dropdown when clicking outside
    useEffect(() => {
        function handleClickOutside(event) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setIsOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    const handleSelect = (model) => {
        setSelectedModel(model);
        setIsOpen(false);
    };

    return (
        <div className="relative" ref={dropdownRef}>
            <button 
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center justify-between space-x-2 px-3 py-1 rounded bg-[#30363d30] border border-[#30363d] hover:border-[#58a6ff] transition-all group w-64"
            >
                <div className="flex flex-col items-start min-w-0">
                    <span className="text-[10px] font-bold text-[#58a6ff] uppercase tracking-tighter truncate w-full text-left">
                        {selectedModel || "SELECT MODEL"}
                    </span>
                    <div className="flex items-center space-x-1">
                        <div className="w-1 h-1 bg-[#3fb950] rounded-full"></div>
                        <span className="text-[8px] text-[#8b949e] font-black uppercase tracking-widest">
                            ACTIVE INSTANCE
                        </span>
                    </div>
                </div>
                <svg className={`w-3 h-3 text-[#8b949e] transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M19 9l-7 7-7-7" />
                </svg>
            </button>

            {isOpen && (
                <div className="absolute top-full left-0 mt-2 w-64 glass-panel rounded-md z-50 overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200">
                    <div className="px-4 py-2 border-b border-[#30363d] bg-[#161b2250]">
                        <span className="text-[9px] font-black text-[#8b949e] uppercase tracking-widest">DEPLOYED REGISTRY</span>
                    </div>
                    <div className="max-h-60 overflow-y-auto custom-scrollbar">
                        {models.map((m) => {
                            const isMultiModal = m.toLowerCase().includes('hybrid') || m.toLowerCase().includes('multi') || m.toLowerCase().includes('sentiment');
                            return (
                                <div 
                                    key={m} 
                                    onClick={() => handleSelect(m)}
                                    className={`dropdown-item ${selectedModel === m ? 'bg-[#58a6ff15] text-white border-l-2 border-[#58a6ff]' : ''}`}
                                >
                                    <div className="flex flex-col">
                                        <span className={selectedModel === m ? 'text-[#58a6ff]' : ''}>{m}</span>
                                        <span className="text-[8px] opacity-60 uppercase">
                                            {isMultiModal ? 'Multi-Modal Architecture' : 'Technical Pipeline'}
                                        </span>
                                    </div>
                                    <span className={`px-1.5 py-0.5 rounded-[2px] text-[8px] font-black ${isMultiModal ? 'bg-[#3fb95020] text-[#3fb950]' : 'bg-[#58a6ff20] text-[#58a6ff]'}`}>
                                        {isMultiModal ? 'M' : 'T'}
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}
