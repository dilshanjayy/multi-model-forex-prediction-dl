import { useState } from "react";
import toast from "react-hot-toast";

export default function Register({ setView }) {
    const [username, setUsername] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");

    const handleRegister = async (e) => {
        e.preventDefault();
        try {
            const res = await fetch("http://localhost:8000/api/v1/auth/register", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ username, email, password }),
            });

            if (res.ok) {
                toast.success("Registration successful! Please login.");
                setView("login");
            } else {
                const err = await res.json();
                toast.error(err.detail || "Registration failed");
            }
        } catch (error) {
            console.error(error);
            toast.error("Network error");
        }
    };

    return (
        <div className="flex flex-col items-center justify-center h-screen bg-[#06090f] text-white">
            <div className="w-96 p-8 bg-[#0d1117] border border-[#30363d] rounded shadow-lg">
                <div className="flex items-center space-x-2 mb-6 justify-center">
                    <div className="w-3 h-3 bg-[#3fb950] rounded-full shadow-[0_0_8px_#3fb950]"></div>
                    <h1 className="text-xl font-black tracking-tighter text-white">
                        FALCON.OS
                    </h1>
                </div>
                <h2 className="text-center text-lg font-bold mb-4 tracking-widest text-[#8b949e]">NEW OPERATOR</h2>
                <form onSubmit={handleRegister} className="flex flex-col space-y-4">
                    <input
                        type="text"
                        placeholder="Username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        className="bg-[#06090f] border border-[#30363d] p-2 rounded text-sm focus:outline-none focus:border-[#58a6ff] text-white"
                        required
                    />
                    <input
                        type="email"
                        placeholder="Email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        className="bg-[#06090f] border border-[#30363d] p-2 rounded text-sm focus:outline-none focus:border-[#58a6ff] text-white"
                        required
                    />
                    <input
                        type="password"
                        placeholder="Password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="bg-[#06090f] border border-[#30363d] p-2 rounded text-sm focus:outline-none focus:border-[#58a6ff] text-white"
                        required
                    />
                    <button
                        type="submit"
                        className="bg-[#58a6ff] text-[#06090f] font-bold tracking-widest py-2 rounded mt-2 hover:bg-[#388bfd] transition-colors"
                    >
                        INITIALIZE PROFILE
                    </button>
                </form>
                <div className="mt-4 text-center">
                    <button 
                        onClick={() => setView('login')} 
                        className="text-[#8b949e] text-xs hover:text-white transition-colors tracking-widest"
                    >
                        &lt; RETURN TO LOGIN
                    </button>
                </div>
            </div>
        </div>
    );
}
