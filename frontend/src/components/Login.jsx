import { useState } from "react";
import { useStore } from "../store";
import toast from "react-hot-toast";

export default function Login({ setView }) {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const setToken = useStore((state) => state.setToken);

    const handleLogin = async (e) => {
        e.preventDefault();
        try {
            const formData = new URLSearchParams();
            formData.append("username", username);
            formData.append("password", password);

            const res = await fetch("http://localhost:8000/api/v1/auth/login", {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                body: formData,
            });

            if (res.ok) {
                const data = await res.json();
                setToken(data.access_token);
                toast.success("Login successful!");
            } else {
                const err = await res.json();
                toast.error(err.detail || "Login failed");
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
                    <div className="w-3 h-3 bg-[#3fb950] rounded-full animate-pulse shadow-[0_0_8px_#3fb950]"></div>
                    <h1 className="text-xl font-black tracking-tighter text-white">
                        FALCON.OS
                    </h1>
                </div>
                <h2 className="text-center text-lg font-bold mb-4 tracking-widest text-[#8b949e]">SYSTEM LOGIN</h2>
                <form onSubmit={handleLogin} className="flex flex-col space-y-4">
                    <input
                        type="text"
                        placeholder="Username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
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
                        className="bg-[#3fb950] text-[#06090f] font-bold tracking-widest py-2 rounded mt-2 hover:bg-[#2ea043] transition-colors"
                    >
                        AUTHENTICATE
                    </button>
                </form>
                <div className="mt-4 text-center">
                    <button 
                        onClick={() => setView('register')} 
                        className="text-[#58a6ff] text-xs hover:underline tracking-widest"
                    >
                        REQUEST ACCESS (REGISTER)
                    </button>
                </div>
            </div>
        </div>
    );
}
