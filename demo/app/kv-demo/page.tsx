"use client";

import { useState } from "react";

export default function Home() {
  const [playerMessage, setPlayerMessage] = useState("");
  const [npcResponse, setNpcResponse] = useState(
    "Hello traveler. Ask me anything about this world."
  );
  const [latency, setLatency] = useState<number | null>(null);
  const [cacheMode, setCacheMode] = useState("KV Cache ON");
  const [loading, setLoading] = useState(false);

  function generatePlaceholderResponse(message: string) {
    if (message.toLowerCase().includes("dragon")) {
      return "The dragon guards the cave because it protects an ancient energy source hidden beneath the mountain.";
    }

    if (message.toLowerCase().includes("city")) {
      return "The city gates are locked because the council detected movement outside the walls last night.";
    }

    if (message.toLowerCase().includes("treasure")) {
      return "The treasure is not just gold. It is a memory crystal that stores the history of this kingdom.";
    }

    return "That is an interesting question. The answer depends on the history of this world and the choices you make next.";
  }

  async function handleSend() {
    if (!playerMessage.trim()) return;

    setLoading(true);

    const startTime = performance.now();

    await new Promise((resolve) => setTimeout(resolve, cacheMode === "KV Cache ON" ? 500 : 1800));

    const response = generatePlaceholderResponse(playerMessage);

    const endTime = performance.now();

    setNpcResponse(response);
    setLatency(Math.round(endTime - startTime));
    setLoading(false);
    setPlayerMessage("");
  }

  return (
    <main className="min-h-screen bg-slate-950 text-white p-8">
      <div className="mx-auto max-w-5xl">
        <h1 className="text-4xl font-bold mb-2">
          AI NPC KV Cache Demo
        </h1>

        <p className="text-slate-300 mb-8">
          A simple prototype showing how AI response speed can affect gameplay feel.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <section className="md:col-span-2 rounded-2xl bg-slate-900 border border-slate-700 p-6">
            <div className="h-96 rounded-xl bg-gradient-to-b from-slate-800 to-slate-950 border border-slate-700 relative overflow-hidden">
              <div className="absolute bottom-10 left-16 text-center">
                <div className="w-16 h-16 rounded-full bg-blue-500 mx-auto mb-2"></div>
                <p className="text-sm text-slate-300">Player</p>
              </div>

              <div className="absolute bottom-10 right-16 text-center">
                <div className="w-16 h-16 rounded-full bg-purple-500 mx-auto mb-2 animate-pulse"></div>
                <p className="text-sm text-slate-300">AI NPC</p>
              </div>

              <div className="absolute top-6 left-6 right-6 rounded-xl bg-black/50 border border-slate-600 p-4">
                <p className="text-sm text-slate-400 mb-1">NPC Response</p>
                <p className="text-lg">
                  {loading ? "Thinking..." : npcResponse}
                </p>
              </div>
            </div>

            <div className="mt-6 flex gap-3">
              <input
                value={playerMessage}
                onChange={(e) => setPlayerMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSend();
                }}
                placeholder="Ask the NPC something..."
                className="flex-1 rounded-xl bg-slate-800 border border-slate-600 px-4 py-3 outline-none"
              />

              <button
                onClick={handleSend}
                className="rounded-xl bg-blue-600 px-6 py-3 font-semibold hover:bg-blue-500"
              >
                Send
              </button>
            </div>
          </section>

          <section className="rounded-2xl bg-slate-900 border border-slate-700 p-6">
            <h2 className="text-xl font-bold mb-4">Metrics</h2>

            <div className="space-y-4">
              <div className="rounded-xl bg-slate-800 p-4">
                <p className="text-sm text-slate-400">Cache Mode</p>
                <select
                  value={cacheMode}
                  onChange={(e) => setCacheMode(e.target.value)}
                  className="mt-2 w-full rounded-lg bg-slate-700 p-2"
                >
                  <option>KV Cache ON</option>
                  <option>KV Cache OFF</option>
                </select>
              </div>

              <div className="rounded-xl bg-slate-800 p-4">
                <p className="text-sm text-slate-400">Response Latency</p>
                <p className="text-2xl font-bold">
                  {latency === null ? "Not measured" : `${latency} ms`}
                </p>
              </div>

              <div className="rounded-xl bg-slate-800 p-4">
                <p className="text-sm text-slate-400">Tokens Per Second</p>
                <p className="text-2xl font-bold">
                  {cacheMode === "KV Cache ON" ? "Simulated faster" : "Simulated slower"}
                </p>
              </div>

              <div className="rounded-xl bg-slate-800 p-4">
                <p className="text-sm text-slate-400">Purpose</p>
                <p className="text-sm text-slate-300">
                  Later this panel will show real GPU inference metrics from the AMD MI300X backend.
                </p>
              </div>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}