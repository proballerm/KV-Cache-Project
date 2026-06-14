"use client";

import { useEffect, useState } from "react";

export default function Home() {
  const [player, setPlayer] = useState({ x: 110, y: 330 });
  const [message, setMessage] = useState("");
  const [npcResponse, setNpcResponse] = useState(
    "Hi! I am Elder Nova. Walk closer and ask me about the dragon, the cave, or the castle!"
  );
  const [latency, setLatency] = useState<number | null>(null);
  const [cacheMode, setCacheMode] = useState("KV Cache ON");
  const [loading, setLoading] = useState(false);

  const npc = { x: 690, y: 300 };
  const speed = 18;

  const distance = Math.sqrt(
    Math.pow(player.x - npc.x, 2) + Math.pow(player.y - npc.y, 2)
  );

  const isNearNpc = distance < 130;

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      setPlayer((prev) => {
        let nextX = prev.x;
        let nextY = prev.y;

        if (event.key === "ArrowUp" || event.key.toLowerCase() === "w") nextY -= speed;
        if (event.key === "ArrowDown" || event.key.toLowerCase() === "s") nextY += speed;
        if (event.key === "ArrowLeft" || event.key.toLowerCase() === "a") nextX -= speed;
        if (event.key === "ArrowRight" || event.key.toLowerCase() === "d") nextX += speed;

        nextX = Math.max(30, Math.min(900, nextX));
        nextY = Math.max(90, Math.min(450, nextY));

        return { x: nextX, y: nextY };
      });
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);


  async function handleSend() {
  if (!message.trim() || !isNearNpc) return;

  setLoading(true);

  try {
    const response = await fetch(
      "http://localhost:8000/chat",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
          use_cache: cacheMode === "KV Cache ON",
        }),
      }
    );

    const data = await response.json();

    setNpcResponse(data.response);
    setLatency(data.latency_ms);
  } catch (error) {
    setNpcResponse(
      "Could not connect to backend."
    );
  }

  setMessage("");
  setLoading(false);
}

  return (
    <main className="min-h-screen bg-gradient-to-b from-sky-300 via-cyan-100 to-lime-100 p-8 text-slate-900">
      <div className="mx-auto max-w-6xl">
        <div className="mb-6 rounded-3xl border-4 border-white bg-white/80 p-6 shadow-xl">
          <p className="text-sm font-bold uppercase tracking-[0.25em] text-purple-600">
            Week 4 Prototype
          </p>
          <h1 className="text-5xl font-black text-blue-700">
            CacheQuest: AI NPC Demo
          </h1>
          <p className="mt-2 text-lg font-medium text-slate-700">
            Move around the cartoon world, talk to the AI Elder, and compare fast vs slow responses.
          </p>
        </div>

        <section className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_330px]">
          <div className="rounded-3xl border-4 border-white bg-white/80 p-5 shadow-2xl">
            <div
              className="relative h-[560px] overflow-hidden rounded-3xl border-4 border-green-500 bg-green-300"
              style={{
                backgroundImage:
                  "linear-gradient(rgba(255,255,255,0.35) 2px, transparent 2px), linear-gradient(90deg, rgba(255,255,255,0.35) 2px, transparent 2px)",
                backgroundSize: "48px 48px",
              }}
            >
              {/* sky / grass decorations */}
              <div className="absolute left-8 top-8 text-5xl">☀️</div>
              <div className="absolute right-10 top-8 text-5xl">☁️</div>
              <div className="absolute left-36 top-12 text-4xl">☁️</div>

              {/* paths */}
              <div className="absolute left-0 top-[265px] h-28 w-full bg-yellow-200 border-y-4 border-yellow-400" />
              <div className="absolute left-[420px] top-0 h-full w-28 bg-yellow-200 border-x-4 border-yellow-400" />

              {/* places */}
              <div className="absolute left-24 top-130 text-center">
                <div className="text-6xl">🌲</div>
                <p className="rounded-full bg-green-700 px-3 py-1 text-sm font-bold text-white">
                  Forest
                </p>
              </div>

              <div className="absolute right-20 top-24 text-center">
                <div className="text-7xl">🏰</div>
                <p className="rounded-full bg-indigo-600 px-3 py-1 text-sm font-bold text-white">
                  Castle
                </p>
              </div>

              <div className="absolute left-[390px] bottom-20 text-center">
                <div className="text-7xl">🕳️</div>
                <p className="rounded-full bg-stone-700 px-3 py-1 text-sm font-bold text-white">
                  Cave
                </p>
              </div>

              <div className="absolute left-[520px] top-[120px] rounded-2xl border-4 border-orange-400 bg-orange-100 px-4 py-3 text-sm font-bold text-orange-900 shadow-lg">
                Quest: Ask the AI Elder why the dragon guards the cave.
              </div>

              <div className="absolute left-6 bottom-6 rounded-2xl border-4 border-blue-300 bg-white px-4 py-3 text-sm font-bold text-blue-700 shadow-lg">
                Move: WASD / Arrow Keys
              </div>

              {isNearNpc && (
                <div className="absolute left-[570px] top-[205px] animate-bounce rounded-2xl border-4 border-pink-400 bg-white px-4 py-2 text-sm font-black text-pink-600 shadow-xl">
                  You can talk now!
                </div>
              )}

              {/* player */}
              <div
                className="absolute text-center transition-all duration-100"
                style={{ left: player.x, top: player.y }}
              >
                <div className="flex h-20 w-20 items-center justify-center rounded-full border-4 border-white bg-blue-400 text-5xl shadow-xl">
                  🧢
                </div>
                <p className="mt-1 rounded-full bg-blue-600 px-3 py-1 text-xs font-bold text-white">
                  Player
                </p>
              </div>

              {/* npc */}
              <div
                className="absolute text-center"
                style={{ left: npc.x, top: npc.y }}
              >
                <div className="flex h-20 w-20 animate-pulse items-center justify-center rounded-full border-4 border-white bg-purple-400 text-5xl shadow-xl">
                  🧙‍♂️
                </div>
                <p className="mt-1 rounded-full bg-purple-600 px-3 py-1 text-xs font-bold text-white">
                  AI Elder
                </p>
              </div>
            </div>

            {/* dialogue on same page */}
            <div className="mt-5 rounded-3xl border-4 border-purple-300 bg-purple-100 p-5 shadow-lg">
              <h2 className="text-2xl font-black text-purple-700">
                Talk to the AI Elder
              </h2>

              <div className="mt-3 rounded-2xl border-4 border-white bg-white p-4">
                <p className="text-sm font-bold text-purple-500">NPC Response</p>
                <p className="mt-1 text-lg font-medium">
                  {loading ? "Thinking..." : npcResponse}
                </p>
              </div>

              <div className="mt-4 flex gap-3">
                <input
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleSend();
                  }}
                  disabled={!isNearNpc || loading}
                  placeholder={
                    isNearNpc
                      ? "Ask about the dragon, cave, castle, or treasure..."
                      : "Move closer to the AI Elder first..."
                  }
                  className="flex-1 rounded-2xl border-4 border-purple-300 bg-white px-4 py-3 font-medium outline-none disabled:bg-slate-200"
                />

                <button
                  onClick={handleSend}
                  disabled={!isNearNpc || loading}
                  className="rounded-2xl border-4 border-white bg-pink-500 px-6 py-3 font-black text-white shadow-lg hover:bg-pink-400 disabled:bg-slate-400"
                >
                  Send
                </button>
              </div>
            </div>
          </div>

          {/* side metrics */}
          <aside className="rounded-3xl border-4 border-white bg-white/85 p-5 shadow-2xl">
            <h2 className="text-3xl font-black text-blue-700">Metrics</h2>

            <div className="mt-5 space-y-4">
              <div className="rounded-2xl border-4 border-blue-200 bg-blue-100 p-4">
                <p className="text-sm font-bold text-blue-600">Cache Mode</p>
                <select
                  value={cacheMode}
                  onChange={(e) => setCacheMode(e.target.value)}
                  className="mt-2 w-full rounded-xl border-2 border-blue-300 bg-white p-2 font-bold"
                >
                  <option>KV Cache ON</option>
                  <option>KV Cache OFF</option>
                </select>
              </div>

              <div className="rounded-2xl border-4 border-green-200 bg-green-100 p-4">
                <p className="text-sm font-bold text-green-700">Response Latency</p>
                <p className="text-3xl font-black text-green-800">
                  {latency === null ? "Not measured" : `${latency} ms`}
                </p>
              </div>

              <div className="rounded-2xl border-4 border-yellow-200 bg-yellow-100 p-4">
                <p className="text-sm font-bold text-yellow-700">Response Speed</p>
                <p className="text-xl font-black text-yellow-800">
                  {cacheMode === "KV Cache ON" ? "Fast Mode ⚡" : "Slow Mode 🐢"}
                </p>
              </div>

              <div className="rounded-2xl border-4 border-pink-200 bg-pink-100 p-4">
                <p className="text-sm font-bold text-pink-700">Goal</p>
                <p className="text-sm font-medium">
                  This demo shows how response speed can change the feel of talking to an AI game character.
                </p>
              </div>

              <button
                onClick={() => setPlayer({ x: 110, y: 330 })}
                className="w-full rounded-2xl border-4 border-white bg-orange-400 px-5 py-3 font-black text-white shadow-lg hover:bg-orange-300"
              >
                Reset Player
              </button>
            </div>
          </aside>
        </section>
      </div>
    </main>
  );
}