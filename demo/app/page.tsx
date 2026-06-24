"use client";

import { useEffect, useRef, useState } from "react";

type Team = "cache" | "nocache";

type Bot = {
  id: number;
  team: Team;
  name: string;
  emoji: string;
  role: string;
  x: number;
  y: number;
  targetX: number;
  targetY: number;
  goalIndex: number;
  goalsCompleted: number;
  actionsTaken: number;
  latencyTotal: number;
  lastLatency: number | null;
  tokensPerSecond: number | null;
  thought: string;
  isThinking: boolean;
};

type LogEntry = {
  id: number;
  team: Team;
  text: string;
  latency: number;
};

const backendUrl =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

const locations = [
  { name: "Village", x: 210, y: 270, emoji: "🏘️" },
  { name: "Forest", x: 95, y: 135, emoji: "🌲" },
  { name: "Castle", x: 675, y: 120, emoji: "🏰" },
  { name: "Cave", x: 420, y: 420, emoji: "🕳️" },
  { name: "Crystal", x: 705, y: 420, emoji: "💎" },
];

const goalPath = ["Village", "Forest", "Castle", "Cave", "Crystal", "Village"];

const startingBots: Bot[] = [
  {
    id: 1,
    team: "cache",
    name: "Blue Wizard",
    emoji: "🧙‍♂️",
    role: "KV Cache ON",
    x: 120,
    y: 250,
    targetX: 210,
    targetY: 270,
    goalIndex: 0,
    goalsCompleted: 0,
    actionsTaken: 0,
    latencyTotal: 0,
    lastLatency: null,
    tokensPerSecond: null,
    thought: "Ready to complete objectives with KV Cache.",
    isThinking: false,
  },
  {
    id: 2,
    team: "cache",
    name: "Blue Guard",
    emoji: "🛡️",
    role: "KV Cache ON",
    x: 120,
    y: 330,
    targetX: 210,
    targetY: 270,
    goalIndex: 0,
    goalsCompleted: 0,
    actionsTaken: 0,
    latencyTotal: 0,
    lastLatency: null,
    tokensPerSecond: null,
    thought: "Ready to patrol quickly.",
    isThinking: false,
  },
  {
    id: 3,
    team: "nocache",
    name: "Red Wizard",
    emoji: "🧙‍♀️",
    role: "KV Cache OFF",
    x: 820,
    y: 250,
    targetX: 210,
    targetY: 270,
    goalIndex: 0,
    goalsCompleted: 0,
    actionsTaken: 0,
    latencyTotal: 0,
    lastLatency: null,
    tokensPerSecond: null,
    thought: "Ready, but every thought is slower without cache.",
    isThinking: false,
  },
  {
    id: 4,
    team: "nocache",
    name: "Red Guard",
    emoji: "🛡️",
    role: "KV Cache OFF",
    x: 820,
    y: 330,
    targetX: 210,
    targetY: 270,
    goalIndex: 0,
    goalsCompleted: 0,
    actionsTaken: 0,
    latencyTotal: 0,
    lastLatency: null,
    tokensPerSecond: null,
    thought: "Ready, but no KV cache means slower decisions.",
    isThinking: false,
  },
];

export default function Home() {
  const [bots, setBots] = useState<Bot[]>(startingBots);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [running, setRunning] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(60);
  const [backendStatus, setBackendStatus] = useState("Not checked");

  const movementRef = useRef<NodeJS.Timeout | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  async function checkBackend() {
    try {
      const response = await fetch(`${backendUrl}/`);
      const data = await response.json();
      setBackendStatus(data.status || "Backend connected");
    } catch {
      setBackendStatus("Backend not connected");
    }
  }

  function getLocation(name: string) {
    return locations.find((location) => location.name === name) || locations[0];
  }

  function getTeamStats(team: Team) {
    const teamBots = bots.filter((bot) => bot.team === team);

    const goals = teamBots.reduce((sum, bot) => sum + bot.goalsCompleted, 0);
    const actions = teamBots.reduce((sum, bot) => sum + bot.actionsTaken, 0);
    const latencyTotal = teamBots.reduce(
      (sum, bot) => sum + bot.latencyTotal,
      0
    );

    const avgLatency = actions === 0 ? null : Math.round(latencyTotal / actions);

    const avgTpsValues = teamBots
      .map((bot) => bot.tokensPerSecond)
      .filter((value): value is number => value !== null);

    const avgTps =
      avgTpsValues.length === 0
        ? null
        : Math.round(
            avgTpsValues.reduce((sum, value) => sum + value, 0) /
              avgTpsValues.length
          );

    return { goals, actions, avgLatency, avgTps };
  }

  async function completeBotDecision(bot: Bot) {
    const currentGoal = goalPath[bot.goalIndex];
    const nextGoal = goalPath[(bot.goalIndex + 1) % goalPath.length];

    const message =
      `${bot.name} is trying to finish objectives in an AI village race. ` +
      `Current location: ${currentGoal}. Next objective: ${nextGoal}. ` +
      `Team mode: ${bot.role}. Explain the bot's next action briefly. ` +
      `Mention dragon, cave, castle, treasure, or crystal if useful.`;

    setBots((prev) =>
      prev.map((item) =>
        item.id === bot.id ? { ...item, isThinking: true } : item
      )
    );

    try {
      const response = await fetch(`${backendUrl}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
          use_cache: bot.team === "cache",
        }),
      });

      const data = await response.json();

      const newGoalIndex = (bot.goalIndex + 1) % goalPath.length;
      const newTarget = getLocation(goalPath[newGoalIndex]);
      const completedFullLoop = newGoalIndex === 0;

      setBots((prev) =>
        prev.map((item) =>
          item.id === bot.id
            ? {
                ...item,
                goalIndex: newGoalIndex,
                targetX: newTarget.x,
                targetY: newTarget.y,
                goalsCompleted: completedFullLoop
                  ? item.goalsCompleted + 1
                  : item.goalsCompleted,
                actionsTaken: item.actionsTaken + 1,
                latencyTotal: item.latencyTotal + data.latency_ms,
                lastLatency: data.latency_ms,
                tokensPerSecond: data.tokens_per_second,
                thought: data.response,
                isThinking: false,
              }
            : item
        )
      );

      setLogs((prev) => [
        {
          id: Date.now() + Math.random(),
          team: bot.team,
          text: `${bot.name} reached ${currentGoal} → heading to ${nextGoal}. ${data.response}`,
          latency: data.latency_ms,
        },
        ...prev.slice(0, 14),
      ]);
    } catch {
      setBots((prev) =>
        prev.map((item) =>
          item.id === bot.id
            ? {
                ...item,
                thought: "Backend request failed.",
                isThinking: false,
              }
            : item
        )
      );
    }
  }

  function updateMovement() {
    setBots((prevBots) => {
      const updatedBots = prevBots.map((bot) => {
        const dx = bot.targetX - bot.x;
        const dy = bot.targetY - bot.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < 8 && !bot.isThinking) {
          completeBotDecision(bot);
          return bot;
        }

        if (distance < 8) return bot;

        const speed = bot.team === "cache" ? 0.075 : 0.045;

        return {
          ...bot,
          x: bot.x + dx * speed,
          y: bot.y + dy * speed,
        };
      });

      return updatedBots;
    });
  }

  function startRace() {
    if (running) return;

    setRunning(true);

    movementRef.current = setInterval(() => {
      updateMovement();
    }, 100);

    timerRef.current = setInterval(() => {
      setSecondsLeft((prev) => {
        if (prev <= 1) {
          stopRace();
          return 0;
        }

        return prev - 1;
      });
    }, 1000);
  }

  function stopRace() {
    setRunning(false);

    if (movementRef.current) {
      clearInterval(movementRef.current);
      movementRef.current = null;
    }

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  function resetRace() {
    stopRace();
    setBots(startingBots);
    setLogs([]);
    setSecondsLeft(60);
  }

  useEffect(() => {
    checkBackend();

    return () => {
      if (movementRef.current) clearInterval(movementRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const cacheStats = getTeamStats("cache");
  const noCacheStats = getTeamStats("nocache");

  let winner = "Race in progress";
  if (secondsLeft === 0 || !running) {
    if (cacheStats.goals > noCacheStats.goals) winner = "🏆 KV Cache ON wins";
    else if (noCacheStats.goals > cacheStats.goals)
      winner = "🏆 KV Cache OFF wins";
    else if (cacheStats.actions > noCacheStats.actions)
      winner = "🏆 KV Cache ON leads by actions";
    else if (noCacheStats.actions > cacheStats.actions)
      winner = "🏆 KV Cache OFF leads by actions";
    else winner = "Tie so far";
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-sky-300 via-cyan-100 to-lime-100 p-8 text-slate-900">
      <div className="mx-auto max-w-7xl">
        <div className="mb-6 rounded-3xl border-4 border-white bg-white/85 p-6 shadow-xl">
          <p className="text-sm font-bold uppercase tracking-[0.25em] text-purple-600">
            Version 3 Demo
          </p>

          <h1 className="text-5xl font-black text-blue-700">
            CacheQuest: AI Agent Race
          </h1>

          <p className="mt-2 text-lg font-medium text-slate-700">
            Two AI teams race to complete objectives. Blue agents use KV Cache.
            Red agents do not. Lower inference latency means faster decisions
            and more completed goals.
          </p>
        </div>

        <section className="grid grid-cols-1 gap-6 xl:grid-cols-[1fr_390px]">
          <div className="rounded-3xl border-4 border-white bg-white/80 p-5 shadow-2xl">
            <div
              className="relative h-[640px] overflow-hidden rounded-3xl border-4 border-green-500 bg-green-300"
              style={{
                backgroundImage:
                  "linear-gradient(rgba(255,255,255,0.35) 2px, transparent 2px), linear-gradient(90deg, rgba(255,255,255,0.35) 2px, transparent 2px)",
                backgroundSize: "48px 48px",
              }}
            >
              <div className="absolute left-8 top-8 text-5xl">☀️</div>
              <div className="absolute right-10 top-8 text-5xl">☁️</div>
              <div className="absolute left-36 top-12 text-4xl">☁️</div>

              <div className="absolute left-0 top-[300px] h-28 w-full border-y-4 border-yellow-400 bg-yellow-200" />
              <div className="absolute left-[440px] top-0 h-full w-28 border-x-4 border-yellow-400 bg-yellow-200" />

              {locations.map((location) => (
                <div
                  key={location.name}
                  className="absolute text-center"
                  style={{ left: location.x, top: location.y }}
                >
                  <div className="text-6xl">{location.emoji}</div>
                  <p className="rounded-full bg-slate-800 px-3 py-1 text-sm font-bold text-white">
                    {location.name}
                  </p>
                </div>
              ))}

              {bots.map((bot) => (
                <div
                  key={bot.id}
                  className="absolute text-center transition-all duration-100"
                  style={{ left: bot.x, top: bot.y }}
                >
                  <div
                    className={`flex h-20 w-20 items-center justify-center rounded-full border-4 border-white text-5xl shadow-xl ${
                      bot.team === "cache" ? "bg-blue-500" : "bg-red-500"
                    }`}
                  >
                    {bot.emoji}
                  </div>

                  <p
                    className={`mt-1 rounded-full px-3 py-1 text-xs font-bold text-white ${
                      bot.team === "cache" ? "bg-blue-700" : "bg-red-700"
                    }`}
                  >
                    {bot.name}
                  </p>

                  {bot.isThinking && (
                    <p className="mt-1 animate-pulse rounded-full bg-yellow-300 px-2 py-1 text-xs font-black text-yellow-900">
                      Thinking...
                    </p>
                  )}
                </div>
              ))}

              <div className="absolute bottom-5 left-5 rounded-2xl border-4 border-white bg-blue-100 p-4 shadow-xl">
                <p className="text-lg font-black text-blue-700">
                  🔵 KV Cache ON
                </p>
                <p className="text-sm font-bold">
                  Faster decisions, more actions
                </p>
              </div>

              <div className="absolute bottom-5 right-5 rounded-2xl border-4 border-white bg-red-100 p-4 shadow-xl">
                <p className="text-lg font-black text-red-700">
                  🔴 KV Cache OFF
                </p>
                <p className="text-sm font-bold">
                  Slower decisions, fewer actions
                </p>
              </div>
            </div>

            <div className="mt-5 rounded-3xl border-4 border-purple-300 bg-purple-100 p-5 shadow-lg">
              <h2 className="text-2xl font-black text-purple-700">
                AI Decision Log
              </h2>

              <div className="mt-3 max-h-72 space-y-3 overflow-y-auto rounded-2xl border-4 border-white bg-white p-4">
                {logs.length === 0 ? (
                  <p className="font-medium text-slate-500">
                    Start the race to watch autonomous agents make decisions.
                  </p>
                ) : (
                  logs.map((log) => (
                    <div
                      key={log.id}
                      className={`rounded-xl border-2 p-3 ${
                        log.team === "cache"
                          ? "border-blue-200 bg-blue-50"
                          : "border-red-200 bg-red-50"
                      }`}
                    >
                      <p className="text-sm font-medium">{log.text}</p>
                      <p
                        className={`mt-1 text-xs font-bold ${
                          log.team === "cache"
                            ? "text-blue-700"
                            : "text-red-700"
                        }`}
                      >
                        {log.team === "cache"
                          ? "KV Cache ON"
                          : "KV Cache OFF"}{" "}
                        · {log.latency} ms
                      </p>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          <aside className="rounded-3xl border-4 border-white bg-white/85 p-5 shadow-2xl">
            <h2 className="text-3xl font-black text-blue-700">Race Metrics</h2>

            <div className="mt-5 space-y-4">
              <div className="rounded-2xl border-4 border-blue-200 bg-blue-100 p-4">
                <p className="text-sm font-bold text-blue-600">Backend</p>
                <p className="text-lg font-black text-blue-800">
                  {backendStatus}
                </p>
              </div>

              <div className="rounded-2xl border-4 border-yellow-300 bg-yellow-100 p-4 text-center">
                <p className="text-sm font-bold text-yellow-700">
                  Time Remaining
                </p>
                <p className="text-5xl font-black text-yellow-800">
                  {secondsLeft}s
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={startRace}
                  disabled={running}
                  className="rounded-2xl border-4 border-white bg-green-500 px-4 py-3 font-black text-white shadow-lg hover:bg-green-400 disabled:bg-slate-400"
                >
                  Start
                </button>

                <button
                  onClick={stopRace}
                  disabled={!running}
                  className="rounded-2xl border-4 border-white bg-red-500 px-4 py-3 font-black text-white shadow-lg hover:bg-red-400 disabled:bg-slate-400"
                >
                  Stop
                </button>
              </div>

              <button
                onClick={resetRace}
                className="w-full rounded-2xl border-4 border-white bg-orange-400 px-5 py-3 font-black text-white shadow-lg hover:bg-orange-300"
              >
                Reset Race
              </button>

              <div className="rounded-2xl border-4 border-purple-300 bg-purple-100 p-4 text-center">
                <p className="text-sm font-bold text-purple-700">Winner</p>
                <p className="text-2xl font-black text-purple-900">{winner}</p>
              </div>

              <div className="rounded-2xl border-4 border-blue-300 bg-blue-100 p-4">
                <h3 className="text-xl font-black text-blue-700">
                  🔵 KV Cache ON
                </h3>
                <p className="mt-2 font-bold">
                  Goals Completed: {cacheStats.goals}
                </p>
                <p className="font-bold">Actions Taken: {cacheStats.actions}</p>
                <p className="font-bold">
                  Avg Latency:{" "}
                  {cacheStats.avgLatency === null
                    ? "N/A"
                    : `${cacheStats.avgLatency} ms`}
                </p>
                <p className="font-bold">
                  Avg Tokens/sec:{" "}
                  {cacheStats.avgTps === null ? "N/A" : cacheStats.avgTps}
                </p>
              </div>

              <div className="rounded-2xl border-4 border-red-300 bg-red-100 p-4">
                <h3 className="text-xl font-black text-red-700">
                  🔴 KV Cache OFF
                </h3>
                <p className="mt-2 font-bold">
                  Goals Completed: {noCacheStats.goals}
                </p>
                <p className="font-bold">
                  Actions Taken: {noCacheStats.actions}
                </p>
                <p className="font-bold">
                  Avg Latency:{" "}
                  {noCacheStats.avgLatency === null
                    ? "N/A"
                    : `${noCacheStats.avgLatency} ms`}
                </p>
                <p className="font-bold">
                  Avg Tokens/sec:{" "}
                  {noCacheStats.avgTps === null ? "N/A" : noCacheStats.avgTps}
                </p>
              </div>

              <div className="rounded-2xl border-4 border-pink-200 bg-pink-100 p-4">
                <p className="text-sm font-bold text-pink-700">
                  Demo Explanation
                </p>
                <p className="text-sm font-medium">
                  Every objective requires an AI backend call. KV Cache ON has
                  lower simulated inference latency, so those agents can make
                  decisions faster and complete more work in the same time.
                </p>
              </div>

              <div className="rounded-2xl border-4 border-slate-200 bg-slate-100 p-4">
                <p className="text-sm font-bold text-slate-700">
                  Objective Path
                </p>
                <p className="text-sm font-medium">
                  Village → Forest → Castle → Cave → Crystal → Village
                </p>
              </div>
            </div>
          </aside>
        </section>
      </div>
    </main>
  );
}