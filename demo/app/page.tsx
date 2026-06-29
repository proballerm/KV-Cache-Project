"use client";

import { useEffect, useRef, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Mode = "gpu" | "cpu";

type Result = {
  id: number;
  mode: Mode;
  round: number;
  latency: number;
  tps: number;
  response: string;
  contextTokens: number;
};

const backendUrl =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

const basePrompt = `
You are an AI NPC in a fantasy game.
The player is exploring a village, forest, castle, cave, and crystal chamber.
A dragon guards the cave because it protects an ancient crystal.
`;

const tasks = [
  "Explain why the dragon guards the cave.",
  "Explain how the crystal powers the village.",
  "Explain what the player should do next.",
  "Explain why the castle gates are locked.",
  "Explain how the forest connects to the cave.",
  "Explain the danger near the treasure.",
  "Explain how the NPC guides the player.",
  "Explain the secret history of the crystal.",
];

export default function Home() {
  const [backendStatus, setBackendStatus] = useState("Not checked");
  const [running, setRunning] = useState(false);
  const [round, setRound] = useState(0);
  const [secondsLeft, setSecondsLeft] = useState(60);
  const [logs, setLogs] = useState<Result[]>([]);
  const [gpuThinking, setGpuThinking] = useState(false);
  const [cpuThinking, setCpuThinking] = useState(false);

  const runningRef = useRef(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  async function checkBackend() {
    try {
      const res = await fetch(`${backendUrl}/`);
      const data = await res.json();
      setBackendStatus(data.status || "Backend connected");
    } catch {
      setBackendStatus("Backend not connected");
    }
  }

  function buildPrompt(currentRound: number) {
    const history = Array.from({ length: currentRound + 1 })
      .map((_, i) => `Round ${i + 1}: ${tasks[i % tasks.length]}`)
      .join("\n");

    return `${basePrompt}

Conversation history:
${history}

Answer the latest task in one short sentence.`;
  }

  async function runInference(mode: Mode, currentRound: number) {
    if (mode === "gpu") setGpuThinking(true);
    else setCpuThinking(true);

    const prompt = buildPrompt(currentRound);

    try {
      const res = await fetch(`${backendUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: prompt,
          use_cache: mode === "gpu",
        }),
      });

      const data = await res.json();

      const result: Result = {
        id: Date.now() + Math.random(),
        mode,
        round: currentRound + 1,
        latency: Math.round(data.latency_ms || 0),
        tps: Math.round(data.tokens_per_second || 0),
        response: data.response || "No response returned.",
        contextTokens: Math.round(prompt.length / 4),
      };

      setLogs((prev) => [result, ...prev.slice(0, 39)]);
    } catch {
      setLogs((prev) => [
        {
          id: Date.now() + Math.random(),
          mode,
          round: currentRound + 1,
          latency: 0,
          tps: 0,
          response: "Backend request failed.",
          contextTokens: 0,
        },
        ...prev.slice(0, 39),
      ]);
    }

    if (mode === "gpu") setGpuThinking(false);
    else setCpuThinking(false);
  }

  async function modeLoop(mode: Mode) {
    let currentRound = 0;

    while (runningRef.current) {
      setRound((prev) => Math.max(prev, currentRound + 1));
      await runInference(mode, currentRound);
      currentRound++;
    }
  }

  function start() {
    if (running) return;

    setRunning(true);
    runningRef.current = true;
    setSecondsLeft(60);

    timerRef.current = setInterval(() => {
      setSecondsLeft((prev) => {
        if (prev <= 1) {
          stop();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    modeLoop("gpu");
    modeLoop("cpu");
  }

  function stop() {
    setRunning(false);
    runningRef.current = false;

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  function reset() {
    stop();
    setRound(0);
    setSecondsLeft(60);
    setLogs([]);
  }

  useEffect(() => {
    checkBackend();
    return () => stop();
  }, []);

  const successfulLogs = logs.filter((l) => l.latency > 0 && l.tps > 0);
  const gpuLogs = successfulLogs.filter((l) => l.mode === "gpu");
  const cpuLogs = successfulLogs.filter((l) => l.mode === "cpu");

  const avg = (arr: number[]) =>
    arr.length === 0
      ? null
      : Math.round(arr.reduce((a, b) => a + b, 0) / arr.length);

  const sum = (arr: number[]) => arr.reduce((a, b) => a + b, 0);

  const gpuAvgLatency = avg(gpuLogs.map((l) => l.latency));
  const cpuAvgLatency = avg(cpuLogs.map((l) => l.latency));
  const gpuAvgTps = avg(gpuLogs.map((l) => l.tps));
  const cpuAvgTps = avg(cpuLogs.map((l) => l.tps));

  const gpuLast = gpuLogs[0];
  const cpuLast = cpuLogs[0];

  const gpuTotalLatency = sum(gpuLogs.map((l) => l.latency));
  const cpuTotalLatency = sum(cpuLogs.map((l) => l.latency));

  const gpuTotalTokensApprox = sum(
    gpuLogs.map((l) => Math.round((l.tps * l.latency) / 1000))
  );
  const cpuTotalTokensApprox = sum(
    cpuLogs.map((l) => Math.round((l.tps * l.latency) / 1000))
  );

  const latest = successfulLogs[0];
  const contextTokens = latest?.contextTokens || 0;

  const speedup =
    gpuAvgLatency && cpuAvgLatency && gpuAvgLatency > 0
      ? (cpuAvgLatency / gpuAvgLatency).toFixed(2)
      : "N/A";

  const maxCompleted = Math.max(gpuLogs.length, cpuLogs.length, 1);
  const gpuCompletedWidth = Math.max(6, (gpuLogs.length / maxCompleted) * 100);
  const cpuCompletedWidth = Math.max(6, (cpuLogs.length / maxCompleted) * 100);

  const chartData = Array.from({
    length: Math.max(gpuLogs.length, cpuLogs.length),
  }).map((_, index) => {
    const gpu = gpuLogs[gpuLogs.length - 1 - index];
    const cpu = cpuLogs[cpuLogs.length - 1 - index];

    return {
      round: index + 1,
      gpuLatency: gpu?.latency ?? null,
      cpuLatency: cpu?.latency ?? null,
      gpuTps: gpu?.tps ?? null,
      cpuTps: cpu?.tps ?? null,
    };
  });

  return (
    <main className="min-h-screen bg-slate-950 p-3 text-white">
      <div className="mx-auto max-w-[1700px] space-y-2">
        <header className="rounded-2xl border border-cyan-500/40 bg-slate-900 p-3">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.25em] text-cyan-400">
                KV Cache Performance Lab
              </p>
              <h1 className="text-3xl font-black">
                Transformer Inference Control Room
              </h1>
              <p className="text-sm text-slate-300">
                Same GPU, same model, two modes: GPU KV Cache vs CPU KV Cache.
              </p>
            </div>

            <div className="grid min-w-[680px] grid-cols-4 gap-2">
              <Metric label="Backend" value={backendStatus} />
              <Metric label="Round" value={round.toString()} />
              <Metric label="Time" value={`${secondsLeft}s`} />
              <Metric label="Speedup" value={`${speedup}x`} />
            </div>
          </div>
        </header>

        <section className="grid grid-cols-1 gap-2 xl:grid-cols-[0.85fr_1.35fr_0.85fr]">
          <Panel
            title="🔵 GPU KV Cache"
            subtitle="KV tensors use the GPU fast path."
            thinking={gpuThinking}
            score={gpuLogs.length}
            latency={gpuAvgLatency}
            tps={gpuAvgTps}
            lastLatency={gpuLast?.latency ?? null}
            lastTps={gpuLast?.tps ?? null}
            totalLatency={gpuTotalLatency}
            totalTokens={gpuTotalTokensApprox}
            color="blue"
          />

          <div className="space-y-2">
            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-3">
              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={start}
                  disabled={running}
                  className="rounded-xl bg-green-500 px-4 py-2 font-black hover:bg-green-400 disabled:bg-slate-600"
                >
                  Start
                </button>

                <button
                  onClick={stop}
                  disabled={!running}
                  className="rounded-xl bg-red-500 px-4 py-2 font-black hover:bg-red-400 disabled:bg-slate-600"
                >
                  Stop
                </button>

                <button
                  onClick={reset}
                  className="rounded-xl bg-orange-500 px-4 py-2 font-black hover:bg-orange-400"
                >
                  Reset
                </button>
              </div>

              <div className="mt-2 rounded-xl border border-purple-500 bg-purple-950/50 p-2 text-sm text-slate-300">
                Progress bars tell the story. Line graphs prove the measured
                trend.
              </div>
            </div>

            <PerformanceRace
              gpuCompleted={gpuLogs.length}
              cpuCompleted={cpuLogs.length}
              gpuCompletedWidth={gpuCompletedWidth}
              cpuCompletedWidth={cpuCompletedWidth}
              gpuAvgLatency={gpuAvgLatency}
              cpuAvgLatency={cpuAvgLatency}
              gpuAvgTps={gpuAvgTps}
              cpuAvgTps={cpuAvgTps}
              gpuLast={gpuLast}
              cpuLast={cpuLast}
              gpuTotalLatency={gpuTotalLatency}
              cpuTotalLatency={cpuTotalLatency}
              gpuTotalTokens={gpuTotalTokensApprox}
              cpuTotalTokens={cpuTotalTokensApprox}
              contextTokens={contextTokens}
              speedup={speedup}
            />

            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-3">
              <h2 className="text-xl font-black">Live Graphs</h2>

              <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
                <div className="h-48 rounded-xl bg-slate-950/60 p-2">
                  <p className="mb-1 text-xs font-bold text-slate-300">
                    Latency Over Time
                  </p>

                  <ResponsiveContainer width="100%" height="88%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="round" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="gpuLatency"
                        name="GPU"
                        stroke="#60a5fa"
                        strokeWidth={3}
                        dot={false}
                        connectNulls
                      />
                      <Line
                        type="monotone"
                        dataKey="cpuLatency"
                        name="CPU"
                        stroke="#fb7185"
                        strokeWidth={3}
                        dot={false}
                        connectNulls
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="h-48 rounded-xl bg-slate-950/60 p-2">
                  <p className="mb-1 text-xs font-bold text-slate-300">
                    Tokens/sec Over Time
                  </p>

                  <ResponsiveContainer width="100%" height="88%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="round" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="gpuTps"
                        name="GPU"
                        stroke="#38bdf8"
                        strokeWidth={3}
                        dot={false}
                        connectNulls
                      />
                      <Line
                        type="monotone"
                        dataKey="cpuTps"
                        name="CPU"
                        stroke="#f43f5e"
                        strokeWidth={3}
                        dot={false}
                        connectNulls
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-3">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-black">Live Inference Log</h2>
                <p className="text-xs text-slate-400">
                  Context: ~{contextTokens} tokens
                </p>
              </div>

              <div className="mt-2 max-h-[150px] space-y-2 overflow-y-auto pr-1">
                {logs.length === 0 ? (
                  <p className="text-sm text-slate-400">
                    Start the benchmark to collect live llama.cpp results.
                  </p>
                ) : (
                  logs.slice(0, 6).map((log) => (
                    <div
                      key={log.id}
                      className={`rounded-xl border p-2 ${
                        log.mode === "gpu"
                          ? "border-blue-500 bg-blue-950/40"
                          : "border-red-500 bg-red-950/40"
                      }`}
                    >
                      <div className="flex flex-wrap justify-between gap-2">
                        <p className="text-sm font-black">
                          {log.mode === "gpu"
                            ? "🔵 GPU KV Cache"
                            : "🔴 CPU KV Cache"}{" "}
                          · Round {log.round}
                        </p>

                        <p className="text-xs font-bold text-slate-300">
                          {log.latency} ms · {log.tps} tok/s · ~
                          {log.contextTokens} ctx
                        </p>
                      </div>

                      <p className="mt-1 line-clamp-1 text-xs text-slate-300">
                        {log.response}
                      </p>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          <Panel
            title="🔴 CPU KV Cache"
            subtitle="KV tensors are not offloaded to GPU."
            thinking={cpuThinking}
            score={cpuLogs.length}
            latency={cpuAvgLatency}
            tps={cpuAvgTps}
            lastLatency={cpuLast?.latency ?? null}
            lastTps={cpuLast?.tps ?? null}
            totalLatency={cpuTotalLatency}
            totalTokens={cpuTotalTokensApprox}
            color="red"
          />
        </section>
      </div>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-950/60 p-2 text-center">
      <p className="text-xs font-bold text-slate-400">{label}</p>
      <p className="mt-1 text-sm font-black text-cyan-300">{value}</p>
    </div>
  );
}

function PerformanceRace({
  gpuCompleted,
  cpuCompleted,
  gpuCompletedWidth,
  cpuCompletedWidth,
  gpuAvgLatency,
  cpuAvgLatency,
  gpuAvgTps,
  cpuAvgTps,
  gpuLast,
  cpuLast,
  gpuTotalLatency,
  cpuTotalLatency,
  gpuTotalTokens,
  cpuTotalTokens,
  contextTokens,
  speedup,
}: {
  gpuCompleted: number;
  cpuCompleted: number;
  gpuCompletedWidth: number;
  cpuCompletedWidth: number;
  gpuAvgLatency: number | null;
  cpuAvgLatency: number | null;
  gpuAvgTps: number | null;
  cpuAvgTps: number | null;
  gpuLast: Result | undefined;
  cpuLast: Result | undefined;
  gpuTotalLatency: number;
  cpuTotalLatency: number;
  gpuTotalTokens: number;
  cpuTotalTokens: number;
  contextTokens: number;
  speedup: string;
}) {
  const maxLatency = Math.max(gpuAvgLatency ?? 0, cpuAvgLatency ?? 0, 1);
  const maxTps = Math.max(gpuAvgTps ?? 0, cpuAvgTps ?? 0, 1);

  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-900 p-3">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-xl font-black">Performance Race</h2>
        <p className="rounded-full bg-cyan-950 px-3 py-1 text-sm font-black text-cyan-300">
          {speedup}x speedup
        </p>
      </div>

      <div className="space-y-2">
        <RaceBar
          label="GPU KV Cache"
          value={gpuCompleted}
          width={gpuCompletedWidth}
          color="blue"
        />

        <RaceBar
          label="CPU KV Cache"
          value={cpuCompleted}
          width={cpuCompletedWidth}
          color="red"
        />
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2">
        <StatsCard
          title="GPU"
          color="blue"
          completed={gpuCompleted}
          avgLatency={gpuAvgLatency}
          avgTps={gpuAvgTps}
          lastLatency={gpuLast?.latency ?? null}
          lastTps={gpuLast?.tps ?? null}
          totalLatency={gpuTotalLatency}
          totalTokens={gpuTotalTokens}
        />

        <StatsCard
          title="CPU"
          color="red"
          completed={cpuCompleted}
          avgLatency={cpuAvgLatency}
          avgTps={cpuAvgTps}
          lastLatency={cpuLast?.latency ?? null}
          lastTps={cpuLast?.tps ?? null}
          totalLatency={cpuTotalLatency}
          totalTokens={cpuTotalTokens}
        />
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2">
        <MiniBars
          title="Latency"
          leftLabel="GPU"
          rightLabel="CPU"
          leftValue={gpuAvgLatency}
          rightValue={cpuAvgLatency}
          max={maxLatency}
          unit="ms"
          lower
        />

        <MiniBars
          title="Tokens/sec"
          leftLabel="GPU"
          rightLabel="CPU"
          leftValue={gpuAvgTps}
          rightValue={cpuAvgTps}
          max={maxTps}
          unit="tok/s"
        />
      </div>

      <div className="mt-3 rounded-xl bg-slate-950/60 p-3">
        <div className="mb-1 flex justify-between text-xs font-bold text-slate-400">
          <span>Context Length</span>
          <span>{contextTokens} tokens</span>
        </div>

        <div className="h-4 overflow-hidden rounded-full bg-slate-800">
          <div
            className="h-full rounded-full bg-cyan-400"
            style={{
              width: `${Math.min(100, Math.max(4, contextTokens / 30))}%`,
            }}
          />
        </div>
      </div>
    </div>
  );
}

function RaceBar({
  label,
  value,
  width,
  color,
}: {
  label: string;
  value: number;
  width: number;
  color: "blue" | "red";
}) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-sm font-bold">
        <span>{label}</span>
        <span>{value} completed</span>
      </div>

      <div className="h-7 overflow-hidden rounded-full bg-slate-800">
        <div
          className={`flex h-full items-center justify-end rounded-full pr-3 text-sm font-black text-white ${
            color === "blue" ? "bg-blue-500" : "bg-red-500"
          }`}
          style={{ width: `${width}%` }}
        >
          {value}
        </div>
      </div>
    </div>
  );
}

function StatsCard({
  title,
  color,
  completed,
  avgLatency,
  avgTps,
  lastLatency,
  lastTps,
  totalLatency,
  totalTokens,
}: {
  title: string;
  color: "blue" | "red";
  completed: number;
  avgLatency: number | null;
  avgTps: number | null;
  lastLatency: number | null;
  lastTps: number | null;
  totalLatency: number;
  totalTokens: number;
}) {
  const colorClass = color === "blue" ? "text-blue-300" : "text-red-300";

  return (
    <div className="rounded-xl bg-slate-950/60 p-3">
      <p className={`text-sm font-black ${colorClass}`}>{title} Summary</p>

      <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
        <p className="text-slate-400">Completed</p>
        <p className="text-right font-bold">{completed}</p>

        <p className="text-slate-400">Avg Latency</p>
        <p className="text-right font-bold">
          {avgLatency === null ? "N/A" : `${avgLatency} ms`}
        </p>

        <p className="text-slate-400">Avg TPS</p>
        <p className="text-right font-bold">
          {avgTps === null ? "N/A" : `${avgTps} tok/s`}
        </p>

        <p className="text-slate-400">Last Latency</p>
        <p className="text-right font-bold">
          {lastLatency === null ? "N/A" : `${lastLatency} ms`}
        </p>

        <p className="text-slate-400">Last TPS</p>
        <p className="text-right font-bold">
          {lastTps === null ? "N/A" : `${lastTps} tok/s`}
        </p>

        <p className="text-slate-400">Total Latency</p>
        <p className="text-right font-bold">{totalLatency} ms</p>

        <p className="text-slate-400">Approx Tokens</p>
        <p className="text-right font-bold">{totalTokens}</p>
      </div>
    </div>
  );
}

function MiniBars({
  title,
  leftLabel,
  rightLabel,
  leftValue,
  rightValue,
  max,
  unit,
  lower = false,
}: {
  title: string;
  leftLabel: string;
  rightLabel: string;
  leftValue: number | null;
  rightValue: number | null;
  max: number;
  unit: string;
  lower?: boolean;
}) {
  return (
    <div className="rounded-xl bg-slate-950/60 p-3">
      <div className="mb-2 flex justify-between text-xs font-bold text-slate-400">
        <span>{title}</span>
        <span>{lower ? "Lower is better" : "Higher is better"}</span>
      </div>

      <MiniBarRow
        label={leftLabel}
        value={leftValue}
        unit={unit}
        width={leftValue ? (leftValue / max) * 100 : 4}
        color="blue"
      />

      <MiniBarRow
        label={rightLabel}
        value={rightValue}
        unit={unit}
        width={rightValue ? (rightValue / max) * 100 : 4}
        color="red"
      />
    </div>
  );
}

function MiniBarRow({
  label,
  value,
  unit,
  width,
  color,
}: {
  label: string;
  value: number | null;
  unit: string;
  width: number;
  color: "blue" | "red";
}) {
  return (
    <div className="mb-1 grid grid-cols-[38px_1fr_80px] items-center gap-2 text-xs">
      <span className={color === "blue" ? "text-blue-300" : "text-red-300"}>
        {label}
      </span>

      <div className="h-3 rounded-full bg-slate-800">
        <div
          className={`h-full rounded-full ${
            color === "blue" ? "bg-blue-500" : "bg-red-500"
          }`}
          style={{ width: `${Math.max(4, width)}%` }}
        />
      </div>

      <span className="text-right font-bold">
        {value === null ? "N/A" : `${value} ${unit}`}
      </span>
    </div>
  );
}

function Panel({
  title,
  subtitle,
  thinking,
  score,
  latency,
  tps,
  lastLatency,
  lastTps,
  totalLatency,
  totalTokens,
  color,
}: {
  title: string;
  subtitle: string;
  thinking: boolean;
  score: number;
  latency: number | null;
  tps: number | null;
  lastLatency: number | null;
  lastTps: number | null;
  totalLatency: number;
  totalTokens: number;
  color: "blue" | "red";
}) {
  const colorClasses =
    color === "blue"
      ? "border-blue-500 bg-blue-950/50 text-blue-300"
      : "border-red-500 bg-red-950/50 text-red-300";

  return (
    <div className={`rounded-2xl border p-4 ${colorClasses}`}>
      <h2 className="text-2xl font-black">{title}</h2>
      <p className="mt-1 text-sm text-slate-300">{subtitle}</p>

      <div className="mt-5 text-center">
        <p className="text-sm font-bold text-slate-300">Answers Completed</p>
        <p className="text-8xl font-black">{score}</p>
      </div>

      <div className="mt-5 rounded-2xl bg-slate-950/50 p-4 text-white">
        <p>Status: {thinking ? "Thinking..." : "Ready"}</p>
        <p>Avg Latency: {latency ?? "N/A"} ms</p>
        <p>Avg Tokens/sec: {tps ?? "N/A"}</p>
        <p>Last Latency: {lastLatency ?? "N/A"} ms</p>
        <p>Last Tokens/sec: {lastTps ?? "N/A"}</p>
        <p>Total Latency: {totalLatency} ms</p>
        <p>Approx Tokens: {totalTokens}</p>
      </div>

      <div className="mt-5">
        <p className="mb-2 text-sm font-bold text-slate-300">
          Decode Activity
        </p>
        <div className="h-5 overflow-hidden rounded-full bg-slate-800">
          <div
            className={`h-full transition-all ${
              color === "blue" ? "bg-blue-300" : "bg-red-300"
            } ${thinking ? "animate-pulse" : ""}`}
            style={{ width: thinking ? "100%" : "18%" }}
          />
        </div>
      </div>

      <div className="mt-5 rounded-xl border border-cyan-500/30 bg-cyan-950/20 p-4 text-sm text-slate-300">
        <p className="font-bold text-cyan-300">Transformer Path</p>

        <div className="mt-3 grid grid-cols-3 items-center gap-2 text-center text-xs">
          <div className="rounded-lg bg-slate-900 p-2">Prompt</div>
          <div>→</div>
          <div className="rounded-lg bg-slate-900 p-2">KV Cache</div>
          <div className="col-span-3">↓</div>
          <div className="col-span-3 rounded-lg bg-slate-900 p-2">
            Decode next token
          </div>
        </div>
      </div>
    </div>
  );
}