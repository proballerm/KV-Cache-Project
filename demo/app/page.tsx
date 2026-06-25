"use client";

import { useEffect, useRef, useState } from "react";

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
        latency: Math.round(data.latency_ms),
        tps: Math.round(data.tokens_per_second),
        response: data.response,
        contextTokens: Math.round(prompt.length / 4),
      };

      setLogs((prev) => [result, ...prev.slice(0, 23)]);
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
        ...prev.slice(0, 23),
      ]);
    }

    if (mode === "gpu") setGpuThinking(false);
    else setCpuThinking(false);
  }

  async function loop() {
    let currentRound = 0;

    while (runningRef.current) {
      setRound(currentRound + 1);

      await Promise.all([
        runInference("gpu", currentRound),
        runInference("cpu", currentRound),
      ]);

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

    loop();
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

  const gpuLogs = logs.filter((l) => l.mode === "gpu");
  const cpuLogs = logs.filter((l) => l.mode === "cpu");

  const avg = (arr: number[]) =>
    arr.length === 0
      ? null
      : Math.round(arr.reduce((a, b) => a + b, 0) / arr.length);

  const gpuAvgLatency = avg(gpuLogs.map((l) => l.latency));
  const cpuAvgLatency = avg(cpuLogs.map((l) => l.latency));
  const gpuAvgTps = avg(gpuLogs.map((l) => l.tps));
  const cpuAvgTps = avg(cpuLogs.map((l) => l.tps));

  const latest = logs[0];
  const contextTokens = latest?.contextTokens || 0;

  const speedup =
    gpuAvgLatency && cpuAvgLatency && gpuAvgLatency > 0
      ? (cpuAvgLatency / gpuAvgLatency).toFixed(2)
      : "N/A";

  return (
    <main className="min-h-screen overflow-y-auto bg-slate-950 p-4 text-white">
      <div className="mx-auto max-w-[1600px] space-y-3">
        <header className="rounded-2xl border border-cyan-500/40 bg-slate-900 p-4">
          <p className="text-xs font-bold uppercase tracking-[0.3em] text-cyan-400">
            KV Cache Performance Lab
          </p>

          <h1 className="text-3xl font-black">
            Transformer Inference Control Room
          </h1>

          <p className="mt-1 text-sm text-slate-300">
            Same GPU, same model, two modes: GPU KV Cache vs CPU KV Cache.
          </p>

          <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-4">
            <Metric label="Backend" value={backendStatus} />
            <Metric label="Round" value={round.toString()} />
            <Metric label="Time" value={`${secondsLeft}s`} />
            <Metric label="Speedup" value={`${speedup}x`} />
          </div>
        </header>

        <section className="grid grid-cols-1 gap-3 xl:grid-cols-[1fr_1.1fr_1fr]">
          <Panel
            title="🔵 GPU KV Cache"
            subtitle="KV tensors use the GPU fast path."
            thinking={gpuThinking}
            score={gpuLogs.length}
            latency={gpuAvgLatency}
            tps={gpuAvgTps}
            color="blue"
          />

          <div className="space-y-3">
            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
              <div className="grid grid-cols-3 gap-3">
                <button
                  onClick={start}
                  disabled={running}
                  className="rounded-xl bg-green-500 px-4 py-3 font-black hover:bg-green-400 disabled:bg-slate-600"
                >
                  Start
                </button>

                <button
                  onClick={stop}
                  disabled={!running}
                  className="rounded-xl bg-red-500 px-4 py-3 font-black hover:bg-red-400 disabled:bg-slate-600"
                >
                  Stop
                </button>

                <button
                  onClick={reset}
                  className="rounded-xl bg-orange-500 px-4 py-3 font-black hover:bg-orange-400"
                >
                  Reset
                </button>
              </div>

              <div className="mt-3 rounded-xl border border-purple-500 bg-purple-950/50 p-3 text-sm text-slate-300">
                Lower latency means more answers completed in the same time.
                Growing context makes KV placement matter more.
              </div>
            </div>

            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
              <h2 className="text-xl font-black">Live Performance Graphs</h2>

              <div className="mt-3 space-y-3">
                <BarGraph
                  title="Latency"
                  leftLabel="GPU"
                  rightLabel="CPU"
                  leftValue={gpuAvgLatency}
                  rightValue={cpuAvgLatency}
                  unit="ms"
                  lowerIsBetter
                />

                <BarGraph
                  title="Tokens/sec"
                  leftLabel="GPU"
                  rightLabel="CPU"
                  leftValue={gpuAvgTps}
                  rightValue={cpuAvgTps}
                  unit="tok/s"
                />

                <SingleBar
                  title="Context Length"
                  value={contextTokens}
                  max={3000}
                  unit="tokens"
                />
              </div>
            </div>

            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
              <h2 className="text-xl font-black">Live Inference Log</h2>

              <div className="mt-3 max-h-[360px] space-y-2 overflow-y-auto pr-1">
                {logs.length === 0 ? (
                  <p className="text-sm text-slate-400">
                    Start the benchmark to collect live llama.cpp results.
                  </p>
                ) : (
                  logs.map((log) => (
                    <div
                      key={log.id}
                      className={`rounded-xl border p-3 ${
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
                          ~{log.contextTokens} tokens · {log.latency} ms ·{" "}
                          {log.tps} tok/s
                        </p>
                      </div>

                      <p className="mt-1 line-clamp-2 text-xs text-slate-300">
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
            color="red"
          />
        </section>
      </div>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-950/60 p-3 text-center">
      <p className="text-xs font-bold text-slate-400">{label}</p>
      <p className="mt-1 text-sm font-black text-cyan-300">{value}</p>
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
  color,
}: {
  title: string;
  subtitle: string;
  thinking: boolean;
  score: number;
  latency: number | null;
  tps: number | null;
  color: "blue" | "red";
}) {
  const colorClasses =
    color === "blue"
      ? "border-blue-500 bg-blue-950/50 text-blue-300"
      : "border-red-500 bg-red-950/50 text-red-300";

  return (
    <div className={`rounded-2xl border p-5 ${colorClasses}`}>
      <h2 className="text-2xl font-black">{title}</h2>
      <p className="mt-1 text-sm text-slate-300">{subtitle}</p>

      <div className="mt-6 text-center">
        <p className="text-sm font-bold text-slate-300">Answers Completed</p>
        <p className="text-8xl font-black">{score}</p>
      </div>

      <div className="mt-6 rounded-2xl bg-slate-950/50 p-4 text-white">
        <p>Status: {thinking ? "Thinking..." : "Ready"}</p>
        <p>Avg Latency: {latency ?? "N/A"} ms</p>
        <p>Avg Tokens/sec: {tps ?? "N/A"}</p>
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

      <div className="mt-6 rounded-xl border border-cyan-500/30 bg-cyan-950/20 p-4 text-sm text-slate-300">
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

function BarGraph({
  title,
  leftLabel,
  rightLabel,
  leftValue,
  rightValue,
  unit,
  lowerIsBetter = false,
}: {
  title: string;
  leftLabel: string;
  rightLabel: string;
  leftValue: number | null;
  rightValue: number | null;
  unit: string;
  lowerIsBetter?: boolean;
}) {
  const left = leftValue ?? 0;
  const right = rightValue ?? 0;
  const max = Math.max(left, right, 1);

  const leftWidth = Math.max(4, (left / max) * 100);
  const rightWidth = Math.max(4, (right / max) * 100);

  return (
    <div>
      <div className="mb-1 flex justify-between text-xs font-bold text-slate-400">
        <span>{title}</span>
        <span>{lowerIsBetter ? "Lower is better" : "Higher is better"}</span>
      </div>

      <GraphRow
        label={leftLabel}
        value={leftValue}
        unit={unit}
        width={leftWidth}
        color="blue"
      />

      <GraphRow
        label={rightLabel}
        value={rightValue}
        unit={unit}
        width={rightWidth}
        color="red"
      />
    </div>
  );
}

function GraphRow({
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
    <div className="mb-1 grid grid-cols-[42px_1fr_90px] items-center gap-2">
      <p className="text-xs font-bold text-slate-300">{label}</p>

      <div className="h-4 overflow-hidden rounded-full bg-slate-800">
        <div
          className={`h-full rounded-full ${
            color === "blue" ? "bg-blue-400" : "bg-red-400"
          }`}
          style={{ width: `${width}%` }}
        />
      </div>

      <p className="text-right text-xs font-bold text-slate-300">
        {value === null ? "N/A" : `${value} ${unit}`}
      </p>
    </div>
  );
}

function SingleBar({
  title,
  value,
  max,
  unit,
}: {
  title: string;
  value: number;
  max: number;
  unit: string;
}) {
  const width = Math.min(100, Math.max(4, (value / max) * 100));

  return (
    <div>
      <div className="mb-1 flex justify-between text-xs font-bold text-slate-400">
        <span>{title}</span>
        <span>
          {value} {unit}
        </span>
      </div>

      <div className="h-4 overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-full bg-cyan-400"
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}