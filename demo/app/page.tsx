"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Precision = "F16" | "Q8_0" | "Q4_0";
type QualityPriority =
  | "maximum_quality"
  | "balanced"
  | "maximum_capacity";

type SystemInfo = {
  backend?: string;
  device?: string;
  model_description?: string;
  model_layers?: number;
  attention_heads?: number;
  key_value_heads?: number;
  head_dimension?: number;
  model_size_bytes?: number;
  device_memory?: {
    free_bytes?: number;
    total_bytes?: number;
    used_bytes?: number;
  };
  supported_kv_precisions?: string[];
};

type CacheConfiguration = {
  key_precision?: Precision;
  value_precision?: Precision;
  precision_name?: string;
  estimated_cache_bytes?: number;
  estimated_cache_megabytes?: number;
  fits_available_vram?: boolean;
  used_fallback?: boolean;
  reason?: string;
};

type MemoryResult = {
  before_context_bytes?: number;
  after_context_bytes?: number;
  context_allocation_delta_bytes?: number;
};

type InferenceResult = {
  response?: string;
  backend?: string;
  device?: string;
  mode?: string;
  configuration?: CacheConfiguration;
  offload_kqv?: boolean;
  flash_attention?: boolean;
  context_tokens?: number;
  generated_tokens?: number;
  prefill_ms?: number;
  decode_ms?: number;
  latency_ms?: number;
  time_to_first_token_ms?: number;
  tokens_per_second?: number;
  memory?: MemoryResult;
  error?: string;
};

type BenchmarkRow = {
  target_context_tokens: number;
  actual_context_tokens: number;
  generated_tokens: number;

  key_precision: Precision;
  value_precision: Precision;

  estimated_cache_bytes: number;

  memory_before_context_bytes: number;
  memory_after_context_bytes: number;
  context_allocation_delta_bytes: number;

  prefill_ms: number;
  decode_ms: number;
  total_latency_ms: number;
  time_to_first_token_ms: number;

  decode_tokens_per_second: number;
  end_to_end_tokens_per_second: number;

  success: boolean;
  error: string;
};

type BenchmarkResponse = {
  benchmark_type?: string;
  backend?: string;
  device?: string;
  flash_attention?: boolean;
  max_tokens?: number;
  results?: BenchmarkRow[];
  error?: string;
};

const backendUrl =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

const defaultPrompt = `
The player is exploring a fantasy village near an ancient cave.
A dragon protects a crystal that powers the village.
Explain why the dragon guards the crystal and what the player should do next.
`.trim();

const contextOptions = [1024, 2048, 4096, 8192, 16384];

function formatBytes(bytes?: number) {
  if (!bytes || bytes <= 0) {
    return "N/A";
  }

  const gigabytes = bytes / 1024 ** 3;

  if (gigabytes >= 1) {
    return `${gigabytes.toFixed(2)} GB`;
  }

  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function formatMilliseconds(value?: number) {
  if (value === undefined || !Number.isFinite(value)) {
    return "N/A";
  }

  return `${value.toFixed(1)} ms`;
}

function formatRate(value?: number) {
  if (value === undefined || !Number.isFinite(value)) {
    return "N/A";
  }

  return `${value.toFixed(1)} tok/s`;
}

function percentageReduction(
  baseline?: number,
  optimized?: number
): number | null {
  if (
    baseline === undefined ||
    optimized === undefined ||
    baseline <= 0 ||
    optimized < 0
  ) {
    return null;
  }

  return ((baseline - optimized) / baseline) * 100;
}

async function readJson<T>(response: Response): Promise<T> {
  const data = (await response.json()) as T;

  if (!response.ok) {
    const possibleError = data as { error?: string };

    throw new Error(
      possibleError.error ||
        `Backend request failed with status ${response.status}`
    );
  }

  return data;
}

export default function Home() {
  const [backendStatus, setBackendStatus] =
    useState("Checking backend...");

  const [systemInfo, setSystemInfo] =
    useState<SystemInfo | null>(null);

  const [prompt, setPrompt] =
    useState(defaultPrompt);

  const [maxTokens, setMaxTokens] =
    useState(64);

  const [concurrency, setConcurrency] =
    useState(1);

  const [qualityPriority, setQualityPriority] =
    useState<QualityPriority>("balanced");

  const [flashAttention, setFlashAttention] =
    useState(true);

  const [baselineRunning, setBaselineRunning] =
    useState(false);

  const [optimizedRunning, setOptimizedRunning] =
    useState(false);

  const [benchmarkRunning, setBenchmarkRunning] =
    useState(false);

  const [baseline, setBaseline] =
    useState<InferenceResult | null>(null);

  const [optimized, setOptimized] =
    useState<InferenceResult | null>(null);

  const [benchmarkRows, setBenchmarkRows] =
    useState<BenchmarkRow[]>([]);

  const [selectedContexts, setSelectedContexts] =
    useState<number[]>([1024, 2048, 4096, 8192]);

  const [error, setError] =
    useState<string | null>(null);

  async function loadSystemInfo() {
    setError(null);

    try {
      const healthResponse =
        await fetch(`${backendUrl}/`);

      const health =
        await readJson<{ status?: string }>(
          healthResponse
        );

      setBackendStatus(
        health.status || "Backend connected"
      );

      const systemResponse =
        await fetch(`${backendUrl}/system`);

      const system =
        await readJson<SystemInfo>(
          systemResponse
        );

      setSystemInfo(system);
    } catch (caughtError) {
      const message =
        caughtError instanceof Error
          ? caughtError.message
          : "Backend is not connected";

      setBackendStatus("Backend not connected");
      setError(message);
    }
  }

  async function runBaseline() {
    setBaselineRunning(true);
    setError(null);

    try {
      const response = await fetch(
        `${backendUrl}/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: prompt,
            mode: "manual",
            key_precision: "F16",
            value_precision: "F16",
            max_tokens: maxTokens,
            concurrency,
            offload_kqv: true,
            flash_attention: flashAttention,
          }),
        }
      );

      const result =
        await readJson<InferenceResult>(
          response
        );

      setBaseline(result);
    } catch (caughtError) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "Baseline inference failed"
      );
    } finally {
      setBaselineRunning(false);
    }
  }

  async function runOptimized() {
    setOptimizedRunning(true);
    setError(null);

    try {
      const response = await fetch(
        `${backendUrl}/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: prompt,
            mode: "auto",
            quality_priority: qualityPriority,
            max_tokens: maxTokens,
            concurrency,
            offload_kqv: true,
            flash_attention: flashAttention,
          }),
        }
      );

      const result =
        await readJson<InferenceResult>(
          response
        );

      setOptimized(result);
    } catch (caughtError) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "Optimized inference failed"
      );
    } finally {
      setOptimizedRunning(false);
    }
  }

  async function runComparison() {
    setBaseline(null);
    setOptimized(null);
    setError(null);

    await runBaseline();
    await runOptimized();
  }

  async function runBenchmark() {
    setBenchmarkRunning(true);
    setError(null);
    setBenchmarkRows([]);

    try {
      const response = await fetch(
        `${backendUrl}/benchmark`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            context_lengths: selectedContexts,
            max_tokens: maxTokens,
            flash_attention: flashAttention,
            precisions: [
              "F16",
              "Q8_0",
              "Q4_0",
            ],
          }),
        }
      );

      const data =
        await readJson<BenchmarkResponse>(
          response
        );

      setBenchmarkRows(
        data.results || []
      );
    } catch (caughtError) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "Benchmark failed"
      );
    } finally {
      setBenchmarkRunning(false);
    }
  }

  function toggleContext(context: number) {
    setSelectedContexts((current) => {
      if (current.includes(context)) {
        if (current.length === 1) {
          return current;
        }

        return current.filter(
          (value) => value !== context
        );
      }

      return [...current, context].sort(
        (left, right) => left - right
      );
    });
  }

  function resetResults() {
    setBaseline(null);
    setOptimized(null);
    setBenchmarkRows([]);
    setError(null);
  }

  useEffect(() => {
    loadSystemInfo();
  }, []);

  const baselineEstimated =
    baseline?.configuration
      ?.estimated_cache_bytes;

  const optimizedEstimated =
    optimized?.configuration
      ?.estimated_cache_bytes;

  const estimatedMemoryReduction =
    percentageReduction(
      baselineEstimated,
      optimizedEstimated
    );

  const baselineMeasured =
    baseline?.memory
      ?.context_allocation_delta_bytes;

  const optimizedMeasured =
    optimized?.memory
      ?.context_allocation_delta_bytes;

  const measuredMemoryReduction =
    percentageReduction(
      baselineMeasured,
      optimizedMeasured
    );

  const latencyChange =
    baseline?.latency_ms &&
    optimized?.latency_ms
      ? ((optimized.latency_ms -
          baseline.latency_ms) /
          baseline.latency_ms) *
        100
      : null;

  const throughputChange =
    baseline?.tokens_per_second &&
    optimized?.tokens_per_second
      ? ((optimized.tokens_per_second -
          baseline.tokens_per_second) /
          baseline.tokens_per_second) *
        100
      : null;

  const comparisonChart = useMemo(
    () => [
      {
        metric: "Estimated KV MB",
        baseline:
          (baselineEstimated || 0) /
          1024 ** 2,
        optimized:
          (optimizedEstimated || 0) /
          1024 ** 2,
      },
      {
        metric: "Measured Delta MB",
        baseline:
          (baselineMeasured || 0) /
          1024 ** 2,
        optimized:
          (optimizedMeasured || 0) /
          1024 ** 2,
      },
    ],
    [
      baselineEstimated,
      optimizedEstimated,
      baselineMeasured,
      optimizedMeasured,
    ]
  );

  const benchmarkChart = useMemo(() => {
    const grouped = new Map<
      number,
      Record<string, number | null>
    >();

    for (const row of benchmarkRows) {
      if (!row.success) {
        continue;
      }

      const current =
        grouped.get(
          row.target_context_tokens
        ) || {
          context:
            row.target_context_tokens,
          F16: null,
          Q8_0: null,
          Q4_0: null,
        };

      current[row.key_precision] =
        row.decode_tokens_per_second;

      grouped.set(
        row.target_context_tokens,
        current
      );
    }

    return Array.from(
      grouped.values()
    ).sort(
      (left, right) =>
        Number(left.context) -
        Number(right.context)
    );
  }, [benchmarkRows]);

  const memoryBenchmarkChart = useMemo(() => {
    return benchmarkRows
      .filter((row) => row.success)
      .map((row) => ({
        context:
          row.target_context_tokens,
        precision:
          row.key_precision,
        label:
          `${row.target_context_tokens / 1024}K ${row.key_precision}`,
        estimatedMB:
          row.estimated_cache_bytes /
          1024 ** 2,
        measuredMB:
          row.context_allocation_delta_bytes /
          1024 ** 2,
      }));
  }, [benchmarkRows]);

  const totalMemory =
    systemInfo?.device_memory
      ?.total_bytes;

  const freeMemory =
    systemInfo?.device_memory
      ?.free_bytes;

  const deviceMemoryUsage =
    totalMemory && freeMemory
      ? ((totalMemory - freeMemory) /
          totalMemory) *
        100
      : null;

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-5 text-white">
      <div className="mx-auto max-w-[1700px] space-y-4">
        <header className="rounded-2xl border border-cyan-500/40 bg-slate-900 p-5">
          <div className="flex flex-col justify-between gap-5 xl:flex-row xl:items-center">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.28em] text-cyan-400">
                AMD MI300X · ROCm · llama.cpp
              </p>

              <h1 className="mt-1 text-3xl font-black md:text-4xl">
                Adaptive KV-Cache Optimization Lab
              </h1>

              <p className="mt-2 max-w-3xl text-sm text-slate-300">
                Compare an F16 KV-cache baseline
                against a memory-aware optimizer that
                selects Q8_0, Q4_0, or mixed
                key/value precision for real
                llama.cpp inference.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:min-w-[760px]">
              <Metric
                label="Backend"
                value={
                  systemInfo?.backend ||
                  backendStatus
                }
              />

              <Metric
                label="Device"
                value={
                  systemInfo?.device ||
                  "Unknown"
                }
              />

              <Metric
                label="HBM / VRAM"
                value={
                  totalMemory
                    ? formatBytes(totalMemory)
                    : "N/A"
                }
              />

              <Metric
                label="Memory Used"
                value={
                  deviceMemoryUsage === null
                    ? "N/A"
                    : `${deviceMemoryUsage.toFixed(1)}%`
                }
              />
            </div>
          </div>
        </header>

        {error && (
          <div className="rounded-xl border border-red-500 bg-red-950/50 p-3 text-sm text-red-200">
            {error}
          </div>
        )}

        <section className="grid grid-cols-1 gap-4 xl:grid-cols-[360px_1fr]">
          <aside className="space-y-4">
            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
              <h2 className="text-xl font-black">
                Experiment Controls
              </h2>

              <label className="mt-4 block text-sm font-bold text-slate-300">
                Prompt
              </label>

              <textarea
                value={prompt}
                onChange={(event) =>
                  setPrompt(
                    event.target.value
                  )
                }
                rows={7}
                className="mt-1 w-full rounded-xl border border-slate-600 bg-slate-950 p-3 text-sm outline-none focus:border-cyan-400"
              />

              <label className="mt-4 block text-sm font-bold text-slate-300">
                Maximum generated tokens
              </label>

              <input
                type="number"
                min={1}
                max={512}
                value={maxTokens}
                onChange={(event) =>
                  setMaxTokens(
                    Math.max(
                      1,
                      Number(
                        event.target.value
                      )
                    )
                  )
                }
                className="mt-1 w-full rounded-xl border border-slate-600 bg-slate-950 px-3 py-2"
              />

              <label className="mt-4 block text-sm font-bold text-slate-300">
                Capacity concurrency
              </label>

              <input
                type="number"
                min={1}
                max={64}
                value={concurrency}
                onChange={(event) =>
                  setConcurrency(
                    Math.max(
                      1,
                      Number(
                        event.target.value
                      )
                    )
                  )
                }
                className="mt-1 w-full rounded-xl border border-slate-600 bg-slate-950 px-3 py-2"
              />

              <p className="mt-1 text-xs text-slate-400">
                Values above one currently represent
                estimated KV capacity unless the backend
                executes multiple sequences concurrently.
              </p>

              <label className="mt-4 block text-sm font-bold text-slate-300">
                Optimizer priority
              </label>

              <select
                value={qualityPriority}
                onChange={(event) =>
                  setQualityPriority(
                    event.target
                      .value as QualityPriority
                  )
                }
                className="mt-1 w-full rounded-xl border border-slate-600 bg-slate-950 px-3 py-2"
              >
                <option value="maximum_quality">
                  Maximum quality
                </option>

                <option value="balanced">
                  Balanced
                </option>

                <option value="maximum_capacity">
                  Maximum capacity
                </option>
              </select>

              <label className="mt-4 flex items-center justify-between rounded-xl border border-slate-700 bg-slate-950/60 p-3">
                <span>
                  <span className="block text-sm font-bold">
                    Flash Attention
                  </span>

                  <span className="text-xs text-slate-400">
                    Apply the same setting to both
                    comparison runs.
                  </span>
                </span>

                <input
                  type="checkbox"
                  checked={flashAttention}
                  onChange={(event) =>
                    setFlashAttention(
                      event.target.checked
                    )
                  }
                  className="h-5 w-5"
                />
              </label>

              <div className="mt-4 grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={runComparison}
                  disabled={
                    baselineRunning ||
                    optimizedRunning ||
                    !prompt.trim()
                  }
                  className="rounded-xl bg-cyan-500 px-3 py-2 font-black text-slate-950 hover:bg-cyan-400 disabled:bg-slate-600 disabled:text-slate-300"
                >
                  {baselineRunning ||
                  optimizedRunning
                    ? "Running..."
                    : "Run Comparison"}
                </button>

                <button
                  type="button"
                  onClick={resetResults}
                  className="rounded-xl bg-orange-500 px-3 py-2 font-black hover:bg-orange-400"
                >
                  Reset
                </button>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
              <h2 className="text-lg font-black">
                Loaded Model
              </h2>

              <div className="mt-3 space-y-2 text-sm">
                <InfoRow
                  label="Description"
                  value={
                    systemInfo?.model_description ||
                    "N/A"
                  }
                />

                <InfoRow
                  label="Model size"
                  value={formatBytes(
                    systemInfo?.model_size_bytes
                  )}
                />

                <InfoRow
                  label="Layers"
                  value={
                    systemInfo?.model_layers?.toString() ||
                    "N/A"
                  }
                />

                <InfoRow
                  label="Attention heads"
                  value={
                    systemInfo?.attention_heads?.toString() ||
                    "N/A"
                  }
                />

                <InfoRow
                  label="KV heads"
                  value={
                    systemInfo?.key_value_heads?.toString() ||
                    "N/A"
                  }
                />

                <InfoRow
                  label="Head dimension"
                  value={
                    systemInfo?.head_dimension?.toString() ||
                    "N/A"
                  }
                />

                <InfoRow
                  label="Free device memory"
                  value={formatBytes(
                    freeMemory
                  )}
                />
              </div>

              <button
                type="button"
                onClick={loadSystemInfo}
                className="mt-4 w-full rounded-xl border border-cyan-500 px-3 py-2 text-sm font-black text-cyan-300 hover:bg-cyan-950/40"
              >
                Refresh System Data
              </button>
            </div>
          </aside>

          <div className="space-y-4">
            <section className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <ResultPanel
                title="F16 Baseline"
                subtitle="Manual F16 key and value caches"
                result={baseline}
                running={baselineRunning}
                color="blue"
              />

              <ResultPanel
                title="Adaptive Optimizer"
                subtitle="Precision selected from available memory and quality priority"
                result={optimized}
                running={optimizedRunning}
                color="purple"
              />
            </section>

            <section className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
              <div className="flex flex-col justify-between gap-3 md:flex-row md:items-center">
                <div>
                  <h2 className="text-xl font-black">
                    Optimization Outcome
                  </h2>

                  <p className="text-sm text-slate-400">
                    Estimated KV memory is calculated
                    from model architecture. Measured
                    allocation delta may include other
                    llama.cpp context buffers.
                  </p>
                </div>

                <span className="rounded-full bg-cyan-950 px-4 py-2 text-sm font-black text-cyan-300">
                  {optimized?.configuration
                    ?.precision_name ||
                    "Waiting for optimizer"}
                </span>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
                <SummaryMetric
                  label="Estimated Memory Saved"
                  value={
                    estimatedMemoryReduction ===
                    null
                      ? "N/A"
                      : `${estimatedMemoryReduction.toFixed(1)}%`
                  }
                />

                <SummaryMetric
                  label="Measured Delta Reduction"
                  value={
                    measuredMemoryReduction ===
                    null
                      ? "N/A"
                      : `${measuredMemoryReduction.toFixed(1)}%`
                  }
                />

                <SummaryMetric
                  label="Latency Change"
                  value={
                    latencyChange === null
                      ? "N/A"
                      : `${latencyChange >= 0 ? "+" : ""}${latencyChange.toFixed(1)}%`
                  }
                />

                <SummaryMetric
                  label="Throughput Change"
                  value={
                    throughputChange === null
                      ? "N/A"
                      : `${throughputChange >= 0 ? "+" : ""}${throughputChange.toFixed(1)}%`
                  }
                />
              </div>

              <div className="mt-4 h-72">
                <ResponsiveContainer
                  width="100%"
                  height="100%"
                >
                  <BarChart
                    data={comparisonChart}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#334155"
                    />

                    <XAxis
                      dataKey="metric"
                      stroke="#94a3b8"
                    />

                    <YAxis
                      stroke="#94a3b8"
                      unit=" MB"
                    />

                    <Tooltip />

                    <Legend />

                    <Bar
                      dataKey="baseline"
                      name="F16 Baseline"
                      fill="#3b82f6"
                    />

                    <Bar
                      dataKey="optimized"
                      name="Adaptive"
                      fill="#a855f7"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>
          </div>
        </section>

        <section className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
          <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-center">
            <div>
              <h2 className="text-2xl font-black">
                KV Precision Benchmark
              </h2>

              <p className="text-sm text-slate-400">
                Run real F16, Q8_0, and Q4_0
                llama.cpp contexts across several
                context lengths.
              </p>
            </div>

            <button
              type="button"
              onClick={runBenchmark}
              disabled={
                benchmarkRunning ||
                selectedContexts.length === 0
              }
              className="rounded-xl bg-green-500 px-5 py-3 font-black text-slate-950 hover:bg-green-400 disabled:bg-slate-600 disabled:text-slate-300"
            >
              {benchmarkRunning
                ? "Benchmark Running..."
                : "Run Precision Benchmark"}
            </button>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            {contextOptions.map(
              (context) => {
                const active =
                  selectedContexts.includes(
                    context
                  );

                return (
                  <button
                    key={context}
                    type="button"
                    onClick={() =>
                      toggleContext(context)
                    }
                    className={`rounded-full border px-4 py-2 text-sm font-bold ${
                      active
                        ? "border-cyan-400 bg-cyan-950 text-cyan-300"
                        : "border-slate-600 bg-slate-950 text-slate-400"
                    }`}
                  >
                    {context / 1024}K
                  </button>
                );
              }
            )}
          </div>

          {benchmarkRows.length === 0 ? (
            <div className="mt-5 rounded-xl border border-dashed border-slate-600 p-8 text-center text-slate-400">
              No benchmark data yet.
            </div>
          ) : (
            <>
              <div className="mt-5 grid grid-cols-1 gap-4 xl:grid-cols-2">
                <ChartPanel title="Decode Throughput by Context">
                  <ResponsiveContainer
                    width="100%"
                    height="100%"
                  >
                    <LineChart
                      data={benchmarkChart}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#334155"
                      />

                      <XAxis
                        dataKey="context"
                        stroke="#94a3b8"
                      />

                      <YAxis
                        stroke="#94a3b8"
                        unit=" tok/s"
                      />

                      <Tooltip />
                      <Legend />

                      <Line
                        type="monotone"
                        dataKey="F16"
                        stroke="#3b82f6"
                        strokeWidth={3}
                        connectNulls
                      />

                      <Line
                        type="monotone"
                        dataKey="Q8_0"
                        stroke="#a855f7"
                        strokeWidth={3}
                        connectNulls
                      />

                      <Line
                        type="monotone"
                        dataKey="Q4_0"
                        stroke="#22c55e"
                        strokeWidth={3}
                        connectNulls
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartPanel>

                <ChartPanel title="Estimated vs Measured Memory">
                  <ResponsiveContainer
                    width="100%"
                    height="100%"
                  >
                    <BarChart
                      data={
                        memoryBenchmarkChart
                      }
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#334155"
                      />

                      <XAxis
                        dataKey="label"
                        stroke="#94a3b8"
                      />

                      <YAxis
                        stroke="#94a3b8"
                        unit=" MB"
                      />

                      <Tooltip />
                      <Legend />

                      <Bar
                        dataKey="estimatedMB"
                        name="Estimated KV"
                        fill="#06b6d4"
                      />

                      <Bar
                        dataKey="measuredMB"
                        name="Measured Allocation Delta"
                        fill="#f59e0b"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartPanel>
              </div>

              <div className="mt-5 overflow-x-auto">
                <table className="w-full min-w-[1100px] border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-slate-700 text-left text-slate-400">
                      <th className="p-2">
                        Context
                      </th>

                      <th className="p-2">
                        K / V
                      </th>

                      <th className="p-2">
                        Estimated KV
                      </th>

                      <th className="p-2">
                        Measured Delta
                      </th>

                      <th className="p-2">
                        Prefill
                      </th>

                      <th className="p-2">
                        TTFT
                      </th>

                      <th className="p-2">
                        Decode TPS
                      </th>

                      <th className="p-2">
                        Generated
                      </th>

                      <th className="p-2">
                        Status
                      </th>
                    </tr>
                  </thead>

                  <tbody>
                    {benchmarkRows.map(
                      (row, index) => (
                        <tr
                          key={`${row.target_context_tokens}-${row.key_precision}-${row.value_precision}-${index}`}
                          className="border-b border-slate-800"
                        >
                          <td className="p-2 font-bold">
                            {
                              row.actual_context_tokens
                            }
                          </td>

                          <td className="p-2">
                            {row.key_precision} /{" "}
                            {row.value_precision}
                          </td>

                          <td className="p-2">
                            {formatBytes(
                              row.estimated_cache_bytes
                            )}
                          </td>

                          <td className="p-2">
                            {formatBytes(
                              row.context_allocation_delta_bytes
                            )}
                          </td>

                          <td className="p-2">
                            {formatMilliseconds(
                              row.prefill_ms
                            )}
                          </td>

                          <td className="p-2">
                            {formatMilliseconds(
                              row.time_to_first_token_ms
                            )}
                          </td>

                          <td className="p-2">
                            {formatRate(
                              row.decode_tokens_per_second
                            )}
                          </td>

                          <td className="p-2">
                            {
                              row.generated_tokens
                            }
                          </td>

                          <td className="p-2">
                            {row.success ? (
                              <span className="rounded-full bg-green-950 px-3 py-1 text-xs font-bold text-green-300">
                                Success
                              </span>
                            ) : (
                              <span
                                title={
                                  row.error
                                }
                                className="rounded-full bg-red-950 px-3 py-1 text-xs font-bold text-red-300"
                              >
                                Failed
                              </span>
                            )}
                          </td>
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      </div>
    </main>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-950/60 p-3 text-center">
      <p className="text-xs font-bold text-slate-400">
        {label}
      </p>

      <p className="mt-1 line-clamp-2 text-sm font-black text-cyan-300">
        {value}
      </p>
    </div>
  );
}

function SummaryMetric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl bg-slate-950/60 p-3">
      <p className="text-xs font-bold text-slate-400">
        {label}
      </p>

      <p className="mt-1 text-2xl font-black text-cyan-300">
        {value}
      </p>
    </div>
  );
}

function InfoRow({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="grid grid-cols-[120px_1fr] gap-3 border-b border-slate-800 pb-2">
      <span className="text-slate-400">
        {label}
      </span>

      <span className="text-right font-bold">
        {value}
      </span>
    </div>
  );
}

function ResultPanel({
  title,
  subtitle,
  result,
  running,
  color,
}: {
  title: string;
  subtitle: string;
  result: InferenceResult | null;
  running: boolean;
  color: "blue" | "purple";
}) {
  const borderColor =
    color === "blue"
      ? "border-blue-500"
      : "border-purple-500";

  const headingColor =
    color === "blue"
      ? "text-blue-300"
      : "text-purple-300";

  return (
    <div
      className={`rounded-2xl border ${borderColor} bg-slate-900 p-4`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2
            className={`text-2xl font-black ${headingColor}`}
          >
            {title}
          </h2>

          <p className="text-sm text-slate-400">
            {subtitle}
          </p>
        </div>

        <span className="rounded-full bg-slate-950 px-3 py-1 text-xs font-black">
          {running
            ? "Running"
            : result
              ? "Complete"
              : "Ready"}
        </span>
      </div>

      <div className="mt-4 rounded-xl bg-slate-950/60 p-4">
        <p className="text-xs font-bold uppercase tracking-wide text-slate-400">
          Applied KV Configuration
        </p>

        <p
          className={`mt-1 text-3xl font-black ${headingColor}`}
        >
          {result?.configuration
            ?.precision_name || "N/A"}
        </p>

        <p className="mt-2 text-xs text-slate-400">
          {result?.configuration?.reason ||
            "Run inference to select and apply a configuration."}
        </p>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2">
        <SmallResult
          label="Estimated KV"
          value={formatBytes(
            result?.configuration
              ?.estimated_cache_bytes
          )}
        />

        <SmallResult
          label="Measured Delta"
          value={formatBytes(
            result?.memory
              ?.context_allocation_delta_bytes
          )}
        />

        <SmallResult
          label="Prefill"
          value={formatMilliseconds(
            result?.prefill_ms
          )}
        />

        <SmallResult
          label="TTFT"
          value={formatMilliseconds(
            result?.time_to_first_token_ms
          )}
        />

        <SmallResult
          label="Total Latency"
          value={formatMilliseconds(
            result?.latency_ms
          )}
        />

        <SmallResult
          label="Decode Throughput"
          value={formatRate(
            result?.tokens_per_second
          )}
        />

        <SmallResult
          label="Prompt Tokens"
          value={
            result?.context_tokens?.toString() ||
            "N/A"
          }
        />

        <SmallResult
          label="Generated Tokens"
          value={
            result?.generated_tokens?.toString() ||
            "N/A"
          }
        />
      </div>

      <div className="mt-4 rounded-xl border border-slate-700 bg-slate-950/50 p-3">
        <p className="text-xs font-bold uppercase tracking-wide text-slate-400">
          Model Response
        </p>

        <p className="mt-2 min-h-[72px] text-sm text-slate-200">
          {running
            ? "Running llama.cpp inference..."
            : result?.response ||
              "No response yet."}
        </p>
      </div>
    </div>
  );
}

function SmallResult({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl bg-slate-950/60 p-3">
      <p className="text-xs font-bold text-slate-400">
        {label}
      </p>

      <p className="mt-1 font-black">
        {value}
      </p>
    </div>
  );
}

function ChartPanel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="h-[340px] rounded-xl bg-slate-950/50 p-3">
      <p className="mb-2 text-sm font-black text-slate-300">
        {title}
      </p>

      <div className="h-[290px]">
        {children}
      </div>
    </div>
  );
}