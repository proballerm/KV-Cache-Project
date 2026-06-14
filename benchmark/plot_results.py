import pandas as pd
import matplotlib.pyplot as plt

csv_file = "benchmark/results/benchmark_results.csv"

df = pd.read_csv(csv_file)

plt.figure()
for cache_mode in [True, False]:
    subset = df[df["use_cache"] == cache_mode]
    label = "KV Cache ON" if cache_mode else "KV Cache OFF"
    plt.plot(subset["prompt_tokens"], subset["total_time_seconds"], marker="o", label=label)

plt.xlabel("Prompt Tokens")
plt.ylabel("Total Time Seconds")
plt.title("Prompt Length vs Latency")
plt.legend()
plt.grid(True)
plt.savefig("benchmark/plots/prompt_length_vs_latency.png")

plt.figure()
for cache_mode in [True, False]:
    subset = df[df["use_cache"] == cache_mode]
    label = "KV Cache ON" if cache_mode else "KV Cache OFF"
    plt.plot(subset["prompt_tokens"], subset["tokens_per_second"], marker="o", label=label)

plt.xlabel("Prompt Tokens")
plt.ylabel("Tokens Per Second")
plt.title("Prompt Length vs Tokens Per Second")
plt.legend()
plt.grid(True)
plt.savefig("benchmark/plots/prompt_length_vs_tokens_per_second.png")

print("Graphs saved in benchmark/plots/")