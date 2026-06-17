import time
import csv
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# GPT2-XL (1.5B parameters)
model_name = "gpt2-xl"

# Generate more tokens so KV cache effects become visible
max_new_tokens = 500

# Longer prompts
repeat_counts = [1, 10, 20, 40]

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

base_prompt = (
    "You are an AI game character. "
    "A player asks: Why is the ancient dragon guarding the cave? "
)

max_context_length = model.config.n_positions
max_input_length = max_context_length - max_new_tokens

results = []


def run_inference(prompt, repeat_count, use_cache_value):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length
    ).to(device)

    input_token_count = inputs["input_ids"].shape[1]

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print("\n----------------------------------")
    print(f"Repeat count: {repeat_count}")
    print(f"use_cache = {use_cache_value}")
    print(f"Input tokens = {input_token_count}")

    start_time = time.time()

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache_value,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None
        )

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    total_latency = end_time - start_time

    total_token_count = output.shape[1]

    generated_token_count = (
        total_token_count - input_token_count
    )

    tokens_per_second = (
        generated_token_count / total_latency
    )

    peak_memory_mb = None

    if device == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

    print(f"Generated tokens: {generated_token_count}")
    print(f"Latency: {total_latency:.2f} sec")
    print(f"Throughput: {tokens_per_second:.2f} tok/sec")

    if peak_memory_mb is not None:
        print(
            f"Peak GPU Memory: "
            f"{peak_memory_mb:.2f} MB"
        )

    return {
        "repeat_count": repeat_count,
        "input_tokens": input_token_count,
        "use_cache": use_cache_value,
        "latency_seconds": total_latency,
        "tokens_per_second": tokens_per_second,
        "peak_memory_mb": peak_memory_mb
    }


print(f"Using device: {device}")
print(f"Model: {model_name}")
print(f"Context length: {max_context_length}")
print(f"Max input length: {max_input_length}")
print(f"Generation length: {max_new_tokens}")

for repeat_count in repeat_counts:

    prompt = base_prompt * repeat_count

    result_cache_on = run_inference(
        prompt,
        repeat_count,
        True
    )

    result_cache_off = run_inference(
        prompt,
        repeat_count,
        False
    )

    results.append(result_cache_on)
    results.append(result_cache_off)

csv_file = "kv_cache_results.csv"

with open(csv_file, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "repeat_count",
            "input_tokens",
            "use_cache",
            "latency_seconds",
            "tokens_per_second",
            "peak_memory_mb"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved results to {csv_file}")

cache_on = [
    r for r in results
    if r["use_cache"] is True
]

cache_off = [
    r for r in results
    if r["use_cache"] is False
]

plt.figure(figsize=(8, 5))

plt.plot(
    [r["input_tokens"] for r in cache_on],
    [r["latency_seconds"] for r in cache_on],
    marker="o",
    linewidth=2,
    label="KV Cache ON"
)

plt.plot(
    [r["input_tokens"] for r in cache_off],
    [r["latency_seconds"] for r in cache_off],
    marker="o",
    linewidth=2,
    label="KV Cache OFF"
)

plt.xlabel("Input Tokens")
plt.ylabel("Latency (seconds)")
plt.title("KV Cache Impact on Latency")
plt.legend()
plt.grid(True)

plt.savefig(
    "latency_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(8, 5))

plt.plot(
    [r["input_tokens"] for r in cache_on],
    [r["tokens_per_second"] for r in cache_on],
    marker="o",
    linewidth=2,
    label="KV Cache ON"
)

plt.plot(
    [r["input_tokens"] for r in cache_off],
    [r["tokens_per_second"] for r in cache_off],
    marker="o",
    linewidth=2,
    label="KV Cache OFF"
)

plt.xlabel("Input Tokens")
plt.ylabel("Tokens / Second")
plt.title("KV Cache Impact on Throughput")
plt.legend()
plt.grid(True)

plt.savefig(
    "throughput_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved latency_graph.png")
print("Saved throughput_graph.png")