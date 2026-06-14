import time
import csv
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# This is the small language model we are testing.
# distilgpt2 is small enough to run on CPU.
model_name = "distilgpt2"

# This controls how many new tokens the model generates after reading the prompt.
max_new_tokens = 80

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

# Use GPU if available, otherwise use CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# eval mode means we are doing inference, not training.
model.eval()

# This prompt simulates a game NPC dialogue scenario.
base_prompt = "You are an AI game character. A player asks: Why is the ancient dragon guarding the cave? "

# These repeat counts create short, medium, and long prompts.
repeat_counts = [1, 5, 10, 20, 40]

# distilgpt2 has a max context length of 1024 tokens.
max_context_length = model.config.n_positions

# We reserve space for generated tokens so input plus output does not exceed model limit.
max_input_length = max_context_length - max_new_tokens

results = []

def run_inference(prompt, repeat_count, use_cache_value):
    # Convert text prompt into token IDs.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length
    ).to(device)

    input_token_count = inputs["input_ids"].shape[1]

    # Clear GPU memory stats only when using CUDA.
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print("\nRunning inference...")
    print(f"Repeat count: {repeat_count}")
    print(f"use_cache = {use_cache_value}")
    print(f"Input tokens: {input_token_count}")

    # Start timing generation.
    start_time = time.time()

    # no_grad disables training related memory tracking.
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            use_cache=use_cache_value,
            pad_token_id=tokenizer.eos_token_id
        )

    # Synchronize GPU timing if CUDA is being used.
    if device == "cuda":
        torch.cuda.synchronize()

    # Stop timing generation.
    end_time = time.time()

    total_latency = end_time - start_time
    total_token_count = output.shape[1]
    generated_token_count = total_token_count - input_token_count
    tokens_per_second = generated_token_count / total_latency

    peak_memory_mb = None
    if device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    print("\nPerformance:")
    print(f"Generated tokens: {generated_token_count}")
    print(f"Total latency: {total_latency:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    if peak_memory_mb is not None:
        print(f"Peak GPU memory: {peak_memory_mb:.2f} MB")

    return {
        "repeat_count": repeat_count,
        "input_tokens": input_token_count,
        "use_cache": use_cache_value,
        "latency_seconds": total_latency,
        "tokens_per_second": tokens_per_second,
        "peak_memory_mb": peak_memory_mb
    }

print(f"Using device: {device}")
print(f"Model max context length: {max_context_length}")
print(f"Max allowed input length: {max_input_length}")

# Run experiments for different prompt lengths.
for repeat_count in repeat_counts:
    prompt = base_prompt * repeat_count

    result_cache_on = run_inference(prompt, repeat_count, True)
    result_cache_off = run_inference(prompt, repeat_count, False)

    results.append(result_cache_on)
    results.append(result_cache_off)

# Save results to CSV.
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

# Separate cache on and cache off results.
cache_on = [r for r in results if r["use_cache"] == True]
cache_off = [r for r in results if r["use_cache"] == False]

# Graph 1: Input tokens vs latency.
plt.figure()
plt.plot(
    [r["input_tokens"] for r in cache_on],
    [r["latency_seconds"] for r in cache_on],
    marker="o",
    label="KV cache ON"
)
plt.plot(
    [r["input_tokens"] for r in cache_off],
    [r["latency_seconds"] for r in cache_off],
    marker="o",
    label="KV cache OFF"
)
plt.xlabel("Input Tokens")
plt.ylabel("Latency Seconds")
plt.title("KV Cache Impact on Latency")
plt.legend()
plt.grid(True)
plt.savefig("latency_graph.png")

# Graph 2: Input tokens vs tokens per second.
plt.figure()
plt.plot(
    [r["input_tokens"] for r in cache_on],
    [r["tokens_per_second"] for r in cache_on],
    marker="o",
    label="KV cache ON"
)
plt.plot(
    [r["input_tokens"] for r in cache_off],
    [r["tokens_per_second"] for r in cache_off],
    marker="o",
    label="KV cache OFF"
)
plt.xlabel("Input Tokens")
plt.ylabel("Tokens per Second")
plt.title("KV Cache Impact on Throughput")
plt.legend()
plt.grid(True)
plt.savefig("throughput_graph.png")

print("Saved latency_graph.png")
print("Saved throughput_graph.png")