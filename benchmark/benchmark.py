import time
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

prompt = """
You are an AI game character. A player asks:
Why is the ancient dragon guarding the cave?
"""

output_file = "benchmark/results/benchmark_results.csv"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Using device: {device}")

tests = [
    {"prompt_repeat": 1, "max_new_tokens": 30, "use_cache": True},
    {"prompt_repeat": 1, "max_new_tokens": 30, "use_cache": False},
    {"prompt_repeat": 5, "max_new_tokens": 30, "use_cache": True},
    {"prompt_repeat": 5, "max_new_tokens": 30, "use_cache": False},
    {"prompt_repeat": 10, "max_new_tokens": 30, "use_cache": True},
    {"prompt_repeat": 10, "max_new_tokens": 30, "use_cache": False},
]

rows = []

for test in tests:
    full_prompt = prompt * test["prompt_repeat"]

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("\nRunning test:")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Max new tokens: {test['max_new_tokens']}")
    print(f"KV cache: {test['use_cache']}")

    start_time = time.time()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=test["max_new_tokens"],
            do_sample=False,
            use_cache=test["use_cache"],
            pad_token_id=tokenizer.eos_token_id
        )

    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = output.shape[1]
    generated_tokens = total_tokens - prompt_tokens
    tokens_per_second = generated_tokens / total_time

    if device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_memory_mb = 0

    row = {
        "model": model_name,
        "device": device,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "use_cache": test["use_cache"],
        "total_time_seconds": round(total_time, 4),
        "tokens_per_second": round(tokens_per_second, 4),
        "peak_memory_mb": round(peak_memory_mb, 2),
    }

    rows.append(row)

    print(row)

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved results to {output_file}")