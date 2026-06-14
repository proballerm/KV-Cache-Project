import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "distilgpt2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

prompt = "You are an AI game character. A player asks: Why is the ancient dragon guarding the cave?"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\nRunning inference...")
start_time = time.time()

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        use_cache=True
    )

end_time = time.time()

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nGenerated Output:")
print(generated_text)

print("\nPerformance:")
print(f"Total latency: {end_time - start_time:.2f} seconds")