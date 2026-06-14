# GPU KV Cache Optimization Project

## Project Goal

Build a GPU based LLM inference demo that shows how KV cache optimization improves response latency, throughput, and user experience in a real time AI application.

## Demo Idea

The final demo will be a simple AI game NPC.

A player sends messages to an AI character. The project will compare response speed with and without KV cache optimization.

## Why This Matters

Large language models generate text token by token. Without KV cache, the model may recompute previous attention information again and again. KV cache stores previous key and value tensors so future tokens can be generated faster.

## Metrics

We will measure:

1. Time to first token
2. Total response latency
3. Tokens per second
4. GPU memory usage
5. User experience impact

## Tech Stack

Initial version:

- Python
- Hugging Face Transformers
- PyTorch
- Small open source model

Future version:

- vLLM
- ROCm
- AMD GPU or AMD cloud
- FastAPI
- React demo

## Weekly Plan

Week 1: Learn KV cache and build baseline script  
Week 2: Compare inference with and without cache  
Week 3: Add better benchmarking  
Week 4: Learn vLLM and PagedAttention  
Week 5: Run model using vLLM  
Week 6: Build simple AI NPC backend  
Week 7: Build simple frontend demo  
Week 8: Add performance dashboard  
Week 9: Move to AMD ROCm environment  
Week 10: Optimize and collect results  
Week 11: Write final report and blogs  
Week 12: Final demo and presentation