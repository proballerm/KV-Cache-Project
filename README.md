# Adaptive KV Cache Optimization for Transformer Inference

A C++/`llama.cpp` inference project for exploring how KV-cache precision affects memory usage and inference performance on modern GPUs.

The final demo compares a fixed **F16 KV-cache baseline** against a **memory-aware adaptive optimizer** that can select `F16`, `Q8_0`, `Q4_0`, or mixed key/value precision. The system includes a C++ backend, real `llama.cpp` inference, AMD ROCm/HIP support, GPU-memory reporting, and a Next.js benchmarking dashboard.

The project was developed and tested on an **AMD Instinct MI300X** using **ROCm**.

---

## Why this project exists

Large language models generate text one token at a time. During attention, each processed token produces **Key** and **Value** tensors. Recomputing those tensors for the entire previous context during every decode step would waste a large amount of work, so inference engines store them in a **KV cache** and reuse them.

That improves decode performance, but it introduces a second problem: **the KV cache grows with context length and active sequences**.

This project explores that memory/performance tradeoff. Instead of assuming one KV-cache format is always best, the backend can estimate cache requirements and select a precision based on the workload and optimization objective.

---

## Final system

```text
Browser / Next.js Dashboard
            |
            | HTTP / JSON
            v
      C++ REST Backend
            |
            +----------------------+
            |                      |
            v                      v
   KVCacheOptimizer           LlamaRunner
            |                      |
            +----------+-----------+
                       |
                       v
                   llama.cpp
                       |
                ggml GPU backend
                       |
              ROCm/HIP or CUDA
                       |
                       v
                     GPU
```

The frontend is only the experiment and visualization layer. Precision selection, model execution, timing, tokenization, and device-memory measurements are performed by the C++ backend.

---

## Main features

### Real `llama.cpp` inference

The backend loads a GGUF model directly through `llama.cpp` and performs autoregressive generation in C++.

### Adaptive KV-cache precision

Supported cache formats:

- `F16`
- `Q8_0`
- `Q4_0`

The optimizer can also use mixed key/value configurations such as:

- `K=F16, V=Q8_0`
- `K=Q8_0, V=Q4_0`

### Three optimization priorities

**Maximum Quality**

Prefers the highest-precision configuration that fits the safe memory budget, beginning with `F16/F16` and falling back only when necessary.

**Balanced**

Starts with `Q8_0/Q8_0`, then falls back to mixed `Q8_0/Q4_0` or `Q4_0/Q4_0` when required.

**Maximum Capacity**

Uses `Q4_0/Q4_0` to minimize KV-cache storage and maximize estimated context/concurrency capacity.

### GPU-memory-aware selection

The optimizer estimates the combined Key + Value cache size using:

```text
layers
x total tokens
x concurrency
x KV heads
x head dimension
x storage cost of selected precision
```

`total tokens` includes both the current context and the requested maximum generation length.

The optimizer reserves **10% of reported free device memory** for runtime overhead, temporary buffers, alignment, fragmentation, and compute-graph allocations. A candidate must fit inside the remaining 90% safe budget.

### Real model metadata

The backend reads model information from the loaded `llama.cpp` model, including:

- Number of transformer layers
- Attention heads
- KV heads
- Head dimension
- Model size

### Device-memory reporting

The backend reports free and total device memory through the ggml backend. On AMD MI300X this represents HBM; on NVIDIA GPUs it represents VRAM.

### Precision benchmarking

The benchmark endpoint can execute `F16`, `Q8_0`, and `Q4_0` KV-cache configurations across multiple context lengths.

The dashboard supports context sizes including:

- 1K
- 2K
- 4K
- 8K
- 16K

### Interactive comparison dashboard

The Next.js frontend provides a side-by-side comparison between:

- Manual `F16/F16` baseline
- Adaptive optimizer configuration

It displays:

- Applied KV configuration
- Estimated KV-cache size
- Measured context-allocation delta
- Prefill time
- Time to First Token (TTFT)
- Total latency
- Decode throughput
- Prompt tokens
- Generated tokens
- Model response
- Device name
- Backend type
- Total HBM/VRAM
- Current device-memory usage

The benchmark section also plots decode throughput by context length and compares estimated KV memory against measured allocation deltas.

---

## Metrics

### Prefill time

Time spent processing the input context before normal token-by-token decoding begins.

### Time to First Token (TTFT)

Latency from the beginning of the request until the first generated token is available.

### Decode time

Time spent generating tokens after the first token.

### Decode throughput

```text
generated decode tokens / decode time
```

Reported in tokens per second.

### Total latency

End-to-end measured inference latency for the request.

### Estimated KV memory

Calculated from model architecture, token capacity, concurrency, and KV precision.

### Measured allocation delta

Change in device-memory allocation surrounding context creation. This value can include more than the theoretical KV tensors because `llama.cpp` may allocate additional context buffers, runtime structures, alignment, and backend resources.

For that reason, the dashboard intentionally displays **estimated KV memory** and **measured allocation delta** as separate metrics.

---

## Important note about concurrency

The optimizer accepts a `concurrency` value when estimating capacity.

In the current implementation, benchmark requests execute sequences one at a time. Therefore, concurrency values greater than one represent **estimated KV-cache capacity for multiple sequences**, not a claim that the backend is executing that many requests simultaneously.

Real concurrent contexts / continuous batching would be a future extension.

---

## Repository structure

```text
KV-Cache-Project/
|
|-- backend/
|   |-- main.cpp
|   |-- llama_runner.cpp
|   |-- llama_runner.h
|   |-- kv_cache_optimizer.cpp
|   |-- kv_cache_optimizer.h
|   |-- CMakeLists.txt
|   `-- build/                 # generated locally
|
|-- demo/
|   |-- app/
|   |-- package.json
|   `-- ...                    # Next.js dashboard
|
|-- third_party/
|   `-- llama.cpp/             # git submodule
|
|-- experiments/               # earlier experiments / analysis
|-- benchmark/                 # benchmark-related work
|-- blogs/                     # technical writing
|-- notes/                     # project notes
`-- README.md
```

---

## Backend API

The backend uses `cpp-httplib` and JSON over HTTP.

Default port: `8000`

### `GET /`

Basic health endpoint.

Returns the backend status and available routes.

### `GET /system`

Returns runtime and model information, including:

- Backend (`ROCm/HIP`, CUDA, or CPU depending on the build)
- GPU/device name
- Model description
- Model architecture
- Model size
- Free/total device memory
- Supported KV precisions

### `POST /optimize`

Runs the adaptive optimizer without performing inference.

Example request:

```json
{
  "context_tokens": 4096,
  "max_tokens": 128,
  "concurrency": 1,
  "quality_priority": "balanced"
}
```

Example priority values:

```text
maximum_quality
balanced
maximum_capacity
```

### `POST /chat`

Runs real inference.

#### Adaptive mode

```json
{
  "message": "Explain why KV cache matters.",
  "mode": "auto",
  "quality_priority": "balanced",
  "max_tokens": 128,
  "concurrency": 1,
  "offload_kqv": true,
  "flash_attention": true
}
```

#### Manual baseline mode

```json
{
  "message": "Explain why KV cache matters.",
  "mode": "manual",
  "key_precision": "F16",
  "value_precision": "F16",
  "max_tokens": 128,
  "concurrency": 1,
  "offload_kqv": true,
  "flash_attention": true
}
```

### `POST /benchmark`

Runs multiple KV-cache configurations across selected context lengths and returns timing and memory results for each run.

The frontend uses this endpoint for the precision benchmark graphs and result table.

---

## Build configuration

The backend uses CMake and C++17.

Three build modes are supported:

- AMD ROCm/HIP
- NVIDIA CUDA
- CPU-only fallback

Only one GPU backend should be enabled at a time.

---

# AMD MI300X / ROCm setup

The final project was tested on an AMD Instinct MI300X (`gfx942`).

A complete ROCm development environment is recommended because building `llama.cpp` requires the HIP compiler and ROCm development libraries such as hipBLAS/rocBLAS.

## 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/proballerm/KV-Cache-Project.git
cd KV-Cache-Project

git submodule update --init --recursive
```

## 2. Set up ROCm environment

The exact ROCm installation path can vary by image/container. A typical setup is:

```bash
export ROCM_PATH=/opt/rocm
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
```

Some ROCm installations place runtime libraries in a versioned directory. If needed, include that directory in `LD_LIBRARY_PATH` as well.

Example from the ROCm environment used during final MI300X testing:

```bash
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
```

## 3. Build the backend

```bash
cd backend
rm -rf build
mkdir build
cd build

cmake .. \
  -DKV_USE_ROCM=ON \
  -DKV_USE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DAMDGPU_TARGETS=gfx942

cmake --build . -j"$(nproc)"
```

If the ROCm image requires an explicit HIP compiler, configure CMake using the Clang path reported by `hipconfig` rather than the `hipcc` wrapper.

Example:

```bash
HIPCXX="$(hipconfig -l)/clang" \
HIP_PATH="$(hipconfig -R)" \
cmake .. \
  -DKV_USE_ROCM=ON \
  -DKV_USE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DAMDGPU_TARGETS=gfx942
```

## 4. Set the model path

The backend expects a GGUF model through the `MODEL_PATH` environment variable.

The final demo used a Llama 3.1 8B Instruct GGUF model.

```bash
export MODEL_PATH=/path/to/model.gguf
```

## 5. Run the backend

From the backend build directory:

```bash
./kv_backend
```

The server listens on:

```text
http://0.0.0.0:8000
```

Test it with:

```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/system
```

---

# NVIDIA CUDA build

The same backend can be configured for NVIDIA hardware:

```bash
cd backend
rm -rf build
mkdir build
cd build

cmake .. \
  -DKV_USE_CUDA=ON \
  -DKV_USE_ROCM=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . -j"$(nproc)"
```

Then set `MODEL_PATH` and start `./kv_backend` normally.

---

# CPU-only build

If neither GPU backend is enabled, CMake builds the project using the CPU backend:

```bash
cmake .. \
  -DKV_USE_CUDA=OFF \
  -DKV_USE_ROCM=OFF \
  -DCMAKE_BUILD_TYPE=Release
```

This is useful for development, but the main project is intended for GPU inference benchmarking.

---

# Frontend setup

The dashboard is built with:

- Next.js
- React
- TypeScript
- Tailwind CSS
- Recharts

Node.js 20+ is required by the current Next.js version; Node.js 22 is a good choice.

## 1. Configure the backend URL

```bash
cd demo

cat > .env.local <<'EOF'
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
EOF
```

## 2. Install dependencies

```bash
npm install
```

## 3. Start the dashboard

```bash
npm run dev -- -H 0.0.0.0
```

Open:

```text
http://localhost:3000
```

---

## Running through SSH

If the application is running on a remote GPU machine and ports 3000/8000 are not publicly exposed, use SSH local forwarding:

```bash
ssh -N \
  -L 3000:127.0.0.1:3000 \
  -L 8000:127.0.0.1:8000 \
  user@REMOTE_HOST
```

Then open:

```text
http://localhost:3000
```

---

## Monitoring the AMD GPU

While running inference on ROCm, GPU activity can be observed with:

```bash
amd-smi
```

or:

```bash
watch -n 1 amd-smi
```

Depending on the ROCm installation, `rocm-smi` can also be used.

This is useful for confirming that inference is actually executing on the accelerator and for observing HBM usage, utilization, power, and temperature during benchmarks.

---

# Experiment workflow

A typical demo run is:

1. Start the C++ backend.
2. Start the Next.js frontend.
3. Confirm the dashboard reports the expected GPU and backend.
4. Run the **F16 Baseline vs Adaptive Optimizer** comparison.
5. Change optimization priority or capacity concurrency and run again.
6. Run the precision benchmark across selected context lengths.
7. Compare estimated memory, measured allocation delta, TTFT, latency, and decode throughput.
8. Monitor the GPU using `amd-smi` while inference is active.

The demo is designed to make the memory/performance tradeoff visible rather than presenting only a generated text response.

---

# What I learned

This project started as an experiment to understand why KV caching improves transformer inference. It eventually became a deeper exploration of AI infrastructure and GPU memory behavior.

Major takeaways included:

- LLM inference is autoregressive: generation happens one token at a time.
- KV caching primarily avoids recomputing previous Key and Value tensors during decode.
- Prefill and decode have different performance characteristics.
- KV-cache memory grows with context length and sequence capacity.
- KV heads matter when estimating cache size; using total attention heads would overestimate models that use grouped-query attention.
- Lower KV precision can substantially reduce the theoretical memory footprint.
- Smaller memory usage does **not** automatically mean lower latency; quantization/dequantization and backend kernels also matter.
- Estimated KV tensor size and observed GPU allocation delta are not the same measurement.
- GPU software setup, runtime libraries, compiler configuration, and backend compatibility can be just as important as the inference code itself.
- Real benchmarking requires separating TTFT, prefill, decode throughput, total latency, and memory measurements instead of relying on one number.

---

# Limitations

This repository is an educational / experimental inference project, not a production LLM serving engine.

Current limitations include:

- The concurrency setting is primarily a capacity estimate; requests are not continuously batched across multiple real contexts.
- The optimizer uses a heuristic memory budget and predefined precision candidate ordering rather than a learned policy.
- Measured context allocation can include non-KV `llama.cpp` buffers.
- Output-quality evaluation across KV precisions is not yet a full automated benchmark.
- The project does not implement distributed or multi-GPU serving.
- The project does not currently implement KV eviction, prefix sharing, or paged KV allocation.

These limitations are intentional areas for future exploration rather than claims of production parity with large inference-serving systems.

---

# Future work

Potential extensions include:

- Real concurrent sequence execution / continuous batching
- Paged KV-cache allocation
- Prefix caching and KV reuse across requests
- Sliding-window / cache eviction policies
- Host-memory KV offload
- Automated output-quality evaluation across cache precisions
- Longer-context stress testing
- Multi-GPU inference
- More detailed ROCm profiling
- Integration with additional GGUF models
- Comparing memory-aware selection against fixed precision policies at higher request concurrency

---

# Technical blog

A companion technical blog documents the concepts learned while building this project, including:

- Autoregressive token generation
- Tokenization and embeddings
- Attention
- Query, Key, and Value
- Why repeated KV computation is expensive
- How the KV Cache works
- GPU-memory tradeoffs
- Adaptive KV-cache precision
- C++ / `llama.cpp` implementation
- AMD MI300X / ROCm deployment

The blog is intended to explain the project from first principles rather than treating KV caching as a black box.

---

## Tech stack

**Inference / Systems**

- C++17
- llama.cpp
- ggml
- CMake
- cpp-httplib
- nlohmann/json

**GPU**

- AMD ROCm / HIP
- AMD Instinct MI300X (`gfx942`)
- Optional NVIDIA CUDA backend

**Frontend**

- Next.js
- React
- TypeScript
- Tailwind CSS
- Recharts

**Infrastructure / Development**

- Linux
- Docker
- SSH
- Git / Git submodules

---

## Project status

The core project is complete as an educational adaptive KV-cache inference demo:

- [x] Learn transformer inference and KV-cache fundamentals
- [x] Build baseline inference experiments
- [x] Move the core backend to C++
- [x] Integrate `llama.cpp`
- [x] Implement adaptive KV-cache precision selection
- [x] Add F16 / Q8_0 / Q4_0 support
- [x] Add memory estimation and device-memory reporting
- [x] Measure prefill, TTFT, latency, throughput, and allocation deltas
- [x] Build a Next.js visualization dashboard
- [x] Run the system on AMD Instinct MI300X with ROCm/HIP
- [x] Add multi-context precision benchmarking
- [x] Record/demo the final system
- [x] Write a technical blog explaining the concepts and implementation

---

## Author

**Prabal Malavalli**

Computer Science, Arizona State University

GitHub: [@proballerm](https://github.com/proballerm)
