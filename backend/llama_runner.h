#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "kv_cache_optimizer.h"

/*
 * Configuration applied directly to llama_context_params.
 */
struct InferenceConfiguration {
    KVCachePrecision key_precision =
        KVCachePrecision::F16;

    KVCachePrecision value_precision =
        KVCachePrecision::F16;

    bool offload_kqv = true;
    bool flash_attention = true;

    int max_tokens = 80;
};

/*
 * Device memory information reported through the ggml backend.
 *
 * On AMD MI300X, these values represent HBM memory.
 * On NVIDIA hardware, these values represent GPU VRAM.
 */
struct DeviceMemoryInfo {
    std::uint64_t free_bytes = 0;
    std::uint64_t total_bytes = 0;
};

/*
 * Results from one real llama.cpp inference request.
 */
struct LlamaResult {
    std::string response;

    int context_tokens = 0;
    int generated_tokens = 0;

    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double total_latency_ms = 0.0;
    double time_to_first_token_ms = 0.0;

    double decode_tokens_per_second = 0.0;
    double end_to_end_tokens_per_second = 0.0;

    std::uint64_t memory_before_context_bytes = 0;
    std::uint64_t memory_after_context_bytes = 0;
    std::uint64_t context_allocation_delta_bytes = 0;
};

/*
 * Results from one KV-cache precision benchmark run.
 */
struct BenchmarkResult {
    int target_context_tokens = 0;
    int actual_context_tokens = 0;
    int generated_tokens = 0;

    std::string key_precision;
    std::string value_precision;

    std::uint64_t estimated_cache_bytes = 0;

    std::uint64_t memory_before_context_bytes = 0;
    std::uint64_t memory_after_context_bytes = 0;
    std::uint64_t context_allocation_delta_bytes = 0;

    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double total_latency_ms = 0.0;
    double time_to_first_token_ms = 0.0;

    double decode_tokens_per_second = 0.0;
    double end_to_end_tokens_per_second = 0.0;

    bool success = false;
    std::string error;
};

class LlamaRunner {
public:
    LlamaRunner(
        const std::string& model_path,
        int n_gpu_layers = 999
    );

    ~LlamaRunner();

    LlamaRunner(const LlamaRunner&) = delete;
    LlamaRunner& operator=(const LlamaRunner&) = delete;

    LlamaRunner(LlamaRunner&&) = delete;
    LlamaRunner& operator=(LlamaRunner&&) = delete;

    /*
     * Runs one real inference request using the supplied
     * KV-cache precision and GPU configuration.
     */
    LlamaResult generate(
        const std::string& user_message,
        const InferenceConfiguration& configuration
    );

    /*
     * Benchmarks multiple KV-cache configurations across
     * multiple context lengths.
     */
    std::vector<BenchmarkResult>
    benchmark_kv_configurations(
        const std::vector<int>& context_lengths,
        const std::vector<KVCacheConfiguration>& configurations,
        int max_tokens = 64,
        bool flash_attention = true
    );

    /*
     * Counts the actual tokenizer tokens in a string.
     */
    int count_tokens(
        const std::string& text
    ) const;

    /*
     * Model architecture information read directly from
     * the loaded llama.cpp model.
     */
    int model_layers() const;

    int attention_heads() const;

    int key_value_heads() const;

    int head_dimension() const;

    std::uint64_t model_size_bytes() const;

    std::string model_description() const;

    /*
     * Runtime/backend information.
     */
    std::string backend_name() const;

    std::string device_name() const;

    DeviceMemoryInfo device_memory_info() const;

private:
    struct Impl;

    std::unique_ptr<Impl> impl;
    mutable std::mutex mutex;
};