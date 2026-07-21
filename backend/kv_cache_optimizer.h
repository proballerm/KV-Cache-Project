#pragma once

#include <cstdint>
#include <string>

/*
 * KV-cache storage formats supported by this project.
 *
 * These values are converted to llama.cpp ggml_type values
 * inside llama_runner.cpp.
 */
enum class KVCachePrecision {
    F16,
    Q8_0,
    Q4_0
};

/*
 * Controls how aggressively the optimizer reduces KV-cache
 * precision when available GPU memory is limited.
 */
enum class QualityPriority {
    MaximumQuality,
    Balanced,
    MaximumCapacity
};

/*
 * Inputs used by the adaptive KV-cache optimizer.
 */
struct KVCacheRequest {
    /*
     * Number of prompt or input-context tokens.
     */
    int context_tokens = 4096;

    /*
     * Maximum number of tokens expected to be generated.
     *
     * The cache must have enough capacity for both the input
     * context and the generated tokens.
     */
    int max_generated_tokens = 128;

    /*
     * Number of simultaneous sequences whose KV caches must
     * fit in device memory.
     *
     * The current benchmark executes one sequence at a time,
     * so values greater than one are capacity estimates unless
     * real concurrent contexts are created.
     */
    int concurrency = 1;

    /*
     * Device memory available after model weights and other
     * major runtime allocations have been loaded.
     *
     * On AMD MI300X, this represents available HBM.
     * On NVIDIA hardware, this represents available VRAM.
     *
     * A value of zero means no memory limit was provided.
     */
    std::uint64_t available_vram_bytes = 0;

    QualityPriority quality_priority =
        QualityPriority::Balanced;
};

/*
 * Configuration selected by the optimizer.
 */
struct KVCacheConfiguration {
    /*
     * Precision used for the key cache.
     */
    KVCachePrecision key_precision =
        KVCachePrecision::F16;

    /*
     * Precision used for the value cache.
     */
    KVCachePrecision value_precision =
        KVCachePrecision::F16;

    /*
     * Estimated combined size of the key and value caches.
     */
    std::uint64_t estimated_cache_bytes = 0;

    /*
     * True when the estimated cache fits within the safe
     * available-memory budget.
     */
    bool fits_available_vram = false;

    /*
     * True when the optimizer had to select a lower-priority
     * configuration because a preferred configuration did not fit.
     */
    bool used_fallback = false;

    /*
     * Human-readable configuration name.
     *
     * Examples:
     *
     * F16
     * Q8_0
     * K=Q8_0, V=Q4_0
     */
    std::string precision_name;

    /*
     * Explanation for why this configuration was selected.
     */
    std::string reason;
};

class KVCacheOptimizer {
public:
    /*
     * model_layers:
     *     Number of transformer layers.
     *
     * key_value_heads:
     *     Number of KV heads used by grouped-query or
     *     multi-query attention.
     *
     * head_dimension:
     *     Number of elements in each attention head.
     */
    KVCacheOptimizer(
        int model_layers,
        int key_value_heads,
        int head_dimension
    );

    /*
     * Selects the highest-priority KV-cache configuration that
     * fits within the safe available-memory budget.
     */
    KVCacheConfiguration select_configuration(
        const KVCacheRequest& request
    ) const;

    /*
     * Estimates the combined key-cache and value-cache size.
     *
     * total_tokens should include:
     *
     *     context tokens + maximum generated tokens
     */
    std::uint64_t estimate_cache_bytes(
        int total_tokens,
        int concurrency,
        KVCachePrecision key_precision,
        KVCachePrecision value_precision
    ) const;

    /*
     * Returns the approximate number of bytes used per element.
     *
     * F16:
     *     2 bytes per element.
     *
     * Q8_0:
     *     Includes the quantized values and block-scale metadata.
     *
     * Q4_0:
     *     Includes the packed values and block-scale metadata.
     */
    static double bytes_per_element(
        KVCachePrecision precision
    );

    /*
     * Converts a precision enum to a human-readable name.
     */
    static std::string precision_to_string(
        KVCachePrecision precision
    );

private:
    /*
     * Estimates a candidate configuration and determines whether
     * it fits within the safe memory budget.
     */
    bool configuration_fits(
        const KVCacheRequest& request,
        KVCachePrecision key_precision,
        KVCachePrecision value_precision,
        std::uint64_t& estimated_bytes
    ) const;

    /*
     * Performs checked uint64_t multiplication and throws an
     * overflow error when the calculation is too large.
     */
    static std::uint64_t checked_multiply(
        std::uint64_t left,
        std::uint64_t right
    );

    /*
     * Safely calculates:
     *
     * context_tokens + max_generated_tokens
     */
    static std::uint64_t calculate_total_tokens(
        const KVCacheRequest& request
    );

    /*
     * Builds a display name for same-precision or mixed-precision
     * key/value configurations.
     */
    static std::string configuration_name(
        KVCachePrecision key_precision,
        KVCachePrecision value_precision
    );

    int model_layers_;
    int key_value_heads_;
    int head_dimension_;
};