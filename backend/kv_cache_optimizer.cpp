#include "kv_cache_optimizer.h"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

struct PrecisionCandidate {
    KVCachePrecision key_precision;
    KVCachePrecision value_precision;
    const char* description;
};

/*
 * llama.cpp Q8_0 and Q4_0 use blocks of 32 elements.
 *
 * Q8_0:
 *     32 quantized bytes + 2-byte scale = 34 bytes
 *
 * Q4_0:
 *     16 packed quantized bytes + 2-byte scale = 18 bytes
 */
constexpr std::uint64_t QUANT_BLOCK_ELEMENTS = 32;
constexpr std::uint64_t Q8_0_BLOCK_BYTES = 34;
constexpr std::uint64_t Q4_0_BLOCK_BYTES = 18;

std::uint64_t divideRoundUp(
    std::uint64_t value,
    std::uint64_t divisor
) {
    if (divisor == 0) {
        throw std::invalid_argument(
            "divisor cannot be zero"
        );
    }

    return value / divisor +
           (value % divisor != 0 ? 1 : 0);
}

std::uint64_t precisionStorageBytes(
    std::uint64_t element_count,
    KVCachePrecision precision
) {
    switch (precision) {
        case KVCachePrecision::F16:
            if (
                element_count >
                std::numeric_limits<std::uint64_t>::max() / 2
            ) {
                throw std::overflow_error(
                    "F16 KV-cache size calculation overflowed"
                );
            }

            return element_count * 2;

        case KVCachePrecision::Q8_0: {
            const std::uint64_t block_count =
                divideRoundUp(
                    element_count,
                    QUANT_BLOCK_ELEMENTS
                );

            if (
                block_count >
                std::numeric_limits<std::uint64_t>::max() /
                    Q8_0_BLOCK_BYTES
            ) {
                throw std::overflow_error(
                    "Q8_0 KV-cache size calculation overflowed"
                );
            }

            return block_count * Q8_0_BLOCK_BYTES;
        }

        case KVCachePrecision::Q4_0: {
            const std::uint64_t block_count =
                divideRoundUp(
                    element_count,
                    QUANT_BLOCK_ELEMENTS
                );

            if (
                block_count >
                std::numeric_limits<std::uint64_t>::max() /
                    Q4_0_BLOCK_BYTES
            ) {
                throw std::overflow_error(
                    "Q4_0 KV-cache size calculation overflowed"
                );
            }

            return block_count * Q4_0_BLOCK_BYTES;
        }
    }

    throw std::invalid_argument(
        "Unsupported KV-cache precision"
    );
}

} // namespace

KVCacheOptimizer::KVCacheOptimizer(
    int model_layers,
    int key_value_heads,
    int head_dimension
)
    : model_layers_(model_layers),
      key_value_heads_(key_value_heads),
      head_dimension_(head_dimension) {
    if (model_layers_ <= 0) {
        throw std::invalid_argument(
            "model_layers must be greater than zero"
        );
    }

    if (key_value_heads_ <= 0) {
        throw std::invalid_argument(
            "key_value_heads must be greater than zero"
        );
    }

    if (head_dimension_ <= 0) {
        throw std::invalid_argument(
            "head_dimension must be greater than zero"
        );
    }
}

KVCacheConfiguration KVCacheOptimizer::select_configuration(
    const KVCacheRequest& request
) const {
    if (request.context_tokens <= 0) {
        throw std::invalid_argument(
            "context_tokens must be greater than zero"
        );
    }

    if (request.max_generated_tokens < 0) {
        throw std::invalid_argument(
            "max_generated_tokens cannot be negative"
        );
    }

    if (request.concurrency <= 0) {
        throw std::invalid_argument(
            "concurrency must be greater than zero"
        );
    }

    std::vector<PrecisionCandidate> candidates;

    switch (request.quality_priority) {
        case QualityPriority::MaximumQuality:
            candidates = {
                {
                    KVCachePrecision::F16,
                    KVCachePrecision::F16,
                    "Selected F16 key and value caches for maximum quality."
                },
                {
                    KVCachePrecision::F16,
                    KVCachePrecision::Q8_0,
                    "Selected F16 keys and Q8_0 values because full F16 exceeded the safe memory budget."
                },
                {
                    KVCachePrecision::Q8_0,
                    KVCachePrecision::Q8_0,
                    "Selected Q8_0 key and value caches because higher-precision configurations exceeded the safe memory budget."
                },
                {
                    KVCachePrecision::Q8_0,
                    KVCachePrecision::Q4_0,
                    "Selected Q8_0 keys and Q4_0 values because higher-precision configurations exceeded the safe memory budget."
                },
                {
                    KVCachePrecision::Q4_0,
                    KVCachePrecision::Q4_0,
                    "Selected Q4_0 key and value caches because all higher-precision configurations exceeded the safe memory budget."
                }
            };
            break;

        case QualityPriority::Balanced:
            candidates = {
                {
                    KVCachePrecision::Q8_0,
                    KVCachePrecision::Q8_0,
                    "Selected Q8_0 key and value caches for balanced quality and memory usage."
                },
                {
                    KVCachePrecision::Q8_0,
                    KVCachePrecision::Q4_0,
                    "Selected Q8_0 keys and Q4_0 values because Q8_0 for both caches exceeded the safe memory budget."
                },
                {
                    KVCachePrecision::Q4_0,
                    KVCachePrecision::Q4_0,
                    "Selected Q4_0 key and value caches because the balanced configurations exceeded the safe memory budget."
                }
            };
            break;

        case QualityPriority::MaximumCapacity:
            candidates = {
                {
                    KVCachePrecision::Q4_0,
                    KVCachePrecision::Q4_0,
                    "Selected Q4_0 key and value caches to maximize context and concurrency capacity."
                }
            };
            break;

        default:
            throw std::invalid_argument(
                "Unsupported quality priority"
            );
    }

    for (
        std::size_t index = 0;
        index < candidates.size();
        ++index
    ) {
        const PrecisionCandidate& candidate =
            candidates[index];

        std::uint64_t estimated_bytes = 0;

        const bool fits = configuration_fits(
            request,
            candidate.key_precision,
            candidate.value_precision,
            estimated_bytes
        );

        if (!fits) {
            continue;
        }

        KVCacheConfiguration selected{};

        selected.key_precision =
            candidate.key_precision;

        selected.value_precision =
            candidate.value_precision;

        selected.estimated_cache_bytes =
            estimated_bytes;

        selected.fits_available_vram = true;
        selected.used_fallback = index > 0;

        selected.precision_name =
            configuration_name(
                candidate.key_precision,
                candidate.value_precision
            );

        selected.reason =
            candidate.description;

        if (request.available_vram_bytes == 0) {
            selected.reason +=
                " No device-memory limit was provided, so the "
                "preferred configuration was accepted without "
                "a memory-budget check.";
        }

        return selected;
    }

    /*
     * None of the supported configurations fit.
     *
     * Return the smallest supported configuration so the caller can
     * display the estimated requirement and report the failure.
     */
    KVCacheConfiguration selected{};

    selected.key_precision =
        KVCachePrecision::Q4_0;

    selected.value_precision =
        KVCachePrecision::Q4_0;

    const std::uint64_t total_tokens =
        calculate_total_tokens(request);

    if (
        total_tokens >
        static_cast<std::uint64_t>(
            std::numeric_limits<int>::max()
        )
    ) {
        throw std::overflow_error(
            "Total token count exceeds the supported integer range"
        );
    }

    selected.estimated_cache_bytes =
        estimate_cache_bytes(
            static_cast<int>(total_tokens),
            request.concurrency,
            selected.key_precision,
            selected.value_precision
        );

    selected.fits_available_vram = false;
    selected.used_fallback = true;

    selected.precision_name =
        configuration_name(
            selected.key_precision,
            selected.value_precision
        );

    selected.reason =
        "Even the Q4_0 key and value caches exceed the safe "
        "device-memory budget. Reduce context length, maximum "
        "generated tokens, or concurrency.";

    return selected;
}

std::uint64_t KVCacheOptimizer::estimate_cache_bytes(
    int total_tokens,
    int concurrency,
    KVCachePrecision key_precision,
    KVCachePrecision value_precision
) const {
    if (total_tokens <= 0) {
        throw std::invalid_argument(
            "total_tokens must be greater than zero"
        );
    }

    if (concurrency <= 0) {
        throw std::invalid_argument(
            "concurrency must be greater than zero"
        );
    }

    /*
     * Number of elements in either the K cache or the V cache:
     *
     * model layers
     * × total tokens
     * × concurrent sequences
     * × KV heads
     * × head dimension
     */
    std::uint64_t elements_per_cache =
        static_cast<std::uint64_t>(
            model_layers_
        );

    elements_per_cache = checked_multiply(
        elements_per_cache,
        static_cast<std::uint64_t>(
            total_tokens
        )
    );

    elements_per_cache = checked_multiply(
        elements_per_cache,
        static_cast<std::uint64_t>(
            concurrency
        )
    );

    elements_per_cache = checked_multiply(
        elements_per_cache,
        static_cast<std::uint64_t>(
            key_value_heads_
        )
    );

    elements_per_cache = checked_multiply(
        elements_per_cache,
        static_cast<std::uint64_t>(
            head_dimension_
        )
    );

    const std::uint64_t key_bytes =
        precisionStorageBytes(
            elements_per_cache,
            key_precision
        );

    const std::uint64_t value_bytes =
        precisionStorageBytes(
            elements_per_cache,
            value_precision
        );

    if (
        key_bytes >
        std::numeric_limits<std::uint64_t>::max() -
            value_bytes
    ) {
        throw std::overflow_error(
            "Combined KV-cache size calculation overflowed"
        );
    }

    return key_bytes + value_bytes;
}

double KVCacheOptimizer::bytes_per_element(
    KVCachePrecision precision
) {
    switch (precision) {
        case KVCachePrecision::F16:
            return 2.0;

        case KVCachePrecision::Q8_0:
            return static_cast<double>(
                Q8_0_BLOCK_BYTES
            ) / static_cast<double>(
                QUANT_BLOCK_ELEMENTS
            );

        case KVCachePrecision::Q4_0:
            return static_cast<double>(
                Q4_0_BLOCK_BYTES
            ) / static_cast<double>(
                QUANT_BLOCK_ELEMENTS
            );
    }

    throw std::invalid_argument(
        "Unsupported KV-cache precision"
    );
}

std::string KVCacheOptimizer::precision_to_string(
    KVCachePrecision precision
) {
    switch (precision) {
        case KVCachePrecision::F16:
            return "F16";

        case KVCachePrecision::Q8_0:
            return "Q8_0";

        case KVCachePrecision::Q4_0:
            return "Q4_0";
    }

    throw std::invalid_argument(
        "Unsupported KV-cache precision"
    );
}

bool KVCacheOptimizer::configuration_fits(
    const KVCacheRequest& request,
    KVCachePrecision key_precision,
    KVCachePrecision value_precision,
    std::uint64_t& estimated_bytes
) const {
    const std::uint64_t total_tokens =
        calculate_total_tokens(request);

    if (
        total_tokens >
        static_cast<std::uint64_t>(
            std::numeric_limits<int>::max()
        )
    ) {
        throw std::overflow_error(
            "Total token count exceeds the supported integer range"
        );
    }

    estimated_bytes =
        estimate_cache_bytes(
            static_cast<int>(total_tokens),
            request.concurrency,
            key_precision,
            value_precision
        );

    /*
     * Zero means the caller did not provide a device-memory limit.
     */
    if (request.available_vram_bytes == 0) {
        return true;
    }

    /*
     * Reserve 10% for:
     *
     * - compute-graph buffers
     * - temporary allocations
     * - allocator fragmentation
     * - alignment
     * - ROCm/CUDA runtime overhead
     */
    constexpr long double safety_factor =
        0.90L;

    const std::uint64_t safe_budget =
        static_cast<std::uint64_t>(
            static_cast<long double>(
                request.available_vram_bytes
            ) * safety_factor
        );

    return estimated_bytes <= safe_budget;
}

std::uint64_t KVCacheOptimizer::checked_multiply(
    std::uint64_t left,
    std::uint64_t right
) {
    if (
        right != 0 &&
        left >
            std::numeric_limits<std::uint64_t>::max() /
                right
    ) {
        throw std::overflow_error(
            "KV-cache size multiplication overflowed"
        );
    }

    return left * right;
}

std::uint64_t KVCacheOptimizer::calculate_total_tokens(
    const KVCacheRequest& request
) {
    if (request.context_tokens <= 0) {
        throw std::invalid_argument(
            "context_tokens must be greater than zero"
        );
    }

    if (request.max_generated_tokens < 0) {
        throw std::invalid_argument(
            "max_generated_tokens cannot be negative"
        );
    }

    return
        static_cast<std::uint64_t>(
            request.context_tokens
        ) +
        static_cast<std::uint64_t>(
            request.max_generated_tokens
        );
}

std::string KVCacheOptimizer::configuration_name(
    KVCachePrecision key_precision,
    KVCachePrecision value_precision
) {
    const std::string key_name =
        precision_to_string(
            key_precision
        );

    const std::string value_name =
        precision_to_string(
            value_precision
        );

    if (key_precision == value_precision) {
        return key_name;
    }

    return
        "K=" + key_name +
        ", V=" + value_name;
}