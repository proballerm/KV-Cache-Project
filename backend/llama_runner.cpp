#include "llama_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

using Clock = std::chrono::high_resolution_clock;

namespace {

using ContextPtr = std::unique_ptr<
    llama_context,
    decltype(&llama_free)
>;

using SamplerPtr = std::unique_ptr<
    llama_sampler,
    decltype(&llama_sampler_free)
>;

double elapsedMilliseconds(
    const Clock::time_point& start,
    const Clock::time_point& end
) {
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

std::vector<llama_token> tokenizePrompt(
    const llama_vocab* vocab,
    const std::string& prompt
) {
    if (vocab == nullptr) {
        throw std::invalid_argument(
            "Vocabulary cannot be null"
        );
    }

    if (
        prompt.size() >
        static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max()
        )
    ) {
        throw std::overflow_error(
            "Prompt is too large to tokenize"
        );
    }

    const std::int32_t prompt_length =
        static_cast<std::int32_t>(prompt.size());

    const std::int32_t required_tokens =
        -llama_tokenize(
            vocab,
            prompt.c_str(),
            prompt_length,
            nullptr,
            0,
            true,
            true
        );

    if (required_tokens <= 0) {
        throw std::runtime_error(
            "Could not determine prompt token count"
        );
    }

    std::vector<llama_token> tokens(
        static_cast<std::size_t>(required_tokens)
    );

    const std::int32_t tokenized_count =
        llama_tokenize(
            vocab,
            prompt.c_str(),
            prompt_length,
            tokens.data(),
            static_cast<std::int32_t>(tokens.size()),
            true,
            true
        );

    if (tokenized_count < 0) {
        throw std::runtime_error(
            "Prompt tokenization failed"
        );
    }

    tokens.resize(
        static_cast<std::size_t>(tokenized_count)
    );

    return tokens;
}

enum ggml_type toGgmlType(
    KVCachePrecision precision
) {
    switch (precision) {
        case KVCachePrecision::F16:
            return GGML_TYPE_F16;

        case KVCachePrecision::Q8_0:
            return GGML_TYPE_Q8_0;

        case KVCachePrecision::Q4_0:
            return GGML_TYPE_Q4_0;
    }

    throw std::invalid_argument(
        "Unsupported KV-cache precision"
    );
}

enum llama_flash_attn_type toFlashAttentionType(
    bool enabled
) {
    return enabled
        ? LLAMA_FLASH_ATTN_TYPE_ENABLED
        : LLAMA_FLASH_ATTN_TYPE_DISABLED;
}

std::uint64_t usedMemoryBytes(
    const DeviceMemoryInfo& memory
) {
    if (memory.total_bytes < memory.free_bytes) {
        return 0;
    }

    return memory.total_bytes - memory.free_bytes;
}

std::uint64_t positiveMemoryDelta(
    std::uint64_t before,
    std::uint64_t after
) {
    if (after <= before) {
        return 0;
    }

    return after - before;
}

std::string buildBenchmarkPrompt(
    const llama_vocab* vocab,
    int target_tokens
) {
    if (target_tokens <= 0) {
        throw std::invalid_argument(
            "target_tokens must be greater than zero"
        );
    }

    const std::string repeated_text =
        "The player explores a fantasy world containing villages, "
        "forests, caves, castles, dragons, ancient crystals, "
        "puzzles, treasure, and AI-controlled characters. ";

    std::string prompt;
    prompt.reserve(
        static_cast<std::size_t>(target_tokens) * 5
    );

    /*
     * Add a reasonable amount of text before tokenizing.
     * If it is not enough, continue growing it.
     */
    for (int i = 0; i < target_tokens / 20 + 2; ++i) {
        prompt += repeated_text;
    }

    while (
        static_cast<int>(
            tokenizePrompt(vocab, prompt).size()
        ) < target_tokens
    ) {
        prompt += repeated_text;
    }

    return prompt;
}

std::uint64_t estimateKvBytes(
    int model_layers,
    int key_value_heads,
    int head_dimension,
    int total_tokens,
    KVCachePrecision key_precision,
    KVCachePrecision value_precision
) {
    if (
        model_layers <= 0 ||
        key_value_heads <= 0 ||
        head_dimension <= 0 ||
        total_tokens <= 0
    ) {
        return 0;
    }

    const long double elements_per_cache =
        static_cast<long double>(model_layers) *
        static_cast<long double>(key_value_heads) *
        static_cast<long double>(head_dimension) *
        static_cast<long double>(total_tokens);

    const long double key_bytes =
        elements_per_cache *
        static_cast<long double>(
            KVCacheOptimizer::bytes_per_element(
                key_precision
            )
        );

    const long double value_bytes =
        elements_per_cache *
        static_cast<long double>(
            KVCacheOptimizer::bytes_per_element(
                value_precision
            )
        );

    const long double total_bytes =
        key_bytes + value_bytes;

    if (
        total_bytes >
        static_cast<long double>(
            std::numeric_limits<std::uint64_t>::max()
        )
    ) {
        throw std::overflow_error(
            "KV-cache estimate exceeded uint64_t"
        );
    }

    return static_cast<std::uint64_t>(
        std::ceil(total_bytes)
    );
}

ContextPtr createContext(
    llama_model* model,
    int context_tokens,
    int max_tokens,
    KVCachePrecision key_precision,
    KVCachePrecision value_precision,
    bool offload_kqv,
    bool flash_attention
) {
    if (model == nullptr) {
        throw std::invalid_argument(
            "Model cannot be null"
        );
    }

    if (context_tokens <= 0) {
        throw std::invalid_argument(
            "context_tokens must be greater than zero"
        );
    }

    if (max_tokens <= 0) {
        throw std::invalid_argument(
            "max_tokens must be greater than zero"
        );
    }

    const std::uint64_t requested_context =
        static_cast<std::uint64_t>(context_tokens) +
        static_cast<std::uint64_t>(max_tokens);

    if (
        requested_context >
        std::numeric_limits<std::uint32_t>::max()
    ) {
        throw std::overflow_error(
            "Requested context exceeds llama.cpp limits"
        );
    }

    llama_context_params context_params =
        llama_context_default_params();

    context_params.n_ctx =
        static_cast<std::uint32_t>(requested_context);

    context_params.n_batch =
        static_cast<std::uint32_t>(context_tokens);

    context_params.type_k =
        toGgmlType(key_precision);

    context_params.type_v =
        toGgmlType(value_precision);

    context_params.offload_kqv =
        offload_kqv;

    context_params.flash_attn_type =
        toFlashAttentionType(flash_attention);

    context_params.no_perf = false;

    llama_context* raw_context =
        llama_init_from_model(
            model,
            context_params
        );

    if (raw_context == nullptr) {
        throw std::runtime_error(
            "Failed to create llama.cpp context. "
            "The selected KV-cache precision may not be "
            "supported by the active GPU backend."
        );
    }

    return ContextPtr(
        raw_context,
        &llama_free
    );
}

SamplerPtr createGreedySampler() {
    llama_sampler_chain_params sampler_params =
        llama_sampler_chain_default_params();

    sampler_params.no_perf = false;

    llama_sampler* raw_sampler =
        llama_sampler_chain_init(
            sampler_params
        );

    if (raw_sampler == nullptr) {
        throw std::runtime_error(
            "Failed to create llama sampler"
        );
    }

    llama_sampler_chain_add(
        raw_sampler,
        llama_sampler_init_greedy()
    );

    return SamplerPtr(
        raw_sampler,
        &llama_sampler_free
    );
}

} // namespace

struct LlamaRunner::Impl {
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;

    ggml_backend_dev_t device = nullptr;

    int n_gpu_layers = 999;

    int model_layers = 0;
    int attention_heads = 0;
    int key_value_heads = 0;
    int head_dimension = 0;

    std::uint64_t model_size_bytes = 0;

    std::string model_description;
    std::string device_name;
};

LlamaRunner::LlamaRunner(
    const std::string& model_path,
    int n_gpu_layers
) {
    if (model_path.empty()) {
        throw std::invalid_argument(
            "model_path cannot be empty"
        );
    }

    impl = std::make_unique<Impl>();
    impl->n_gpu_layers = n_gpu_layers;

    /*
     * Load CUDA, HIP/ROCm, CPU, and other compiled backends.
     */
    ggml_backend_load_all();
    llama_backend_init();

    /*
     * Locate the first available GPU backend.
     *
     * On MI300X this should resolve to the HIP/ROCm device.
     * On NVIDIA systems it resolves to the CUDA device.
     */
    const std::size_t device_count =
        ggml_backend_dev_count();

    for (
        std::size_t index = 0;
        index < device_count;
        ++index
    ) {
        ggml_backend_dev_t candidate =
            ggml_backend_dev_get(index);

        if (
            candidate != nullptr &&
            ggml_backend_dev_type(candidate) ==
                GGML_BACKEND_DEVICE_TYPE_GPU
        ) {
            impl->device = candidate;
            break;
        }
    }

    /*
     * CPU fallback if no GPU backend was found.
     */
    if (impl->device == nullptr) {
        impl->device =
            ggml_backend_dev_by_type(
                GGML_BACKEND_DEVICE_TYPE_CPU
            );
    }

    if (impl->device != nullptr) {
        const char* description =
            ggml_backend_dev_description(
                impl->device
            );

        if (description != nullptr) {
            impl->device_name = description;
        } else {
            const char* name =
                ggml_backend_dev_name(
                    impl->device
                );

            impl->device_name =
                name != nullptr
                    ? name
                    : "Unknown device";
        }
    } else {
        impl->device_name =
            "No backend device detected";
    }

    llama_model_params model_params =
        llama_model_default_params();

    model_params.n_gpu_layers =
        n_gpu_layers;

    impl->model =
        llama_model_load_from_file(
            model_path.c_str(),
            model_params
        );

    if (impl->model == nullptr) {
        llama_backend_free();

        throw std::runtime_error(
            "Failed to load model: " + model_path
        );
    }

    impl->vocab =
        llama_model_get_vocab(
            impl->model
        );

    if (impl->vocab == nullptr) {
        llama_model_free(impl->model);
        impl->model = nullptr;

        llama_backend_free();

        throw std::runtime_error(
            "Failed to load model vocabulary"
        );
    }

    impl->model_layers =
        llama_model_n_layer(
            impl->model
        );

    impl->attention_heads =
        llama_model_n_head(
            impl->model
        );

    impl->key_value_heads =
        llama_model_n_head_kv(
            impl->model
        );

    const int embedding_dimension =
        llama_model_n_embd(
            impl->model
        );

    if (impl->attention_heads > 0) {
        impl->head_dimension =
            embedding_dimension /
            impl->attention_heads;
    }

    impl->model_size_bytes =
        llama_model_size(
            impl->model
        );

    std::vector<char> description_buffer(1024);

    const int description_length =
        llama_model_desc(
            impl->model,
            description_buffer.data(),
            description_buffer.size()
        );

    if (description_length > 0) {
        impl->model_description =
            description_buffer.data();
    } else {
        impl->model_description =
            "Unknown llama.cpp model";
    }
}

LlamaRunner::~LlamaRunner() {
    if (
        impl != nullptr &&
        impl->model != nullptr
    ) {
        llama_model_free(
            impl->model
        );

        impl->model = nullptr;
    }

    llama_backend_free();
}

LlamaResult LlamaRunner::generate(
    const std::string& user_message,
    const InferenceConfiguration& configuration
) {
    std::lock_guard<std::mutex> lock(mutex);

    if (user_message.empty()) {
        throw std::invalid_argument(
            "user_message cannot be empty"
        );
    }

    if (configuration.max_tokens <= 0) {
        throw std::invalid_argument(
            "max_tokens must be greater than zero"
        );
    }

    const std::string prompt =
        "You are a friendly AI NPC in a fantasy game. "
        "Answer in one or two short sentences.\n"
        "Player: " +
        user_message +
        "\nNPC:";

    std::vector<llama_token> prompt_tokens =
        tokenizePrompt(
            impl->vocab,
            prompt
        );

    const int prompt_token_count =
        static_cast<int>(
            prompt_tokens.size()
        );

    const DeviceMemoryInfo memory_before =
        device_memory_info();

    const std::uint64_t used_before_context =
        usedMemoryBytes(
            memory_before
        );

    ContextPtr context =
        createContext(
            impl->model,
            prompt_token_count,
            configuration.max_tokens,
            configuration.key_precision,
            configuration.value_precision,
            configuration.offload_kqv,
            configuration.flash_attention
        );

    const DeviceMemoryInfo memory_after =
        device_memory_info();

    const std::uint64_t used_after_context =
        usedMemoryBytes(
            memory_after
        );

    const std::uint64_t allocation_delta =
        positiveMemoryDelta(
            used_before_context,
            used_after_context
        );

    SamplerPtr sampler =
        createGreedySampler();

    /*
     * PREFILL
     *
     * Process the full prompt and populate the selected KV cache.
     */
    llama_batch prompt_batch =
        llama_batch_get_one(
            prompt_tokens.data(),
            static_cast<std::int32_t>(
                prompt_tokens.size()
            )
        );

    const auto total_start =
        Clock::now();

    const auto prefill_start =
        Clock::now();

    const std::int32_t prefill_status =
        llama_decode(
            context.get(),
            prompt_batch
        );

    llama_synchronize(
        context.get()
    );

    const auto prefill_end =
        Clock::now();

    if (prefill_status != 0) {
        throw std::runtime_error(
            "Prompt prefill failed with llama_decode status " +
            std::to_string(prefill_status)
        );
    }

    const double prefill_ms =
        elapsedMilliseconds(
            prefill_start,
            prefill_end
        );

    /*
     * FIRST TOKEN
     */
    const auto first_token_start =
        Clock::now();

    llama_token current_token =
        llama_sampler_sample(
            sampler.get(),
            context.get(),
            -1
        );

    const auto first_token_end =
        Clock::now();

    const double first_sample_ms =
        elapsedMilliseconds(
            first_token_start,
            first_token_end
        );

    const double time_to_first_token_ms =
        prefill_ms + first_sample_ms;

    std::string output;
    int generated_tokens = 0;
    int decode_steps = 0;

    double decode_ms = 0.0;

    if (
        !llama_vocab_is_eog(
            impl->vocab,
            current_token
        )
    ) {
        char piece_buffer[512];

        const int piece_length =
            llama_token_to_piece(
                impl->vocab,
                current_token,
                piece_buffer,
                sizeof(piece_buffer),
                0,
                true
            );

        if (piece_length > 0) {
            output.append(
                piece_buffer,
                static_cast<std::size_t>(
                    piece_length
                )
            );
        }

        generated_tokens = 1;
    }

    /*
     * DECODE
     *
     * Each decode step reuses the previously populated KV cache.
     */
    while (
        generated_tokens <
            configuration.max_tokens &&
        !llama_vocab_is_eog(
            impl->vocab,
            current_token
        )
    ) {
        llama_batch token_batch =
            llama_batch_get_one(
                &current_token,
                1
            );

        const auto decode_start =
            Clock::now();

        const std::int32_t decode_status =
            llama_decode(
                context.get(),
                token_batch
            );

        llama_synchronize(
            context.get()
        );

        if (decode_status != 0) {
            throw std::runtime_error(
                "Token decode failed with llama_decode status " +
                std::to_string(decode_status)
            );
        }

        current_token =
            llama_sampler_sample(
                sampler.get(),
                context.get(),
                -1
            );

        const auto decode_end =
            Clock::now();

        decode_ms +=
            elapsedMilliseconds(
                decode_start,
                decode_end
            );

        ++decode_steps;

        if (
            llama_vocab_is_eog(
                impl->vocab,
                current_token
            )
        ) {
            break;
        }

        char piece_buffer[512];

        const int piece_length =
            llama_token_to_piece(
                impl->vocab,
                current_token,
                piece_buffer,
                sizeof(piece_buffer),
                0,
                true
            );

        if (piece_length > 0) {
            output.append(
                piece_buffer,
                static_cast<std::size_t>(
                    piece_length
                )
            );
        }

        ++generated_tokens;
    }

    const auto total_end =
        Clock::now();

    const double total_latency_ms =
        elapsedMilliseconds(
            total_start,
            total_end
        );

    const double decode_seconds =
        decode_ms / 1000.0;

    const double total_seconds =
        total_latency_ms / 1000.0;

    const double decode_tokens_per_second =
        decode_seconds > 0.0
            ? static_cast<double>(decode_steps) /
                decode_seconds
            : 0.0;

    const double end_to_end_tokens_per_second =
        total_seconds > 0.0
            ? static_cast<double>(
                generated_tokens
            ) / total_seconds
            : 0.0;

    LlamaResult result;

    result.response = output;
    result.context_tokens =
        prompt_token_count;

    result.generated_tokens =
        generated_tokens;

    result.prefill_ms =
        prefill_ms;

    result.decode_ms =
        decode_ms;

    result.total_latency_ms =
        total_latency_ms;

    result.time_to_first_token_ms =
        time_to_first_token_ms;

    result.decode_tokens_per_second =
        decode_tokens_per_second;

    result.end_to_end_tokens_per_second =
        end_to_end_tokens_per_second;

    result.memory_before_context_bytes =
        used_before_context;

    result.memory_after_context_bytes =
        used_after_context;

    result.context_allocation_delta_bytes =
        allocation_delta;

    return result;
}

std::vector<BenchmarkResult>
LlamaRunner::benchmark_kv_configurations(
    const std::vector<int>& context_lengths,
    const std::vector<KVCacheConfiguration>& configurations,
    int max_tokens,
    bool flash_attention
) {
    std::lock_guard<std::mutex> lock(mutex);

    if (context_lengths.empty()) {
        throw std::invalid_argument(
            "At least one context length is required"
        );
    }

    if (configurations.empty()) {
        throw std::invalid_argument(
            "At least one KV-cache configuration is required"
        );
    }

    if (max_tokens <= 0) {
        throw std::invalid_argument(
            "max_tokens must be greater than zero"
        );
    }

    std::vector<BenchmarkResult> results;

    results.reserve(
        context_lengths.size() *
        configurations.size()
    );

    for (
        const KVCacheConfiguration& configuration :
        configurations
    ) {
        for (
            const int target_tokens :
            context_lengths
        ) {
            BenchmarkResult result;

            result.target_context_tokens =
                target_tokens;

            result.key_precision =
                KVCacheOptimizer::precision_to_string(
                    configuration.key_precision
                );

            result.value_precision =
                KVCacheOptimizer::precision_to_string(
                    configuration.value_precision
                );

            try {
                if (target_tokens <= 0) {
                    throw std::invalid_argument(
                        "Context lengths must be greater than zero"
                    );
                }

                const std::string prompt =
                    buildBenchmarkPrompt(
                        impl->vocab,
                        target_tokens
                    );

                std::vector<llama_token> prompt_tokens =
                    tokenizePrompt(
                        impl->vocab,
                        prompt
                    );

                if (
                    static_cast<int>(
                        prompt_tokens.size()
                    ) > target_tokens
                ) {
                    prompt_tokens.resize(
                        static_cast<std::size_t>(
                            target_tokens
                        )
                    );
                }

                const int actual_context_tokens =
                    static_cast<int>(
                        prompt_tokens.size()
                    );

                result.actual_context_tokens =
                    actual_context_tokens;

                result.estimated_cache_bytes =
                    estimateKvBytes(
                        impl->model_layers,
                        impl->key_value_heads,
                        impl->head_dimension,
                        actual_context_tokens +
                            max_tokens,
                        configuration.key_precision,
                        configuration.value_precision
                    );

                const DeviceMemoryInfo memory_before =
                    device_memory_info();

                result.memory_before_context_bytes =
                    usedMemoryBytes(
                        memory_before
                    );

                ContextPtr context =
                    createContext(
                        impl->model,
                        actual_context_tokens,
                        max_tokens,
                        configuration.key_precision,
                        configuration.value_precision,
                        true,
                        flash_attention
                    );

                const DeviceMemoryInfo memory_after =
                    device_memory_info();

                result.memory_after_context_bytes =
                    usedMemoryBytes(
                        memory_after
                    );

                result.context_allocation_delta_bytes =
                    positiveMemoryDelta(
                        result.memory_before_context_bytes,
                        result.memory_after_context_bytes
                    );

                SamplerPtr sampler =
                    createGreedySampler();

                llama_batch prompt_batch =
                    llama_batch_get_one(
                        prompt_tokens.data(),
                        static_cast<std::int32_t>(
                            prompt_tokens.size()
                        )
                    );

                const auto total_start =
                    Clock::now();

                const auto prefill_start =
                    Clock::now();

                const std::int32_t prefill_status =
                    llama_decode(
                        context.get(),
                        prompt_batch
                    );

                llama_synchronize(
                    context.get()
                );

                const auto prefill_end =
                    Clock::now();

                if (prefill_status != 0) {
                    throw std::runtime_error(
                        "Benchmark prefill failed with status " +
                        std::to_string(prefill_status)
                    );
                }

                result.prefill_ms =
                    elapsedMilliseconds(
                        prefill_start,
                        prefill_end
                    );

                const auto first_token_start =
                    Clock::now();

                llama_token current_token =
                    llama_sampler_sample(
                        sampler.get(),
                        context.get(),
                        -1
                    );

                const auto first_token_end =
                    Clock::now();

                result.time_to_first_token_ms =
                    result.prefill_ms +
                    elapsedMilliseconds(
                        first_token_start,
                        first_token_end
                    );

                int generated_tokens = 0;
                int decode_steps = 0;

                if (
                    !llama_vocab_is_eog(
                        impl->vocab,
                        current_token
                    )
                ) {
                    generated_tokens = 1;
                }

                while (
                    generated_tokens < max_tokens &&
                    !llama_vocab_is_eog(
                        impl->vocab,
                        current_token
                    )
                ) {
                    llama_batch token_batch =
                        llama_batch_get_one(
                            &current_token,
                            1
                        );

                    const auto decode_start =
                        Clock::now();

                    const std::int32_t decode_status =
                        llama_decode(
                            context.get(),
                            token_batch
                        );

                    llama_synchronize(
                        context.get()
                    );

                    if (decode_status != 0) {
                        throw std::runtime_error(
                            "Benchmark decode failed with status " +
                            std::to_string(decode_status)
                        );
                    }

                    current_token =
                        llama_sampler_sample(
                            sampler.get(),
                            context.get(),
                            -1
                        );

                    const auto decode_end =
                        Clock::now();

                    result.decode_ms +=
                        elapsedMilliseconds(
                            decode_start,
                            decode_end
                        );

                    ++decode_steps;

                    if (
                        llama_vocab_is_eog(
                            impl->vocab,
                            current_token
                        )
                    ) {
                        break;
                    }

                    ++generated_tokens;
                }

                const auto total_end =
                    Clock::now();

                result.generated_tokens =
                    generated_tokens;

                result.total_latency_ms =
                    elapsedMilliseconds(
                        total_start,
                        total_end
                    );

                const double decode_seconds =
                    result.decode_ms / 1000.0;

                const double total_seconds =
                    result.total_latency_ms /
                    1000.0;

                result.decode_tokens_per_second =
                    decode_seconds > 0.0
                        ? static_cast<double>(
                            decode_steps
                        ) / decode_seconds
                        : 0.0;

                result.end_to_end_tokens_per_second =
                    total_seconds > 0.0
                        ? static_cast<double>(
                            generated_tokens
                        ) / total_seconds
                        : 0.0;

                result.success = true;
                result.error.clear();
            } catch (const std::exception& error) {
                result.success = false;
                result.error = error.what();
            }

            results.push_back(
                std::move(result)
            );
        }
    }

    return results;
}

int LlamaRunner::count_tokens(
    const std::string& text
) const {
    if (text.empty()) {
        return 0;
    }

    const std::vector<llama_token> tokens =
        tokenizePrompt(
            impl->vocab,
            text
        );

    return static_cast<int>(
        tokens.size()
    );
}

int LlamaRunner::model_layers() const {
    return impl->model_layers;
}

int LlamaRunner::attention_heads() const {
    return impl->attention_heads;
}

int LlamaRunner::key_value_heads() const {
    return impl->key_value_heads;
}

int LlamaRunner::head_dimension() const {
    return impl->head_dimension;
}

std::uint64_t LlamaRunner::model_size_bytes() const {
    return impl->model_size_bytes;
}

std::string LlamaRunner::model_description() const {
    return impl->model_description;
}

std::string LlamaRunner::backend_name() const {
#if defined(KV_BACKEND_ROCM)
    return "ROCm/HIP llama.cpp";
#elif defined(KV_BACKEND_CUDA)
    return "CUDA llama.cpp";
#elif defined(KV_BACKEND_CPU)
    return "CPU llama.cpp";
#else
    return "llama.cpp";
#endif
}

std::string LlamaRunner::device_name() const {
    return impl->device_name;
}

DeviceMemoryInfo LlamaRunner::device_memory_info() const {
    DeviceMemoryInfo result;

    if (impl->device == nullptr) {
        return result;
    }

    std::size_t free_memory = 0;
    std::size_t total_memory = 0;

    ggml_backend_dev_memory(
        impl->device,
        &free_memory,
        &total_memory
    );

    result.free_bytes =
        static_cast<std::uint64_t>(
            free_memory
        );

    result.total_bytes =
        static_cast<std::uint64_t>(
            total_memory
        );

    return result;
}