#include "llama_runner.h"

#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"

struct LlamaRunner::Impl {
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    int n_gpu_layers = 999;
};

LlamaRunner::LlamaRunner(const std::string& model_path, int n_gpu_layers) {
    impl = std::make_unique<Impl>();
    impl->n_gpu_layers = n_gpu_layers;

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    impl->model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (!impl->model) {
        throw std::runtime_error("Failed to load model: " + model_path);
    }

    impl->vocab = llama_model_get_vocab(impl->model);
}

LlamaRunner::~LlamaRunner() {
    if (impl && impl->model) {
        llama_model_free(impl->model);
    }
}

LlamaResult LlamaRunner::generate(
    const std::string& user_message,
    bool use_cache,
    int max_tokens
) {
    std::lock_guard<std::mutex> lock(mutex);

    std::string prompt =
        "You are a friendly AI NPC in a fantasy game. "
        "Answer in 1 or 2 short sentences.\n"
        "Player: " + user_message + "\n"
        "NPC:";

    int n_prompt = -llama_tokenize(
        impl->vocab,
        prompt.c_str(),
        prompt.size(),
        nullptr,
        0,
        true,
        true
    );

    std::vector<llama_token> prompt_tokens(n_prompt);

    if (llama_tokenize(
            impl->vocab,
            prompt.c_str(),
            prompt.size(),
            prompt_tokens.data(),
            prompt_tokens.size(),
            true,
            true
        ) < 0) {
        throw std::runtime_error("Tokenization failed");
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + max_tokens;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = false;

    // true  = GPU KV Cache / KQV offload enabled
    // false = CPU KV Cache / KQV offload disabled
    ctx_params.offload_kqv = use_cache;

    llama_context* ctx = llama_init_from_model(impl->model, ctx_params);

    if (!ctx) {
        throw std::runtime_error("Failed to create llama context");
    }

    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;

    llama_sampler* sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(
        prompt_tokens.data(),
        prompt_tokens.size()
    );

    std::string output;
    int generated_tokens = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + max_tokens;) {
        if (llama_decode(ctx, batch)) {
            llama_sampler_free(sampler);
            llama_free(ctx);
            throw std::runtime_error("llama_decode failed");
        }

        n_pos += batch.n_tokens;

        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);

        if (llama_vocab_is_eog(impl->vocab, new_token)) {
            break;
        }

        char buffer[256];
        int n = llama_token_to_piece(
            impl->vocab,
            new_token,
            buffer,
            sizeof(buffer),
            0,
            true
        );

        if (n > 0) {
            output.append(buffer, n);
        }

        batch = llama_batch_get_one(&new_token, 1);
        generated_tokens++;
    }

    auto end = std::chrono::high_resolution_clock::now();

    llama_sampler_free(sampler);
    llama_free(ctx);

    int latency_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start
        ).count()
    );

    double seconds = latency_ms / 1000.0;
    double tps = seconds > 0 ? generated_tokens / seconds : 0.0;

    return {
        output,
        latency_ms,
        tps,
        generated_tokens
    };
}