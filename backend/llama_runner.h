#pragma once

#include <memory>
#include <mutex>
#include <string>

struct LlamaResult {
    std::string response;
    int latency_ms;
    double tokens_per_second;
    int generated_tokens;
};

class LlamaRunner {
public:
    LlamaRunner(const std::string& model_path, int n_gpu_layers = 999);
    ~LlamaRunner();

    LlamaResult generate(const std::string& user_message, int max_tokens = 80);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    std::mutex mutex;
};