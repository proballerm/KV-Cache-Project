#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "httplib.h"
#include "json.hpp"
#include "llama_runner.h"

using json = nlohmann::json;

int main() {
    const char* model_env = std::getenv("MODEL_PATH");

    if (!model_env) {
        std::cerr << "ERROR: MODEL_PATH is not set\n";
        return 1;
    }

    std::string model_path = model_env;

    std::cout << "Loading llama.cpp model...\n";
    LlamaRunner runner(model_path, 999);
    std::cout << "Model loaded successfully\n";

    httplib::Server app;

    app.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "*");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    app.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");

        json response = {
            {"status", "Backend is running"},
            {"backend", "embedded llama.cpp"},
            {"modes", "GPU KV Cache vs CPU KV Cache"}
        };

        res.set_content(response.dump(), "application/json");
    });

    app.Post("/chat", [&runner](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "*");
        res.set_header("Access-Control-Allow-Headers", "*");

        try {
            json request = json::parse(req.body);

            std::string message = request.value("message", "");
            bool use_cache = request.value("use_cache", true);

            LlamaResult result = runner.generate(message, use_cache, 80);

            std::string mode = use_cache
                ? "GPU KV Cache"
                : "CPU KV Cache";

            json response = {
                {"response", result.response},
                {"latency_ms", result.latency_ms},
                {"tokens_per_second", result.tokens_per_second},
                {"generated_tokens", result.generated_tokens},
                {"use_cache", use_cache},
                {"mode", mode},
                {"backend", "embedded llama.cpp"}
            };

            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json error = {
                {"error", e.what()}
            };

            res.status = 500;
            res.set_content(error.dump(), "application/json");
        }
    });

    std::cout << "Backend running on http://localhost:8000\n";
    app.listen("0.0.0.0", 8000);

    return 0;
}