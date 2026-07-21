#include <cstdlib>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "httplib.h"
#include "json.hpp"
#include "kv_cache_optimizer.h"
#include "llama_runner.h"

using json = nlohmann::json;

namespace {

void addCorsHeaders(httplib::Response& response) {
    response.set_header(
        "Access-Control-Allow-Origin",
        "*"
    );

    response.set_header(
        "Access-Control-Allow-Methods",
        "GET, POST, OPTIONS"
    );

    response.set_header(
        "Access-Control-Allow-Headers",
        "Content-Type"
    );
}

KVCachePrecision parsePrecision(
    const std::string& precision
) {
    if (
        precision == "f16" ||
        precision == "F16"
    ) {
        return KVCachePrecision::F16;
    }

    if (
        precision == "q8_0" ||
        precision == "Q8_0" ||
        precision == "q8" ||
        precision == "Q8"
    ) {
        return KVCachePrecision::Q8_0;
    }

    if (
        precision == "q4_0" ||
        precision == "Q4_0" ||
        precision == "q4" ||
        precision == "Q4"
    ) {
        return KVCachePrecision::Q4_0;
    }

    throw std::invalid_argument(
        "Unsupported KV-cache precision: " + precision
    );
}

QualityPriority parseQualityPriority(
    const std::string& priority
) {
    if (
        priority == "maximum_quality" ||
        priority == "quality"
    ) {
        return QualityPriority::MaximumQuality;
    }

    if (
        priority == "balanced" ||
        priority == "auto"
    ) {
        return QualityPriority::Balanced;
    }

    if (
        priority == "maximum_capacity" ||
        priority == "capacity"
    ) {
        return QualityPriority::MaximumCapacity;
    }

    throw std::invalid_argument(
        "Unsupported quality priority: " + priority
    );
}

json configurationToJson(
    const KVCacheConfiguration& configuration
) {
    return {
        {
            "key_precision",
            KVCacheOptimizer::precision_to_string(
                configuration.key_precision
            )
        },
        {
            "value_precision",
            KVCacheOptimizer::precision_to_string(
                configuration.value_precision
            )
        },
        {
            "precision_name",
            configuration.precision_name
        },
        {
            "estimated_cache_bytes",
            configuration.estimated_cache_bytes
        },
        {
            "estimated_cache_megabytes",
            static_cast<double>(
                configuration.estimated_cache_bytes
            ) / (1024.0 * 1024.0)
        },
        {
            "fits_available_vram",
            configuration.fits_available_vram
        },
        {
            "used_fallback",
            configuration.used_fallback
        },
        {
            "reason",
            configuration.reason
        }
    };
}

InferenceConfiguration buildInferenceConfiguration(
    const json& request,
    const KVCacheConfiguration& cache_configuration,
    int max_tokens
) {
    InferenceConfiguration configuration;

    configuration.key_precision =
        cache_configuration.key_precision;

    configuration.value_precision =
        cache_configuration.value_precision;

    configuration.max_tokens = max_tokens;

    configuration.offload_kqv =
        request.value("offload_kqv", true);

    configuration.flash_attention =
        request.value("flash_attention", true);

    return configuration;
}

KVCacheConfiguration buildManualConfiguration(
    KVCacheOptimizer& optimizer,
    int context_tokens,
    int max_tokens,
    int concurrency,
    KVCachePrecision key_precision,
    KVCachePrecision value_precision,
    std::uint64_t available_vram_bytes
) {
    KVCacheConfiguration configuration;

    configuration.key_precision = key_precision;
    configuration.value_precision = value_precision;

    configuration.estimated_cache_bytes =
        optimizer.estimate_cache_bytes(
            context_tokens + max_tokens,
            concurrency,
            key_precision,
            value_precision
        );

    configuration.precision_name =
        "K=" +
        KVCacheOptimizer::precision_to_string(
            key_precision
        ) +
        ", V=" +
        KVCacheOptimizer::precision_to_string(
            value_precision
        );

    if (key_precision == value_precision) {
        configuration.precision_name =
            KVCacheOptimizer::precision_to_string(
                key_precision
            );
    }

    configuration.used_fallback = false;

    if (available_vram_bytes == 0) {
        configuration.fits_available_vram = true;

        configuration.reason =
            "Manual KV-cache configuration selected. "
            "No VRAM limit was provided.";
    } else {
        const std::uint64_t safe_budget =
            static_cast<std::uint64_t>(
                static_cast<long double>(
                    available_vram_bytes
                ) * 0.90L
            );

        configuration.fits_available_vram =
            configuration.estimated_cache_bytes <=
            safe_budget;

        configuration.reason =
            configuration.fits_available_vram
                ? "Manual KV-cache configuration fits the safe VRAM budget."
                : "Manual KV-cache configuration exceeds the safe VRAM budget.";
    }

    return configuration;
}

} // namespace

int main() {
    const char* model_env =
        std::getenv("MODEL_PATH");

    if (model_env == nullptr) {
        std::cerr
            << "ERROR: MODEL_PATH is not set\n";

        return 1;
    }

    const std::string model_path =
        model_env;

    try {
        std::cout
            << "Loading llama.cpp model...\n";

        LlamaRunner runner(
            model_path,
            999
        );

        std::cout
            << "Model loaded successfully\n";

        KVCacheOptimizer optimizer(
            runner.model_layers(),
            runner.key_value_heads(),
            runner.head_dimension()
        );

        httplib::Server app;

        app.Options(
            ".*",
            [](
                const httplib::Request&,
                httplib::Response& response
            ) {
                addCorsHeaders(response);
                response.status = 204;
            }
        );

        /*
         * Basic backend health endpoint.
         */
        app.Get(
            "/",
            [&runner](
                const httplib::Request&,
                httplib::Response& response
            ) {
                addCorsHeaders(response);

                const json body = {
                    {
                        "status",
                        "Backend is running"
                    },
                    {
                        "backend",
                        runner.backend_name()
                    },
                    {
                        "project",
                        "Adaptive KV-Cache Optimizer"
                    },
                    {
                        "endpoints",
                        {
                            "/system",
                            "/optimize",
                            "/chat",
                            "/benchmark"
                        }
                    }
                };

                response.set_content(
                    body.dump(4),
                    "application/json"
                );
            }
        );

        /*
         * Return GPU, model, and llama.cpp information.
         */
        app.Get(
            "/system",
            [&runner](
                const httplib::Request&,
                httplib::Response& response
            ) {
                addCorsHeaders(response);

                try {
                    const DeviceMemoryInfo memory =
                        runner.device_memory_info();

                    const json body = {
                        {
                            "backend",
                            runner.backend_name()
                        },
                        {
                            "device",
                            runner.device_name()
                        },
                        {
                            "model_description",
                            runner.model_description()
                        },
                        {
                            "model_layers",
                            runner.model_layers()
                        },
                        {
                            "attention_heads",
                            runner.attention_heads()
                        },
                        {
                            "key_value_heads",
                            runner.key_value_heads()
                        },
                        {
                            "head_dimension",
                            runner.head_dimension()
                        },
                        {
                            "model_size_bytes",
                            runner.model_size_bytes()
                        },
                        {
                            "device_memory",
                            {
                                {
                                    "free_bytes",
                                    memory.free_bytes
                                },
                                {
                                    "total_bytes",
                                    memory.total_bytes
                                },
                                {
                                    "used_bytes",
                                    memory.total_bytes -
                                    memory.free_bytes
                                }
                            }
                        },
                        {
                            "supported_kv_precisions",
                            {
                                "F16",
                                "Q8_0",
                                "Q4_0"
                            }
                        }
                    };

                    response.set_content(
                        body.dump(4),
                        "application/json"
                    );
                } catch (const std::exception& error) {
                    response.status = 500;

                    response.set_content(
                        json{
                            {
                                "error",
                                error.what()
                            }
                        }.dump(),
                        "application/json"
                    );
                }
            }
        );

        /*
         * Run the optimizer without performing inference.
         */
        app.Post(
            "/optimize",
            [&runner, &optimizer](
                const httplib::Request& request,
                httplib::Response& response
            ) {
                addCorsHeaders(response);

                try {
                    const json body =
                        request.body.empty()
                            ? json::object()
                            : json::parse(request.body);

                    KVCacheRequest optimizer_request;

                    optimizer_request.context_tokens =
                        body.value(
                            "context_tokens",
                            4096
                        );

                    optimizer_request.max_generated_tokens =
                        body.value(
                            "max_tokens",
                            128
                        );

                    optimizer_request.concurrency =
                        body.value(
                            "concurrency",
                            1
                        );

                    const DeviceMemoryInfo memory =
                        runner.device_memory_info();

                    optimizer_request.available_vram_bytes =
                        body.value(
                            "available_vram_bytes",
                            static_cast<std::uint64_t>(
                                memory.free_bytes
                            )
                        );

                    optimizer_request.quality_priority =
                        parseQualityPriority(
                            body.value(
                                "quality_priority",
                                std::string("balanced")
                            )
                        );

                    const KVCacheConfiguration selected =
                        optimizer.select_configuration(
                            optimizer_request
                        );

                    const json output = {
                        {
                            "request",
                            {
                                {
                                    "context_tokens",
                                    optimizer_request.context_tokens
                                },
                                {
                                    "max_tokens",
                                    optimizer_request.max_generated_tokens
                                },
                                {
                                    "concurrency",
                                    optimizer_request.concurrency
                                },
                                {
                                    "available_vram_bytes",
                                    optimizer_request.available_vram_bytes
                                }
                            }
                        },
                        {
                            "selected_configuration",
                            configurationToJson(
                                selected
                            )
                        },
                        {
                            "backend",
                            runner.backend_name()
                        },
                        {
                            "device",
                            runner.device_name()
                        }
                    };

                    response.set_content(
                        output.dump(4),
                        "application/json"
                    );
                } catch (const json::parse_error& error) {
                    response.status = 400;

                    response.set_content(
                        json{
                            {
                                "error",
                                std::string(
                                    "Invalid JSON: "
                                ) + error.what()
                            }
                        }.dump(),
                        "application/json"
                    );
                } catch (const std::exception& error) {
                    response.status = 400;

                    response.set_content(
                        json{
                            {
                                "error",
                                error.what()
                            }
                        }.dump(),
                        "application/json"
                    );
                }
            }
        );

        /*
         * Run one real inference request using either:
         *
         * mode = "auto"
         * mode = "manual"
         */
        app.Post(
            "/chat",
            [&runner, &optimizer](
                const httplib::Request& request,
                httplib::Response& response
            ) {
                addCorsHeaders(response);

                try {
                    const json body =
                        json::parse(request.body);

                    const std::string message =
                        body.value(
                            "message",
                            std::string()
                        );

                    if (message.empty()) {
                        response.status = 400;

                        response.set_content(
                            json{
                                {
                                    "error",
                                    "message cannot be empty"
                                }
                            }.dump(),
                            "application/json"
                        );

                        return;
                    }

                    const int max_tokens =
                        body.value(
                            "max_tokens",
                            80
                        );

                    const int concurrency =
                        body.value(
                            "concurrency",
                            1
                        );

                    const std::string mode =
                        body.value(
                            "mode",
                            std::string("auto")
                        );

                    const int prompt_tokens =
                        runner.count_tokens(
                            message
                        );

                    const DeviceMemoryInfo memory_before =
                        runner.device_memory_info();

                    KVCacheConfiguration cache_configuration;

                    if (mode == "manual") {
                        const KVCachePrecision key_precision =
                            parsePrecision(
                                body.value(
                                    "key_precision",
                                    std::string("F16")
                                )
                            );

                        const KVCachePrecision value_precision =
                            parsePrecision(
                                body.value(
                                    "value_precision",
                                    std::string("F16")
                                )
                            );

                        cache_configuration =
                            buildManualConfiguration(
                                optimizer,
                                prompt_tokens,
                                max_tokens,
                                concurrency,
                                key_precision,
                                value_precision,
                                memory_before.free_bytes
                            );
                    } else if (mode == "auto") {
                        KVCacheRequest optimizer_request;

                        optimizer_request.context_tokens =
                            prompt_tokens;

                        optimizer_request.max_generated_tokens =
                            max_tokens;

                        optimizer_request.concurrency =
                            concurrency;

                        optimizer_request.available_vram_bytes =
                            memory_before.free_bytes;

                        optimizer_request.quality_priority =
                            parseQualityPriority(
                                body.value(
                                    "quality_priority",
                                    std::string("balanced")
                                )
                            );

                        cache_configuration =
                            optimizer.select_configuration(
                                optimizer_request
                            );
                    } else {
                        throw std::invalid_argument(
                            "mode must be either auto or manual"
                        );
                    }

                    if (
                        !cache_configuration.fits_available_vram &&
                        memory_before.free_bytes != 0
                    ) {
                        response.status = 400;

                        response.set_content(
                            json{
                                {
                                    "error",
                                    "Selected KV-cache configuration does not fit available device memory"
                                },
                                {
                                    "configuration",
                                    configurationToJson(
                                        cache_configuration
                                    )
                                }
                            }.dump(4),
                            "application/json"
                        );

                        return;
                    }

                    const InferenceConfiguration inference_configuration =
                        buildInferenceConfiguration(
                            body,
                            cache_configuration,
                            max_tokens
                        );

                    const LlamaResult result =
                        runner.generate(
                            message,
                            inference_configuration
                        );

                    const json output = {
                        {
                            "response",
                            result.response
                        },
                        {
                            "backend",
                            runner.backend_name()
                        },
                        {
                            "device",
                            runner.device_name()
                        },
                        {
                            "mode",
                            mode
                        },
                        {
                            "configuration",
                            configurationToJson(
                                cache_configuration
                            )
                        },
                        {
                            "offload_kqv",
                            inference_configuration.offload_kqv
                        },
                        {
                            "flash_attention",
                            inference_configuration.flash_attention
                        },
                        {
                            "context_tokens",
                            result.context_tokens
                        },
                        {
                            "generated_tokens",
                            result.generated_tokens
                        },
                        {
                            "prefill_ms",
                            result.prefill_ms
                        },
                        {
                            "decode_ms",
                            result.decode_ms
                        },
                        {
                            "latency_ms",
                            result.total_latency_ms
                        },
                        {
                            "time_to_first_token_ms",
                            result.time_to_first_token_ms
                        },
                        {
                            "tokens_per_second",
                            result.decode_tokens_per_second
                        },
                        {
                            "memory",
                            {
                                {
                                    "before_context_bytes",
                                    result.memory_before_context_bytes
                                },
                                {
                                    "after_context_bytes",
                                    result.memory_after_context_bytes
                                },
                                {
                                    "context_allocation_delta_bytes",
                                    result.context_allocation_delta_bytes
                                }
                            }
                        }
                    };

                    response.set_content(
                        output.dump(4),
                        "application/json"
                    );
                } catch (const json::parse_error& error) {
                    response.status = 400;

                    response.set_content(
                        json{
                            {
                                "error",
                                std::string(
                                    "Invalid JSON: "
                                ) + error.what()
                            }
                        }.dump(),
                        "application/json"
                    );
                } catch (const std::exception& error) {
                    response.status = 500;

                    response.set_content(
                        json{
                            {
                                "error",
                                error.what()
                            }
                        }.dump(),
                        "application/json"
                    );
                }
            }
        );

        /*
         * Compare real KV-cache configurations across context sizes.
         */
        app.Post(
            "/benchmark",
            [&runner, &optimizer](
                const httplib::Request& request,
                httplib::Response& response
            ) {
                addCorsHeaders(response);

                try {
                    const json body =
                        request.body.empty()
                            ? json::object()
                            : json::parse(request.body);

                    const std::vector<int> context_lengths =
                        body.value(
                            "context_lengths",
                            std::vector<int>{
                                1024,
                                2048,
                                4096,
                                8192,
                                16384
                            }
                        );

                    const int max_tokens =
                        body.value(
                            "max_tokens",
                            64
                        );

                    const bool flash_attention =
                        body.value(
                            "flash_attention",
                            true
                        );

                    if (context_lengths.empty()) {
                        throw std::invalid_argument(
                            "context_lengths cannot be empty"
                        );
                    }

                    if (max_tokens <= 0) {
                        throw std::invalid_argument(
                            "max_tokens must be greater than zero"
                        );
                    }

                    std::vector<KVCacheConfiguration>
                        configurations;

                    const std::vector<std::string>
                        requested_precisions =
                            body.value(
                                "precisions",
                                std::vector<std::string>{
                                    "F16",
                                    "Q8_0",
                                    "Q4_0"
                                }
                            );

                    const DeviceMemoryInfo memory =
                        runner.device_memory_info();

                    for (
                        const std::string& precision_name :
                        requested_precisions
                    ) {
                        const KVCachePrecision precision =
                            parsePrecision(
                                precision_name
                            );

                        configurations.push_back(
                            buildManualConfiguration(
                                optimizer,
                                context_lengths.back(),
                                max_tokens,
                                1,
                                precision,
                                precision,
                                memory.free_bytes
                            )
                        );
                    }

                    const std::vector<BenchmarkResult> results =
                        runner.benchmark_kv_configurations(
                            context_lengths,
                            configurations,
                            max_tokens,
                            flash_attention
                        );

                    json benchmark_results =
                        json::array();

                    for (const BenchmarkResult& result : results) {
                        benchmark_results.push_back({
                            {
                                "target_context_tokens",
                                result.target_context_tokens
                            },
                            {
                                "actual_context_tokens",
                                result.actual_context_tokens
                            },
                            {
                                "generated_tokens",
                                result.generated_tokens
                            },
                            {
                                "key_precision",
                                result.key_precision
                            },
                            {
                                "value_precision",
                                result.value_precision
                            },
                            {
                                "estimated_cache_bytes",
                                result.estimated_cache_bytes
                            },
                            {
                                "memory_before_context_bytes",
                                result.memory_before_context_bytes
                            },
                            {
                                "memory_after_context_bytes",
                                result.memory_after_context_bytes
                            },
                            {
                                "context_allocation_delta_bytes",
                                result.context_allocation_delta_bytes
                            },
                            {
                                "prefill_ms",
                                result.prefill_ms
                            },
                            {
                                "decode_ms",
                                result.decode_ms
                            },
                            {
                                "total_latency_ms",
                                result.total_latency_ms
                            },
                            {
                                "time_to_first_token_ms",
                                result.time_to_first_token_ms
                            },
                            {
                                "decode_tokens_per_second",
                                result.decode_tokens_per_second
                            },
                            {
                                "end_to_end_tokens_per_second",
                                result.end_to_end_tokens_per_second
                            },
                            {
                                "success",
                                result.success
                            },
                            {
                                "error",
                                result.error
                            }
                        });
                    }

                    const json output = {
                        {
                            "benchmark_type",
                            "real_kv_cache_precision_comparison"
                        },
                        {
                            "backend",
                            runner.backend_name()
                        },
                        {
                            "device",
                            runner.device_name()
                        },
                        {
                            "flash_attention",
                            flash_attention
                        },
                        {
                            "max_tokens",
                            max_tokens
                        },
                        {
                            "results",
                            benchmark_results
                        }
                    };

                    response.set_content(
                        output.dump(4),
                        "application/json"
                    );
                } catch (const json::parse_error& error) {
                    response.status = 400;

                    response.set_content(
                        json{
                            {
                                "error",
                                std::string(
                                    "Invalid JSON: "
                                ) + error.what()
                            }
                        }.dump(),
                        "application/json"
                    );
                } catch (const std::exception& error) {
                    response.status = 500;

                    response.set_content(
                        json{
                            {
                                "error",
                                error.what()
                            }
                        }.dump(),
                        "application/json"
                    );
                }
            }
        );

        std::cout
            << "Backend running on http://0.0.0.0:8000\n";

        if (!app.listen("0.0.0.0", 8000)) {
            std::cerr
                << "ERROR: Failed to start backend on port 8000\n";

            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr
            << "FATAL ERROR: "
            << error.what()
            << '\n';

        return 1;
    }

    return 0;
}