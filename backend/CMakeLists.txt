cmake_minimum_required(VERSION 3.20)

project(kv_cache_backend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------
# GPU backend selection
# ------------------------------------------------------------

option(
    KV_USE_CUDA
    "Build llama.cpp with NVIDIA CUDA support"
    OFF
)

option(
    KV_USE_ROCM
    "Build llama.cpp with AMD ROCm/HIP support"
    OFF
)

if(KV_USE_CUDA AND KV_USE_ROCM)
    message(
        FATAL_ERROR
        "Enable only one GPU backend: KV_USE_CUDA or KV_USE_ROCM"
    )
endif()

# ------------------------------------------------------------
# Disable unnecessary llama.cpp targets
# ------------------------------------------------------------

set(
    LLAMA_BUILD_EXAMPLES
    OFF
    CACHE BOOL
    "Disable llama.cpp examples"
    FORCE
)

set(
    LLAMA_BUILD_TESTS
    OFF
    CACHE BOOL
    "Disable llama.cpp tests"
    FORCE
)

set(
    LLAMA_BUILD_SERVER
    OFF
    CACHE BOOL
    "Disable llama.cpp server"
    FORCE
)

set(
    LLAMA_BUILD_TOOLS
    OFF
    CACHE BOOL
    "Disable llama.cpp tools"
    FORCE
)

# ------------------------------------------------------------
# Configure NVIDIA CUDA
# ------------------------------------------------------------

if(KV_USE_CUDA)
    message(
        STATUS
        "Configuring kv_backend for NVIDIA CUDA"
    )

    set(
        GGML_CUDA
        ON
        CACHE BOOL
        "Enable CUDA backend"
        FORCE
    )

    set(
        GGML_HIP
        OFF
        CACHE BOOL
        "Disable HIP backend"
        FORCE
    )

    add_compile_definitions(
        KV_BACKEND_CUDA=1
    )

# ------------------------------------------------------------
# Configure AMD ROCm/HIP
# ------------------------------------------------------------

elseif(KV_USE_ROCM)
    message(
        STATUS
        "Configuring kv_backend for AMD ROCm/HIP"
    )

    set(
        GGML_CUDA
        OFF
        CACHE BOOL
        "Disable CUDA backend"
        FORCE
    )

    set(
        GGML_HIP
        ON
        CACHE BOOL
        "Enable HIP backend"
        FORCE
    )

    # AMD MI300X uses the gfx942 architecture.
    if(NOT DEFINED AMDGPU_TARGETS)
        set(
            AMDGPU_TARGETS
            "gfx942"
            CACHE STRING
            "AMD GPU architecture targets"
            FORCE
        )
    endif()

    add_compile_definitions(
        KV_BACKEND_ROCM=1
    )

# ------------------------------------------------------------
# CPU-only fallback
# ------------------------------------------------------------

else()
    message(
        WARNING
        "No GPU backend selected. Building CPU-only."
    )

    set(
        GGML_CUDA
        OFF
        CACHE BOOL
        "Disable CUDA backend"
        FORCE
    )

    set(
        GGML_HIP
        OFF
        CACHE BOOL
        "Disable HIP backend"
        FORCE
    )

    add_compile_definitions(
        KV_BACKEND_CPU=1
    )
endif()

# ------------------------------------------------------------
# Add llama.cpp
# ------------------------------------------------------------

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/llama_build
)

# ------------------------------------------------------------
# Backend executable
# ------------------------------------------------------------

add_executable(
    kv_backend

    main.cpp
    llama_runner.cpp
    kv_cache_optimizer.cpp
)

# ------------------------------------------------------------
# Header locations
# ------------------------------------------------------------

target_include_directories(
    kv_backend

    PRIVATE

    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include

    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp/ggml/include
)

# ------------------------------------------------------------
# Link llama.cpp
# ------------------------------------------------------------

target_link_libraries(
    kv_backend

    PRIVATE

    llama
)

# ------------------------------------------------------------
# Compiler warnings
# ------------------------------------------------------------

if(
    CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    OR
    CMAKE_CXX_COMPILER_ID MATCHES "Clang"
)
    target_compile_options(
        kv_backend

        PRIVATE

        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# ------------------------------------------------------------
# Print selected configuration
# ------------------------------------------------------------

message(
    STATUS
    "C++ standard: ${CMAKE_CXX_STANDARD}"
)

message(
    STATUS
    "KV_USE_CUDA: ${KV_USE_CUDA}"
)

message(
    STATUS
    "KV_USE_ROCM: ${KV_USE_ROCM}"
)

if(KV_USE_ROCM)
    message(
        STATUS
        "AMDGPU_TARGETS: ${AMDGPU_TARGETS}"
    )
endif()