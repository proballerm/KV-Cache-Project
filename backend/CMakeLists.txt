cmake_minimum_required(VERSION 3.20)
project(kv_cache_backend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GGML_CUDA ON CACHE BOOL "Enable CUDA")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples")
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable tests")
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable server")

add_subdirectory(../third_party/llama.cpp llama_build)

add_executable(kv_backend
    main.cpp
    llama_runner.cpp
)

target_include_directories(kv_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ../third_party/llama.cpp/include
    ../third_party/llama.cpp/ggml/include
)

target_link_libraries(kv_backend PRIVATE
    llama
)