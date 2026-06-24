cmake_minimum_required(VERSION 3.10)

project(kv_backend)

set(CMAKE_CXX_STANDARD 17)

add_executable(
    kv_backend
    main.cpp
)

target_include_directories(
    kv_backend PRIVATE include
)

target_link_libraries(
    kv_backend
    ws2_32
)