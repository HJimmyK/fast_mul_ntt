cmake_minimum_required(VERSION 3.10)
project(fast_mul_ntt LANGUAGES C)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 代码使用 GNU 内联汇编（x86-64），仅支持 GCC / Clang
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -g)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ffast-math -fomit-frame-pointer")
endif()

# 核心静态库
add_library(fast_mul STATIC src/fast_mul.c)
target_include_directories(fast_mul PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# 基准测试程序
add_executable(bench bench/bench.c)
target_link_libraries(bench PRIVATE fast_mul)
