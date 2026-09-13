cmake_minimum_required(VERSION 3.10)
project(fast_mul_ntt LANGUAGES C)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 自研实现使用 GNU 内联汇编（x86-64），仅支持 GCC / Clang
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -g)
endif()

# ---------------- 3 模数 NTT ----------------

add_library(fast_mul STATIC src/fast_mul.c)
target_include_directories(fast_mul PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(fast_mul PRIVATE -ffast-math -fomit-frame-pointer)

add_executable(bench bench/bench.c)
target_link_libraries(bench PRIVATE fast_mul)
target_compile_options(bench PRIVATE -ffast-math -fomit-frame-pointer)

# ---------------- fft_small：FLINT 移植 ----------------
# 对外接口只有 fft_small.h（abs_mul64 / abs_sqr64 / fft_small_init / fft_small_clear），
# 其余 fsd_*.c/h 均为模块内部实现。

add_library(fft_small STATIC
    fft_small/fft_small.c
    fft_small/fsd_mpn.c
    fft_small/fsd_fft_ctx.c
    fft_small/fsd_fft.c
    fft_small/fsd_ifft.c
    fft_small/fsd_mpn_mul.c)
target_include_directories(fft_small PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/fft_small)
# 需要 AVX2 + FMA；严禁 -ffast-math（FFT 的浮点模乘依赖严格的舍入语义）
target_compile_options(fft_small PRIVATE -mavx2 -mfma -O3 -g)

add_executable(fsd_test fft_small/fsd_test.c)
target_link_libraries(fsd_test PRIVATE fft_small)
target_compile_options(fsd_test PRIVATE -mavx2 -mfma -O2 -g)

add_executable(bench_fft_small bench/bench_fft_small.c)
target_link_libraries(bench_fft_small PRIVATE fft_small)
target_compile_options(bench_fft_small PRIVATE -mavx2 -mfma -O3 -g)
