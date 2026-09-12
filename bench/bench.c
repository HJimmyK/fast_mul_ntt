// MIT License
//
// Copyright (c) 2025 Jecricho Knox
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// bench.c —— 性能基准测试：按操作数长度扫描，统计乘法耗时并导出 CSV

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "fast_mul.h"

void test_mul_time(int len1, int len2) {
    u64* in1 = (u64*)malloc(len1 * sizeof(u64));
    u64* in2 = (u64*)malloc(len2 * sizeof(u64));
    u64* out = (u64*)malloc((len1 + len2) * sizeof(u64));

    srand(11);

    for (int i = 0; i < len1; i++) {
        in1[i] = (u64)rand() * rand();
        in2[i] = in1[i];
    }

    abs_mul64(in1, len1, in2, len2, out);

    free(in1);
    free(in2);
    free(out);
}

void test_sqr_time(int len1) {
    u64* in1 = (u64*)malloc(len1 * sizeof(u64));
    u64* out = (u64*)malloc((len1 * 2) * sizeof(u64));

    srand(12);

    for (int i = 0; i < len1; i++) {
        in1[i] = (u64)rand() * rand();
    }
    clock_t start = clock();
    abs_sqr64(in1, len1, out);
    clock_t end = clock();

    double elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("%.3f\n", elapsed);

    free(in1);
    free(out);
}

long long measure_real_time(int len1, int len2) {
    u64* in1 = (u64*)malloc(len1 * sizeof(u64));
    u64* in2 = (u64*)malloc(len2 * sizeof(u64));
    u64* out = (u64*)malloc((len1 + len2) * sizeof(u64));

    srand(time(NULL));

    for (int i = 0; i < len1; i++) {
        in1[i] = (u64)rand() * rand();
        in2[i] = in1[i];
    }
    struct timespec start, end;

    // 获取起始时间（TIME_UTC：UTC时间标准，保证跨平台一致性）
    timespec_get(&start, TIME_UTC);

    abs_mul64(in1, len1, in2, len2, out);

    // 获取结束时间
    timespec_get(&end, TIME_UTC);

    free(in1);
    free(in2);
    free(out);

    // 计算时间差：先转成纳秒，再转换为微秒（避免精度丢失）
    long long start_ns = (long long)start.tv_sec * 1000000000LL + start.tv_nsec;
    long long end_ns = (long long)end.tv_sec * 1000000000LL + end.tv_nsec;
    long long duration_us = (end_ns - start_ns) / 1000LL;

    return duration_us;
}

int write_longlong_arr_to_csv(const long long* arr, int arr_len, const char* csv_path) {
    FILE* fp = fopen(csv_path, "w");
    if (fp == NULL) {
        perror("文件打开失败");
        return -1;
    }

    for (int i = 0; i < arr_len; i++) {
        fprintf(fp, "%lld", arr[i]);

        if (i != arr_len - 1) {
            fprintf(fp, ",");
        }
    }

    fprintf(fp, "\n");

    fclose(fp);
    printf("success write %s\n", csv_path);
    return 0;
}

int main() {
    const size_t min_len = 10000;
    const size_t max_len = 10000000;
    const size_t step = 33300;
    const size_t time_len = (max_len - min_len) / step + 1;
    long long times[time_len];

    for (size_t i = 0; i < time_len; i++) {
        size_t len1 = min_len + i * step;
        size_t len2 = len1;
        long long t1 = measure_real_time(len1, len2);
        long long t2 = measure_real_time(len1, len2);
        long long t3 = measure_real_time(len1, len2);
        times[i] = (t1 + t2 + t3) / 3;
        printf("%llu, %llu\n", i, times[i]);
    }
    write_longlong_arr_to_csv(times, time_len, "cc_ntt-crt_times.csv");

    return 0;
}
