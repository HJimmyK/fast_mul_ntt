// fft_small 性能基准：与 bench/bench.c 相同的长度扫描方式
//
// 操作数长度从 10 000 到 10 000 000 limb，步长 33 300，每点 3 次取平均，
// 结果写入运行目录下 cc_fft_small_times.csv（单位微秒）。

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "fft_small.h"

static long long measure_real_time(long long len1, long long len2) {
    u64* in1 = (u64*)malloc(len1 * sizeof(u64));
    u64* in2 = (u64*)malloc(len2 * sizeof(u64));
    u64* out = (u64*)malloc((len1 + len2) * sizeof(u64));

    srand((unsigned)time(NULL));

    for (long long i = 0; i < len1; i++) {
        in1[i] = ((u64)rand() << 45) ^ ((u64)rand() << 30) ^ ((u64)rand() << 15) ^ (u64)rand();
        in2[i] = in1[i];
    }
    struct timespec start, end;

    timespec_get(&start, TIME_UTC);

    abs_mul64(in1, (u64)len1, in2, (u64)len2, out);

    timespec_get(&end, TIME_UTC);

    free(in1);
    free(in2);
    free(out);

    long long start_ns = (long long)start.tv_sec * 1000000000LL + start.tv_nsec;
    long long end_ns = (long long)end.tv_sec * 1000000000LL + end.tv_nsec;
    return (end_ns - start_ns) / 1000LL;
}

int main(void) {
    const long long min_len = 10000;
    const long long max_len = 10000000;
    const long long step = 33300;
    const long long time_len = (max_len - min_len) / step + 1;
    long long* times = (long long*)malloc(time_len * sizeof(long long));

    for (long long i = 0; i < time_len; i++) {
        long long len1 = min_len + i * step;
        long long t1 = measure_real_time(len1, len1);
        long long t2 = measure_real_time(len1, len1);
        long long t3 = measure_real_time(len1, len1);
        times[i] = (t1 + t2 + t3) / 3;
        printf("%lld, %lld\n", i, times[i]);
        fflush(stdout);
    }

    FILE* fp = fopen("cc_fft_small_times.csv", "w");
    if (fp == NULL) {
        perror("文件打开失败");
        return 1;
    }
    for (long long i = 0; i < time_len; i++) {
        fprintf(fp, "%lld", times[i]);
        if (i != time_len - 1)
            fprintf(fp, ",");
    }
    fprintf(fp, "\n");
    fclose(fp);
    printf("success write cc_fft_small_times.csv\n");

    free(times);
    fft_small_clear();
    return 0;
}
