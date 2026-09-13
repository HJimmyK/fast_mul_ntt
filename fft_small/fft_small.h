// fft_small —— 从 FLINT 提炼的单线程 AVX2 大整数乘法（公开接口）
//
// 本目录代码移植自 FLINT (https://flintlib.org)，遵循 LGPL-3.0-or-later，
// 与仓库其余 MIT 代码相互独立。要求编译期 -mavx2 -mfma、运行期 CPU 支持。

#ifndef FFT_SMALL_MUL_H
#define FFT_SMALL_MUL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long long u64;

/*
 * 大整数乘法：out = in1 * in2
 * in1、in2、out 均为小端 64 位 limb 数组，out 长度必须为 len1 + len2
 * in1 与 in2 指向不同内存块；同一块内存且等长时自动走平方快速路径
 */
void abs_mul64(const u64* in1, u64 len1, const u64* in2, u64 len2, u64* out);

/*
 * 大整数平方：out = in1 * in1
 * out 长度必须为 2 * len1
 */
void abs_sqr64(const u64* in1, u64 len1, u64* out);

/*
 * 可选：手动初始化/释放全局上下文（约 4 MB 常量表）。
 * 不调用也可以：首次乘法时自动初始化，进程退出时由操作系统回收。
 * 限制：len1、len2 均至少为 1；较短操作数不超过约 25 亿 limb（约 20 GB）。
 */
void fft_small_init(void);
void fft_small_clear(void);

#ifdef __cplusplus
}
#endif

#endif // FFT_SMALL_MUL_H
