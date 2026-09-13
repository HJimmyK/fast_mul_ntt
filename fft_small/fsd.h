// fsd.h —— fft_small 移植版的内部公共定义（不属于公开接口）
//
// ★ 本头文件只供 fft_small 模块自身的 .c 文件（以及 fsd_test.c 自测）使用，
//   库的使用者只应包含 fft_small.h（见其中的 abs_mul64 / abs_sqr64）。
//   除 fft_small.h 声明的符号外，本模块其余函数均为内部实现细节，
//   单个 .c 文件内使用的函数一律 static，跨文件的内部函数集中声明于本文件。
//
// 模块概览（详见各 .c 文件）：
//   sd_fft_ctx        单个 ~50 bit NTT 素数的上下文（旋转因子表等）
//   sd_fft_trunc      截断正变换（DIF，输出按位翻转序）
//   sd_ifft_trunc     截断逆变换（输出 = 2^L * 原序列）
//   mpn_ctx           8 个素数 + CRT 数据 + 2 的幂表 + 尺寸方案表
//   mpn_ctx_mpn_mul   大整数乘法主流程（limb 数组 -> FFT -> 点乘 -> CRT）

#ifndef FSD_H
#define FSD_H

#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "fsd_vec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 注意：Windows 的 long 只有 32 位（LLP64），必须用 long long */
typedef unsigned long long ulong;
typedef long long slong;
typedef unsigned long long u64;

#define VEC_SZ 4

/* 标识拼接宏：CAT(vec4d, add) 展开为 vec4d_add */
#define CAT_(a, b) a##_##b
#define CAT(a, b) CAT_(a, b)
#define CAT3_(a, b, c) a##_##b##_##c
#define CAT3(a, b, c) CAT3_(a, b, c)
#define CAT4_(a, b, c, d) a##_##b##_##c##_##d
#define CAT4(a, b, c, d) CAT4_(a, b, c, d)

#if defined(__GNUC__)
#define FSD_UNUSED(x) __attribute__((unused)) x
#else
#define FSD_UNUSED(x) x
#endif

#ifdef NDEBUG
/* 发布版：把断言编译成不求值的表达式（保留对表达式的语法检查） */
#define FSD_ASSERT(x) ((void)sizeof(!(x) ? 1 : 0))
#else
#define FSD_ASSERT(x) assert(x)
#endif

/* 出错即终止（打印消息后 abort） */
void fsd_abort(const char* msg);

/* 对齐内存分配（Windows 用 _aligned_malloc，其余用 C11 aligned_alloc） */
void* fsd_aligned_alloc(size_t align, size_t size);
void fsd_aligned_free(void* p);

/************************* 小工具函数（对应 ulong_extras） ******************/

#define UWORD(x) ((ulong)(x##ULL))

static inline ulong n_pow2(ulong k) { return UWORD(1) << k; }

/* 有效位数：n_nbits(x) = floor(log2(x)) + 1，x == 0 时为 0 */
static inline ulong n_nbits(ulong x) { return x == 0 ? 0 : (ulong)(64 - __builtin_clzll(x)); }

static inline ulong n_trailing_zeros(ulong x) { return x == 0 ? 0 : (ulong)__builtin_ctzll(x); }

static inline ulong n_min(ulong x, ulong y) { return x < y ? x : y; }
static inline ulong n_max(ulong x, ulong y) { return x > y ? x : y; }

/* 向上取整除法与向上取整 log2 */
static inline ulong n_cdiv(ulong a, ulong b) { return (a + b - 1) / b; }
static inline ulong n_clog2(ulong x) { return x <= 1 ? 0 : n_nbits(x - 1); }
static inline ulong n_round_up(ulong x, ulong y) { return y * n_cdiv(x, y); }

/* 取 x 的低 len 位并反转（位翻转序，len 为位数） */
static inline ulong n_revbin(ulong x, ulong len) {
    ulong r = 0;
    for (ulong i = 0; i < len; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

/* 64x64 -> 128 位乘法，返回 (hi, lo) */
#define umul_ppmm(hi, lo, a, b)               \
    do {                                      \
        unsigned __int128 _p =                \
            (unsigned __int128)(a) * (unsigned __int128)(b); \
        (hi) = (ulong)(_p >> 64);             \
        (lo) = (ulong)_p;                     \
    } while (0)

/************************* nmod / 素数 **************************************/

#if defined(_MSC_VER)
#include <intrin.h> /* fft_small.c 的 __cpuid 需要 */
#endif

/* 128 位除以 64 位（要求 hi < d），逐位长除法。
   只用于初始化与每个素数一次的换算，属冷路径；这样实现可以
   不依赖 _udiv128 / __udivti3 等编译器运行时。 */
static inline ulong fsd_udiv128(ulong hi, ulong lo, ulong d, ulong* rem) {
    ulong q = 0;
    ulong r = hi;
    for (int i = 63; i >= 0; i--) {
        unsigned __int128 t = ((unsigned __int128)r << 1) | ((lo >> i) & 1);
        if (t >= d) {
            t -= d;
            q |= (ulong)1 << i;
        }
        r = (ulong)t;
    }
    *rem = r;
    return q;
}

typedef struct {
    ulong n; /* 模数 */
} nmod_t;

static inline ulong nmod_mul(ulong a, ulong b, nmod_t mod) {
    ulong hi, lo, rem;
    umul_ppmm(hi, lo, a, b);
    (void)fsd_udiv128(hi, lo, mod.n, &rem);
    return rem;
}

/* (hi*2^64 + lo) mod mod.n，要求 hi < mod.n（内部经 fsd_udiv128 逐位除） */
static inline ulong nmod_red2(ulong hi, ulong lo, nmod_t mod) {
    ulong rem;
    (void)fsd_udiv128(hi, lo, mod.n, &rem);
    return rem;
}

ulong nmod_pow_ui(ulong a, ulong exp, nmod_t mod);
/* mod.n 为素数，用费马小定理求逆 */
static inline ulong nmod_inv(ulong a, nmod_t mod) { return nmod_pow_ui(a, mod.n - 2, mod); }

/* 确定性 Miller-Rabin 素性检测（对 2^64 以内全部有效） */
int n_is_prime(ulong n);
/* 求模 p 的一个二次非剩余（用于构造 2 的幂次本原根） */
ulong n_quadratic_nonresidue(ulong p);

/* 检验素数能否用于 double 域模乘（见 fsd_mpn.c 的误差分析） */
int fft_small_mulmod_satisfies_bounds(ulong nn);

/************************* mpn 小函数 ***************************************/

/* rp[0..n) = xp[0..n) * limb + 进位，返回最高进位 */
ulong fsd_mpn_mul_1(ulong* rp, const ulong* xp, ulong n, ulong limb);
ulong fsd_mpn_add_n(ulong* rp, const ulong* xp, const ulong* yp, ulong n);
ulong fsd_mpn_sub_n(ulong* rp, const ulong* xp, const ulong* yp, ulong n);
ulong fsd_mpn_sub_1(ulong* rp, const ulong* xp, ulong n, ulong limb);
ulong fsd_mpn_rshift(ulong* rp, const ulong* xp, ulong n, ulong cnt);
ulong fsd_mpn_mod_1(const ulong* xp, ulong n, ulong d);
/* 精确除法：要求 d 整除 (xp, n) */
void fsd_mpn_divexact_1(ulong* rp, const ulong* xp, ulong n, ulong d);

#define flint_mpn_copyi(dst, src, n) memcpy((dst), (src), (n) * sizeof(ulong))

/* cmp(a, b*2^e)，a 不必规范化（用于校验 CRT 素数乘积是否足够大） */
int flint_mpn_cmp_ui_2exp(const ulong* a, ulong an, ulong b, ulong e);

/* 带进位加，编译器可识别为 adc 模式 */
static inline unsigned char _addcarry_ulong(unsigned char cf, ulong x, ulong y, ulong* sum) {
    ulong s = x + y;
    ulong c = s < x;
    ulong t = s + (ulong)cf;
    c += t < s;
    *sum = t;
    return (unsigned char)c;
}

/************************* sd_fft_ctx **************************************/

#define BLK_SZ 256           /* 数据块大小（double 个数） */
#define LG_BLK_SZ 8
/* 初始化时连续填到第 9 张表：basecase_9 (512 点) 需要读 w2tab[8] */
#define SD_FFT_CTX_W2TAB_INIT 9
#define SD_FFT_CTX_W2TAB_SIZE 50

typedef struct sd_fft_ctx_struct {
    double p;    /* 素数（以 double 精确表示） */
    double pinv; /* 1/p（就近舍入的 double） */
    nmod_t mod;
    ulong primitive_2power_root;
    /* w2tab[k] 是长 2^(k-1)（k>=1）的 2^(k+1) 次单位根表，按 revbin 排列；
       前 SD_FFT_CTX_W2TAB_INIT 张连续存放在 w2tab[0] 指向的缓冲区里 */
    double* w2tab[SD_FFT_CTX_W2TAB_SIZE];
    unsigned int w2tab_depth;
} sd_fft_ctx_struct;

typedef sd_fft_ctx_struct sd_fft_ctx_t[1];

/* 第 I 块的起始下标 / 指针；深度 depth 的变换共 n_pow2(depth) 个 double */
#define sd_fft_ctx_blk_offset(I) ((I)*BLK_SZ)
#define sd_fft_ctx_blk_index(ptr, I) ((ptr) + sd_fft_ctx_blk_offset(I))
#define sd_fft_ctx_data_size(depth) n_pow2(depth)

/* j != 0 时：j_bits = j 的有效位数，j_r = 去掉最高位后的剩余（正向蝶形用） */
#define SET_J_BITS_AND_J_R(j_bits, j_r, j) \
    j_bits = n_nbits(j);                   \
    j_r = (j_bits > 0) ? ((j) - n_pow2(j_bits - 1)) : 0;

/* 镜像下标（逆向蝶形用）：恰好有 w2tab[jb][2^jb-1-j] = -w^-1 */
#define SET_J_BITS_AND_J_MR(j_bits, j_mr, j) \
    j_bits = n_nbits(j);                     \
    j_mr = (j_bits > 0) ? (n_pow2(j_bits) - 1 - (j)) : 0;

/* 基数 2 蝶形在扭转 j 处的旋转因子（与 doc/fft.jl 的 w^revbits(2j,L) 一致） */
static inline double sd_fft_ctx_w2(const sd_fft_ctx_t Q, ulong j) {
    ulong j_bits, j_r;
    SET_J_BITS_AND_J_R(j_bits, j_r, j);
    return Q->w2tab[j_bits][j_r];
}

void sd_fft_ctx_init_prime(sd_fft_ctx_t Q, ulong pp);
void sd_fft_ctx_clear(sd_fft_ctx_t Q);
/* 保证 w2tab 已建到 depth 层（不足则现场扩展） */
void sd_fft_ctx_fit_depth(sd_fft_ctx_t Q, ulong depth);

/* 截断正 / 逆变换（内部于 fsd_fft.c / fsd_ifft.c） */
void sd_fft_trunc(sd_fft_ctx_t Q, double* d, ulong L, ulong itrunc, ulong otrunc);
void sd_ifft_trunc(sd_fft_ctx_t Q, double* d, ulong L, ulong trunc);

/************************* crt_data ****************************************/

typedef struct {
    ulong prime;
    ulong coeff_len; /* m 或 m+1 */
    ulong nprimes;
    /* 布局: nprimes*coeff_len (cofactors) + coeff_len (prod) + nprimes (red) */
    ulong* data;
} crt_data_struct;

typedef crt_data_struct crt_data_t[1];

static inline ulong* crt_data_co_prime(crt_data_t C, ulong i) { return C->data + i * C->coeff_len; }
static inline ulong* crt_data_prod_primes(crt_data_t C) { return C->data + C->nprimes * C->coeff_len; }
static inline ulong* crt_data_co_prime_red(crt_data_t C, ulong i) {
    return crt_data_prod_primes(C) + C->coeff_len + i;
}

/************************* mpn_ctx *****************************************/

#define MPN_CTX_NCRTS 8
#define MPN_CTX_TWO_POWER_TAB_SIZE 192
#define MAX_NPROFILES 15

typedef void (*to_ffts_func)(sd_fft_ctx_struct* Rffts, double* d, ulong dstride,
                             const ulong* a, ulong an, ulong atrunc, const vec4d* two_pow,
                             ulong start_easy, ulong stop_easy, ulong start_hard, ulong stop_hard);

/* 一个尺寸方案：np 个素数、每组 bits 位、较短操作数上限 bn_bound */
typedef struct {
    ulong np;
    ulong bits;
    ulong bn_bound;
    to_ffts_func to_ffts;
} profile_struct;

typedef struct mpn_ctx_struct {
    sd_fft_ctx_struct ffts[MPN_CTX_NCRTS];
    crt_data_struct crts[MPN_CTX_NCRTS];
    /* vec_two_pow_tab[nvs-1][i*nvs+l] 第 k 路为 2^i mod ffts[4*l+k].p */
    vec4d* vec_two_pow_tab[MPN_CTX_NCRTS / VEC_SZ];
    vec4d* vec_two_pow_buffer;
    profile_struct profiles[MAX_NPROFILES];
    ulong profiles_size;
    void* buffer;      /* 乘法期间的临时缓冲 */
    ulong buffer_alloc;
} mpn_ctx_struct;

typedef mpn_ctx_struct mpn_ctx_t[1];

void mpn_ctx_init(mpn_ctx_t R, ulong p);
void mpn_ctx_clear(mpn_ctx_t R);
/* z = a * b，z 需要能容纳 an+bn 个 limb；a 与 b 可以是同一块内存（此时走平方路径） */
void mpn_ctx_mpn_mul(mpn_ctx_t R, ulong* z, const ulong* a, ulong an, const ulong* b, ulong bn);

/* CRT 大数乘加 / 缩减模板（multi_add / _big_mul / _reduce_big_sum） */
#include "fsd_crt.h"

#ifdef __cplusplus
}
#endif

#endif // FSD_H
