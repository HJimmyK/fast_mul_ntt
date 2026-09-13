// AVX2 向量原语：vec1d / vec4d / vec8d
//
// 约定：
//   - 模数 n 是小于 2^50 的素数，ninv = 1.0/n（就近舍入的 double）
//   - 剩余用 (-n, n) 内的整数值 double 表示（部分场合为 (-3n/2, 3n/2)）
//   - mulmod 用 FMA 的 double-double 技巧，结果精确同余，见
//     mulmod_satisfies_bounds.c 的误差分析：乘积 |a*b| < 4n^2 时结果落在 (-3n/2, 3n/2)

#ifndef FSD_VEC_H
#define FSD_VEC_H

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

#if !defined(__AVX2__) || !defined(__FMA__)
#error "fft_small 模块需要 AVX2 与 FMA 指令集（编译时加 -mavx2 -mfma）"
#endif

/****************************** vec1d **************************************/

typedef double vec1d;

static inline vec1d vec1d_set_d(double x) { return x; }
static inline vec1d vec1d_add(vec1d a, vec1d b) { return a + b; }
static inline vec1d vec1d_sub(vec1d a, vec1d b) { return a - b; }
static inline vec1d vec1d_mul(vec1d a, vec1d b) { return a * b; }
static inline vec1d vec1d_fnmadd(vec1d a, vec1d b, vec1d c) { return __builtin_fma(-a, b, c); }
static inline vec1d vec1d_fmadd(vec1d a, vec1d b, vec1d c) { return __builtin_fma(a, b, c); }
static inline vec1d vec1d_fmsub(vec1d a, vec1d b, vec1d c) { return __builtin_fma(a, b, -c); }

static inline vec1d vec1d_round(vec1d a) {
    return _mm_cvtsd_f64(_mm_round_sd(_mm_setzero_pd(), _mm_set_sd(a),
                                       _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

static inline vec1d vec1d_zero(void) { return 0.0; }
static inline vec1d vec1d_one(void) { return 1.0; }

/* a*b mod n，精确同余，结果在 (-3n/2, 3n/2)（当 |a*b| < 4n^2） */
static inline vec1d vec1d_mulmod(vec1d a, vec1d b, vec1d n, vec1d ninv) {
    vec1d h = a * b;
    vec1d l = vec1d_fmsub(a, b, h); /* a*b - h，由 FMA 单次舍入，精确 */
    vec1d q = vec1d_round(h * ninv);
    return vec1d_add(l, vec1d_fnmadd(q, n, h));
}

static inline vec1d vec1d_nmulmod(vec1d a, vec1d b, vec1d n, vec1d ninv) {
    return -vec1d_mulmod(a, b, n, ninv);
}

static inline vec1d vec1d_reduce_to_pm1n(vec1d a, vec1d n, vec1d ninv) {
    vec1d q = vec1d_round(a * ninv);
    return vec1d_fnmadd(q, n, a);
}

static inline vec1d vec1d_reduce_to_0n(vec1d a, vec1d n, vec1d ninv) {
    vec1d r = vec1d_reduce_to_pm1n(a, n, ninv);
    return (r < 0) ? (r + n) : r;
}

/* [0, n) -> (-n/2, n/2] */
static inline vec1d vec1d_reduce_0n_to_pmhn(vec1d a, vec1d n) {
    return (a + a > n) ? (a - n) : a;
}

/* (-3n/2, 3n/2) -> (-n/2, n/2] */
static inline vec1d vec1d_reduce_pm1n_to_pmhn(vec1d a, vec1d n) {
    if (a > 0.5 * n)
        return a - n;
    if (a < -0.5 * n)
        return a + n;
    return a;
}

/****************************** vec4d **************************************/

typedef __m256d vec4d; /* 4 路 double */
typedef __m256i vec4n; /* 4 路 u64 */

static inline vec4d vec4d_set_d(double x) { return _mm256_set1_pd(x); }
static inline vec4d vec4d_set_d4(double a, double b, double c, double d) {
    return _mm256_set_pd(d, c, b, a);
}
static inline vec4d vec4d_load(const double* p) { return _mm256_loadu_pd(p); }
static inline vec4d vec4d_load_aligned(const double* p) { return _mm256_load_pd(p); }
static inline void vec4d_store(double* p, vec4d a) { _mm256_storeu_pd(p, a); }
static inline void vec4d_store_aligned(double* p, vec4d a) { _mm256_store_pd(p, a); }

static inline vec4d vec4d_add(vec4d a, vec4d b) { return _mm256_add_pd(a, b); }
static inline vec4d vec4d_sub(vec4d a, vec4d b) { return _mm256_sub_pd(a, b); }
static inline vec4d vec4d_mul(vec4d a, vec4d b) { return _mm256_mul_pd(a, b); }
static inline vec4d vec4d_fmadd(vec4d a, vec4d b, vec4d c) { return _mm256_fmadd_pd(a, b, c); }
static inline vec4d vec4d_fnmadd(vec4d a, vec4d b, vec4d c) { return _mm256_fnmadd_pd(a, b, c); }
static inline vec4d vec4d_fmsub(vec4d a, vec4d b, vec4d c) { return _mm256_fmsub_pd(a, b, c); }

static inline vec4d vec4d_round(vec4d a) {
    return _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static inline vec4d vec4d_zero(void) { return _mm256_setzero_pd(); }
static inline vec4d vec4d_one(void) { return _mm256_set1_pd(1.0); }

static inline vec4d vec4d_mulmod(vec4d a, vec4d b, vec4d n, vec4d ninv) {
    vec4d h = vec4d_mul(a, b);
    vec4d l = vec4d_fmsub(a, b, h);
    vec4d q = vec4d_round(vec4d_mul(h, ninv));
    return vec4d_add(l, vec4d_fnmadd(q, n, h));
}

static inline vec4d vec4d_nmulmod(vec4d a, vec4d b, vec4d n, vec4d ninv) {
    return _mm256_sub_pd(_mm256_setzero_pd(), vec4d_mulmod(a, b, n, ninv));
}

static inline vec4d vec4d_reduce_to_pm1n(vec4d a, vec4d n, vec4d ninv) {
    vec4d q = vec4d_round(vec4d_mul(a, ninv));
    return vec4d_fnmadd(q, n, a);
}

static inline vec4d vec4d_reduce_to_0n(vec4d a, vec4d n, vec4d ninv) {
    vec4d r = vec4d_reduce_to_pm1n(a, n, ninv);
    return _mm256_blendv_pd(r, _mm256_add_pd(r, n), _mm256_cmp_pd(r, _mm256_setzero_pd(), _CMP_LT_OQ));
}

static inline vec4d vec4d_reduce_0n_to_pmhn(vec4d a, vec4d n) {
    vec4d t = _mm256_add_pd(a, a);
    __m256d mask = _mm256_cmp_pd(t, n, _CMP_GT_OQ);
    return _mm256_blendv_pd(a, _mm256_sub_pd(a, n), mask);
}

static inline vec4d vec4d_reduce_pm1n_to_pmhn(vec4d a, vec4d n) {
    vec4d half = _mm256_mul_pd(n, _mm256_set1_pd(0.5));
    vec4d am = _mm256_sub_pd(a, n);
    vec4d ap = _mm256_add_pd(a, n);
    vec4d r = _mm256_blendv_pd(a, am, _mm256_cmp_pd(a, half, _CMP_GT_OQ));
    r = _mm256_blendv_pd(r, ap, _mm256_cmp_pd(a, _mm256_sub_pd(_mm256_setzero_pd(), half), _CMP_LT_OQ));
    return r;
}

static inline double vec4d_get_index(vec4d a, size_t i) {
    double t[4] __attribute__((aligned(32)));
    vec4d_store_aligned(t, a);
    return t[i];
}

/* 逆序 [d,c,b,a] */
static inline vec4d vec4d_permute_3_2_1_0(vec4d a) {
    return _mm256_permute4x64_pd(a, _MM_SHUFFLE(0, 1, 2, 3));
}

/* unpacklo(a,b) = [a0,b0,a2,b2] 再按 0,2,1,3 重排 = [a0,a2,b0,b2] */
static inline vec4d vec4d_unpack_lo_permute_0_2_1_3(vec4d a, vec4d b) {
    return _mm256_permute4x64_pd(_mm256_unpacklo_pd(a, b), _MM_SHUFFLE(3, 1, 2, 0));
}

/* unpackhi(a,b) = [a1,b1,a3,b3] 再按 0,2,1,3 重排 = [a1,a3,b1,b3] */
static inline vec4d vec4d_unpack_hi_permute_0_2_1_3(vec4d a, vec4d b) {
    return _mm256_permute4x64_pd(_mm256_unpackhi_pd(a, b), _MM_SHUFFLE(3, 1, 2, 0));
}

/* unpacklo(a,b) = [a0,b0,a2,b2] 再按 3,1,2,0 重排 = [b2,b0,a2,a0] */
static inline vec4d vec4d_unpacklo_permute_3_1_2_0(vec4d a, vec4d b) {
    return _mm256_permute4x64_pd(_mm256_unpacklo_pd(a, b), _MM_SHUFFLE(0, 2, 1, 3));
}

/* unpackhi(a,b) = [a1,b1,a3,b3] 再按 3,1,2,0 重排 = [b3,b1,a3,a1] */
static inline vec4d vec4d_unpackhi_permute_3_1_2_0(vec4d a, vec4d b) {
    return _mm256_permute4x64_pd(_mm256_unpackhi_pd(a, b), _MM_SHUFFLE(0, 2, 1, 3));
}

/* 4x4 转置：x0..x3 为 4 个 4 路 double（按值传递，输入输出可以同名） */
#define VEC4D_TRANSPOSE(x0, x1, x2, x3, y0, y1, y2, y3)             \
    do {                                                             \
        vec4d _t0 = _mm256_unpacklo_pd(x0, x1);                      \
        vec4d _t1 = _mm256_unpackhi_pd(x0, x1);                      \
        vec4d _t2 = _mm256_unpacklo_pd(x2, x3);                      \
        vec4d _t3 = _mm256_unpackhi_pd(x2, x3);                      \
        y0 = _mm256_permute2f128_pd(_t0, _t2, 0x20);                 \
        y1 = _mm256_permute2f128_pd(_t1, _t3, 0x20);                 \
        y2 = _mm256_permute2f128_pd(_t0, _t2, 0x31);                 \
        y3 = _mm256_permute2f128_pd(_t1, _t3, 0x31);                 \
    } while (0)

/* [0, 2^50) 的整 double -> 4 路 u64。AVX2 没有 pd->epi64 指令，
   按 2^25 拆成两段分别走 int32 魔数转换再拼接。 */
static inline __m128i fsd_dbl4_to_i32x4(vec4d d) {
    /* d 中每路属于 [0, 2^31)：加 2^52+2^51 后低 32 位尾数恰为整数值 */
    const vec4d magic = _mm256_set1_pd(6755399441055744.0);
    __m256i bits = _mm256_castpd_si256(_mm256_add_pd(d, magic));
    __m128i lo = _mm256_castsi256_si128(bits);             /* 车道 0,1 的低 32 位在字节 0,8 */
    __m128i hi = _mm256_extracti128_si256(bits, 1);        /* 车道 2,3 的低 32 位在字节 0,8 */
    __m128i l = _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 2, 2, 0));
    __m128i h = _mm_shuffle_epi32(hi, _MM_SHUFFLE(2, 2, 2, 0));
    return _mm_unpacklo_epi64(l, h);                       /* [d0, d1, d2, d3] */
}

static inline vec4n vec4d_convert_limited_vec4n(vec4d a) {
    const vec4d two25 = _mm256_set1_pd(33554432.0);       /* 2^25 */
    const vec4d inv25 = _mm256_set1_pd(1.0 / 33554432.0); /* 2^-25，乘法精确 */
    vec4d hi = _mm256_round_pd(_mm256_mul_pd(a, inv25), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    vec4d lo = _mm256_fnmadd_pd(hi, two25, a);            /* a - hi*2^25，精确 */
    __m256i hi64 = _mm256_cvtepu32_epi64(fsd_dbl4_to_i32x4(hi));
    __m256i lo64 = _mm256_cvtepu32_epi64(fsd_dbl4_to_i32x4(lo));
    return _mm256_or_si256(_mm256_slli_epi64(hi64, 25), lo64);
}

static inline void vec4n_store_unaligned(void* p, vec4n a) {
    _mm256_storeu_si256((__m256i*)p, a);
}

/****************************** vec8d **************************************/

/* AVX2 上 vec8d 是两个 __m256d；编译器内联后做标量替换，寄存器压力与
   FLINT 的 AVX2 构建一致 */
typedef struct {
    __m256d x, y;
} vec8d;

static inline vec8d vec8d_set_d(double d) {
    vec8d r = { _mm256_set1_pd(d), _mm256_set1_pd(d) };
    return r;
}
static inline vec8d vec8d_load(const double* p) {
    vec8d r = { _mm256_loadu_pd(p), _mm256_loadu_pd(p + 4) };
    return r;
}
static inline void vec8d_store(double* p, vec8d a) {
    _mm256_storeu_pd(p, a.x);
    _mm256_storeu_pd(p + 4, a.y);
}
static inline vec8d vec8d_add(vec8d a, vec8d b) {
    vec8d r = { _mm256_add_pd(a.x, b.x), _mm256_add_pd(a.y, b.y) };
    return r;
}
static inline vec8d vec8d_sub(vec8d a, vec8d b) {
    vec8d r = { _mm256_sub_pd(a.x, b.x), _mm256_sub_pd(a.y, b.y) };
    return r;
}
static inline vec8d vec8d_mul(vec8d a, vec8d b) {
    vec8d r = { _mm256_mul_pd(a.x, b.x), _mm256_mul_pd(a.y, b.y) };
    return r;
}
static inline vec8d vec8d_fmadd(vec8d a, vec8d b, vec8d c) {
    vec8d r = { _mm256_fmadd_pd(a.x, b.x, c.x), _mm256_fmadd_pd(a.y, b.y, c.y) };
    return r;
}
static inline vec8d vec8d_fnmadd(vec8d a, vec8d b, vec8d c) {
    vec8d r = { _mm256_fnmadd_pd(a.x, b.x, c.x), _mm256_fnmadd_pd(a.y, b.y, c.y) };
    return r;
}
static inline vec8d vec8d_fmsub(vec8d a, vec8d b, vec8d c) {
    vec8d r = { _mm256_fmsub_pd(a.x, b.x, c.x), _mm256_fmsub_pd(a.y, b.y, c.y) };
    return r;
}
static inline vec8d vec8d_round(vec8d a) {
    vec8d r = { _mm256_round_pd(a.x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
                _mm256_round_pd(a.y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) };
    return r;
}
static inline vec8d vec8d_zero(void) {
    vec8d r = { _mm256_setzero_pd(), _mm256_setzero_pd() };
    return r;
}

static inline vec8d vec8d_mulmod(vec8d a, vec8d b, vec8d n, vec8d ninv) {
    vec8d h = vec8d_mul(a, b);
    vec8d l = vec8d_fmsub(a, b, h);
    vec8d q = vec8d_round(vec8d_mul(h, ninv));
    return vec8d_add(l, vec8d_fnmadd(q, n, h));
}

static inline vec8d vec8d_nmulmod(vec8d a, vec8d b, vec8d n, vec8d ninv) {
    return vec8d_sub(vec8d_zero(), vec8d_mulmod(a, b, n, ninv));
}

static inline vec8d vec8d_reduce_to_pm1n(vec8d a, vec8d n, vec8d ninv) {
    vec8d q = vec8d_round(vec8d_mul(a, ninv));
    return vec8d_fnmadd(q, n, a);
}

static inline vec8d vec8d_reduce_pm1n_to_pmhn(vec8d a, vec8d n) {
    vec8d half = { _mm256_mul_pd(n.x, _mm256_set1_pd(0.5)), _mm256_mul_pd(n.y, _mm256_set1_pd(0.5)) };
    vec8d am = { _mm256_sub_pd(a.x, n.x), _mm256_sub_pd(a.y, n.y) };
    vec8d ap = { _mm256_add_pd(a.x, n.x), _mm256_add_pd(a.y, n.y) };
    vec8d neghalf = { _mm256_sub_pd(_mm256_setzero_pd(), half.x), _mm256_sub_pd(_mm256_setzero_pd(), half.y) };
    vec8d r;
    r.x = _mm256_blendv_pd(a.x, am.x, _mm256_cmp_pd(a.x, half.x, _CMP_GT_OQ));
    r.y = _mm256_blendv_pd(a.y, am.y, _mm256_cmp_pd(a.y, half.y, _CMP_GT_OQ));
    r.x = _mm256_blendv_pd(r.x, ap.x, _mm256_cmp_pd(a.x, neghalf.x, _CMP_LT_OQ));
    r.y = _mm256_blendv_pd(r.y, ap.y, _mm256_cmp_pd(a.y, neghalf.y, _CMP_LT_OQ));
    return r;
}

static inline vec8d vec8d_load_aligned(const double* p) {
    vec8d r = { _mm256_load_pd(p), _mm256_load_pd(p + 4) };
    return r;
}

static inline void vec8d_store_aligned(double* p, vec8d a) {
    _mm256_store_pd(p, a.x);
    _mm256_store_pd(p + 4, a.y);
}

#endif // FSD_VEC_H
