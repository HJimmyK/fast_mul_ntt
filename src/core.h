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

#ifndef CORE_H
#define CORE_H

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "data.h"


typedef u64 u192[3];
typedef u64 u128[2];

// return i192 < j192 
INLINE bool _u192cmp(const u192 i192, const u192 j192) {
    if (i192[2] != j192[2])
        return i192[2] < j192[2];
    if (i192[1] != j192[1])
        return i192[1] < j192[1];
    return i192[0] < j192[0];
}

#define define_mont_qpow(_mod, _i)                                        \
    INLINE mont64 _mont_qpow_##_mod(mont64 m, u64 n) {                    \
        mont64 result = g_one(_i);                                        \
        while (n > 0) {                                                   \
            if ((n & 1) != 0) {                                           \
                _mont_mulinto(result, m, g_mod(_i), g_modInvNeg(_i));     \
            }                                                             \
            _mont_mulinto(m, m, g_mod(_i), g_modInvNeg(_i));              \
            n >>= 1;                                                      \
        }                                                                 \
        return result;                                                    \
    }

define_mont_qpow(global_mod1, 1) define_mont_qpow(global_mod2, 2) define_mont_qpow(global_mod3, 3)


// 参数 _i 为第几个模数
#define _mont_qpow_func_name(_i) _mont_qpow_global_mod##_i
#define _mont_tomont_func(r, _i) _mont_tomont(r, g_r2(_i), g_mod(_i), g_modInvNeg(_i))
#define _mont_toint_func(r, _i) _mont_toint(r, g_mod(_i), g_modInvNeg(_i))
#define _mont_mul_func(r, a, b, _i) _mont_mul(r, a, b, g_mod(_i), g_modInvNeg(_i))
#define _mont_mulinto_func(r, a, _i) _mont_mulinto(r, a, g_mod(_i), g_modInvNeg(_i))
#define _mont_add_func(r, a, b, _i) _mont_add(r, a, b, g_mod2(_i))
// mont64 r = a - b
#define _mont_sub_func(r, a, b, _i) _mont_sub(r, a, b, g_mod2(_i))
#define _mont_norm2_func(r, a, _i) _mont_norm2(r, a, g_mod2(_i))
#define _raw_add_func(r, a, b, _i) _raw_add(r, a, b)
// mont64 r = a - b
#define _raw_sub_func(r, a, b, _i) _raw_sub(r, a, b, g_mod2(_i))

/* 已指定a, b, c的模数为1，2，3，分别为 mod1，mod2，mod3 */
INLINE void crt3(mont64 a, mont64 b, mont64 c, u192 res) {
    static const u192 mod123 = {8610882487532388353ull, 1266215182732886016ull, 59403314713853952ull};
    static const u128 mod12 = {4431542033332568065ull, 262193940805976064ull};
    static const u128 mod23 = {6124895493223874561ull, 440789813528887296ull};
    static const u128 mod13 = {6665327448508334081ull, 563231428398022656ull};
    static const mont64 mont_mod23_inv1 = 84629344487098392ull;
    static const mont64 mont_mod13_inv2 = 75311807962221426ull;
    static const mont64 mont_mod12_inv3 = 3875284113674644599ull;
    _mont_mul_func(a, a, mont_mod23_inv1, 1);
    _mont_mul_func(b, b, mont_mod13_inv2, 2);
    _mont_mul_func(c, c, mont_mod12_inv3, 3);

    _mont_toint_func(a, 1);
    _mont_toint_func(b, 2);
    _mont_toint_func(c, 3);
    u192 tmp = {0, 0, 0};
    _u128x64to192(res, mod23, a);
    _u128x64to192(tmp, mod13, b);
    _u192add(res, tmp);
    _u128x64to192(tmp, mod12, c);
    _u192add(res, tmp);
    if (!_u192cmp(res, mod123)) {
        _u192sub(res, mod123);
    }
    if (!_u192cmp(res, mod123)) {
        _u192sub(res, mod123);
    }
}

#define _transform2(sum, diff, _i)           \
    do {                                     \
        mont64 _t = sum, _u = diff;          \
        _mont_add(sum, _t, _u, g_mod2(_i));  \
        _mont_sub(diff, _t, _u, g_mod2(_i)); \
    } while (0)

#define _mul_w41(x, _i) _mont_mul(x, x, g_w41(_i), g_mod(_i), g_modInvNeg(_i))
#define _mul_w41inv(x, _i) _mont_mul(x, x, g_w41inv(_i), g_mod(_i), g_modInvNeg(_i))

#define _dif_butterfly244(r0, r1, r2, r3, _i) \
    do {                                      \
        mont64 t0, t1, t2, t3;                \
        _raw_add_func(t0, r0, r2, _i);        \
        _mont_sub_func(t2, r0, r2, _i);       \
        _raw_add_func(t1, r1, r3, _i);        \
        _raw_sub_func(t3, r1, r3, _i);        \
        _mul_w41(t3, _i);                     \
        _mont_norm2_func(r0, t0, _i);         \
        _mont_norm2_func(r1, t1, _i);         \
        _raw_add_func(r2, t2, t3, _i);        \
        _raw_sub_func(r3, t2, t3, _i);        \
    } while (0)

#define _idit_butterfly244(r0, r1, r2, r3, _i) \
    do {                                       \
        mont64 t0, t1, t2, t3;                 \
        _mont_norm2_func(t0, r0, _i);          \
        _mont_norm2_func(t1, r1, _i);          \
        _mont_add_func(t2, r2, r3, _i);        \
        _raw_sub_func(t3, r2, r3, _i);         \
        _mul_w41inv(t3, _i);                   \
        _raw_add_func(r0, t0, t2, _i);         \
        _raw_sub_func(r2, t0, t2, _i);         \
        _raw_add_func(r1, t1, t3, _i);         \
        _raw_sub_func(r3, t1, t3, _i);         \
    } while (0)

#define _dit_butterfly2(r0, r1, o, _i) \
    do {                               \
        mont64 x, y;                   \
        _mont_norm2_func(x, r0, _i);   \
        _mont_mul_func(y, r1, o, _i);  \
        _raw_add_func(r0, x, y, _i);   \
        _raw_sub_func(r1, x, y, _i);   \
    } while (0)

#define _dif_butterfly2(r0, r1, o, _i) \
    do {                               \
        mont64 x, y;                   \
        _mont_add_func(x, r0, r1, _i); \
        _raw_sub_func(y, r0, r1, _i);  \
        r0 = x;                        \
        _mont_mul_func(r1, y, o, _i);  \
    } while (0)


typedef struct NTTshort {
    size_t ntt_len;
    size_t log_len;
    mont64* omega;
    mont64* iomega;
} ntt_short;

typedef struct NTTshort intt_short;

#define define_ntt_short_create(_i)                                                \
    INLINE void create_nttshort_##_i(const size_t lg_len, ntt_short* in) {         \
        assert(in->log_len == lg_len);                                             \
        assert(in->ntt_len == (1ull << lg_len));                                   \
        if (in->omega != NULL) {                                                   \
            free(in->omega);                                                       \
            in->omega = NULL;                                                      \
        }                                                                          \
        in->omega = (mont64*)malloc(in->ntt_len * sizeof(mont64));                 \
        if (in->omega == NULL) {                                                   \
            in = NULL;                                                             \
            return;                                                                \
        }                                                                          \
        if (in->iomega != NULL) {                                                  \
            free(in->iomega);                                                      \
            in->iomega = NULL;                                                     \
        }                                                                          \
        in->iomega = (mont64*)malloc(in->ntt_len * sizeof(mont64));                \
        if (in->iomega == NULL) {                                                  \
            in = NULL;                                                             \
            return;                                                                \
        }                                                                          \
        *(in->omega) = 0;                                                          \
        *(in->iomega) = 0;                                                         \
        for (size_t omega_log_len = 0; omega_log_len <= lg_len; omega_log_len++) { \
            size_t omega_len = 1ull << omega_log_len, omega_count = omega_len / 2; \
            mont64* it1 = (in->omega) + omega_len / 2;                             \
            mont64 root1 = g_mont_root(_i);                                        \
            mont64* it2 = (in->iomega) + omega_len / 2;                            \
            mont64 root2 = g_mont_rootinv(_i);                                     \
            root1 = _mont_qpow_func_name(_i)(root1, (g_mod(_i) - 1) / omega_len);  \
            root2 = _mont_qpow_func_name(_i)(root2, (g_mod(_i) - 1) / omega_len);  \
            mont64 omega_one1 = g_one(_i), omega_one2 = g_one(_i);                 \
            for (size_t ii = 0; ii < omega_count; ii++) {                          \
                it1[ii] = omega_one1;                                              \
                _mont_mul_func(omega_one1, omega_one1, root1, _i);                 \
                it2[ii] = omega_one2;                                              \
                _mont_mul_func(omega_one2, omega_one2, root2, _i);                 \
            }                                                                      \
        }                                                                          \
    }

#define define_ntt_short_cover(_i)                                                 \
    INLINE void cover_nttshort_##_i(const size_t lg_len, ntt_short* in) {          \
        assert(in->log_len == lg_len);                                             \
        assert(in->ntt_len == (1ull << lg_len));                                   \
        assert(in->omega != NULL);                                                 \
        assert(in->iomega != NULL);                                                \
        *(in->omega) = 0;                                                          \
        *(in->iomega) = 0;                                                         \
        for (size_t omega_log_len = 0; omega_log_len <= lg_len; omega_log_len++) { \
            size_t omega_len = 1ull << omega_log_len, omega_count = omega_len / 2; \
            mont64* it1 = (in->omega) + omega_len / 2;                             \
            mont64 root1 = g_mont_root(_i);                                        \
            mont64* it2 = (in->iomega) + omega_len / 2;                            \
            mont64 root2 = g_mont_rootinv(_i);                                     \
            root1 = _mont_qpow_func_name(_i)(root1, (g_mod(_i) - 1) / omega_len);  \
            root2 = _mont_qpow_func_name(_i)(root2, (g_mod(_i) - 1) / omega_len);  \
            mont64 omega_one1 = g_one(_i), omega_one2 = g_one(_i);                 \
            for (size_t ii = 0; ii < omega_count; ii++) {                          \
                it1[ii] = omega_one1;                                              \
                _mont_mul_func(omega_one1, omega_one1, root1, _i);                 \
                it2[ii] = omega_one2;                                              \
                _mont_mul_func(omega_one2, omega_one2, root2, _i);                 \
            }                                                                      \
        }                                                                          \
    }

void destroy_nttshort(struct NTTshort** in) {
    if (in == NULL) {
        return;
    }
    if (*in == NULL) {
        return;
    }
    if ((*in)->omega != NULL) {
        free((*in)->omega);
        (*in)->omega = NULL;
    }
    if ((*in)->iomega != NULL) {
        free((*in)->iomega);
        (*in)->iomega = NULL;
    }
    free(*in);
    *in = NULL;
}

void destroy_nttshort_stack(struct NTTshort* in) {
    if (in == NULL) {
        return;
    }
    if (in->omega != NULL) {
        free(in->omega);
        in->omega = NULL;
    }
    if (in->iomega != NULL) {
        free(in->iomega);
        in->iomega = NULL;
    }
}

define_ntt_short_create(1) 
define_ntt_short_create(2) 
define_ntt_short_create(3) 

define_ntt_short_cover(1) 
define_ntt_short_cover(2) 
define_ntt_short_cover(3) 

#define create_nttshort_func(lg_len, in, _i) create_nttshort_##_i(lg_len, in)
#define cover_nttshort_func(lg_len, in, _i) cover_nttshort_##_i(lg_len, in)

#define define_ntt_short_di_0(_i)                                        \
    INLINE void ntt_short_dif_0_##_i(mont64 in_out[]) {}                 \
    INLINE void ntt_short_dif_0_len_##_i(mont64 in_out[], size_t len) {} \
    INLINE void intt_short_dit_0_##_i(mont64 in_out[]) {}                \
    INLINE void intt_short_dit_0_len_##_i(mont64 in_out[], size_t len) {}

#define define_ntt_short_di_1(_i)                                        \
    INLINE void ntt_short_dif_1_##_i(mont64 in_out[]) {}                 \
    INLINE void ntt_short_dif_1_len_##_i(mont64 in_out[], size_t len) {} \
    INLINE void intt_short_dit_1_##_i(mont64 in_out[]) {}                \
    INLINE void intt_short_dit_1_len_##_i(mont64 in_out[], size_t len) {} 

define_ntt_short_di_0(1) define_ntt_short_di_0(2) define_ntt_short_di_0(3)

define_ntt_short_di_1(1) define_ntt_short_di_1(2) define_ntt_short_di_1(3)

#define define_ntt_short_di_2(_i)                                                                 \
    INLINE void ntt_short_dif_2_##_i(mont64 in_out[]) { _transform2(in_out[0], in_out[1], _i); }  \
    INLINE void ntt_short_dif_2_len_##_i(mont64 in_out[], size_t len) {                           \
        if (len < 2)                                                                              \
            return;                                                                               \
        _transform2(in_out[0], in_out[1], _i);                                                    \
    }                                                                                             \
    INLINE void intt_short_dit_2_##_i(mont64 in_out[]) { _transform2(in_out[0], in_out[1], _i); } \
    INLINE void intt_short_dit_2_len_##_i(mont64 in_out[], size_t len) {                          \
        if (len < 2)                                                                              \
            return;                                                                               \
        _transform2(in_out[0], in_out[1], _i);                                                    \
    }                                                                                             

define_ntt_short_di_2(1) define_ntt_short_di_2(2) define_ntt_short_di_2(3)

#define define_ntt_short_di_4(_i)                                              \
    INLINE void ntt_short_dif_4_##_i(mont64 in_out[]) {                        \
        mont64 t0 = in_out[0], t1 = in_out[1], t2 = in_out[2], t3 = in_out[3]; \
        _transform2(t0, t2, _i);                                               \
        _transform2(t1, t3, _i);                                               \
        _mul_w41(t3, _i);                                                      \
        _mont_add_func(in_out[0], t0, t1, _i);                                 \
        _mont_sub_func(in_out[1], t0, t1, _i);                                 \
        _mont_add_func(in_out[2], t2, t3, _i);                                 \
        _mont_sub_func(in_out[3], t2, t3, _i);                                 \
    }                                                                          \
    INLINE void ntt_short_dif_4_len_##_i(mont64 in_out[], size_t len) {        \
        if (len < 4) {                                                         \
            ntt_short_dif_2_len_##_i(in_out, len);                             \
            return;                                                            \
        }                                                                      \
        ntt_short_dif_4_##_i(in_out);                                          \
    }                                                                          \
    INLINE void intt_short_dit_4_##_i(mont64 in_out[]) {                       \
        mont64 t0 = in_out[0], t1 = in_out[1], t2 = in_out[2], t3 = in_out[3]; \
        _transform2(t0, t1, _i);                                               \
        _transform2(t2, t3, _i);                                               \
        _mul_w41inv(t3, _i);                                                   \
        _mont_add_func(in_out[0], t0, t2, _i);                                 \
        _mont_add_func(in_out[1], t1, t3, _i);                                 \
        _mont_sub_func(in_out[2], t0, t2, _i);                                 \
        _mont_sub_func(in_out[3], t1, t3, _i);                                 \
    }                                                                          \
    INLINE void intt_short_dit_4_len_##_i(mont64 in_out[], size_t len) {       \
        if (len < 4) {                                                         \
            intt_short_dit_2_len_##_i(in_out, len);                            \
            return;                                                            \
        }                                                                      \
        intt_short_dit_4_##_i(in_out);                                         \
    }

#define define_ntt_short_di_8(_i)                                              \
    INLINE void ntt_short_dif_8_##_i(mont64 in_out[]) {                        \
        mont64 t0 = in_out[0], t1 = in_out[1], t2 = in_out[2], t3 = in_out[3]; \
        mont64 t4 = in_out[4], t5 = in_out[5], t6 = in_out[6], t7 = in_out[7]; \
        _transform2(t0, t4, _i);                                               \
        _transform2(t1, t5, _i);                                               \
        _transform2(t2, t6, _i);                                               \
        _transform2(t3, t7, _i);                                               \
        _mont_mul_func(t5, t5, g_w1(_i), _i);                                  \
        _mont_mul_func(t6, t6, g_w2(_i), _i);                                  \
        _mont_mul_func(t7, t7, g_w3(_i), _i);                                  \
        _transform2(t0, t2, _i);                                               \
        _transform2(t1, t3, _i);                                               \
        _transform2(t4, t6, _i);                                               \
        _transform2(t5, t7, _i);                                               \
        _mul_w41(t3, _i);                                                      \
        _mul_w41(t7, _i);                                                      \
        _mont_add_func(in_out[0], t0, t1, _i);                                 \
        _mont_sub_func(in_out[1], t0, t1, _i);                                 \
        _mont_add_func(in_out[2], t2, t3, _i);                                 \
        _mont_sub_func(in_out[3], t2, t3, _i);                                 \
        _mont_add_func(in_out[4], t4, t5, _i);                                 \
        _mont_sub_func(in_out[5], t4, t5, _i);                                 \
        _mont_add_func(in_out[6], t6, t7, _i);                                 \
        _mont_sub_func(in_out[7], t6, t7, _i);                                 \
    }                                                                          \
    INLINE void ntt_short_dif_8_len_##_i(mont64 in_out[], size_t len) {        \
        if (len < 8) {                                                         \
            ntt_short_dif_4_len_##_i(in_out, len);                             \
            return;                                                            \
        }                                                                      \
        ntt_short_dif_8_##_i(in_out);                                          \
    }                                                                          \
    INLINE void intt_short_dit_8_##_i(mont64 in_out[]) {                       \
        mont64 t0 = in_out[0], t1 = in_out[1], t2 = in_out[2], t3 = in_out[3]; \
        mont64 t4 = in_out[4], t5 = in_out[5], t6 = in_out[6], t7 = in_out[7]; \
        _transform2(t0, t1, _i);                                               \
        _transform2(t2, t3, _i);                                               \
        _transform2(t4, t5, _i);                                               \
        _transform2(t6, t7, _i);                                               \
        _mul_w41inv(t3, _i);                                                   \
        _mul_w41inv(t7, _i);                                                   \
        _transform2(t0, t2, _i);                                               \
        _transform2(t1, t3, _i);                                               \
        _transform2(t4, t6, _i);                                               \
        _transform2(t5, t7, _i);                                               \
        _mont_mul_func(t5, t5, g_w1inv(_i), _i);                               \
        _mont_mul_func(t6, t6, g_w2inv(_i), _i);                               \
        _mont_mul_func(t7, t7, g_w3inv(_i), _i);                               \
        _mont_add_func(in_out[0], t0, t4, _i);                                 \
        _mont_add_func(in_out[1], t1, t5, _i);                                 \
        _mont_add_func(in_out[2], t2, t6, _i);                                 \
        _mont_add_func(in_out[3], t3, t7, _i);                                 \
        _mont_sub_func(in_out[4], t0, t4, _i);                                 \
        _mont_sub_func(in_out[5], t1, t5, _i);                                 \
        _mont_sub_func(in_out[6], t2, t6, _i);                                 \
        _mont_sub_func(in_out[7], t3, t7, _i);                                 \
    }                                                                          \
    INLINE void intt_short_dit_8_len_##_i(mont64 in_out[], size_t len) {       \
        if (len < 8) {                                                         \
            intt_short_dit_4_len_##_i(in_out, len);                            \
            return;                                                            \
        }                                                                      \
        intt_short_dit_8_##_i(in_out);                                         \
    }                                                                          

define_ntt_short_di_4(1) define_ntt_short_di_4(2) define_ntt_short_di_4(3)

define_ntt_short_di_8(1) define_ntt_short_di_8(2) define_ntt_short_di_8(3)

#define ntt_short_dif_func(in_out, _N, _i) ntt_short_dif_##_N##_##_i(in_out)
#define ntt_short_dif_len_func(in_out, len, _N, _i) ntt_short_dif_##_N##_len_##_i(in_out, len)

#define intt_short_dit_func(in_out, _N, _i) intt_short_dit_##_N##_##_i(in_out)
#define intt_short_dit_len_func(in_out, len, _N, _i) intt_short_dit_##_N##_len_##_i(in_out, len)

#define L2_BYTES (1ull << 20)

static const u64 long_threshold = L2_BYTES / sizeof(mont64);
#define long_threshold 131072ULL

INLINE u64 log2_64(u64 n) {
    if (n == 0) {
        return 0;
    }
    return 63 - __builtin_clzll(n);
}

#define get_omega_it(table, len) (((table)->omega) + (len) / 2)
#define get_iomega_it(table, len) (((table)->iomega) + (len) / 2)

#define define_dif(_i)                                                                                             \
    INLINE void dif_##_i(mont64 in_out[], ntt_short* table, size_t len) {                                          \
        assert(len <= long_threshold);                                                                             \
        size_t rank = len;                                                                                         \
        for (; rank >= 16; rank /= 4) {                                                                            \
            size_t gap = rank / 4;                                                                                 \
            mont64 *omega_it = get_omega_it(table, rank), *last_omega_it = get_omega_it(table, rank / 2);          \
            mont64 *it0 = in_out, *it1 = in_out + gap, *it2 = in_out + gap * 2, *it3 = in_out + gap * 3;           \
            for (size_t jj = 0; jj < len; jj += rank) {                                                            \
                for (size_t ii = 0; ii < gap; ii++) {                                                              \
                    mont64 temp0 = it0[jj + ii], temp1 = it1[jj + ii], temp2 = it2[jj + ii], temp3 = it3[jj + ii], \
                           omega = last_omega_it[ii];                                                              \
                    _dif_butterfly2(temp0, temp2, omega_it[ii], _i);                                               \
                    _dif_butterfly2(temp1, temp3, omega_it[gap + ii], _i);                                         \
                    _dif_butterfly2(temp0, temp1, omega, _i);                                                      \
                    _dif_butterfly2(temp2, temp3, omega, _i);                                                      \
                    it0[jj + ii] = temp0, it1[jj + ii] = temp1, it2[jj + ii] = temp2, it3[jj + ii] = temp3;        \
                }                                                                                                  \
            }                                                                                                      \
        }                                                                                                          \
        if (log2_64(rank) % 2 == 0) {                                                                              \
            ntt_short_dif_len_func(in_out, len, 4, _i);                                                            \
            for (size_t ii = 4; ii < len; ii += 4) {                                                               \
                ntt_short_dif_func((in_out + ii), 4, _i);                                                          \
            }                                                                                                      \
        } else {                                                                                                   \
            ntt_short_dif_len_func(in_out, len, 8, _i);                                                            \
            for (size_t ii = 8; ii < len; ii += 8) {                                                               \
                ntt_short_dif_func((in_out + ii), 8, _i);                                                          \
            }                                                                                                      \
        }                                                                                                          \
    }

#define define_idit(_i)                                                                                            \
    INLINE void idit_##_i(mont64 in_out[], intt_short* table, size_t len) {                                        \
        assert(len <= long_threshold);                                                                             \
        size_t rank = len;                                                                                         \
        if (log2_64(len) % 2 == 0) {                                                                               \
            intt_short_dit_len_func(in_out, len, 4, _i);                                                           \
            for (size_t ii = 4; ii < len; ii += 4) {                                                               \
                intt_short_dit_func((in_out + ii), 4, _i);                                                         \
            }                                                                                                      \
            rank = 16;                                                                                             \
        } else {                                                                                                   \
            intt_short_dit_len_func(in_out, len, 8, _i);                                                           \
            for (size_t ii = 8; ii < len; ii += 8) {                                                               \
                intt_short_dit_func((in_out + ii), 8, _i);                                                         \
            }                                                                                                      \
            rank = 32;                                                                                             \
        }                                                                                                          \
        for (; rank <= len; rank *= 4) {                                                                           \
            size_t gap = rank / 4;                                                                                 \
            mont64 *omega_it = get_iomega_it(table, rank), *last_omega_it = get_iomega_it(table, rank / 2);        \
            mont64 *it0 = in_out, *it1 = in_out + gap, *it2 = in_out + gap * 2, *it3 = in_out + gap * 3;           \
            for (size_t jj = 0; jj < len; jj += rank) {                                                            \
                for (size_t ii = 0; ii < gap; ii++) {                                                              \
                    mont64 temp0 = it0[jj + ii], temp1 = it1[jj + ii], temp2 = it2[jj + ii], temp3 = it3[jj + ii], \
                           omega = last_omega_it[ii];                                                              \
                    _dit_butterfly2(temp0, temp1, omega, _i);                                                      \
                    _dit_butterfly2(temp2, temp3, omega, _i);                                                      \
                    _dit_butterfly2(temp0, temp2, omega_it[ii], _i);                                               \
                    _dit_butterfly2(temp1, temp3, omega_it[gap + ii], _i);                                         \
                    it0[jj + ii] = temp0, it1[jj + ii] = temp1, it2[jj + ii] = temp2, it3[jj + ii] = temp3;        \
                }                                                                                                  \
            }                                                                                                      \
        }                                                                                                          \
    }

define_dif(1) define_dif(2) define_dif(3)

define_idit(1) define_idit(2) define_idit(3)


#define dif_func(in_out, table, len, _i) dif_##_i(in_out, table, len)
#define idit_func(in_out, table, len, _i) idit_##_i(in_out, table, len)

#endif // CORE_H
