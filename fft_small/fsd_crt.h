//   N = crt 的 coeff_len（能装下 np*prod_primes），M = N 或 N-1
//   f 是 M 个 limb 的 CRT 余因子（P/p_i），x 是单 limb 的剩余 (< 2^50)

#ifndef FSD_CRT_H
#define FSD_CRT_H

/* z[0..n-1] += r[0..n-1]，进位向 z 的高位传播（结果不超过总乘积，安全） */
#define DEFINE_MULTI_ADD(n)                                        \
    static inline void CAT(multi_add, n)(ulong* z, const ulong* r) { \
        unsigned char cf = 0;                                      \
        for (ulong i = 0; i < (n); i++)                            \
            cf = _addcarry_ulong(cf, z[i], r[i], &z[i]);           \
        for (ulong i = (n); cf != 0; i++) {                        \
            FSD_ASSERT(cf == 1);                                   \
            cf = _addcarry_ulong(0, z[i], 1, &z[i]);               \
        }                                                          \
    }

DEFINE_MULTI_ADD(4)
DEFINE_MULTI_ADD(5)
DEFINE_MULTI_ADD(6)
DEFINE_MULTI_ADD(7)
DEFINE_MULTI_ADD(8)
#undef DEFINE_MULTI_ADD

/* r[0..N] = f[0..M-1] * x；N==M+1 时 f 的第 M 个 limb 为 0，乘积正好 N+1 limb */
#define DEFINE_BIG_MUL(N, M)                                               \
    static inline void CAT3(_big_mul, N, M)(ulong* r, ulong* FSD_UNUSED(t), \
                                            const ulong* f, ulong x) {     \
        unsigned __int128 c = 0;                                           \
        for (ulong i = 0; i < (M); i++) {                                  \
            c += (unsigned __int128)f[i] * x;                              \
            r[i] = (ulong)c;                                               \
            c >>= 64;                                                      \
        }                                                                  \
        r[M] = (ulong)c;                                                   \
        if ((N) > (M))                                                     \
            r[N] = 0;                                                      \
    }                                                                      \
    static inline void CAT3(_big_addmul, N, M)(ulong* r, ulong* FSD_UNUSED(t), \
                                               const ulong* f, ulong x) {  \
        ulong c = 0;                                                       \
        for (ulong i = 0; i < (M); i++) {                                  \
            unsigned __int128 s = (unsigned __int128)f[i] * x + r[i] + c;  \
            r[i] = (ulong)s;                                               \
            c = (ulong)(s >> 64);                                          \
        }                                                                  \
        for (ulong i = (M); c != 0 && i <= (N); i++) {                     \
            ulong s = r[i] + c;                                            \
            r[i] = s;                                                      \
            c = s < c;                                                     \
        }                                                                  \
    }

DEFINE_BIG_MUL(4, 3)
DEFINE_BIG_MUL(4, 4)
DEFINE_BIG_MUL(5, 4)
DEFINE_BIG_MUL(6, 5)
DEFINE_BIG_MUL(7, 6)
#undef DEFINE_BIG_MUL

/* r[0..N] (< np*prod, np <= 8) -> r mod prod（N 个 limb，r[N] 清零） */
#define DEFINE_REDUCE_BIG_SUM(N)                                           \
    static inline void CAT(_reduce_big_sum, N)(ulong* r, ulong* FSD_UNUSED(t), \
                                               const ulong* prod) {         \
        for (;;) {                                                         \
            int ge; /* r >= prod ? */                                       \
            if (r[N] != 0) {                                               \
                ge = 1;                                                    \
            } else {                                                       \
                ulong i = (N);                                             \
                while (i > 0 && r[i - 1] == prod[i - 1])                   \
                    i--;                                                   \
                ge = (i == 0) ? 1 : (r[i - 1] > prod[i - 1]);              \
            }                                                              \
            if (!ge)                                                       \
                return;                                                    \
            r[N] -= fsd_mpn_sub_n(r, r, prod, (N));                        \
        }                                                                  \
    }

DEFINE_REDUCE_BIG_SUM(4)
DEFINE_REDUCE_BIG_SUM(5)
DEFINE_REDUCE_BIG_SUM(6)
DEFINE_REDUCE_BIG_SUM(7)
#undef DEFINE_REDUCE_BIG_SUM

/* 与 FLINT 一致的 3 参数访问形式：n == C->coeff_len */
#define _crt_data_co_prime(C, i, n) (FSD_ASSERT((n) == (C)->coeff_len), crt_data_co_prime(C, i))

#endif // FSD_CRT_H
