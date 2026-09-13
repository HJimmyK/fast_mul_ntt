// 基础工具实现：对齐分配、nmod/素数、mpn 小函数、mulmod 精度边界

#include <stdio.h>
#include "fsd.h"

#ifdef __cplusplus
extern "C" {
#endif

void fsd_abort(const char* msg) {
    fprintf(stderr, "fft_small: %s\n", msg);
    abort();
}

void* fsd_aligned_alloc(size_t align, size_t size) {
    if (size == 0)
        size = 1;
#ifdef _WIN32
    return _aligned_malloc(size, align);
#else
    /* C11 aligned_alloc 要求 size 是 align 的倍数 */
    size = (size + align - 1) / align * align;
    return aligned_alloc(align, size);
#endif
}

void fsd_aligned_free(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/************************* nmod / 素数 **************************************/

ulong nmod_pow_ui(ulong a, ulong exp, nmod_t mod) {
    ulong r = 1;
    a %= mod.n;
    while (exp > 0) {
        if (exp & 1)
            r = nmod_mul(r, a, mod);
        a = nmod_mul(a, a, mod);
        exp >>= 1;
    }
    return r;
}

/* 确定性 Miller-Rabin（对 2^64 有效） */
int n_is_prime(ulong n) {
    static const ulong bases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    if (n < 2)
        return 0;
    for (ulong i = 0; i < sizeof(bases) / sizeof(bases[0]); i++) {
        if (n % bases[i] == 0)
            return n == bases[i];
    }

    ulong d = n - 1;
    ulong s = n_trailing_zeros(d);
    d >>= s;

    for (ulong i = 0; i < sizeof(bases) / sizeof(bases[0]); i++) {
        ulong x = nmod_pow_ui(bases[i], d, (nmod_t){n});
        if (x == 1 || x == n - 1)
            continue;
        int composite = 1;
        for (ulong r = 1; r < s; r++) {
            x = nmod_mul(x, x, (nmod_t){n});
            if (x == n - 1) {
                composite = 0;
                break;
            }
        }
        if (composite)
            return 0;
    }
    return 1;
}

ulong n_quadratic_nonresidue(ulong p) {
    nmod_t mod = {{p}};
    for (ulong a = 2; a < p; a++) {
        if (nmod_pow_ui(a, (p - 1) / 2, mod) == p - 1)
            return a;
    }
    fsd_abort("n_quadratic_nonresidue: 找不到二次非剩余");
    return 0;
}

/************************* mpn 小函数 ***************************************/

ulong fsd_mpn_mul_1(ulong* rp, const ulong* xp, ulong n, ulong limb) {
    ulong hi = 0;
    for (ulong i = 0; i < n; i++) {
        unsigned __int128 p = (unsigned __int128)xp[i] * limb + hi;
        rp[i] = (ulong)p;
        hi = (ulong)(p >> 64);
    }
    return hi;
}

ulong fsd_mpn_add_n(ulong* rp, const ulong* xp, const ulong* yp, ulong n) {
    unsigned char cf = 0;
    for (ulong i = 0; i < n; i++)
        cf = _addcarry_ulong(cf, xp[i], yp[i], &rp[i]);
    return cf;
}

ulong fsd_mpn_sub_n(ulong* rp, const ulong* xp, const ulong* yp, ulong n) {
    ulong borrow = 0;
    for (ulong i = 0; i < n; i++) {
        ulong t = xp[i] - yp[i];
        ulong b1 = xp[i] < yp[i];
        ulong s = t - borrow;
        b1 += t < borrow;
        rp[i] = s;
        borrow = b1;
    }
    return borrow;
}

ulong fsd_mpn_sub_1(ulong* rp, const ulong* xp, ulong n, ulong limb) {
    ulong borrow = limb;
    for (ulong i = 0; i < n; i++) {
        ulong t = xp[i] - borrow;
        borrow = xp[i] < borrow;
        rp[i] = t;
    }
    return borrow;
}

ulong fsd_mpn_rshift(ulong* rp, const ulong* xp, ulong n, ulong cnt) {
    ulong ret = xp[0] << (64 - cnt);
    for (ulong i = 0; i + 1 < n; i++)
        rp[i] = (xp[i] >> cnt) | (xp[i + 1] << (64 - cnt));
    rp[n - 1] = xp[n - 1] >> cnt;
    return ret;
}

ulong fsd_mpn_mod_1(const ulong* xp, ulong n, ulong d) {
    ulong rem = 0;
    for (ulong i = n; i > 0; i--) {
        (void)fsd_udiv128(rem, xp[i - 1], d, &rem);
    }
    return rem;
}

void fsd_mpn_divexact_1(ulong* rp, const ulong* xp, ulong n, ulong d) {
    ulong rem = 0;
    for (ulong i = n; i > 0; i--) {
        ulong r;
        rp[i - 1] = fsd_udiv128(rem, xp[i - 1], d, &r);
        rem = r;
    }
    FSD_ASSERT(rem == 0);
}

/************************* mulmod 边界 *************************************
以下逐行移植自 FLINT src/fft_small/mulmod_satisfies_bounds.c：
对 |a*b| < 2*n^2 与 |a*b| < 4*n^2 分别要求界限 < 1 与 < 1.5，
使 mulmod 的输出严格落在 (-n, n) 与 (-3n/2, 3n/2)。
****************************************************************************/

#include <math.h>

int fft_small_mulmod_satisfies_bounds(ulong nn) {
    double n = nn;
    double ninv = 1.0 / n;
    double t1 = fabs(fma(n, ninv, -1.0)); /* epsilon ~= t1/n  good enough */
    double limit2, limit4;
    int B, ok, n1bits, n2bits;
    ulong n2hi, n2lo;

    const int D_BITS = 53;

    n1bits = (int)n_nbits(nn);
    umul_ppmm(n2hi, n2lo, nn, nn);
    if (n2hi != 0)
        n2bits = 64 + (int)n_nbits(n2hi);
    else
        n2bits = (int)n_nbits(n2lo);

    /* for |a*b| < 2*n^2 */
    B = D_BITS - n1bits - 1;
    if (B < 2)
        return 0;
    limit2 = 2 * n * t1 + ldexp(ninv, 1 + n2bits - D_BITS - 1) + 0.5 + ldexp(1.0, -(B + 1));

    /* for |a*b| < 4*n^2 */
    B -= 1;
    limit4 = 4 * n * t1 + ldexp(ninv, 2 + n2bits - D_BITS - 1) + 0.5 + ldexp(1.0, -(B + 1));

    ok = (limit2 < 0.99) && (limit4 < 1.49);
    return ok;
}

#ifdef __cplusplus
}
#endif
