#include <stdio.h>
#include "fsd.h"

void sd_fft_ctx_clear(sd_fft_ctx_t Q) {
    ulong k;
    fsd_aligned_free(Q->w2tab[0]);
    for (k = SD_FFT_CTX_W2TAB_INIT; k < SD_FFT_CTX_W2TAB_SIZE; k++)
        fsd_aligned_free(Q->w2tab[k]);
}

/*
    Return a primitive 2^depth-th root modulo the prime pp.
    Requires depth == valuation(pp - 1, 2).
*/
static ulong sd_fft_ctx_primitive_2power_root(ulong pp, ulong depth, nmod_t mod) {
    ulong a = n_quadratic_nonresidue(pp);
    return nmod_pow_ui(a, (pp - 1) >> depth, mod);
}

/*
    Return the primitive 2^(k+1)-th root used to generate w2tab[k].
    Requires depth == valuation(Q->mod.n - 1, 2).
*/
static ulong sd_fft_ctx_w2tab_root(const sd_fft_ctx_t Q, ulong depth, ulong k) {
    FSD_ASSERT(k + 1 <= depth);
    return nmod_pow_ui(Q->primitive_2power_root, UWORD(1) << (depth - k - 1), Q->mod);
}

/*
    Initialize FFT context.
    pp is a prime with at most ~ 50 bits (exactly representable with a `double`)
    such that pp - 1 has sufficiently high 2-valuation.
    Used in sd_fft_trunc, sd_ifft_trunc, sd_fft_ctx_point_mul, etc.
*/
void sd_fft_ctx_init_prime(sd_fft_ctx_t Q, ulong pp) {
    ulong N, i, k, l, init_depth, two_power_depth;
    double* t;
    double n, ninv, w;

    if (!fft_small_mulmod_satisfies_bounds(pp))
        fsd_abort("FFT prime 素数不满足 double 域模乘的精度边界");

    Q->p = pp;
    Q->pinv = 1.0 / Q->p;
    Q->mod.n = pp;
    two_power_depth = n_trailing_zeros(pp - 1);
    if (two_power_depth == 0)
        fsd_abort("输入是 2 或者不是素数");
    Q->primitive_2power_root = sd_fft_ctx_primitive_2power_root(pp, two_power_depth, Q->mod);
    init_depth = n_min(two_power_depth, SD_FFT_CTX_W2TAB_INIT);
    if (init_depth < 4)
        fsd_abort("素数的 2 进制赋值过低，无法初始化 FFT 上下文");

    n = Q->p;
    ninv = Q->pinv;

    /*
        fill wtab to a depth of init_depth:
        2^(init_depth-1) entries: 1, e(1/4), e(1/8), e(3/8), ...

        Q->w2tab[j] is itself a table of length 2^(j-1) containing 2^(j+1) st
        roots of unity. More documentation on the layout of w2tab can be found
        before the definition of SD_FFT_CTX_W2TAB_SIZE.

        All entries in w2tab are exactly-representable integers modulo pp, but
        they're stored as `double` to make use of the vectorized functions.
    */
    N = n_pow2(init_depth - 1);
    t = (double*)fsd_aligned_alloc(4096, n_round_up(N * sizeof(double), 4096));

    Q->w2tab[0] = t;
    t[0] = 1;

    {
        ulong ww = sd_fft_ctx_w2tab_root(Q, two_power_depth, 3);
        w = vec1d_reduce_0n_to_pmhn(ww, n);
        double w2 = vec1d_reduce_pm1n_to_pmhn(vec1d_mulmod(w, w, n, ninv), n);

        Q->w2tab[1] = t + 1;
        t[1] = vec1d_reduce_pm1n_to_pmhn(vec1d_mulmod(w2, w2, n, ninv), n);

        Q->w2tab[2] = t + 2;
        t[2] = w2;
        t[3] = vec1d_reduce_pm1n_to_pmhn(vec1d_mulmod(t[1], w2, n, ninv), n);
    }

    vec4d n4 = vec4d_set_d(n);
    vec4d ninv4 = vec4d_set_d(ninv);

    for (k = 3, l = 4; k < init_depth; k++, l *= 2) {
        double* curr = t + l;
        vec4d w4 = vec4d_set_d(w);
        Q->w2tab[k] = curr;
        i = 0;
        do {
            vec4d x = vec4d_load_aligned(t + i);
            x = vec4d_mulmod(x, w4, n4, ninv4);
            x = vec4d_reduce_pm1n_to_pmhn(x, n4);
            vec4d_store_aligned(curr + i, x);
        } while (i += 4, i < l);

        if (k + 1 < init_depth)
            w = vec1d_reduce_0n_to_pmhn(sd_fft_ctx_w2tab_root(Q, two_power_depth, k + 1), n);
    }

    Q->w2tab_depth = (unsigned int)k;

    /* the rest of the tables are uninitialized */
    for (; k < SD_FFT_CTX_W2TAB_SIZE; k++)
        Q->w2tab[k] = NULL;

#ifndef NDEBUG
    for (k = 1; k < init_depth; k++) {
        ulong ww = sd_fft_ctx_w2tab_root(Q, two_power_depth, k);
        for (i = 0; i < n_pow2(k - 1); i++) {
            ulong www = nmod_pow_ui(ww, n_revbin(i + n_pow2(k - 1), k), Q->mod);
            FSD_ASSERT(Q->w2tab[k][i] == vec1d_reduce_0n_to_pmhn(www, n));
        }
    }
#endif
}

void sd_fft_ctx_fit_depth(sd_fft_ctx_t Q, ulong depth) {
    if (depth > Q->w2tab_depth)
        sd_fft_ctx_fit_depth_with_lock(Q, depth);
}

void sd_fft_ctx_fit_depth_with_lock(sd_fft_ctx_t Q, ulong depth) {
    ulong two_power_depth = n_trailing_zeros(Q->mod.n - 1);

    if (depth > two_power_depth)
        fsd_abort("FFT prime 素数不支持该变换深度");

    ulong k = (ulong)Q->w2tab_depth;

    while (k < depth) {
        ulong i, j, l, off;
        ulong ww = sd_fft_ctx_w2tab_root(Q, two_power_depth, k);
        vec8d w = vec8d_set_d(vec1d_reduce_0n_to_pmhn(ww, Q->p));
        vec8d n = vec8d_set_d(Q->p);
        vec8d ninv = vec8d_set_d(Q->pinv);
        ulong N = n_pow2(k - 1);
        double* curr = (double*)fsd_aligned_alloc(4096, n_round_up(N * sizeof(double), 4096));
        double* t = Q->w2tab[0];
        Q->w2tab[k] = curr;

        /* The first few tables are stored consecutively, so vec16 is ok. */
        off = 0;
        l = n_pow2(SD_FFT_CTX_W2TAB_INIT - 1);
        for (j = SD_FFT_CTX_W2TAB_INIT - 1; j < k; j++) {
            i = 0;
            do {
                vec8d x0 = vec8d_load_aligned(t + i + 0);
                vec8d x1 = vec8d_load_aligned(t + i + 8);
                x0 = vec8d_mulmod(x0, w, n, ninv);
                x1 = vec8d_mulmod(x1, w, n, ninv);
                x0 = vec8d_reduce_pm1n_to_pmhn(x0, n);
                x1 = vec8d_reduce_pm1n_to_pmhn(x1, n);
                vec8d_store_aligned(curr + off + i + 0, x0);
                vec8d_store_aligned(curr + off + i + 8, x1);
            } while (i += 16, i < l);
            FSD_ASSERT(i == l);
            t = Q->w2tab[j + 1];
            l += off;
            off = l;
        }

#ifndef NDEBUG
        {
            ulong ww = sd_fft_ctx_w2tab_root(Q, two_power_depth, k);
            for (i = 0; i < n_pow2(k - 1); i++) {
                ulong www = nmod_pow_ui(ww, n_revbin(i + n_pow2(k - 1), k), Q->mod);
                FSD_ASSERT(Q->w2tab[k][i] == vec1d_reduce_0n_to_pmhn(www, Q->p));
            }
        }
#endif

        k++;
        Q->w2tab_depth = (unsigned int)k;
    }
}
