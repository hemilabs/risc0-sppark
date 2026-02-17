// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <cstring>

#define MSM_PROFILE

#if defined(FEATURE_BLS12_381)
# include <ff/bls12-381-fp2.hpp>
#elif defined(FEATURE_BLS12_377)
# include <ff/bls12-377-fp2.hpp>
#elif defined(FEATURE_BN254)
# include <ff/alt_bn128-fp2.hpp>
#else
# error "no FEATURE"
#endif

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_inf_t affine_t;
typedef fr_t scalar_t;

#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#include <msm/pippenger.cuh>

// ==================== GLV Endomorphism for BN254 G1 ====================
#ifdef FEATURE_BN254

typedef unsigned __int128 uint128;

static void glv_mul128x256(uint64_t out[6], const uint64_t a[2], const uint64_t b[4])
{
    memset(out, 0, 6 * sizeof(uint64_t));
    for (int i = 0; i < 2; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint128 prod = (uint128)a[i] * b[j] + out[i+j] + carry;
            out[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        out[i+4] = carry;
    }
}

static void glv_mul128x128(uint64_t out[4], const uint64_t a[2], const uint64_t b[2])
{
    memset(out, 0, 4 * sizeof(uint64_t));
    for (int i = 0; i < 2; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 2; j++) {
            uint128 prod = (uint128)a[i] * b[j] + out[i+j] + carry;
            out[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        out[i+2] = carry;
    }
}

static int glv_sub256(uint64_t out[4], const uint64_t a[4], const uint64_t b[4])
{
    uint128 borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint128 diff = (uint128)a[i] - b[i] - borrow;
        out[i] = (uint64_t)diff;
        borrow = (diff >> 64) != 0 ? 1 : 0;
    }
    return (int)borrow;
}

static int glv_add256(uint64_t out[4], const uint64_t a[4], const uint64_t b[4])
{
    uint128 carry = 0;
    for (int i = 0; i < 4; i++) {
        carry += (uint128)a[i] + b[i];
        out[i] = (uint64_t)carry;
        carry >>= 64;
    }
    return (int)carry;
}

static uint64_t glv_submul(uint64_t u[], int j, const uint64_t v[], uint64_t q_hat)
{
    uint64_t carry = 0, borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint128 prod = (uint128)q_hat * v[i] + carry;
        uint64_t prod_lo = (uint64_t)prod;
        carry = (uint64_t)(prod >> 64);
        uint128 diff = (uint128)u[j+i] - prod_lo - borrow;
        u[j+i] = (uint64_t)diff;
        borrow = (diff >> 64) != 0 ? 1 : 0;
    }
    uint128 diff = (uint128)u[j+4] - carry - borrow;
    u[j+4] = (uint64_t)diff;
    return (diff >> 64) != 0 ? 1 : 0;
}

static void glv_addback(uint64_t u[], int j, const uint64_t v[])
{
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint128 sum = (uint128)u[j+i] + v[i] + carry;
        u[j+i] = (uint64_t)sum;
        carry = (uint64_t)(sum >> 64);
    }
    u[j+4] += carry;
}

static void glv_div384by256(uint64_t q[2], const uint64_t u_in[6], const uint64_t v_in[4])
{
    int shift = __builtin_clzll(v_in[3]);
    uint64_t v[4], u[7];

    if (shift > 0) {
        for (int i = 3; i > 0; i--)
            v[i] = (v_in[i] << shift) | (v_in[i-1] >> (64 - shift));
        v[0] = v_in[0] << shift;
        u[6] = u_in[5] >> (64 - shift);
        for (int i = 5; i > 0; i--)
            u[i] = (u_in[i] << shift) | (u_in[i-1] >> (64 - shift));
        u[0] = u_in[0] << shift;
    } else {
        memcpy(v, v_in, 4 * sizeof(uint64_t));
        memcpy(u, u_in, 6 * sizeof(uint64_t));
        u[6] = 0;
    }

    for (int j = 1; j >= 0; j--) {
        uint128 u_top = ((uint128)u[j+4] << 64) | u[j+3];
        uint64_t q_hat;
        if (u[j+4] == v[3]) {
            q_hat = UINT64_MAX;
        } else {
            q_hat = (uint64_t)(u_top / v[3]);
            uint128 rhat = u_top - (uint128)q_hat * v[3];
            while ((uint64_t)(rhat >> 64) == 0 &&
                   (uint128)q_hat * v[2] > (rhat << 64 | u[j+2])) {
                q_hat--;
                rhat += v[3];
            }
        }
        if (glv_submul(u, j, v, q_hat)) {
            q_hat--;
            glv_addback(u, j, v);
        }
        q[j] = q_hat;
    }
}

// GLV precomputed constants
static fp_t glv_beta;          // cube root of unity in Fp
static uint64_t glv_s1[2];     // Babai multiplier: 2x+1 (64-bit, stored in 128-bit)
static uint64_t glv_s2[2];     // Babai multiplier: 6x²+2x (127-bit)
static uint64_t glv_a2[2];     // Lattice component: 6x²+4x+1 (128-bit)
static uint64_t glv_r[4];      // scalar field modulus r as raw limbs
static bool glv_initialized = false;

static fp_t glv_fp_pow(fp_t base, const uint64_t exp[4])
{
    fp_t result = fp_t::one();
    fp_t b = base;
    for (int i = 0; i < 4; i++) {
        uint64_t e = exp[i];
        for (int bit = 0; bit < 64; bit++) {
            if (e & 1) result *= b;
            b *= b;
            e >>= 1;
        }
    }
    return result;
}

static void glv_init()
{
    if (glv_initialized) return;

    // BN254 scalar field modulus
    glv_r[0] = 0x43e1f593f0000001ULL;
    glv_r[1] = 0x2833e84879b97091ULL;
    glv_r[2] = 0xb85045b68181585dULL;
    glv_r[3] = 0x30644e72e131a029ULL;

    // BN254 GLV lattice constants (x = 4965661367192848881)
    // Lattice basis: v1 = (s1, -s2), v2 = (a2, s1), det(B) = r
    // s1 = 2x+1 (64-bit)
    glv_s1[0] = 0x89d3256894d213e3ULL;
    glv_s1[1] = 0;
    // s2 = 6x²+2x (127-bit)
    glv_s2[0] = 0x8211bbeb7d4f1128ULL;
    glv_s2[1] = 0x6f4d8248eeb859fcULL;
    // a2 = 6x²+4x+1 (128-bit)
    glv_a2[0] = 0x0be4e1541221250bULL;
    glv_a2[1] = 0x6f4d8248eeb859fdULL;

    // Compute beta = cube root of unity in Fp
    fp_t one = fp_t::one();
    fp_t two = one + one;
    fp_t three = two + one;
    fp_t neg3 = -three;

    const uint64_t p1_over_4[4] = {
        0x4f082305b61f3f52ULL, 0x65e05aa45a1c72a3ULL,
        0x6e14116da0605617ULL, 0x0c19139cb84c680aULL
    };
    fp_t sqrt_neg3 = glv_fp_pow(neg3, p1_over_4);
    // Try both cube roots to find the one matching our lattice lambda
    // ω₁ = (-1 + sqrt(-3))/2,  ω₂ = (-1 - sqrt(-3))/2
    glv_beta = (sqrt_neg3 - one) / two;  // ω₁

    fp_t beta3 = glv_beta * glv_beta * glv_beta;
    assert(beta3.is_one());

    glv_initialized = true;
}

// Decompose N scalars into 2N half-scalars using GLV.
// Input: scalars[0..N-1] (raw bytes, same representation as MSM expects)
// Output: out[0..N-1] = k1 values, out[N..2N-1] = k2 values
// Sign handling: negative k_i stored as r - |k_i|, which the breakdown
// kernel's abs() detects (> r/2) and negates back, setting the msb flag
// to flip point signs. No offset or correction needed.
static void glv_decompose(scalar_t out[], const scalar_t scalars[], size_t N)
{
    for (size_t idx = 0; idx < N; idx++) {
        uint64_t k[4];
        memcpy(k, &scalars[idx], 32);

        // Babai decomposition: c1 = floor(k * s1 / r), c2 = floor(k * s2 / r)
        uint64_t prod1[6], c1[2];
        glv_mul128x256(prod1, glv_s1, k);
        glv_div384by256(c1, prod1, glv_r);

        uint64_t prod2[6], c2[2];
        glv_mul128x256(prod2, glv_s2, k);
        glv_div384by256(c2, prod2, glv_r);

        // k1 = k - c1*s1 - c2*a2
        uint64_t c1s1[4], c2a2[4];
        glv_mul128x128(c1s1, c1, glv_s1);
        glv_mul128x128(c2a2, c2, glv_a2);

        uint64_t k1_tmp[4], k1[4];
        glv_sub256(k1_tmp, k, c1s1);
        int k1_borrow = glv_sub256(k1, k1_tmp, c2a2);

        // k2 = c1*s2 - c2*s1
        uint64_t c1s2[4], c2s1[4], k2[4];
        glv_mul128x128(c1s2, c1, glv_s2);
        glv_mul128x128(c2s1, c2, glv_s1);
        int k2_borrow = glv_sub256(k2, c1s2, c2s1);

        // Handle sign via field negation:
        // If k_i < 0 (borrow set), k_i is stored as 2^256 + k_true (wrapped).
        // Adding r gives: (r + 2^256 + k_true) mod 2^256 = r + k_true = r - |k_true|.
        // This value is > r/2, so breakdown's abs() negates it back to |k_true|
        // and sets msb=1 to flip the point sign. Net effect: -|k_true| * P = k_true * P.
        // If k_i >= 0, it's < 2^128 << r/2, so abs() leaves it unchanged (msb=0).
        uint64_t k1_out[4], k2_out[4];
        if (k1_borrow) {
            glv_add256(k1_out, glv_r, k1);
        } else {
            memcpy(k1_out, k1, 32);
        }

        if (k2_borrow) {
            glv_add256(k2_out, glv_r, k2);
        } else {
            memcpy(k2_out, k2, 32);
        }

        memcpy(&out[idx], k1_out, 32);
        memcpy(&out[N + idx], k2_out, 32);
    }
}

#endif // FEATURE_BN254

// ==================== Non-preloaded MSM ====================

extern "C"
RustError::by_value mult_pippenger_inf(point_t* out, const affine_t points[],
                                       size_t npoints, const scalar_t scalars[],
                                       size_t ffi_affine_sz)
{
    return mult_pippenger<bucket_t>(out, points, npoints, scalars, false, ffi_affine_sz);
}

// ==================== Pre-loaded MSM API (G1) ====================
// All kernel launches happen in this compilation unit to avoid NVCC
// cross-TU device code generation issues.

static msm_t<bucket_t, point_t, affine_t, scalar_t>* g_msm_g1 = nullptr;
static scalar_t* g_d_scalars_g1 = nullptr;
static const void* g_cached_scalars_g1 = nullptr;
static size_t g_cached_npoints_g1 = 0;

#ifdef FEATURE_BN254
static scalar_t* g_host_glv_scalars_g1 = nullptr;
#endif

extern "C"
RustError::by_value preloaded_msm_init_g1(const affine_t points[],
                                           size_t npoints,
                                           size_t ffi_affine_sz)
{
    try {
        delete g_msm_g1;
        g_msm_g1 = nullptr;

        if (g_d_scalars_g1) {
            cudaFree(g_d_scalars_g1);
            g_d_scalars_g1 = nullptr;
            g_cached_scalars_g1 = nullptr;
            g_cached_npoints_g1 = 0;
        }

#ifdef FEATURE_BN254
        glv_init();

        delete[] g_host_glv_scalars_g1;
        g_host_glv_scalars_g1 = nullptr;

        // Create 2N point array: [P_0,...,P_{N-1}, phi(P_0),...,phi(P_{N-1})]
        affine_t* pts2n = new affine_t[2 * npoints];

        const char* src = reinterpret_cast<const char*>(points);
        for (size_t i = 0; i < npoints; i++)
            memcpy(&pts2n[i], src + i * ffi_affine_sz, sizeof(affine_t));

        // Compute endomorphism points phi(P) = (beta*X, Y)
        for (size_t i = 0; i < npoints; i++) {
            const fp_t* src_xy = reinterpret_cast<const fp_t*>(&pts2n[i]);
            const bool* src_inf = reinterpret_cast<const bool*>(
                reinterpret_cast<const char*>(&pts2n[i]) + 2 * sizeof(fp_t));

            char* dst = reinterpret_cast<char*>(&pts2n[npoints + i]);
            fp_t* dst_xy = reinterpret_cast<fp_t*>(dst);
            bool* dst_inf = reinterpret_cast<bool*>(dst + 2 * sizeof(fp_t));

            if (*src_inf) {
                memset(dst, 0, sizeof(affine_t));
                *dst_inf = true;
            } else {
                dst_xy[0] = glv_beta * src_xy[0];
                dst_xy[1] = src_xy[1];
                *dst_inf = false;
            }
        }

        // Initialize MSM with 2N points (constructor uses default wbits=17)
        g_msm_g1 = new msm_t<bucket_t, point_t, affine_t, scalar_t>(
            pts2n, 2 * npoints, sizeof(affine_t));

        // Switch to wbits=16 for GLV: |k1|, |k2| < 2^128, so 128/16 = 8
        // windows with all 16 bits fully utilized → uniform bucket distribution.
        // (wbits=17 would give top window only 9 significant bits → 128x bucket
        // imbalance → 9.4x accumulate regression)
        g_msm_g1->set_wbits(16);
        g_msm_g1->set_nbits(128);

        delete[] pts2n;
#else
        g_msm_g1 = new msm_t<bucket_t, point_t, affine_t, scalar_t>(
            points, npoints, ffi_affine_sz);
#endif

        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
}

extern "C"
RustError::by_value preloaded_msm_invoke_g1(point_t* out,
                                             size_t npoints,
                                             const scalar_t scalars[])
{
#ifdef FEATURE_BN254
    size_t glv_npoints = 2 * npoints;

    if (g_cached_scalars_g1 != scalars || g_cached_npoints_g1 != npoints) {
        if (!g_host_glv_scalars_g1)
            g_host_glv_scalars_g1 = new scalar_t[glv_npoints];

        // GLV decomposition: k -> (k1, k2) with sign via field negation
        glv_decompose(g_host_glv_scalars_g1, scalars, npoints);

        if (g_d_scalars_g1)
            cudaFree(g_d_scalars_g1);
        size_t aligned = (glv_npoints + WARP_SZ - 1) & ~(size_t)(WARP_SZ - 1);
        CUDA_OK(cudaMalloc(&g_d_scalars_g1, aligned * sizeof(scalar_t)));

        const auto& gpu = g_msm_g1->get_gpu();
        CUDA_OK(cudaMemcpyAsync(g_d_scalars_g1, g_host_glv_scalars_g1,
                                glv_npoints * sizeof(scalar_t),
                                cudaMemcpyHostToDevice, gpu[2]));
        g_cached_scalars_g1 = scalars;
        g_cached_npoints_g1 = npoints;

        g_msm_g1->set_d_scalars_ptr(g_d_scalars_g1);
        g_msm_g1->precompute_digits(glv_npoints, false);
    }

    return g_msm_g1->invoke_precomputed(*out, glv_npoints);
#else
    // Non-GLV path: cache scalars + precomputed digits
    if (g_cached_scalars_g1 != scalars || g_cached_npoints_g1 != npoints) {
        if (g_d_scalars_g1)
            cudaFree(g_d_scalars_g1);
        size_t aligned = (npoints + WARP_SZ - 1) & ~(size_t)(WARP_SZ - 1);
        CUDA_OK(cudaMalloc(&g_d_scalars_g1, aligned * sizeof(scalar_t)));

        const auto& gpu = g_msm_g1->get_gpu();
        CUDA_OK(cudaMemcpyAsync(g_d_scalars_g1, scalars,
                                npoints * sizeof(scalar_t),
                                cudaMemcpyHostToDevice, gpu[2]));
        g_cached_scalars_g1 = scalars;
        g_cached_npoints_g1 = npoints;

        g_msm_g1->set_d_scalars_ptr(g_d_scalars_g1);
        g_msm_g1->precompute_digits(npoints, false);
    }

    return g_msm_g1->invoke_precomputed(*out, npoints);
#endif
}

extern "C"
void preloaded_msm_free_g1()
{
    delete g_msm_g1;
    g_msm_g1 = nullptr;
    if (g_d_scalars_g1) {
        cudaFree(g_d_scalars_g1);
        g_d_scalars_g1 = nullptr;
        g_cached_scalars_g1 = nullptr;
        g_cached_npoints_g1 = 0;
    }
#ifdef FEATURE_BN254
    delete[] g_host_glv_scalars_g1;
    g_host_glv_scalars_g1 = nullptr;
#endif
}

// ==================== G2 ====================
#if defined(FEATURE_BLS12_381) || defined(FEATURE_BLS12_377) || defined(FEATURE_BN254)
typedef jacobian_t<fp2_t> point_fp2_t;
typedef xyzz_t<fp2_t> bucket_fp2_t;
typedef bucket_fp2_t::affine_t affine_fp2_t;  // compact 128 bytes (no inf field)

extern "C"
RustError::by_value mult_pippenger_fp2_inf(point_fp2_t* out, const affine_fp2_t points[],
                                           size_t npoints, const scalar_t scalars[],
                                           size_t ffi_affine_sz)
{
    return mult_pippenger<bucket_fp2_t>(out, points, npoints, scalars, false, ffi_affine_sz);
}

// Pre-loaded MSM API (G2)
static msm_t<bucket_fp2_t, point_fp2_t, affine_fp2_t, scalar_t>* g_msm_g2 = nullptr;
static scalar_t* g_d_scalars_g2 = nullptr;
static const void* g_cached_scalars_g2 = nullptr;
static size_t g_cached_npoints_g2 = 0;

extern "C"
RustError::by_value preloaded_msm_init_g2(const affine_fp2_t points[],
                                           size_t npoints,
                                           size_t ffi_affine_sz)
{
    try {
        delete g_msm_g2;
        g_msm_g2 = new msm_t<bucket_fp2_t, point_fp2_t, affine_fp2_t, scalar_t>(
            points, npoints, ffi_affine_sz);
        g_msm_g2->set_integrate_M(4);
        g_msm_g2->set_integrate_nthreads(32);

        if (g_d_scalars_g2) {
            cudaFree(g_d_scalars_g2);
            g_d_scalars_g2 = nullptr;
            g_cached_scalars_g2 = nullptr;
            g_cached_npoints_g2 = 0;
        }

        return RustError{cudaSuccess};
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
}

extern "C"
RustError::by_value preloaded_msm_invoke_g2(point_fp2_t* out,
                                             size_t npoints,
                                             const scalar_t scalars[])
{
    if (g_cached_scalars_g2 != scalars || g_cached_npoints_g2 != npoints) {
        if (g_d_scalars_g2)
            cudaFree(g_d_scalars_g2);
        size_t aligned = (npoints + WARP_SZ - 1) & ~(size_t)(WARP_SZ - 1);
        CUDA_OK(cudaMalloc(&g_d_scalars_g2, aligned * sizeof(scalar_t)));

        // CRITICAL: upload on stream 2 for ordering with breakdown/sort
        const auto& gpu = g_msm_g2->get_gpu();
        CUDA_OK(cudaMemcpyAsync(g_d_scalars_g2, scalars,
                                npoints * sizeof(scalar_t),
                                cudaMemcpyHostToDevice, gpu[2]));
        g_cached_scalars_g2 = scalars;
        g_cached_npoints_g2 = npoints;

        g_msm_g2->set_d_scalars_ptr(g_d_scalars_g2);
        g_msm_g2->precompute_digits(npoints, false);
    }

    return g_msm_g2->invoke_precomputed(*out, npoints);
}

extern "C"
void preloaded_msm_free_g2()
{
    delete g_msm_g2;
    g_msm_g2 = nullptr;
    if (g_d_scalars_g2) {
        cudaFree(g_d_scalars_g2);
        g_d_scalars_g2 = nullptr;
        g_cached_scalars_g2 = nullptr;
        g_cached_npoints_g2 = 0;
    }
}
#endif
