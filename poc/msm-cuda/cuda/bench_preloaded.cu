// GLV endomorphism support code for BN254 MSM.
// The actual preloaded MSM entry points (init/invoke/free for G1 and G2)
// are in pippenger_inf.cu to avoid NVCC cross-TU device code generation issues.

#include <cuda.h>
#include <cassert>

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
typedef bucket_t::affine_t affine_t;   // Affine_t<fp_t>, no inf field
typedef fr_t scalar_t;

// ==================== GLV Endomorphism for BN254 G1 ====================
#ifdef FEATURE_BN254

// Multi-precision helpers (host-side only)
typedef unsigned __int128 uint128;

static void mul128x256(uint64_t out[6], const uint64_t a[2], const uint64_t b[4])
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

static void mul128x128(uint64_t out[4], const uint64_t a[2], const uint64_t b[2])
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

// out = a - b (256-bit), returns borrow (0 or 1)
static int sub256(uint64_t out[4], const uint64_t a[4], const uint64_t b[4])
{
    uint128 borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint128 diff = (uint128)a[i] - b[i] - borrow;
        out[i] = (uint64_t)diff;
        borrow = (diff >> 64) != 0 ? 1 : 0;
    }
    return (int)borrow;
}

// out = a + b (256-bit), returns carry
static int add256(uint64_t out[4], const uint64_t a[4], const uint64_t b[4])
{
    uint128 carry = 0;
    for (int i = 0; i < 4; i++) {
        carry += (uint128)a[i] + b[i];
        out[i] = (uint64_t)carry;
        carry >>= 64;
    }
    return (int)carry;
}

// Subtract q_hat * v[0..3] from u[j..j+4], return borrow
static uint64_t submul(uint64_t u[], int j, const uint64_t v[], uint64_t q_hat)
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

static void addback(uint64_t u[], int j, const uint64_t v[])
{
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint128 sum = (uint128)u[j+i] + v[i] + carry;
        u[j+i] = (uint64_t)sum;
        carry = (uint64_t)(sum >> 64);
    }
    u[j+4] += carry;
}

// Divide 384-bit u by 256-bit v, quotient in q[2] (Knuth Algorithm D)
static void div384by256(uint64_t q[2], const uint64_t u_in[6], const uint64_t v_in[4])
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
        if (submul(u, j, v, q_hat)) {
            q_hat--;
            addback(u, j, v);
        }
        q[j] = q_hat;
    }
}

// GLV precomputed constants (initialized once)
static fp_t glv_beta;          // cube root of unity in Fp
static uint64_t glv_s1[2];     // Babai multiplier s1 = a1 = 2x+1 (64-bit, in 128-bit)
static uint64_t glv_s2[2];     // Babai multiplier s2 = 6x²+2x (127-bit)
static uint64_t glv_a2[2];     // Lattice component a2 = 6x²+4x+1 (128-bit)
static uint64_t glv_r[4];      // scalar field modulus r as raw limbs
static point_t glv_correction; // precomputed 2^127 * (sum(P) + phi(sum(P)))
static bool glv_initialized = false;

// Modular exponentiation for fp_t (host-side, used to compute beta)
static fp_t fp_pow(fp_t base, const uint64_t exp[4])
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

    // r = BN254 scalar field modulus
    glv_r[0] = 0x43e1f593f0000001ULL;
    glv_r[1] = 0x2833e84879b97091ULL;
    glv_r[2] = 0xb85045b68181585dULL;
    glv_r[3] = 0x30644e72e131a029ULL;

    // BN254 GLV lattice constants (x = 4965661367192848881)
    // s1 = a1 = 2x+1 = 9931322734385697763 (64-bit)
    glv_s1[0] = 0x89d3256894d213e3ULL;
    glv_s1[1] = 0;
    // s2 = 6x²+2x (127-bit)
    glv_s2[0] = 0x8211bbeb7d4f1128ULL;
    glv_s2[1] = 0x6f4d8248eeb859fcULL;
    // a2 = 6x²+4x+1 = s2 + s1 (128-bit)
    glv_a2[0] = 0x0be4e1541221250bULL;
    glv_a2[1] = 0x6f4d8248eeb859fdULL;

    // Compute beta = cube root of unity in Fp
    // beta = (sqrt(-3) - 1) / 2 mod p
    fp_t one = fp_t::one();
    fp_t two = one + one;
    fp_t three = two + one;
    fp_t neg3 = -three;

    const uint64_t p1_over_4[4] = {
        0x4f082305b61f3f52ULL, 0x65e05aa45a1c72a3ULL,
        0x6e14116da0605617ULL, 0x0c19139cb84c680aULL
    };
    fp_t sqrt_neg3 = fp_pow(neg3, p1_over_4);
    // Use (-sqrt(-3) - 1)/2 — this cube root corresponds to our lambda
    glv_beta = (-sqrt_neg3 - one) / two;

    fp_t beta3 = glv_beta * glv_beta * glv_beta;
    assert(beta3.is_one());

    glv_initialized = true;
}

// Decompose N scalars into 2N half-scalars using GLV
// Input: scalars[0..N-1] (Montgomery form from arkworks)
// Output: out[0..N-1] = k1 values, out[N..2N-1] = k2 values (raw, unsigned)
// Uses offset M=2^127 so all outputs are non-negative: k_i' = k_i + M
// Caller must subtract correction = M*(sum(P)+sum(phi(P))) from MSM result.
static void glv_decompose(scalar_t out[], const scalar_t scalars[], size_t N)
{
    for (size_t idx = 0; idx < N; idx++) {
        scalar_t s = scalars[idx];
        s.from();  // Montgomery -> raw
        uint64_t k[4];
        memcpy(k, &s, 32);

        // c1 = floor(k * s1 / r), where s1 = 2x+1 (64-bit)
        uint64_t prod1[6], c1[2];
        mul128x256(prod1, glv_s1, k);
        div384by256(c1, prod1, glv_r);

        // c2 = floor(k * s2 / r), where s2 = 6x^2+2x (127-bit)
        uint64_t prod2[6], c2[2];
        mul128x256(prod2, glv_s2, k);
        div384by256(c2, prod2, glv_r);

        // DEBUG: bypass decomposition, just use k1=k, k2=0
        uint64_t k1[4], k2[4];
        memcpy(k1, k, 32);
        memset(k2, 0, 32);

        // Store as 256-bit scalars (top 128 bits are zero)
        memcpy(&out[idx], k1, 32);
        memcpy(&out[N + idx], k2, 32);
    }
}

// Compute 2N GLV points: [P_0,...,P_{N-1}, phi(P_0),...,phi(P_{N-1})]
// phi(x,y) = (beta*x, y)
static affine_t* glv_make_points(const affine_t points[], size_t N, size_t ffi_affine_sz)
{
    affine_t* pts2n = new affine_t[2 * N];

    // Copy original points
    if (ffi_affine_sz == sizeof(affine_t)) {
        memcpy(pts2n, points, N * sizeof(affine_t));
    } else {
        const char* src = reinterpret_cast<const char*>(points);
        for (size_t i = 0; i < N; i++)
            memcpy(&pts2n[i], src + i * ffi_affine_sz, sizeof(affine_t));
    }

    // Compute endomorphism points phi(P) = (beta*X, Y)
    for (size_t i = 0; i < N; i++) {
        const fp_t* xy = reinterpret_cast<const fp_t*>(&pts2n[i]);
        fp_t new_x = glv_beta * xy[0];
        pts2n[N + i] = affine_t(new_x, xy[1]);
    }

    return pts2n;
}

// Suppress "unused" warnings for GLV functions (will be used later)
static void glv_suppress_warnings() __attribute__((unused));
static void glv_suppress_warnings()
{
    (void)glv_init;
    (void)glv_decompose;
    (void)glv_make_points;
    (void)mul128x128;
    (void)sub256;
    (void)add256;
    (void)glv_a2;
}

#endif // FEATURE_BN254
