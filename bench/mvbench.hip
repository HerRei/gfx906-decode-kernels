// mvbench: IQ4_NL MoE matvec (8 of 128 experts, one token) on gfx906, the standard ggml block layout, several kernels
// side by side. This is how the matvec kernel was found; each variant is checked against a CPU reference.
//   READ ceiling  stream-read the same 8 expert slices with 16-byte loads (what the card can do at all)
//   direct        each lane computes whole blocks from 2-byte loads (the winner)
//   lds           the rows are copied to LDS with 16-byte loads first (slower: 16-52 % in another fork too)
//   unroll        direct, with several blocks' loads in flight per lane
//   dw            whole blocks via one 16-byte + one 4-byte aligned load and alignbyte (slower)
//   ylds          direct with the activation staged in LDS (best for the expert products, used in the kernel)
// Build and run (rows, K):  hipcc -O3 --offload-arch=gfx906 -o mvbench mvbench.hip && ./mvbench 2304 2304
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

struct __attribute__((packed)) block_iq4_nl { __half d; uint8_t qs[16]; };   // 18 bytes
struct block_q8_1 { __half2 ds; int8_t qs[32]; };                           // 36 bytes
static_assert(sizeof(block_iq4_nl) == 18, "iq4");
static_assert(sizeof(block_q8_1) == 36, "q8");

__constant__ int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static const int8_t kv_h[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ __forceinline__ int dp4a(int a, int b, int c) { return __builtin_amdgcn_sdot4(a, b, c, false); }

__device__ __forceinline__ int2 lut16(uint32_t q4) {
    // same idea as ggml's HIP get_int_from_table_16 (v_perm based)
    const uint32_t * t = (const uint32_t *) kv;
    const uint32_t qe = q4, qo = q4 >> 4;
    uint32_t el = __builtin_amdgcn_perm(t[1], t[0], qe & 0x07070707);
    uint32_t ol = __builtin_amdgcn_perm(t[1], t[0], qo & 0x07070707);
    uint32_t eh = __builtin_amdgcn_perm(t[3], t[2], qe & 0x07070707);
    uint32_t oh = __builtin_amdgcn_perm(t[3], t[2], qo & 0x07070707);
    uint32_t me = 0x03020100 | ((qe & 0x08080808) >> 1);
    uint32_t mo = 0x03020100 | ((qo & 0x08080808) >> 1);
    return make_int2(__builtin_amdgcn_perm(eh, el, me), __builtin_amdgcn_perm(oh, ol, mo));
}

// dot of one iq4_nl block (given as bytes at p, 2-byte aligned) with one q8_1 block
__device__ __forceinline__ float dot_block(const uint8_t * p, const block_q8_1 * y) {
    const uint16_t * p16 = (const uint16_t *) p;
    const float d = __half2float(*(const __half *) p16);
    const int * yq = (const int *) y->qs;
    int s = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t q = (uint32_t) p16[1 + 2*l] | ((uint32_t) p16[2 + 2*l] << 16);
        const int2 v = lut16(q);
        s = dp4a(v.x, yq[l], s);
        s = dp4a(v.y, yq[l + 4], s);
    }
    return d * __low2float(y->ds) * (float) s;
}

// A: rows per block RPB, LPR lanes per row; lane reads its blocks straight from global memory
template <int RPB, int LPR>
__global__ void __launch_bounds__(RPB*LPR) k_direct(const uint8_t * __restrict__ W, const block_q8_1 * __restrict__ y,
        const int * __restrict__ ids, float * __restrict__ dst, int nrows, int nb, size_t expert_stride) {
    const int e = ids[blockIdx.y];
    const int r = blockIdx.x*RPB + threadIdx.x / LPR;
    const int l = threadIdx.x % LPR;
    const uint8_t * row = W + e*expert_stride + (size_t) r*nb*18;
    float acc = 0.f;
    for (int b = l; b < nb; b += LPR) acc += dot_block(row + b*18, y + b);
#pragma unroll
    for (int o = LPR/2; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
    if (l == 0 && r < nrows) dst[blockIdx.y*nrows + r] = acc;
}

// B: copy the RPB contiguous rows into LDS with 16-byte loads, then compute from LDS
template <int RPB, int LPR, int MAXNB>
__global__ void __launch_bounds__(RPB*LPR) k_lds(const uint8_t * __restrict__ W, const block_q8_1 * __restrict__ y,
        const int * __restrict__ ids, float * __restrict__ dst, int nrows, int nb, size_t expert_stride) {
    __shared__ __attribute__((aligned(16))) uint8_t sm[RPB*MAXNB*18];
    const int e = ids[blockIdx.y];
    const int r0 = blockIdx.x*RPB;
    const size_t row_bytes = (size_t) nb*18;
    const uint4 * src = (const uint4 *) (W + e*expert_stride + (size_t) r0*row_bytes);   // row_bytes*RPB is a multiple of 16
    const int n16 = (int) (RPB*row_bytes/16);
    uint4 * s4 = (uint4 *) sm;
    for (int i = threadIdx.x; i < n16; i += RPB*LPR) s4[i] = src[i];
    __syncthreads();
    const int rr = threadIdx.x / LPR, l = threadIdx.x % LPR;
    const uint8_t * row = sm + rr*row_bytes;
    float acc = 0.f;
    for (int b = l; b < nb; b += LPR) acc += dot_block(row + b*18, y + b);
#pragma unroll
    for (int o = LPR/2; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
    if (l == 0) dst[blockIdx.y*nrows + r0 + rr] = acc;
}

// D: like A, but each lane issues U block loads before using them
template <int RPB, int LPR, int U>
__global__ void __launch_bounds__(RPB*LPR) k_unroll(const uint8_t * __restrict__ W, const block_q8_1 * __restrict__ y,
        const int * __restrict__ ids, float * __restrict__ dst, int nrows, int nb, size_t expert_stride) {
    const int e = ids[blockIdx.y];
    const int r = blockIdx.x*RPB + threadIdx.x / LPR;
    const int l = threadIdx.x % LPR;
    const uint8_t * row = W + e*expert_stride + (size_t) r*nb*18;
    float acc = 0.f;
    for (int b0 = 0; b0 < nb; b0 += LPR*U) {
        uint16_t raw[U][9];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int b = b0 + u*LPR + l;
            if (b < nb) {
                const uint16_t * p16 = (const uint16_t *) (row + b*18);
#pragma unroll
                for (int j = 0; j < 9; ++j) raw[u][j] = p16[j];
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int b = b0 + u*LPR + l;
            if (b < nb) {
                const float d = __half2float(*(const __half *) &raw[u][0]);
                const int * yq = (const int *) y[b].qs;
                int s = 0;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint32_t w = (uint32_t) raw[u][1 + 2*q] | ((uint32_t) raw[u][2 + 2*q] << 16);
                    const int2 v = lut16(w);
                    s = dp4a(v.x, yq[q], s);
                    s = dp4a(v.y, yq[q + 4], s);
                }
                acc += d * __low2float(y[b].ds) * (float) s;
            }
        }
    }
#pragma unroll
    for (int o = LPR/2; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
    if (l == 0 && r < nrows) dst[blockIdx.y*nrows + r] = acc;
}

// E: each block read with one 16-byte + one 4-byte load from the 4-byte aligned address below it, then realigned
typedef uint32_t u32x4a __attribute__((ext_vector_type(4), aligned(4)));
__device__ __forceinline__ float dot_block_dw(const uint8_t * p, const block_q8_1 * y) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t * A = (const uint32_t *) (a & ~(uintptr_t) 3);
    const bool odd = (a & 3) != 0;              // 2-byte offset
    const u32x4a w4 = *(const u32x4a *) A;
    const uint32_t w[5] = {w4.x, w4.y, w4.z, w4.w, A[4]};
    const uint16_t dh = odd ? (uint16_t) (w[0] >> 16) : (uint16_t) (w[0] & 0xffff);
    __half dhh; __builtin_memcpy(&dhh, &dh, 2);
    const float d = __half2float(dhh);
    const int * yq = (const int *) y->qs;
    int s = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t q = odd ? w[l + 1] : __builtin_amdgcn_alignbyte(w[l + 1], w[l], 2);
        const int2 v = lut16(q);
        s = dp4a(v.x, yq[l], s);
        s = dp4a(v.y, yq[l + 4], s);
    }
    return d * __low2float(y->ds) * (float) s;
}
template <int RPB, int LPR>
__global__ void __launch_bounds__(RPB*LPR) k_dw(const uint8_t * __restrict__ W, const block_q8_1 * __restrict__ y,
        const int * __restrict__ ids, float * __restrict__ dst, int nrows, int nb, size_t expert_stride) {
    const int e = ids[blockIdx.y];
    const int r = blockIdx.x*RPB + threadIdx.x / LPR;
    const int l = threadIdx.x % LPR;
    const uint8_t * row = W + e*expert_stride + (size_t) r*nb*18;
    float acc = 0.f;
    for (int b = l; b < nb; b += LPR) acc += dot_block_dw(row + b*18, y + b);
#pragma unroll
    for (int o = LPR/2; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
    if (l == 0 && r < nrows) dst[blockIdx.y*nrows + r] = acc;
}

// F: bandwidth ceiling: stream-read the same 8 expert slices with 16-byte loads
__global__ void __launch_bounds__(256) k_read(const uint4 * __restrict__ W, const int * __restrict__ ids, float * __restrict__ dst, size_t n16_per_expert) {
    const uint4 * base = W + ids[blockIdx.y]*n16_per_expert;
    uint32_t x = 0;
    for (size_t i = blockIdx.x*256 + threadIdx.x; i < n16_per_expert; i += (size_t) gridDim.x*256) { const uint4 v = base[i]; x ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (x == 0x12345678) dst[0] = 1.f;
}

// G: activation (q8_1 blocks) staged in LDS once per workgroup, weights straight from global
template <int RPB, int LPR, int MAXNB>
__global__ void __launch_bounds__(RPB*LPR) k_ylds(const uint8_t * __restrict__ W, const block_q8_1 * __restrict__ y,
        const int * __restrict__ ids, float * __restrict__ dst, int nrows, int nb, size_t expert_stride) {
    __shared__ block_q8_1 ys[MAXNB];
    for (int i = threadIdx.x; i < nb*9; i += RPB*LPR) ((int *) ys)[i] = ((const int *) y)[i];
    __syncthreads();
    const int e = ids[blockIdx.y];
    const int r = blockIdx.x*RPB + threadIdx.x / LPR;
    const int l = threadIdx.x % LPR;
    const uint8_t * row = W + e*expert_stride + (size_t) r*nb*18;
    float acc = 0.f;
    for (int b = l; b < nb; b += LPR) acc += dot_block(row + b*18, ys + b);
#pragma unroll
    for (int o = LPR/2; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
    if (l == 0 && r < nrows) dst[blockIdx.y*nrows + r] = acc;
}

static float h2f(uint16_t h) { __half x; memcpy(&x, &h, 2); return __half2float(x); }

int main(int argc, char ** argv) {
    const int nrows = argc > 1 ? atoi(argv[1]) : 2304;   // output rows per expert
    const int K     = argc > 2 ? atoi(argv[2]) : 2304;   // input length
    const int nexp = 128, nused = 8, nb = K/32;
    const size_t estride = (size_t) nrows*nb*18;
    std::vector<uint8_t> hW(estride*nexp);
    srand(1);
    for (size_t i = 0; i < hW.size(); i += 18) {
        __half d = __float2half(0.001f + (rand() % 1000)*1e-5f); memcpy(&hW[i], &d, 2);
        for (int j = 0; j < 16; ++j) hW[i + 2 + j] = rand() & 0xff;
    }
    std::vector<block_q8_1> hy(nb);
    for (int b = 0; b < nb; ++b) { hy[b].ds = __floats2half2_rn(0.01f + (rand() % 100)*1e-4f, 0.f); for (int j = 0; j < 32; ++j) hy[b].qs[j] = (int8_t) ((rand() % 255) - 127); }
    std::vector<int> hids(nused); for (int i = 0; i < nused; ++i) hids[i] = (i*37 + 5) % nexp;
    // CPU reference
    std::vector<float> ref(nused*nrows);
    for (int u = 0; u < nused; ++u) for (int r = 0; r < nrows; ++r) {
        double acc = 0;
        for (int b = 0; b < nb; ++b) {
            const uint8_t * p = &hW[hids[u]*estride + ((size_t) r*nb + b)*18];
            uint16_t dh; memcpy(&dh, p, 2); const float d = h2f(dh);
            int s = 0;
            for (int j = 0; j < 16; ++j) { s += kv_h[p[2 + j] & 15]*hy[b].qs[j]; s += kv_h[p[2 + j] >> 4]*hy[b].qs[j + 16]; }
            acc += (double) d*__low2float(hy[b].ds)*s;
        }
        ref[u*nrows + r] = (float) acc;
    }
    uint8_t * dW; block_q8_1 * dy; int * dids; float * dd;
    CHECK(hipMalloc(&dW, hW.size())); CHECK(hipMalloc(&dy, nb*sizeof(block_q8_1))); CHECK(hipMalloc(&dids, nused*4)); CHECK(hipMalloc(&dd, nused*nrows*4));
    CHECK(hipMemcpy(dW, hW.data(), hW.size(), hipMemcpyHostToDevice)); CHECK(hipMemcpy(dy, hy.data(), nb*sizeof(block_q8_1), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dids, hids.data(), nused*4, hipMemcpyHostToDevice));
    const double bytes = (double) estride*nused;
    hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
    auto run = [&](const char * name, auto launch) {
        CHECK(hipMemset(dd, 0, nused*nrows*4)); launch(); CHECK(hipDeviceSynchronize());
        std::vector<float> out(nused*nrows); CHECK(hipMemcpy(out.data(), dd, out.size()*4, hipMemcpyDeviceToHost));
        double maxrel = 0; for (size_t i = 0; i < out.size(); ++i) maxrel = std::max(maxrel, (double) fabs(out[i] - ref[i])/(fabs(ref[i]) + 1e-3));
        // timing: rotate over 16 different id sets so the weights do not sit in L2
        const int iters = 400; CHECK(hipEventRecord(a));
        for (int it = 0; it < iters; ++it) launch();
        CHECK(hipEventRecord(b)); CHECK(hipEventSynchronize(b));
        float ms; CHECK(hipEventElapsedTime(&ms, a, b));
        const double us = 1000.0*ms/iters;
        printf("%-28s %8.2f us  %6.0f GB/s  maxrel %.2e\n", name, us, bytes/us/1e3, maxrel);
    };
    printf("rows %d K %d, %d of %d experts, %.1f MB per call\n", nrows, K, nused, nexp, bytes/1e6);
#define RUN_DIRECT(RPB, LPR) run("direct rpb" #RPB " lpr" #LPR, [&]{ hipLaunchKernelGGL((k_direct<RPB, LPR>), dim3(nrows/RPB, nused), dim3(RPB*LPR), 0, 0, dW, dy, dids, dd, nrows, nb, estride); })
#define RUN_LDS(RPB, LPR) run("lds rpb" #RPB " lpr" #LPR, [&]{ hipLaunchKernelGGL((k_lds<RPB, LPR, 96>), dim3(nrows/RPB, nused), dim3(RPB*LPR), 0, 0, dW, dy, dids, dd, nrows, nb, estride); })
    RUN_DIRECT(2, 32); RUN_DIRECT(8, 8); RUN_DIRECT(16, 4);
#define RUN_Y(RPB, LPR) run("ylds rpb" #RPB " lpr" #LPR, [&]{ hipLaunchKernelGGL((k_ylds<RPB, LPR, 96>), dim3(nrows/RPB, nused), dim3(RPB*LPR), 0, 0, dW, dy, dids, dd, nrows, nb, estride); })
    RUN_Y(2, 32); RUN_Y(4, 32); RUN_Y(8, 8); RUN_Y(8, 16); RUN_Y(16, 4); RUN_Y(16, 8); RUN_Y(32, 4); RUN_Y(64, 4); RUN_Y(32, 8);
    return 0;
}
