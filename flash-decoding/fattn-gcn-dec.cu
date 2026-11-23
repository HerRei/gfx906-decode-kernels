// Single-token flash attention ("flash decoding") for gfx906 (Radeon VII, MI50, MI60; GCN5, wave64).
//
// Shape: one query token, head size 128, 3 Q heads per K/V head, K and V in q8_0, an F16 mask, optional sinks and
// the optional Inkling 2 relative-position bias (ggml_flash_attn_ext_set_rel_bias). Grid: (chunks, K/V heads). Each
// workgroup takes one chunk of `tiles` x NT keys for all 3 Q heads of one K/V head, so K and V are read once:
//   1. Q of the 3 heads is quantized to int8 per 32 values into LDS (softmax scale folded in).
//   2. Per tile, each lane reads one key's q8_0 row (136 bytes) and computes its 3 scores with dp4a; the next tile's
//      rows are loaded while this tile does steps 3 and 4.
//   3. Online softmax: tile max and sum per head (wave shuffles + LDS), running max and sum, accumulators rescaled.
//   4. V: 8 lanes per row, 16 values per lane, NT/8 rows at a time, 4 rows of loads in flight, half2 FMA accumulators.
//   5. Each chunk writes its partial output and (max, sum); a second kernel combines them and adds the sinks.
// Used from GGML_CUDA_GCN_FA_DEC_MIN_KV keys on (default 6144); GGML_CUDA_GCN_FA_DEC=0 turns it off.
// Radeon VII, q8_0 K/V, 24 Q / 8 K/V heads (test-backend-ops perf), vector kernel -> this kernel:
//   8k keys 111 -> 92 us, 16k 210 -> 163, 32k 409 -> 225, 64k 805 -> 451.
#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-gcn-dec.cuh"
#include <algorithm>

#define GCN_FA_D   128
#define GCN_FA_G   3
#define GCN_FA_NB  (GCN_FA_D/32)       // q8_0 blocks per row
#define GCN_FA_ROW (GCN_FA_NB*34)      // bytes per q8_0 row = 136


typedef uint32_t gcn_u32x4 __attribute__((ext_vector_type(4), aligned(4)));

template <int NT>   // threads per workgroup = keys per tile
static __global__ void __launch_bounds__(NT, 1) gcn_fa_dec_split(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V, const half * __restrict__ mask,
        float * __restrict__ part, float2 * __restrict__ meta, const float scale, const int n_kv, const int n_head,
        const int nb02, const int nb11, const int nb12, const int nb21, const int nb22, const ggml_cuda_fattn_rel_bias rb,
        const int tiles) {
#if defined(GCN)
    constexpr int G = GCN_FA_G, D = GCN_FA_D, NB = GCN_FA_NB, NW = NT/64;
    const int chunk = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int head0 = kvh*G;
    const int tid   = threadIdx.x;
    const int lane  = tid & 63;
    const int wave  = tid >> 6;
    const int k0    = chunk*tiles*NT;            // first key of this workgroup's chunk

    __shared__ int    q_i[G][D/4];       // Q as int8 x4, scale folded in
    __shared__ float  q_d[G][NB];
    __shared__ int8_t q_b[G][D];
    __shared__ float  p_s[G][NT];
    __shared__ float  red[G][NW];
    __shared__ float  acc_s[NW][G][D];

    // 1. quantize Q once per workgroup (each 32-value block sits in 32 consecutive lanes)
    for (int i = tid; i < G*D; i += NT) {
        const int g = i / D, dd = i % D;
        const float v = ((const float *) (Q + (size_t) (head0 + g)*nb02))[dd]*scale;
        float amax = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
        const float d = amax/127.0f;
        q_b[g][dd] = (int8_t) (d > 0.0f ? __float2int_rn(v/d) : 0);
        if ((dd & 31) == 0) q_d[g][dd >> 5] = d;
    }
    __syncthreads();
    for (int i = tid; i < G*D/4; i += NT) {
        const int g = i / (D/4), j = i % (D/4);
        q_i[g][j] = *(const int *) &q_b[g][4*j];
    }
    __syncthreads();

    // V mapping: 16 values of a row per lane (half a q8_0 block), 8 lanes per row, RG rows at a time
    constexpr int RG  = NT/8;
    constexpr int RPL = NT/RG;
    constexpr int VB  = 4;                   // rows whose loads are in flight together
    const int l8 = tid & 7, rg = tid >> 3;
    const int vb = l8 >> 1, hb = l8 & 1;
    const int qbyte = 34*vb + 2 + 16*hb;
    const int qal   = qbyte & ~3;
    const bool qsh  = (qbyte & 3) != 0;

    float m_run[G], l_run[G];
    half2 a2[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m_run[g] = -INFINITY; l_run[g] = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) a2[g][i] = make_half2(0.0f, 0.0f);
    }

    // K of one tile: the mask, the row and the key position, loaded together
    uint32_t w[GCN_FA_ROW/4];
    half mh = __float2half(0.0f);
    int  kpos = 0;
    bool in_range = false;
    auto load_k = [&](const int t) {
        const int k  = k0 + t*NT + tid;
        in_range     = k < n_kv;
        const int kc = in_range ? k : n_kv - 1;
        mh = mask[kc];
        const uint32_t * row = (const uint32_t *) (K + (size_t) kc*nb11 + (size_t) kvh*nb12);   // 4-byte aligned
#pragma unroll
        for (int i = 0; i < GCN_FA_ROW/16; ++i) {
            const gcn_u32x4 v4 = *(const gcn_u32x4 *) (row + 4*i);
            w[4*i] = v4.x; w[4*i + 1] = v4.y; w[4*i + 2] = v4.z; w[4*i + 3] = v4.w;
        }
        w[32] = row[32]; w[33] = row[33];
        kpos = rb.table ? rb.kpos[kc] : 0;
    };
    load_k(0);

    for (int t = 0; t < tiles; ++t) {
        const int kt = k0 + t*NT;            // first key of the tile (uniform)
        if (kt >= n_kv) {
            break;
        }
        // 2. scores of this lane's key
        float s[G];
        const float mf = __half2float(mh);
        const bool valid = in_range && mf > -INFINITY && mf > -65000.0f;
        if (valid) {
            float sc[G] = {};
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                const int  wi  = (34*b) / 4;
                const bool off = (34*b) % 4 != 0;
                const uint16_t dh = off ? (uint16_t) (w[wi] >> 16) : (uint16_t) (w[wi] & 0xFFFF);
                const float dkb = __half2float(*(const half *) &dh);
                int si[G] = {};
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int qk = off ? (int) w[wi + 1 + j] : (int) __builtin_amdgcn_alignbyte(w[wi + 1 + j], w[wi + j], 2);
#pragma unroll
                    for (int g = 0; g < G; ++g) si[g] = ggml_cuda_dp4a(qk, q_i[g][8*b + j], si[g]);
                }
#pragma unroll
                for (int g = 0; g < G; ++g) sc[g] += dkb*q_d[g][b]*(float) si[g];
            }
            const int dpos = rb.table ? rb.qpos[0] - kpos : 0;
            const int r    = dpos >= 0 && dpos < rb.extent ? dpos : rb.extent;
#pragma unroll
            for (int g = 0; g < G; ++g) {
                float bias = mf;
                if (rb.table) {
                    const int hh = rb.head0 + head0 + g;   // the value ggml_cuda_fattn_rel_bias_mask gives
                    bias = __half2float(__float2half(rb.table[r + rb.n_rel*hh] - rb.sinks[hh] + mf));
                }
                s[g] = sc[g] + bias;
            }
        } else {
#pragma unroll
            for (int g = 0; g < G; ++g) s[g] = -INFINITY;
        }
        // the next tile's K is in flight while this tile does its softmax and V
        if (t + 1 < tiles && kt + NT < n_kv) {
            load_k(t + 1);
        }

        // 3. online softmax over the tiles
        float mt[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float v = s[g];
#pragma unroll
            for (int o = 32; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 64));
            if (lane == 0) red[g][wave] = v;
        }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float v = red[g][0];
#pragma unroll
            for (int w2 = 1; w2 < NW; ++w2) v = fmaxf(v, red[g][w2]);
            mt[g] = v;
        }
        __syncthreads();
        float corr[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float m_new = fmaxf(m_run[g], mt[g]);
            corr[g] = m_run[g] == -INFINITY ? 0.0f : expf(m_run[g] - m_new);
            m_run[g] = m_new;
            const float p = valid && m_new > -INFINITY ? expf(s[g] - m_new) : 0.0f;
            p_s[g][tid] = p;
            float v = p;
#pragma unroll
            for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
            if (lane == 0) red[g][wave] = v;
        }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float v = 0.0f;
#pragma unroll
            for (int w2 = 0; w2 < NW; ++w2) v += red[g][w2];
            l_run[g] = l_run[g]*corr[g] + v;
            const half2 c2 = make_half2(corr[g], corr[g]);
#pragma unroll
            for (int i = 0; i < 8; ++i) a2[g][i] = __hmul2(a2[g][i], c2);
        }

        // 4. V of this tile
        const int k_end = min(NT, n_kv - kt);
        const char * vbase = V + (size_t) kt*nb21 + (size_t) kvh*nb22;
#pragma unroll
        for (int u0 = 0; u0 < RPL; u0 += VB) {
            uint32_t ww[VB][5];
            float    dv[VB];
#pragma unroll
            for (int u = 0; u < VB; ++u) {
                const int kk = min(rg + (u0 + u)*RG, k_end - 1);
                const char * vr = vbase + (size_t) kk*nb21;
                const gcn_u32x4 w4 = *(const gcn_u32x4 *) (vr + qal);
                ww[u][0] = w4.x; ww[u][1] = w4.y; ww[u][2] = w4.z; ww[u][3] = w4.w;
                ww[u][4] = *(const uint32_t *) (vr + qal + 16);
                dv[u] = __half2float(*(const half *) (vr + 34*vb));
            }
#pragma unroll
            for (int u = 0; u < VB; ++u) {
                const int kk = rg + (u0 + u)*RG;
                if (kk >= k_end) break;
                half2 pg[G];
                bool any = false;
#pragma unroll
                for (int g = 0; g < G; ++g) { const float pf = p_s[g][kk]; any |= pf != 0.0f; pg[g] = make_half2(pf, pf); }
                if (!any) continue;
                const half2 dv2 = make_half2(dv[u], dv[u]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t q = qsh ? __builtin_amdgcn_alignbyte(ww[u][j + 1], ww[u][j], 2) : ww[u][j];
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const half2 x2 = __hmul2(dv2, make_half2((float) (int8_t) (q >> (16*c)), (float) (int8_t) (q >> (16*c + 8))));
#pragma unroll
                        for (int g = 0; g < G; ++g) a2[g][2*j + c] = __hfma2(pg[g], x2, a2[g][2*j + c]);
                    }
                }
            }
        }
        __syncthreads();                         // p_s and red are reused by the next tile
    }

    // 5. sum the row groups (shuffles within a wave, then the waves in LDS) and write this chunk's partial result
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            float v = (i & 1) ? __high2float(a2[g][i >> 1]) : __low2float(a2[g][i >> 1]);
            v += __shfl_xor(v, 8, 64);
            v += __shfl_xor(v, 16, 64);
            v += __shfl_xor(v, 32, 64);
            if (lane < 8) acc_s[wave][g][32*vb + 16*hb + i] = v;
        }
    __syncthreads();
    for (int i = tid; i < G*D; i += NT) {
        const int g = i / D, d = i % D;
        float v = 0.0f;
#pragma unroll
        for (int w2 = 0; w2 < NW; ++w2) v += acc_s[w2][g][d];
        part[((size_t) chunk*n_head + head0 + g)*D + d] = v;
    }
    if (tid < G) {
        meta[(size_t) chunk*n_head + head0 + tid] = make_float2(m_run[tid], l_run[tid]);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, part, meta, scale, n_kv, n_head, nb02, nb11, nb12, nb21, nb22, rb, tiles);
    NO_DEVICE_CODE;
#endif // defined(GCN)
}

// combine the splits of every head; dst[head*D + d] (one token, one sequence)
static __global__ void __launch_bounds__(GCN_FA_D, 1) gcn_fa_dec_combine(
        const float * __restrict__ part, const float2 * __restrict__ meta, const float * __restrict__ sinks,
        float * __restrict__ dst, const int n_split, const int n_head) {
    const int h = blockIdx.x, d = threadIdx.x;
    float M = sinks ? sinks[h] : -INFINITY;
    for (int s = 0; s < n_split; ++s) M = fmaxf(M, meta[(size_t) s*n_head + h].x);
    float L = sinks ? expf(sinks[h] - M) : 0.0f, O = 0.0f;
    if (M > -INFINITY) {
        for (int s = 0; s < n_split; ++s) {
            const float2 ml = meta[(size_t) s*n_head + h];
            if (ml.x == -INFINITY) continue;
            const float w = expf(ml.x - M);
            L += w*ml.y;
            O += w*part[((size_t) s*n_head + h)*GCN_FA_D + d];
        }
    }
    dst[(size_t) h*GCN_FA_D + d] = L > 0.0f ? O/L : 0.0f;
}

bool ggml_cuda_gcn_fa_dec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_GCN_FA_DEC"); return !(e && e[0] == '0'); }();
    static const int  nt = [] { const char * e = getenv("GGML_CUDA_GCN_FA_DEC_NT"); return e ? atoi(e) : 256; }();
    if (!on) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_GCN(cc)) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3], * sinks = dst->src[4];
    float scale, max_bias, softcap;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    const bool ok = Q->type == GGML_TYPE_F32 && Q->ne[0] == GCN_FA_D && Q->ne[1] == 1 && Q->ne[3] == 1 && Q->nb[0] == sizeof(float) &&
        K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 && K->ne[0] == GCN_FA_D && V->ne[0] == GCN_FA_D &&
        K->ne[2] > 0 && Q->ne[2] == GCN_FA_G*K->ne[2] && V->ne[2] == K->ne[2] && K->ne[3] == 1 && V->ne[3] == 1 &&
        K->nb[0] == sizeof(block_q8_0) && V->nb[0] == sizeof(block_q8_0) && K->nb[1] % 4 == 0 && K->nb[2] % 4 == 0 &&
        V->nb[1] % 2 == 0 && V->nb[2] % 2 == 0 &&
        mask && mask->type == GGML_TYPE_F16 && mask->ne[2] == 1 && mask->ne[3] == 1 && mask->ne[0] >= K->ne[1] &&
        max_bias == 0.0f && softcap == 0.0f && (!sinks || sinks->type == GGML_TYPE_F32) &&
        dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && (nt == 128 || nt == 256 || nt == 512) &&
        (!dst->src[6] || (dst->src[7] && dst->src[8] && dst->src[9]));
    if (!ok) {
        return false;
    }
    ggml_cuda_fattn_rel_bias rb;
    if (dst->src[6]) {
        rb.table  = (const float   *) dst->src[6]->data;
        rb.sinks  = (const float   *) dst->src[7]->data;
        rb.kpos   = (const int32_t *) dst->src[8]->data;
        rb.qpos   = (const int32_t *) dst->src[9]->data;
        rb.n_rel  = dst->src[6]->ne[0];
        rb.n_head = dst->src[6]->ne[1];
        rb.extent = ggml_get_op_params_i32(dst, 5);
        rb.head0  = ggml_get_op_params_i32(dst, 6);
    }
    const int n_kv    = K->ne[1];
    const int n_head  = Q->ne[2];
    // below ~6k keys the generic vector kernel is as fast or faster on the VII (1k: 23.5 vs 24.0 us, 4k: 64 vs 67);
    // from 8k on this one wins (8k -17 %, 16k -23 %, 32k and 64k -45 %)
    static const int min_kv = [] { const char * e = getenv("GGML_CUDA_GCN_FA_DEC_MIN_KV"); return e ? atoi(e) : 6144; }();
    if (n_kv < min_kv) {
        return false;
    }
    // tiles per workgroup: measured on the VII (q8_0, 256 keys per tile): 4k keys 2, 16k 4, 64k 8, i.e. about
    // sqrt(tiles)/2; fewer workgroups than that leave CUs idle, more repeat the per-workgroup latency chain
    static const int tiles_env = [] { const char * e = getenv("GGML_CUDA_GCN_FA_DEC_TILES"); return e ? atoi(e) : 0; }();
    const int n_tiles = (n_kv + nt - 1)/nt;
    const int tiles   = tiles_env > 0 ? tiles_env : std::max(1, (int) lrintf(sqrtf((float) n_tiles)*0.5f));
    const int n_split = (n_tiles + tiles - 1)/tiles;
    ggml_cuda_pool_alloc<float>  part(ctx.pool(), (size_t) n_split*n_head*GCN_FA_D);
    ggml_cuda_pool_alloc<float2> meta(ctx.pool(), (size_t) n_split*n_head);
    cudaStream_t stream = ctx.stream();
    const dim3 grid(n_split, K->ne[2], 1);
#define GCN_FA_LAUNCH(NT_) gcn_fa_dec_split<NT_><<<grid, NT_, 0, stream>>>((const char *) Q->data, (const char *) K->data, \
        (const char *) V->data, (const half *) mask->data, part.ptr, meta.ptr, scale, n_kv, n_head, (int) Q->nb[2], (int) K->nb[1], \
        (int) K->nb[2], (int) V->nb[1], (int) V->nb[2], rb, tiles)
    if (nt == 128) { GCN_FA_LAUNCH(128); } else if (nt == 512) { GCN_FA_LAUNCH(512); } else { GCN_FA_LAUNCH(256); }
#undef GCN_FA_LAUNCH
    gcn_fa_dec_combine<<<n_head, GCN_FA_D, 0, stream>>>(part.ptr, meta.ptr, sinks ? (const float *) sinks->data : nullptr,
            (float *) dst->data, n_split, n_head);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
