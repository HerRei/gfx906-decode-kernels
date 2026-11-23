#pragma once

#include "common.cuh"

// gfx906 single-token flash attention for head size 128, 3 Q heads per K/V head, q8_0 K/V.
// Returns false (and does nothing) when the op is not that case; GGML_CUDA_GCN_FA_DEC=0 turns it off.
bool ggml_cuda_gcn_fa_dec(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
