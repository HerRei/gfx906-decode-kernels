# Inkling 2 Mini on a Radeon VII + RX 9060 XT

How Inkling 2 Mini (Thinking Machines, 26.4B mixture of experts, 8 of 128 experts per token, layers 6 to 17 run
three times) runs on a Radeon VII (gfx906, 16 GB HBM2) together with an RX 9060 XT (gfx1200, 16 GB) on Fedora 44 with
Fedora's ROCm 7.1 packages. It writes about 48 tok/s and reads about 615 tok/s. The six patches go on top of a pinned
llama.cpp tree that already supports Inkling 2; `setup-tree.sh` builds that tree and checks it.

## Results

Writing speed of one chat (two 300-token answers with thinking, temperature 1), both cards:

| Setup | tok/s | reading tok/s |
|---|---|---|
| ggml-org IQ4_NL file, base tree, layers split 18/30 | 41.2 | ~470 |
| + gfx906 matvec (patch 4) | 42.8 | ~460 |
| experts in Q4_1, loop controllers in F16 | 45.0 | 603 |
| + attention in Q4_1 (the file used now) | 48.0 | 615 |

Writing 160 tokens after a long document (131k context, q8_0 cache, mean of two runs):

| Document | patch 6 off | patch 6 on |
|---|---|---|
| 8.6k tokens | 43.9 | 43.5 |
| 34k tokens | 38.5 | 40.4 |
| 68.5k tokens | 33.5 | 37.0 |

The needle in a 64k and a 118k document is found. For comparison, the same machine runs Qwen3.8 27B Q6_K at 33 to
38 tok/s.

## Things that cost time and may save you some

- **"no kernel image is available for execution on the device" with two different AMD GPUs.** rocBLAS loads the
  kernel library of only one GPU per process, so the second card fails as soon as llama.cpp calls rocBLAS
  ([ROCm/rocm-libraries#3413](https://github.com/ROCm/rocm-libraries/issues/3413); the same error on a 9060 XT + MI50
  in [llama.cpp#19893](https://github.com/ggml-org/llama.cpp/issues/19893)).
  - **Upstream fix:** [ROCm/rocm-libraries#4781](https://github.com/ROCm/rocm-libraries/pull/4781) gives each GPU
    architecture its own library. It is in TheRock 7.12 and later, but not in ROCm 7.2.x. Fedora 44 and 45 ship
    rocBLAS 7.1.1 and 7.2.0 without it (checked 2026-11-23).
  - **Not tested here:** the fix, and gfx906 on TheRock (listed there as building, not tested).
  - **Workaround:** patch 1 adds `GGML_CUDA_RDNA_NO_BLAS=1`. The RDNA card then computes float products with
    llama.cpp's own kernels.
- **The layer split of a looped model.** Weights are placed by stored layer, but `-ts` counts the 48 layer passes:
  stored layer i lands on the card that owns logical position i. `-ts 18,30` puts stored layers 0 to 17 (the whole
  loop) on the Radeon VII and layers 18 to 23 plus the output head on the RX 9060 XT. A 50/50 split puts everything on
  one card.
- **Thinking switch.** Inkling 2 reads `reasoning_effort` (none, minimal, low, medium, high), not `enable_thinking`.
- **Sampling.** `--top-k 40` matters: sampling the full 201k vocabulary costs about 1 ms per token on the CPU.
- **ROCm in a container.** With only one card's render node passed in, ROCm reports no device at all. Pass all render
  nodes and pick the card with `ROCR_VISIBLE_DEVICES`.
- **Overclocking the Radeon VII.** 2050 MHz hung after about ten minutes and the reset also took down a job on the other
  card. 1950 MHz at 1150 mV passed a 60-minute soak (generation plus GPU-against-CPU op tests) but gives only 48.2
  instead of 47.4 tok/s, so the card runs at stock.

## The patches

1. **RDNA float products without rocBLAS** (`GGML_CUDA_RDNA_NO_BLAS=1`), see above.
2. **DSpark draft loader** accepts a `markov_w1` padded to the full vocabulary (201024 rows instead of 200058).
3. **Op profiler and perf cases.** `GGML_CUDA_OPPROF=1` with `GGML_CUDA_DISABLE_GRAPHS=1` prints GPU time per op kind.
   It overstates small ops; the decisions here used `test-backend-ops perf` with the Inkling 2 shapes this adds.
4. **gfx906 single-column matvec**, tuned for Inkling's shapes (`GGML_CUDA_GCN_MMV=0` turns it off). The version for
   current upstream llama.cpp in `../matvec/` is the newer one, with a layout rule measured on more shapes.
5. **Fused conv state store and short convolution** (`GGML_INK_NO_CATCONV=1` turns it off). Correct, but no measurable
   gain: in a HIP graph two small dependent kernels cost about as much as one on this card.
6. **gfx906 flash decoding** (`../flash-decoding/`, `GGML_CUDA_GCN_FA_DEC=0` turns it off).

## The model file

From the official weights: `convert_hf_to_gguf.py --outtype bf16`, then `llama-quantize` with ggml-org's importance
matrix and a per-tensor list from `tools/make_types.py` (experts and attention Q4_1, the rest as in ggml-org's IQ4_NL
file), then `tools/ctrl_f16.py` for the loop controllers, which llama-quantize leaves in F32.

| File | wikitext-2 perplexity (60 x 512) |
|---|---|
| ggml-org IQ4_NL | 6.916 |
| experts Q4_1 | 6.936 |
| experts and attention Q4_1 | 6.982 |

## Build and run

```sh
./setup-tree.sh llama.cpp        # pinned base tree (checked), then git am patches/*.patch
podman build -t inkling2-build build/
podman run --rm --security-opt label=disable -v $PWD:/work inkling2-build /work/build/build.sh
```

```sh
GGML_CUDA_RDNA_NO_BLAS=1 llama-server -m Inkling-2-Mini-Q41A.gguf --mmproj mmproj-F16.gguf --no-mmproj-offload \
  -ngl 99 -fa on -dev ROCm1,ROCm0 -sm layer -ts 18,30 -c 131072 -ctk q8_0 -ctv q8_0 -ub 256 -np 1 \
  --jinja --chat-template-kwargs '{"reasoning_effort": "medium"}' --temp 1.0 --top-p 1.0 --top-k 40 --min-p 0
```

## Tests

Run with both cards visible in one process (the setup where rocBLAS fails), `GGML_CUDA_RDNA_NO_BLAS=1`:

| Backend | Tests | Result |
|---|---|---|
| RX 9060 XT | `MUL_MAT`, F32/F16/BF16 | 914/914 |
| RX 9060 XT | Inkling ops (`INK.*`) | 67/67 |
| Radeon VII | `MUL_MAT`, Q4_0/Q4_1/IQ4_NL/Q8_0 | 431/431 |
| Radeon VII | `MUL_MAT_ID`, same types | 262/262 |
| Radeon VII | Inkling ops, including `INK_FA_REL_BIAS` up to 33k keys | 67/67 |
| Radeon VII | `FLASH_ATTN_EXT`, GQA 3, q8_0, 7000 to 16384 keys | 6/6 |

```sh
test-backend-ops test -b ROCm1 -o MUL_MAT,MUL_MAT_ID -p "type_a=(q4_0|q4_1|iq4_nl|q8_0)"
test-backend-ops test -b ROCm1 -o "INK.*"
test-backend-ops perf -b ROCm1 -o INK_FA_REL_BIAS
python3 tools/needle.py 8080 64      # 64k-token needle through the server
```
