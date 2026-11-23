# gfx906-decode-kernels

Two llama.cpp HIP kernels that make single-token decode faster on AMD gfx906 cards (Radeon VII, Instinct MI50,
Instinct MI60), and the setup that runs Inkling 2 Mini on a Radeon VII next to an RX 9060 XT.

| Folder | What |
|---|---|
| [`matvec/`](matvec/) | A matrix-vector product for Q4_0, Q4_1 and IQ4_NL weights, as one patch for current upstream llama.cpp (162 added lines in `mmvq.cu`). |
| [`flash-decoding/`](flash-decoding/) | Attention for one token over a long q8_0 cache, head size 128, 3 query heads per key/value head. |
| [`inkling2/`](inkling2/) | Inkling 2 Mini at about 48 tok/s on a Radeon VII + RX 9060 XT, with a fix for Fedora's rocBLAS failing on mixed GPUs. |
| [`writeup/`](writeup/gfx906-decode-kernels.pdf) | A six-page write-up of both kernels: design, measurements, what did not work. |
| [`bench/`](bench/) | The standalone microbenchmark the matvec kernel came from. |

## How this was made

The code was written with an AI coding agent (Claude Code). The point of publishing it is the checking around it:
every kernel is compared against llama.cpp's CPU reference with `test-backend-ops`, every number below was measured
on the card, the cases where an idea did not help are listed, and the first version of the matvec, which made one model
5 % slower, is described along with why. Similar ideas already exist in community forks; see [Existing work](#existing-work).
If something here is wrong, please open an issue.

## Results

Radeon VII at stock clocks (1801 MHz core, 1000 MHz memory, 250 W power limit), Fedora 44, Fedora's ROCm 7.1,
llama.cpp `94f3af3d0`.

**Matvec, one token, µs per product, upstream / patched** (`test-backend-ops perf`):

| Shape (rows x K) | Example | Q4_0 | Q4_1 | IQ4_NL |
|---|---|---|---|---|
| 4096 x 4096 | Llama 3 8B, q/o | 23.6 / 21.9 | 21.8 / 20.5 | 24.4 / 22.9 |
| 4096 x 14336 | Llama 3 8B, down | 71.6 / 64.2 | 65.4 / 55.2 | 73.6 / 67.0 |
| 14336 x 4096 | Llama 3 8B, up | 68.2 / 56.1 | 61.6 / 53.1 | 70.5 / 57.8 |
| 5120 x 17408 | Qwen 3.8 27B, down | 106.4 / 88.4 | 97.4 / 76.3 | 109.7 / 90.9 |
| 8192 x 8192 | 70B, q/o | 78.2 / 66.2 | 71.2 / 64.0 | 80.5 / 68.3 |
| 28672 x 8192 | 70B, up | 259.0 / 211.3 | 234.2 / 200.3 | 266.3 / 216.8 |
| 151936 x 2048 | output head | 378.7 / 272.9 | 377.4 / 247.9 | 410.6 / 285.3 |

Over 49 shapes with at least 512 rows the median change is -14 % (Q4_0), -18 % (Q4_1) and -15 % (IQ4_NL); the worst
shape is 0 to -5 %. Below 512 rows the patch leaves the work to the generic kernel. Q8_0 was not faster and is not
covered.

**Whole models, Radeon VII alone** (`llama-bench -p 0 -n 128 -fa 1`, tok/s, mean of the runs after the first):

| Model | upstream | patched | change |
|---|---|---|---|
| Gemma 4 26B-A4B, QAT Q4_0 | 91.9 | 101.5 | +10 % |
| Qwen 3.8 27B, Q4_0 | 30.1 | 31.1 | +3 % |
| Qwen 3.8 27B, IQ4_NL | 28.5 | 29.3 | +3 % |
| Qwen 3.8 27B, UD-IQ4_XS (not covered, control) | 25.4 | 25.6 | 0 |

Prompt processing does not change (pp512 350.7 vs 349.2 tok/s). Perplexity with one-token batches, so that every
token goes through the kernel: 6.031 upstream, 6.034 patched (Qwen 3.8 27B Q4_0, wikitext-2, 4 x 512).

**Flash decoding**, 24 query heads on 8 key/value heads, q8_0 cache, µs:

| Keys | vector kernel | this kernel |
|---|---|---|
| 8k | 110.2 | 92.4 |
| 16k | 210.4 | 162.0 |
| 32k | 407.5 | 225.7 |
| 64k | 786.7 | 453.1 |

In Inkling 2 Mini, writing after a 68.5k-token document went from 33.5 to 37.0 tok/s and after 34k tokens from 38.5 to
40.4. Below 6144 keys the stock kernel is as fast, so it keeps running there.

## Use the matvec patch

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 94f3af3d0                     # tested here; newer commits may need a small rebase
git apply ../gfx906-decode-kernels/matvec/gcn-mmv.diff
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B build \
    -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
```

It is on by default for gfx906 (and other GCN cards, untested). `GGML_CUDA_GCN_MMV=0` switches it off, which makes a
before/after comparison one environment variable:

```sh
GGML_CUDA_GCN_MMV=0 build/bin/llama-bench -m model.gguf -p 0 -n 128
GGML_CUDA_GCN_MMV=1 build/bin/llama-bench -m model.gguf -p 0 -n 128
```

The flash-decoding kernel needs the Inkling 2 tree, because upstream llama.cpp does not support Inkling 2 yet; see
[`inkling2/README.md`](inkling2/README.md).

## What was checked

| Tree | Card | Tests | Result |
|---|---|---|---|
| upstream + matvec patch | Radeon VII | `MUL_MAT`, `MUL_MAT_ID` for Q4_0/Q4_1/IQ4_NL/Q8_0 | 683/683 |
| upstream + matvec patch | Radeon VII | `MUL_MAT_FUSION` (fused gate/up) for the same types | 341/341 |
| Inkling 2 tree | Radeon VII | `MUL_MAT`, `MUL_MAT_ID`, Inkling ops incl. attention up to 33k keys, plain GQA-3 attention | 431 + 262 + 67 + 6 |
| Inkling 2 tree | RX 9060 XT | float `MUL_MAT` without rocBLAS, Inkling ops | 914 + 67 |

Not checked: an MI50 or MI60 (same instruction set; the MI60 has 64 instead of 60 compute units), other ROCm versions,
other operating systems.

## Things that may save you time on gfx906

- **A floor at about 2048 one-wavefront workgroups.** Every 64-thread matvec layout tried took about 57 µs when its
  grid had about 2048 workgroups, whatever the matrix size (2048 rows at 1 row per workgroup, 4096 at 2, 8192 at 4).
  128-thread workgroups do not show it. If a small kernel on a Radeon VII is inexplicably slow, check the grid size.
- **IQ4_NL is compute bound on this card.** The table lookup costs more than the memory traffic: a higher memory
  clock changed nothing, a higher core clock did. Q4_1 is the fastest 4-bit format here.
- **"no kernel image is available for execution on the device" with two different AMD GPUs** on Fedora's rocBLAS 7.1:
  it loads the kernels of only one GPU per process. `inkling2/patches/0001` lets the RDNA card skip rocBLAS
  (`GGML_CUDA_RDNA_NO_BLAS=1`).
- **ROCm in a container** reports no device at all when only one card's render node is passed in. Pass all render
  nodes and pick the card with `ROCR_VISIBLE_DEVICES`.
- **Register spills** made the first flash-decoding kernel slower than the stock one. `__launch_bounds__(256, 2)`
  capped it at 128 registers; `hipcc -Rpass-analysis=kernel-resource-usage` shows the scratch use.

## Existing work

- [iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906) computes whole Q4_0, Q4_1 and Q8_0
  blocks per lane at one column too (2 rows x 32 lanes), for matrices with K up to 1024.
- [exabit-io/mx-llama.cpp](https://github.com/exabit-io/mx-llama.cpp) has a gfx906 Q8_0 path with a whole block per
  thread at one column.
- [Martin1701/MI50-llama.cpp](https://github.com/Martin1701/MI50-llama.cpp) has a gfx906 attention kernel that shares
  q8_0 key/value loads across 6 query heads (head size 256) and uses upstream's split and combine.
- [0c33/mi50-vllm-lab](https://github.com/0c33/mi50-vllm-lab) has a vLLM decode kernel with 256-key tiles, GQA sharing
  and an online softmax.
- Upstream llama.cpp merged GCN tuning for prompt processing (#27841, #30021, #30022) and for the tile attention
  kernel (#15769, #15884, #15927).

What is new here, as far as the search went: the matvec for Q4_0, Q4_1 and IQ4_NL at every K with an expert path and
the fused gate, the 2048-workgroup floor, attention for head size 128 with 3 query heads per key/value head, and a
small patch against current upstream instead of a fork.

## Limits

One token at a time only; prompt processing uses the stock kernels. The matvec does not cover Q8_0 or the K-quants
that most downloaded files use. The attention kernel covers head size 128, 3 query heads per key/value head and a
q8_0 cache. Nothing here is upstreamed: llama.cpp asks contributors to disclose AI use and to explain and maintain every
line themselves.

## License

MIT, like llama.cpp, whose files the patches change.
