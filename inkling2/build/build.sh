#!/bin/bash
# build.sh : run inside the image with the patched tree mounted at /work/llama.cpp (see README).
# gfx906 = Radeon VII, gfx1200 = RX 9060 XT. -j 8: the full core count crashed GCC/LLVM on the build machine.
set -e
cd /work/llama.cpp
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS="gfx906;gfx1200" \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON
cmake --build build --config Release -j 8 --target llama-server llama-bench llama-cli llama-perplexity test-backend-ops llama-quantize
