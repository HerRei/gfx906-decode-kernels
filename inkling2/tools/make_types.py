#!/usr/bin/env python3
"""make_types.py REF.gguf SRC.gguf OUT.txt : the per-tensor type list (llama-quantize --tensor-type-file) for the
Q41A file: every tensor keeps the type it has in REF (ggml-org's Inkling-2-Mini-IQ4_NL.gguf), except
  - the experts (ffn_gate_up_exps, ffn_down_exps) and the attention projections (attn_*), which become Q4_1
    (no table lookup, which is what limits IQ4_NL on gfx906; 20-byte blocks are 4-byte aligned),
  - the loop controllers, listed as f16 (llama-quantize skips them anyway; ctrl_f16.py converts them).
SRC is the BF16 GGUF that gets quantized; its tensor names are the ones written.
Needs numpy and gguf-py:  GGUF_PY=llama.cpp/gguf-py python3 make_types.py ..."""
import collections, os, re, sys
sys.path.insert(0, os.environ.get("GGUF_PY", "llama.cpp/gguf-py"))
from gguf import GGUFReader

ref_path, src_path, out_path = sys.argv[1:4]
ref = {t.name: t.tensor_type.name.lower() for t in GGUFReader(ref_path).tensors}
lines, changed = [], collections.Counter()
for t in GGUFReader(src_path).tensors:
    n = t.name
    ty = ref.get(n, t.tensor_type.name.lower())
    if re.match(r"blk\.\d+\.ffn_(gate_up|down)_exps\.weight$", n) or (re.match(r"blk\.\d+\.attn_", n) and ty == "q8_0"):
        ty = "q4_1"
    elif re.match(r"blk\.\d+\.loop_.*_ctrl_(down|gain|forget)\.weight$", n):
        ty = "f16"
    changed[(ref.get(n), ty)] += 1
    lines.append("^" + re.escape(n) + "$=" + ty)
open(out_path, "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} tensors -> {out_path}; (reference type, new type): {dict(changed)}")
