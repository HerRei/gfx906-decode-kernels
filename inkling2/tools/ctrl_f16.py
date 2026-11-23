# ctrl_f16.py IN OUT : copy a GGUF, converting the F32 loop-controller matrices (loop_*_ctrl_{down,gain,forget}) to F16.
# llama-quantize leaves these F32 even with a per-tensor type list. Needs numpy and the gguf-py of the patched tree:
#   GGUF_PY=llama.cpp/gguf-py python3 ctrl_f16.py in.gguf out.gguf
import os, sys, re, numpy as np
sys.path.insert(0, os.environ.get("GGUF_PY", "llama.cpp/gguf-py"))
import gguf
from gguf.scripts.gguf_new_metadata import MetadataDetails
r = gguf.GGUFReader(sys.argv[1])
arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
w = gguf.GGUFWriter(sys.argv[2], arch=arch, endianess=r.endianess)
align = r.fields.get("general.alignment")
if align: w.data_alignment = align.contents()
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."): continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=st)
pat = re.compile(r"blk\.\d+\.loop_.*_ctrl_(down|gain|forget)\.weight$")
datas = []; n = 0
for t in r.tensors:
    d = t.data; ty = t.tensor_type
    if pat.match(t.name) and ty == gguf.GGMLQuantizationType.F32:
        d = np.asarray(d, dtype=np.float32).astype(np.float16); ty = gguf.GGMLQuantizationType.F16; n += 1
    w.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, ty)
    datas.append(d)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for d in datas: w.write_tensor_data(d, tensor_endianess=r.endianess)
w.close(); print("converted", n, "tensors to f16")
