import json, random, sys, time, urllib.request
# needle.py PORT KTOK [effort] : a KTOK-thousand-token filler document with one access code at 30 %, asks for it
# through llama-server's OpenAI API. Inkling 2 reads reasoning_effort (none/minimal/low/medium/high), not enable_thinking.
P, K = sys.argv[1], int(sys.argv[2]); EFF = sys.argv[3] if len(sys.argv) > 3 else "low"
random.seed(7); words = "alpine river cheese clock mountain lake train bank chocolate glacier valley bridge".split()
filler = []; k = 0
while k < K * 1000:
    filler.append(" ".join(random.choice(words) for _ in range(12)) + "."); k += 14
filler.insert(int(len(filler) * 0.3), "The access code for the Basel archive is 7-4-1-9-QUOKKA.")
q = "\n\nWhat is the access code for the Basel archive? Answer with the code only."
b = {"messages": [{"role": "user", "content": " ".join(filler) + q}], "max_tokens": 2000, "temperature": 1.0, "top_p": 1.0,
     "cache_prompt": False, "chat_template_kwargs": {"reasoning_effort": EFF}}
r = urllib.request.Request(f"http://127.0.0.1:{P}/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
t = time.time(); d = json.load(urllib.request.urlopen(r, timeout=7200))
c = d["choices"][0]["message"].get("content") or ""; tm = d["timings"]
print(json.dumps({"ok": "7-4-1-9-QUOKKA" in c, "answer": c[-80:], "prompt_n": tm["prompt_n"], "read_tps": round(tm["prompt_per_second"]),
                  "write_tps": round(tm["predicted_per_second"], 1), "gen": tm["predicted_n"], "wall": round(time.time() - t)}))
