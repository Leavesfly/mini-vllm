"""排查工具：同一 prompt 下 HF greedy 与 mini-vllm(top_k=1) 输出是否逐字符一致。

用途：分清“输出异常”到底是引擎/模板问题还是模型自身行为。一致则引擎忠实。
注意：MiniMind3 的 generation_config.json 没有 eos_token_id，HF 不会在 <|im_end|> 停，
因此 mini-vllm 的输出应是 HF 输出的前缀（而非完全相等）。
需要先启动服务：java -cp target/classes io.leavesfly.minivllm.MiniVllmServer \
        --model-dir models/minimind-3-agent-512 --port 8123
"""
import json
import urllib.request

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

D = "models/minimind-3-agent-512"
MSGS = [{"role": "user", "content": "你好，请介绍一下你自己"}]
N = 120

tok = AutoTokenizer.from_pretrained(D)
try:
    model = AutoModelForCausalLM.from_pretrained(D, dtype=torch.float32)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(D, torch_dtype=torch.float32)
model.eval()

for label, extra in [("nothink", {}), ("think", {"open_thinking": True})]:
    prompt = tok.apply_chat_template(MSGS, tokenize=False, add_generation_prompt=True, **extra)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=N, do_sample=False)
    hf_text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    body = {"model": "mini-vllm", "messages": MSGS, "top_k": 1, "max_tokens": N}
    if extra:
        body["enable_thinking"] = True
    req = urllib.request.Request(
        "http://localhost:8123/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    java_text = json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"]

    print("=== " + label + " ===")
    print("PROMPT:", repr(prompt))
    print("HF    :", repr(hf_text))
    print("JAVA  :", repr(java_text))
    print("PREFIX:", hf_text.startswith(java_text), "(mini-vllm 在 EOS 停，HF 继续写)")
