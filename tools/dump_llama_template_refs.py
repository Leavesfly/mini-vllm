"""导出 Llama-3.2 chat_template 的 jinja2 渲染基准，供 JinjaChatTemplateTest 逐字符对齐。
用法：python tools/dump_llama_template_refs.py（在仓库根目录执行，需 jinja2）
"""
import json
from datetime import datetime
from jinja2 import Environment

env = Environment(trim_blocks=False, lstrip_blocks=False)
env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
cfg = json.load(open("models/Llama-3.2-1B-Instruct/tokenizer_config.json"))
tpl = env.from_string(cfg["chat_template"])

cases = {
    "llama/single": [{"role": "user", "content": "Hello"}],
    "llama/multi": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "1+1=?"},
    ],
}
out = {}
for name, msgs in cases.items():
    expected = tpl.render(messages=msgs, add_generation_prompt=True,
                          bos_token=cfg["bos_token"],
                          strftime_now=lambda f: datetime.now().strftime(f))
    out[name] = {"messages": msgs, "expected": expected}

with open("src/test/resources/llama3/chat_template_cases.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
    f.write("\n")
print("已写入 src/test/resources/llama3/chat_template_cases.json")
