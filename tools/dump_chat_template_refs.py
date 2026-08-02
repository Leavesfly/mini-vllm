"""导出 HF chat_template 的渲染基准，供 ChatMLTemplateTest 逐字符对齐。

Qwen3 与 MiniMind3 同为 qwen3 架构、同用 ChatML 骨架，但思考模式的约定相反：
  Qwen3 官方：思考模式不预填，由模型自己生成 <think>；历史消息丢弃思考内容
  MiniMind3：思考模式预填 <think>\n；历史消息重建 think 块
用法：python tools/dump_chat_template_refs.py（在仓库根目录执行，需 jinja2）
"""
import json
from jinja2 import Environment

env = Environment(trim_blocks=False, lstrip_blocks=False)
env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}

tpl_mini = env.from_string(open("models/minimind-3-agent-512/chat_template.jinja").read())
tpl_qwen = env.from_string(json.load(open("models/Qwen3-0.6B/tokenizer_config.json"))["chat_template"])

single = [{"role": "user", "content": "你好"}]
multi = [
    {"role": "user", "content": "问1"},
    {"role": "assistant", "content": "思考A\n</think>\n\n答案A"},
    {"role": "user", "content": "问2"},
]
plain_hist = [
    {"role": "user", "content": "问1"},
    {"role": "assistant", "content": "纯答案"},
    {"role": "user", "content": "问2"},
]

cases = [
    ("mini/single/think", tpl_mini, single, {"open_thinking": True}),
    ("mini/single/nothink", tpl_mini, single, {}),
    ("mini/multi/think", tpl_mini, multi, {"open_thinking": True}),
    ("mini/plainhist/think", tpl_mini, plain_hist, {"open_thinking": True}),
    ("qwen/single/think", tpl_qwen, single, {}),
    ("qwen/single/nothink", tpl_qwen, single, {"enable_thinking": False}),
    ("qwen/multi/think", tpl_qwen, multi, {}),
]

OUT = "src/test/resources/qwen3/chat_template_cases.json"
out = {}
for name, tpl, msgs, extra in cases:
    out[name] = tpl.render(messages=msgs, add_generation_prompt=True, **extra)
    print("=== " + name + " ===")
    print(repr(out[name]))
with open(OUT, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
    f.write("\n")
print("已写入 " + OUT)
