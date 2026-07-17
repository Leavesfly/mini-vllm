#!/usr/bin/env python3
"""导出 Qwen3-0.6B 的 HF 参考输出，供 Qwen3AlignmentTest 数值对齐。

用法:
    tools/.venv/bin/python tools/dump_reference.py [模型目录]

每个测试 prompt 输出一个 JSON 到 src/test/resources/qwen3/：
  prompt        ChatML 渲染后的完整提示词（apply_chat_template 结果）
  ids           prompt 的 token id
  logits_last   最后位置的完整词表 logits（F32 前向）
  greedy_ids    greedy 生成的 32 个 token id
  greedy_text   greedy 生成的文本
  hidden_L0/L1/L27  第 0/1/27 层 block 输出的最后位置 hidden states
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "你好，请用一句话介绍你自己",
    "What is the capital of France?",
]
# 检查点：layer 0/1 的 block 输出（pre-norm），以及 final norm 之后的 hidden（hs[-1]）
TRACE_LAYERS = [0, 1]
GREEDY_TOKENS = 32


def main():
    if len(sys.argv) > 1:
        model_dir = Path(sys.argv[1])
    else:
        cache = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
        model_dir = next(cache.iterdir())

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float32, attn_implementation="eager")
    model.eval()

    out_dir = Path(__file__).resolve().parent.parent / "src/test/resources/qwen3"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, user in enumerate(PROMPTS):
        messages = [{"role": "user", "content": user}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(prompt)
        input_ids = torch.tensor([ids])

        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
            logits_last = out.logits[0, -1].float().tolist()
            # transformers 5.x: hidden_states[0]=embedding, [i]=第 i-1 层输出,
            # 最后一个元素 hs[-1] 是 final norm 之后的输出（layer27 pre-norm 不暴露）
            hidden = {f"hidden_L{l}": out.hidden_states[l + 1][0, -1].float().tolist()
                      for l in TRACE_LAYERS}
            hidden["hidden_final"] = out.hidden_states[-1][0, -1].float().tolist()
            gen = model.generate(input_ids, max_new_tokens=GREEDY_TOKENS,
                                 do_sample=False, num_beams=1)
            greedy_ids = gen[0, len(ids):].tolist()

        ref = {
            "user": user,
            "prompt": prompt,
            "ids": ids,
            "logits_last": logits_last,
            "greedy_ids": greedy_ids,
            "greedy_text": tok.decode(greedy_ids),
            **hidden,
        }
        path = out_dir / f"reference_{idx}.json"
        path.write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
        print(f"[{idx}] ids={len(ids)} greedy={greedy_ids[:8]}... -> {path}")


if __name__ == "__main__":
    main()
