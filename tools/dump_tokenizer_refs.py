#!/usr/bin/env python3
"""导出 Qwen3-0.6B tokenizer 参考数据，供 BpeTokenizerTest 逐 id 对齐。

用法:
    python3 tools/dump_tokenizer_refs.py [模型目录]

输出: src/test/resources/qwen3/tokenizer_cases.txt
每行一个 JSON: {"text": <原始文本>, "ids": [...]}
"""
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

CASES = [
    "Hello, world!",
    "你好，世界！",
    "The quick brown fox jumps over the lazy dog.",
    "  spaces  and\ttabs\nnewlines",
    "emoji 😀🎉 test",
    "def main():\n    print('hi')",
    "don't can't I'll they're",
    "1234567890",
    "Hello 你好 mixed 混合 test",
    "日本語のテキスト",
    "a",
    " ",
    "Machine learning is a subset of artificial intelligence.",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    "多轮\n换行\r\n混合\r回车",
    "1+1=2, 3.14, -42, 1e10",
]


def main():
    if len(sys.argv) > 1:
        model_dir = Path(sys.argv[1])
    else:
        # 默认取 HF 缓存中的快照目录
        cache = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
        model_dir = next(cache.iterdir())
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    out = Path(__file__).resolve().parent.parent / "src/test/resources/qwen3/tokenizer_cases.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for text in CASES:
            ids = tok.encode(text).ids
            f.write(json.dumps({"text": text, "ids": ids}, ensure_ascii=False) + "\n")
    print(f"已写出 {len(CASES)} 条用例 -> {out}")


if __name__ == "__main__":
    main()
