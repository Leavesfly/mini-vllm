# Getting Started

5 分钟编译、启动并用 curl 测试 mini-vllm 引擎。

## 前置要求

- JDK 17+
- Maven 3.6+
- curl（测试用）

验证环境：

```bash
java -version   # 需 17 及以上
mvn -version    # 任意 3.6+
```

## 1. 编译

```bash
cd /Users/yefei.yf/Qoder/vllm/mini-vllm
mvn compile
```

预期输出：

```
[INFO] Compiling 29 source files with javac [debug release 17] to target/classes
[INFO] BUILD SUCCESS
```

> 零外部依赖，无需下载任何第三方库，编译通常在 1 秒内完成。

## 2. 启动引擎（随机初始化模式）

无需任何权重文件，用随机初始化模型即可跑通全流程（输出为乱码，但 PagedAttention + batching + API 全链路真实运转）：

```bash
java -cp target/classes io.leavesfly.minivllm.MiniVllmServer --random --port 8088
```

启动成功后会打印：

```
使用随机初始化模型（输出无意义，仅用于验证 PagedAttention + batching 流程）
==================================================
  mini-vllm 学习型引擎已启动
  地址: http://localhost:8088
  端点: POST /v1/chat/completions
        GET  /v1/models
  配置: ModelConfig{vocab=256, d=64, h=4, L=2, ffn=256, blk=16}
  并发: maxSeqs=8, blocks=1024
==================================================
```

## 3. 启动引擎（加载真实权重）

如果你已有一个符合命名约定的 safetensors 权重文件（见 [Weight Loading](Weight-Loading.md)）：

```bash
java -cp target/classes io.leavesfly.minivllm.MiniVllmServer --weights ./model.safetensors --port 8088
```

## 4. 测试 API

开另一个终端，用 curl 测试三个端点。

### 查看模型列表

```bash
curl -s http://localhost:8088/v1/models
```

返回：

```json
{"object":"list","data":[{"id":"mini-vllm","object":"model","owned_by":"mini-vllm"}]}
```

### 非流式补全

```bash
curl -s -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mini-vllm","messages":[{"role":"user","content":"Hello"}],"max_tokens":16,"stream":false}'
```

返回标准 OpenAI 格式响应（随机模型下 content 为乱码）：

```json
{
  "id": "chatcmpl-1781751536016",
  "object": "chat.completion",
  "created": 1781751536,
  "model": "mini-vllm",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 12, "completion_tokens": 16, "total_tokens": 28}
}
```

### 流式补全（SSE）

```bash
curl -s -N -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mini-vllm","messages":[{"role":"user","content":"Hi"}],"max_tokens":8,"stream":true}'
```

逐 token 推送 SSE chunk，最后以 `[DONE]` 结束：

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"y"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## 5. 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--port` | 8080 | HTTP 服务端口 |
| `--weights` | 无 | safetensors 权重文件路径 |
| `--random` | false | 使用随机初始化模型 |
| `--max-seqs` | 8 | 最大并发请求数（continuous batching 上限） |
| `--num-blocks` | 1024 | KV cache block 池大小 |
| `--quiet` | false | 关闭引擎每步调度日志 |

示例：

```bash
java -cp target/classes io.leavesfly.minivllm.MiniVllmServer --random --port 9000 --max-seqs 16 --num-blocks 2048
```

## 6. 用 OpenAI SDK 连接

由于接口完全兼容 OpenAI，可直接用官方 SDK，只需把 `base_url` 指向本地：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8088/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="mini-vllm",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=32,
)
print(resp.choices[0].message.content)
```

## 7. 停止服务

`Ctrl+C` 终止前台进程即可。

---

**下一步**：了解整体架构 → [Architecture](Architecture.md)
