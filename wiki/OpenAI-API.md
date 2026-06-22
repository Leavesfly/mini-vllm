# OpenAI API

mini-vllm 提供与 OpenAI 兼容的 HTTP API，用 JDK 内置 HttpServer 实现，零依赖。聚焦 [`api`](../src/main/java/io/leavesfly/minivllm/api) 包。

## 1. 端点一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/v1/models` | 返回模型列表 |
| POST | `/v1/chat/completions` | 聊天补全，支持 stream |
| POST | `/v1/completions` | 文本补全（复用 chat 逻辑） |

## 2. OpenAiHandler 处理流程

[`OpenAiHandler`](../src/main/java/io/leavesfly/minivllm/api/OpenAiHandler.java) 实现 `HttpHandler`，按 path + method 分派：

```java
public void handle(HttpExchange exchange) {
    String path = exchange.getRequestURI().getPath();
    String method = exchange.getRequestMethod();
    if (GET && "/v1/models".equals(path))       → handleModels
    if (POST && "/v1/chat/completions".equals(path)) → handleChatCompletions
    if (POST && "/v1/completions".equals(path))  → handleChatCompletions（复用）
    else → 404
}
```

## 3. 请求解析

`handleChatCompletions` 用手写的 [SimpleJson](../src/main/java/io/leavesfly/minivllm/json/SimpleJson.java) 解析请求体：

```java
Map<String, Object> req = SimpleJson.parseObject(body);

// 从 messages 数组拼接 prompt
String prompt = extractPrompt(req);   // "role: content\n" 逐条拼接
boolean stream = Boolean.TRUE.equals(req.get("stream"));
int maxTokens = intOr(req.get("max_tokens"), 128);
float temperature = (float) doubleOr(req.get("temperature"), 0.8);
float topP = (float) doubleOr(req.get("top_p"), 0.9);
int topK = intOr(req.get("top_k"), 0);
```

支持的参数（与 OpenAI 对齐）：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `messages` | array | 必填 | `[{role, content}]`，拼接为 prompt |
| `stream` | bool | false | true 走 SSE 流式 |
| `max_tokens` | int | 128 | 最大生成 token 数 |
| `temperature` | float | 0.8 | 采样温度 |
| `top_p` | float | 0.9 | nucleus 采样阈值 |
| `top_k` | int | 0 | top-k 采样（0 不限） |

## 4. 非流式响应

`handleNonStream`：加入请求，阻塞等 Sequence 完成，收集所有 token，返回完整 JSON。

```java
List<String> collected = Collections.synchronizedList(new ArrayList<>());
Sequence seq = engine.addRequest(prompt, maxTokens, temp, topK, topP, collected::add);
while (!seq.isFinished()) Thread.sleep(1);   // 等 engine 线程推进

String text = String.join("", collected);
String json = completionJson(id, created, text, promptTokens, completionTokens);
sendJson(exchange, 200, json);
```

返回格式（标准 OpenAI `chat.completion`）：

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

## 5. 流式响应（SSE）

`handleStream`：边生成边推送，每个 token 经 `onToken` 回调写一个 SSE chunk。

```java
SseWriter sse = new SseWriter(exchange);
Sequence seq = engine.addRequest(prompt, maxTokens, temp, topK, topP, token -> {
    sse.write(streamChunk(id, created, token, null));   // engine 线程回调
});
while (!seq.isFinished()) Thread.sleep(1);
sse.write(streamChunk(id, created, "", "stop"));   // 结束 chunk
sse.finish();   // 写 [DONE] 并关闭
```

SSE 输出格式：

```
data: {"id":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}

data: {"id":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"y"},"finish_reason":null}]}

...

data: {"id":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

## 6. SseWriter：流式输出封装

[`SseWriter`](../src/main/java/io/leavesfly/minivllm/api/SseWriter.java) 封装 HTTP 响应为 Server-Sent Events：

```java
public SseWriter(HttpExchange exchange) throws IOException {
    exchange.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
    exchange.sendResponseHeaders(200, 0);   // 0 = chunked 传输
    this.os = exchange.getResponseBody();
}

public synchronized void write(String data) throws IOException {
    os.write(("data: " + data + "\n\n").getBytes(UTF_8));   // SSE 协议格式
    os.flush();
}
```

关键设计：
- **chunked 传输**：`sendResponseHeaders(200, 0)` 启用，无需预知总长度，边生成边写。
- **synchronized**：`onToken` 回调在 engine 线程触发，与 handler 线程并发写需同步。
- **SSE 协议**：每条消息 `data: ` 前缀 + 内容 + 两个换行。

## 7. 线程协作

```
HTTP 线程                          Engine 线程
   │                                  │
   ├─ addRequest(prompt, onToken) ──→ │ 加入 waiting
   │                                  ├─ admitNew: prefill
   │                                  ├─ decodeStep: 生成 token
   │   while(!isFinished()) sleep     │     └─ onToken 回调 ──→ sse.write()  ← 跨线程
   │                                  ├─ sweepFinished
   │   (完成) 写 [DONE] / 返回 JSON    │
```

- HTTP 线程负责接收请求、等待完成、返回响应。
- Engine 线程负责推进 step、生成 token、触发回调。
- `onToken` 是跨线程回调，`SseWriter.write` 用 `synchronized` 保证线程安全。

## 8. /v1/models

```java
private void handleModels(HttpExchange exchange) {
    Map model = {"id": "mini-vllm", "object": "model", "owned_by": "mini-vllm"};
    resp = {"object": "list", "data": [model]};
    sendJson(exchange, 200, SimpleJson.stringify(resp));
}
```

## 9. 兼容性

由于接口完全遵循 OpenAI 规范，可直接用 OpenAI 官方 SDK 连接，只需改 `base_url`：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8088/v1", api_key="not-needed")
```

这意味着任何已对接 OpenAI 的应用，改个地址就能指向自己的 mini-vllm——这是 vLLM 也是本项目易用的关键。

---

**相关**：[Architecture](Architecture.md) · [Getting Started](Getting-Started.md)
