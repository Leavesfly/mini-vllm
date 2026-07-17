# OpenAI API

本文讲清楚 mini-vllm 的对外接口：用 JDK 内置 HttpServer 零依赖实现 OpenAI 兼容 API、SSE 流式输出机制、ChatML 提示词渲染、思考模式，以及内置 Web 对话页。

## 端点总览

由 [OpenAiHandler](../src/main/java/io/leavesfly/minivllm/api/OpenAiHandler.java) 处理，注册在 `/v1` 上下文：

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat/completions` | 对话补全（支持流式/非流式） |
| POST | `/v1/completions` | 文本补全（复用同一逻辑） |
| GET | `/v1/models` | 返回模型列表 |
| GET | `/`、`/index.html` | 内置 Web 对话页（[WebUiHandler](../src/main/java/io/leavesfly/minivllm/api/WebUiHandler.java)） |

服务由 [MiniVllmServer](../src/main/java/io/leavesfly/minivllm/MiniVllmServer.java) 用 `com.sun.net.httpserver.HttpServer` 启动，线程池大小为 `max(16, maxSeqs*4)`（流式请求会阻塞 handler 线程，需足够大）：

```java
HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
server.createContext("/v1", new OpenAiHandler(engine, "mini-vllm"));
server.createContext("/", new WebUiHandler());
server.setExecutor(Executors.newFixedThreadPool(Math.max(16, maxNumSeqs * 4)));
```

## 请求参数

`handleChatCompletions` 从 JSON body 提取参数：

```java
boolean enableThinking = Boolean.TRUE.equals(req.get("enable_thinking"));
String prompt = extractPrompt(req, enableThinking);
boolean stream = Boolean.TRUE.equals(req.get("stream"));
int maxTokens   = intOr(req.get("max_tokens"), 512);
float temperature = (float) doubleOr(req.get("temperature"), 0.8);
float topP      = (float) doubleOr(req.get("top_p"), 0.9);
int topK        = intOr(req.get("top_k"), 0);
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `messages` | — | 对话数组（有则走 ChatML）；也支持 `prompt` 纯文本字段 |
| `stream` | false | true 走 SSE 流式 |
| `max_tokens` | 512 | 最大生成 token 数 |
| `temperature` | 0.8 | 采样温度（≤1e-3 退化为 greedy） |
| `top_p` | 0.9 | nucleus 采样 |
| `top_k` | 0 | top-k 采样（0=不限制） |
| `enable_thinking` | false | Qwen3 思考模式开关 |

采样参数最终传给 [Sampler](../src/main/java/io/leavesfly/minivllm/math/Sampler.java)（temperature → top-k → top-p → 多项采样）。

## ChatML 提示词渲染

`extractPrompt` 把 OpenAI `messages` 渲染为 Qwen3 的 ChatML 格式（[ChatTemplate](../src/main/java/io/leavesfly/minivllm/tokenizer/ChatTemplate.java)）：

```java
private String extractPrompt(Map<String, Object> req, boolean enableThinking) {
    Object msgs = req.get("messages");
    if (msgs == null) return req.getOrDefault("prompt", "").toString(); // 纯文本模式
    List<ChatTemplate.Message> list = /* 遍历 messages 提取 role/content */;
    return ChatTemplate.applyChatML(list, enableThinking);
}
```

ChatML 格式：

```
<|im_start|>system\n{system}<|im_end|>\n
<|im_start|>user\n{user}<|im_end|>\n
<|im_start|>assistant\n            <- 生成起点
```

`applyChatML` 的两个关键行为（与 HF `apply_chat_template` 严格对齐）：

```java
public static String applyChatML(List<Message> messages, boolean enableThinking) {
    StringBuilder sb = new StringBuilder();
    for (Message m : messages)
        sb.append(IM_START).append(m.role).append('\n').append(m.content).append(IM_END).append('\n');
    sb.append(IM_START).append("assistant\n");
    if (!enableThinking)
        sb.append(THINK_OPEN).append("\n\n").append(THINK_CLOSE).append("\n\n"); // 预填空 think 块
    return sb.toString();
}
```

- **不自动补默认 system**：只在首条消息是 system 时才输出 system 块。
- **`<|im_start|>` / `<|im_end|>` 是 special token**：由 BpeTokenizer 直接映射 id，不参与 BPE 合并。

## 思考模式（enable_thinking）

Qwen3 支持"思考模式"，控制模板末尾：

| 模式 | 模板末尾 | 行为 |
|---|---|---|
| `enable_thinking=false`（默认） | 预填空 `<think>\n\n</think>\n\n` | 跳过内部推理，直接作答（更快更简短） |
| `enable_thinking=true` | 仅 `<|im_start|>assistant\n` | 模型自行生成 `<think>...</think>` 推理过程 |

> 这是一个曾经的踩坑点：默认思考模式会让模型先输出长推理再作答，短请求容易被 `max_tokens` 截断在推理中途、看起来"没回答"。默认关闭思考模式让响应直接、简短。

## 分词与增量解码

请求进入引擎后，[BpeTokenizer](../src/main/java/io/leavesfly/minivllm/tokenizer/BpeTokenizer.java) 负责编解码：

- **encode**：special token 切分 → pre-token 正则切分（Qwen2/GPT-4 风格）→ byte-level 字符 → BPE 合并 → 查词表得 id。pre-token 正则不跨越"单词/数字/标点/空白"段，这是与 HF 结果逐 id 一致的关键。
- **decode**：id → token 字符串 → byte-level 字符还原字节 → UTF-8 解码。

流式输出的一个关键细节：**中文/emoji 的多字节 UTF-8 可能被切在两个 token 里**，逐 token 解码会产生乱码 `�`。`IncrementalDecoder` 解决这个问题——缓冲不完整字节，只输出可完整解码的片段：

```java
public synchronized String accept(int tokenId) {
    // 把 token 的 byte-level 字符还原为字节，追加到 pending 缓冲
    for (char c : token) if ((b = charToByte(c)) >= 0) pending.write(b);
    // 用 CharsetDecoder 尝试解码，返回可完整解码部分，剩余不完整字节留在 pending
    CoderResult r = decoder.decode(in, out, false);
    pending 保留未消费字节;
    return out.toString();
}
```

引擎在 `addRequest` 时为 BPE 分词的 Sequence 注入 `incDecoder`，生成结束时 `sweepFinished` 调用 `flush()` 冲刷残留字节（见 [Continuous-Batching](Continuous-Batching.md)）。

## SSE 流式输出

流式由 [SseWriter](../src/main/java/io/leavesfly/minivllm/api/SseWriter.java) 封装 Server-Sent Events。

### SSE 协议要点

```java
public SseWriter(HttpExchange exchange) {
    exchange.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
    exchange.getResponseHeaders().set("Cache-Control", "no-cache");
    exchange.sendResponseHeaders(200, 0); // 0 = chunked 传输，边生成边写
    this.os = exchange.getResponseBody();
}
public synchronized void write(String data) {
    os.write(("data: " + data + "\n\n").getBytes(UTF_8)); // 每条 "data: ...\n\n"
    os.flush();
}
```

- `sendResponseHeaders(200, 0)` 启用 chunked 传输，无需预知总长度。
- 每条消息 `data: <内容>\n\n`；结束发 `data: [DONE]`。
- `write` 是 `synchronized` 的：onToken 回调在**引擎线程**触发，与 **handler 线程**并发写需同步。

### 流式处理流程

```java
private void handleStream(...) {
    SseWriter sse = new SseWriter(exchange);
    Sequence seq = engine.addRequest(prompt, ..., token -> {
        try { sse.write(streamChunk(id, created, token, null)); } // 每 token 一个 chunk
        catch (IOException ignored) {} // 客户端可能已断开
    });
    while (!seq.isFinished()) Thread.sleep(1); // handler 线程阻塞轮询
    sse.write(streamChunk(id, created, "", "stop")); // 结束 chunk
    sse.finish(); // 发 [DONE] 并关闭
}
```

体现了**请求与引擎解耦**：handler 线程只负责入队 + 阻塞等 + 写 SSE，实际推进由后台引擎线程完成。

### 流式响应格式

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"好"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## 非流式响应

`handleNonStream` 收集所有 token 后一次性返回标准 OpenAI JSON，含 usage 统计：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [{"index":0,"message":{"role":"assistant","content":"..."},"finish_reason":"stop"}],
  "usage": {"prompt_tokens": 12, "completion_tokens": 20, "total_tokens": 32}
}
```

`prompt_tokens` = `seq.promptTokens.length`，`completion_tokens` = `seq.outputTokens.size()`。

## JSON 编解码

所有 JSON 由零依赖手写的 [SimpleJson](../src/main/java/io/leavesfly/minivllm/json/SimpleJson.java) 处理（`parseObject` 解析请求、`stringify` 构造响应）。响应对象用 `LinkedHashMap` 保证字段顺序稳定。请求体读取限制 1MB。

## Web 对话页

[WebUiHandler](../src/main/java/io/leavesfly/minivllm/api/WebUiHandler.java) 提供内置单页应用：

- 页面 `src/main/resources/web/index.html` 打包在 classpath，随 jar 分发，无外部依赖。
- 启动时一次性读入内存（`getResourceAsStream("/web/index.html").readAllBytes()`），避免每次请求 IO。
- 仅响应 `GET /` 与 `/index.html`，其余 404。
- 页面本身通过 fetch 调用 `/v1/chat/completions`（SSE 流式），与 OpenaiHandler 解耦。

访问 `http://localhost:8080/` 即可对话。

## curl 速查

```bash
# 模型列表
curl -s http://localhost:8080/v1/models

# 非流式
curl -s -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"mini-vllm","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'

# 流式
curl -s -N -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"mini-vllm","messages":[{"role":"user","content":"Hi"}],"stream":true}'

# 思考模式
curl -s -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"mini-vllm","messages":[{"role":"user","content":"9.11 和 9.9 哪个大"}],"enable_thinking":true}'
```

用 OpenAI Python SDK：`base_url="http://localhost:8080/v1"`, `api_key="not-needed"`（见 [Getting-Started](Getting-Started.md)）。

## 延伸

- 请求进入引擎后如何调度：[Continuous-Batching](Continuous-Batching.md)
- 分词器与 BPE 细节：[Weight-Loading](Weight-Loading.md)
- 采样参数如何生效：[Transformer](Transformer.md)
