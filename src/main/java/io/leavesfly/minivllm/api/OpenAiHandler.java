package io.leavesfly.minivllm.api;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import io.leavesfly.minivllm.core.LLMEngine;
import io.leavesfly.minivllm.core.Sequence;
import io.leavesfly.minivllm.json.SimpleJson;
import io.leavesfly.minivllm.tokenizer.ChatTemplate;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * OpenAiHandler —— OpenAI 兼容 API 的 HTTP 处理器。
 *
 * 学习要点：
 * 1. 用 JDK 自带 com.sun.net.httpserver.HttpServer，零依赖实现 HTTP 服务。
 * 2. 兼容 OpenAI 两个核心端点：
 *      POST /v1/chat/completions —— 支持 stream(true/false)、max_tokens、temperature、top_p、top_k
 *      GET  /v1/models           —— 返回模型列表
 * 3. 流式：每个 token 经 onToken 回调通过 SseWriter 推送一个 chunk，最后发 [DONE]。
 *    非流式：收集所有 token 后一次性返回完整 JSON。
 * 4. handler 线程阻塞等待 Sequence 完成（engine 线程异步推进 step），体现了请求与引擎的解耦。
 */
public final class OpenAiHandler implements HttpHandler {

    private static final int MAX_BODY_BYTES = 1 << 20; // 1MB 请求体上限
    private static final int READ_BUFFER_SIZE = 8192;

    private final LLMEngine engine;
    private final String modelName;

    public OpenAiHandler(LLMEngine engine, String modelName) {
        this.engine = engine;
        this.modelName = modelName;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        try {
            String path = exchange.getRequestURI().getPath();
            String method = exchange.getRequestMethod();
            if ("GET".equals(method) && "/v1/models".equals(path)) {
                handleModels(exchange);
                return;
            }
            if ("POST".equals(method) && ("/v1/chat/completions".equals(path)
                    || "/v1/completions".equals(path))) {
                handleChatCompletions(exchange);
                return;
            }
            sendJson(exchange, 404, "{\"error\":\"not found: " + escape(path) + "\"}");
        } catch (IllegalArgumentException e) {
            sendJson(exchange, 400, "{\"error\":\"bad request: " + escape(e.getMessage()) + "\"}");
        } catch (Exception e) {
            sendJson(exchange, 500, "{\"error\":\"internal error: " + escape(e.getMessage()) + "\"}");
        }
    }

    private void handleChatCompletions(HttpExchange exchange) throws IOException {
        String body = readBody(exchange);
        @SuppressWarnings("unchecked")
        Map<String, Object> req = SimpleJson.parseObject(body);

        boolean enableThinking = Boolean.TRUE.equals(req.get("enable_thinking"));
        String prompt = extractPrompt(req, enableThinking);
        boolean stream = Boolean.TRUE.equals(req.get("stream"));
        int maxTokens = intOr(req.get("max_tokens"), 512);
        float temperature = (float) doubleOr(req.get("temperature"), 0.8);
        float topP = (float) doubleOr(req.get("top_p"), 0.9);
        int topK = intOr(req.get("top_k"), 0);

        if (stream) {
            handleStream(exchange, prompt, maxTokens, temperature, topK, topP);
        } else {
            handleNonStream(exchange, prompt, maxTokens, temperature, topK, topP);
        }
    }

    /** 流式：边生成边推送 SSE chunk */
    private void handleStream(HttpExchange exchange, String prompt, int maxTokens,
                              float temperature, int topK, float topP) throws IOException {
        SseWriter sse = new SseWriter(exchange);
        String id = "chatcmpl-" + System.currentTimeMillis();
        long created = System.currentTimeMillis() / 1000;
        try {
            Sequence seq = engine.addRequest(prompt, maxTokens, temperature, topK, topP, token -> {
                try {
                    sse.write(streamChunk(id, created, token, null));
                } catch (IOException ignored) {
                    // 客户端可能已断开
                }
            });
            seq.awaitDone();
            // 结束 chunk
            sse.write(streamChunk(id, created, "", "stop"));
            sse.finish();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            sse.closeQuietly();
        } catch (Exception e) {
            sse.closeQuietly();
        }
    }

    /** 非流式：等全部生成后返回完整 JSON */
    private void handleNonStream(HttpExchange exchange, String prompt, int maxTokens,
                                 float temperature, int topK, float topP) throws IOException {
        List<String> collected = Collections.synchronizedList(new ArrayList<>());
        Sequence seq = engine.addRequest(prompt, maxTokens, temperature, topK, topP, collected::add);
        try {
            seq.awaitDone();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        String text = String.join("", collected);
        String id = "chatcmpl-" + System.currentTimeMillis();
        long created = System.currentTimeMillis() / 1000;
        int promptTokens = seq.promptTokens().length;
        int completionTokens = seq.outputTokens().size();
        String json = completionJson(id, created, text, promptTokens, completionTokens);
        sendJson(exchange, 200, json);
    }

    private void handleModels(HttpExchange exchange) throws IOException {
        Map<String, Object> model = new LinkedHashMap<>();
        model.put("id", modelName);
        model.put("object", "model");
        model.put("owned_by", "mini-vllm");
        List<Object> data = new ArrayList<>();
        data.add(model);
        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("object", "list");
        resp.put("data", data);
        sendJson(exchange, 200, SimpleJson.stringify(resp));
    }

    // ─── 响应 JSON 构造 ───

    private String streamChunk(String id, long created, String content, String finishReason) {
        Map<String, Object> delta = new LinkedHashMap<>();
        if (!content.isEmpty()) {
            delta.put("content", content);
        }
        Map<String, Object> choice = map("index", 0, "delta", delta, "finish_reason", finishReason);
        Map<String, Object> chunk = map(
                "id", id, "object", "chat.completion.chunk",
                "created", created, "model", modelName,
                "choices", List.of(choice));
        return SimpleJson.stringify(chunk);
    }

    private String completionJson(String id, long created, String content,
                                  int promptTokens, int completionTokens) {
        Map<String, Object> message = map("role", "assistant", "content", content);
        Map<String, Object> choice = map("index", 0, "message", message, "finish_reason", "stop");
        Map<String, Object> usage = map(
                "prompt_tokens", promptTokens,
                "completion_tokens", completionTokens,
                "total_tokens", promptTokens + completionTokens);
        Map<String, Object> resp = map(
                "id", id, "object", "chat.completion",
                "created", created, "model", modelName,
                "choices", List.of(choice), "usage", usage);
        return SimpleJson.stringify(resp);
    }

    // ─── 工具方法 ───

    /** 链式构建有序 Map（避免重复的 put 样板代码） */
    private static Map<String, Object> map(Object... kvs) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kvs.length; i += 2) {
            m.put((String) kvs[i], kvs[i + 1]);
        }
        return m;
    }

    /**
     * 从 messages 数组构造 prompt：
     * - chatML 模式（Qwen3 等对话模型）：渲染为 ChatML 格式（<|im_start|>...<|im_end|>）
     * - 纯文本模式（prompt 字段 / 学习用微模型）：原样返回
     */
    @SuppressWarnings("unchecked")
    private String extractPrompt(Map<String, Object> req, boolean enableThinking) {
        Object msgs = req.get("messages");
        if (msgs == null) {
            // 纯文本 prompt（学习用微模型）：不套 ChatML，原样返回
            Object p = req.get("prompt");
            return p == null ? "" : p.toString();
        }
        List<ChatTemplate.Message> list = new ArrayList<>();
        for (Object o : (List<Object>) msgs) {
            Map<String, Object> m = (Map<String, Object>) o;
            String role = String.valueOf(m.getOrDefault("role", "user"));
            String content = String.valueOf(m.getOrDefault("content", ""));
            list.add(new ChatTemplate.Message(role, content));
        }
        return ChatTemplate.applyChatML(list, enableThinking);
    }

    private String readBody(HttpExchange exchange) throws IOException {
        try (InputStream is = exchange.getRequestBody()) {
            byte[] buf = new byte[READ_BUFFER_SIZE];
            int n, total = 0;
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            while ((n = is.read(buf)) > 0) {
                out.write(buf, 0, n);
                total += n;
                if (total > MAX_BODY_BYTES) break;
            }
            return new String(out.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    private void sendJson(HttpExchange exchange, int status, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static int intOr(Object v, int def) {
        return v instanceof Number ? ((Number) v).intValue() : def;
    }

    private static double doubleOr(Object v, double def) {
        return v instanceof Number ? ((Number) v).doubleValue() : def;
    }

    private static String escape(String s) {
        return s == null ? "" : s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
