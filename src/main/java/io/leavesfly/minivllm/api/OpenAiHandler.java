package io.leavesfly.minivllm.api;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import io.leavesfly.minivllm.core.LLMEngine;
import io.leavesfly.minivllm.core.Sequence;
import io.leavesfly.minivllm.json.SimpleJson;
import io.leavesfly.minivllm.tokenizer.ChatTemplate;

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
            if ("POST".equals(method) && "/v1/chat/completions".equals(path)) {
                handleChatCompletions(exchange);
                return;
            }
            if ("POST".equals(method) && "/v1/completions".equals(path)) {
                handleChatCompletions(exchange); // 复用同一逻辑
                return;
            }
            sendJson(exchange, 404, "{\"error\":\"not found: " + path + "\"}");
        } catch (Exception e) {
            sendJson(exchange, 500, "{\"error\":\"" + escape(e.getMessage()) + "\"}");
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
            while (!seq.isFinished()) {
                Thread.sleep(1);
            }
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
            while (!seq.isFinished()) {
                Thread.sleep(1);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        String text = String.join("", collected);
        String id = "chatcmpl-" + System.currentTimeMillis();
        long created = System.currentTimeMillis() / 1000;
        int promptTokens = seq.promptTokens.length;
        int completionTokens = seq.outputTokens.size();
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

    // ---------- 响应 JSON 构造 ----------

    private String streamChunk(String id, long created, String content, String finishReason) {
        Map<String, Object> delta = new LinkedHashMap<>();
        if (!content.isEmpty()) {
            delta.put("content", content);
        }
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("delta", delta);
        choice.put("finish_reason", finishReason);
        List<Object> choices = new ArrayList<>();
        choices.add(choice);
        Map<String, Object> chunk = new LinkedHashMap<>();
        chunk.put("id", id);
        chunk.put("object", "chat.completion.chunk");
        chunk.put("created", created);
        chunk.put("model", modelName);
        chunk.put("choices", choices);
        return SimpleJson.stringify(chunk);
    }

    private String completionJson(String id, long created, String content,
                                  int promptTokens, int completionTokens) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        message.put("content", content);
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("message", message);
        choice.put("finish_reason", "stop");
        List<Object> choices = new ArrayList<>();
        choices.add(choice);
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", promptTokens);
        usage.put("completion_tokens", completionTokens);
        usage.put("total_tokens", promptTokens + completionTokens);
        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("id", id);
        resp.put("object", "chat.completion");
        resp.put("created", created);
        resp.put("model", modelName);
        resp.put("choices", choices);
        resp.put("usage", usage);
        return SimpleJson.stringify(resp);
    }

    // ---------- 工具方法 ----------

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
            byte[] buf = new byte[8192];
            int n, total = 0;
            ByteArrayOutputStreamLite out = new ByteArrayOutputStreamLite();
            while ((n = is.read(buf)) > 0) {
                out.write(buf, 0, n);
                total += n;
                if (total > 1 << 20) break; // 限制 1MB
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

    /** 极简 ByteArrayOutputStream（避免依赖 java.io.ByteArrayOutputStream 的同步开销，学习用） */
    private static final class ByteArrayOutputStreamLite {
        private byte[] buf = new byte[1024];
        private int count = 0;

        void write(byte[] b, int off, int len) {
            ensure(count + len);
            System.arraycopy(b, off, buf, count, len);
            count += len;
        }

        byte[] toByteArray() {
            byte[] r = new byte[count];
            System.arraycopy(buf, 0, r, 0, count);
            return r;
        }

        private void ensure(int cap) {
            if (cap <= buf.length) return;
            int newCap = buf.length * 2;
            while (newCap < cap) newCap *= 2;
            byte[] nb = new byte[newCap];
            System.arraycopy(buf, 0, nb, 0, count);
            buf = nb;
        }
    }
}
