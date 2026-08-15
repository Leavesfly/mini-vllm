package io.leavesfly.minivllm.api;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import io.leavesfly.minivllm.core.ModelHub;
import io.leavesfly.minivllm.core.ModelRuntime;
import io.leavesfly.minivllm.core.SamplingParams;
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
import java.util.HashMap;
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
 *      GET  /v1/models           —— 返回模型列表（含是否已加载，供 Web 页面标注）
 * 3. 流式：每个 token 经 onToken 回调通过 SseWriter 推送一个 chunk，最后发 [DONE]。
 *    非流式：收集所有 token 后一次性返回完整 JSON。
 * 4. handler 线程阻塞等待 Sequence 完成（engine 线程异步推进 step），体现了请求与引擎的解耦。
 * 5. 多模型：请求体 model 字段经 {@link ModelHub} 路由到对应 {@link ModelRuntime}，
 *    引擎、分词器、对话模板都取自被选中的运行时，本类不感知具体模型的 prompt 格式。
 */
public final class OpenAiHandler implements HttpHandler {

    private static final int MAX_BODY_BYTES = 1 << 20; // 1MB 请求体上限
    private static final int READ_BUFFER_SIZE = 8192;

    private final ModelHub hub;

    public OpenAiHandler(ModelHub hub) {
        this.hub = hub;
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
            if ("GET".equals(method) && "/metrics".equals(path)) {
                handleMetrics(exchange);
                return;
            }
            if ("POST".equals(method) && ("/v1/chat/completions".equals(path)
                    || "/v1/completions".equals(path))) {
                handleChatCompletions(exchange);
                return;
            }
            sendJson(exchange, 404, "{\"error\":\"not found: " + escape(path) + "\"}");
        } catch (PayloadTooLargeException e) {
            sendJson(exchange, 413, "{\"error\":\"" + escape(e.getMessage()) + "\"}");
        } catch (IllegalArgumentException e) {
            sendJson(exchange, 400, "{\"error\":\"bad request: " + escape(e.getMessage()) + "\"}");
        } catch (Exception e) {
            sendJson(exchange, 500, "{\"error\":\"internal error: " + escape(e.getMessage()) + "\"}");
        }
    }

    private void handleChatCompletions(HttpExchange exchange) throws IOException {
        String body = readBody(exchange);
        Map<String, Object> req = SimpleJson.parseObject(body);

        // 按 model 字段选择运行时；未注册的 id 抛 IllegalArgumentException → 400。
        // 该模型尚未加载时，本调用会同步完成加载（首次切换模型的等待就发生在这里）
        Object model = req.get("model");
        ModelRuntime runtime = hub.get(model == null ? null : model.toString());

        boolean enableThinking = Boolean.TRUE.equals(req.get("enable_thinking"));
        String prompt = extractPrompt(runtime.chatTemplate(), req, enableThinking);
        boolean stream = Boolean.TRUE.equals(req.get("stream"));
        SamplingParams params = parseSamplingParams(req);

        if (stream) {
            handleStream(exchange, runtime, prompt, params);
        } else {
            handleNonStream(exchange, runtime, prompt, params);
        }
    }

    /**
     * 未显式传参的字段回退到 {@link SamplingParams#DEFAULT}（与引擎共用同一份默认值）。
     * 除基础四参数外，支持 repetition_penalty / frequency_penalty / min_p / logit_bias
     * （logit_bias 为 token id 字符串 → 偏置值的对象，与 OpenAI 格式一致）。
     */
    @SuppressWarnings("unchecked")
    private static SamplingParams parseSamplingParams(Map<String, Object> req) {
        SamplingParams def = SamplingParams.DEFAULT;
        Map<Integer, Float> logitBias = null;
        Object lb = req.get("logit_bias");
        if (lb instanceof Map<?, ?> m && !m.isEmpty()) {
            logitBias = new HashMap<>();
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (e.getValue() instanceof Number num) {
                    try {
                        logitBias.put(Integer.parseInt(e.getKey().toString()), num.floatValue());
                    } catch (NumberFormatException ignored) {
                        // 非法 token id 忽略（与主流服务容错行为一致）
                    }
                }
            }
        }
        return new SamplingParams(
                intOr(req.get("max_tokens"), def.maxTokens()),
                (float) doubleOr(req.get("temperature"), def.temperature()),
                intOr(req.get("top_k"), def.topK()),
                (float) doubleOr(req.get("top_p"), def.topP()),
                (float) doubleOr(req.get("repetition_penalty"), def.repetitionPenalty()),
                (float) doubleOr(req.get("frequency_penalty"), def.frequencyPenalty()),
                (float) doubleOr(req.get("min_p"), def.minP()),
                logitBias);
    }

    /** 流式：边生成边推送 SSE chunk */
    private void handleStream(HttpExchange exchange, ModelRuntime runtime, String prompt,
                              SamplingParams params) throws IOException {
        SseWriter sse = new SseWriter(exchange);
        String id = "chatcmpl-" + System.currentTimeMillis();
        long created = System.currentTimeMillis() / 1000;
        Sequence seq = null;
        try {
            seq = runtime.engine().addRequest(prompt, params, token -> {
                try {
                    sse.write(streamChunk(runtime.id(), id, created, token, null));
                } catch (IOException e) {
                    // 客户端已断开：抛信号异常，下方捕获后取消请求（引擎清扫时释放 KV）
                    throw new ClientGoneException();
                }
            });
            seq.awaitDone();
            // 结束 chunk
            sse.write(streamChunk(runtime.id(), id, created, "", "stop"));
            sse.finish();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            if (seq != null) {
                seq.cancel();
            }
            sse.closeQuietly();
        } catch (ClientGoneException e) {
            if (seq != null) {
                seq.cancel();
            }
            sse.closeQuietly();
        } catch (Exception e) {
            if (seq != null) {
                seq.cancel();
            }
            sse.closeQuietly();
        }
    }

    /** 流式回调中检测到客户端断连的内部信号（onToken 不能抛受检异常，用运行时异常穿透） */
    private static final class ClientGoneException extends RuntimeException {
    }

    /** 非流式：等全部生成后返回完整 JSON */
    private void handleNonStream(HttpExchange exchange, ModelRuntime runtime, String prompt,
                                 SamplingParams params) throws IOException {
        List<String> collected = Collections.synchronizedList(new ArrayList<>());
        Sequence seq = runtime.engine().addRequest(prompt, params, collected::add);
        try {
            seq.awaitDone();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            seq.cancel(); // 不再等待结果：取消请求，引擎清扫时释放 KV，避免请求泄漏继续生成
            throw new IOException("请求被中断", e);
        }
        String text = String.join("", collected);
        String id = "chatcmpl-" + System.currentTimeMillis();
        long created = System.currentTimeMillis() / 1000;
        int promptTokens = seq.promptTokens().length;
        int completionTokens = seq.outputTokens().size();
        String json = completionJson(runtime.id(), id, created, text, promptTokens, completionTokens);
        sendJson(exchange, 200, json);
    }

    /**
     * 模型列表：除 OpenAI 标准字段外，额外给出 loaded / default，
     * 让 Web 页面能标注「待加载」并选中默认模型（懒加载下这两点用户可感知）。
     */
    private void handleModels(HttpExchange exchange) throws IOException {
        List<Object> data = new ArrayList<>();
        for (String id : hub.ids()) {
            data.add(map("id", id, "object", "model", "owned_by", "mini-vllm",
                    "loaded", hub.isLoaded(id), "default", id.equals(hub.defaultId())));
        }
        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("object", "list");
        resp.put("data", data);
        sendJson(exchange, 200, SimpleJson.stringify(resp));
    }

    /**
     * 引擎指标（对照 vLLM Prometheus 指标的学习版）：每个已加载模型一份，
     * 含队列长度、KV 利用率、TTFT/ITL、吞吐、抢占次数等。
     */
    private void handleMetrics(HttpExchange exchange) throws IOException {
        List<Object> engines = new ArrayList<>();
        for (ModelRuntime runtime : hub.loadedRuntimes()) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("model", runtime.id());
            m.putAll(runtime.engine().metricsSnapshot());
            engines.add(m);
        }
        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("engines", engines);
        sendJson(exchange, 200, SimpleJson.stringify(resp));
    }

    // ─── 响应 JSON 构造 ───

    private String streamChunk(String modelId, String id, long created, String content,
                               String finishReason) {
        Map<String, Object> delta = new LinkedHashMap<>();
        if (!content.isEmpty()) {
            delta.put("content", content);
        }
        Map<String, Object> choice = map("index", 0, "delta", delta, "finish_reason", finishReason);
        Map<String, Object> chunk = map(
                "id", id, "object", "chat.completion.chunk",
                "created", created, "model", modelId,
                "choices", List.of(choice));
        return SimpleJson.stringify(chunk);
    }

    private String completionJson(String modelId, String id, long created, String content,
                                  int promptTokens, int completionTokens) {
        Map<String, Object> message = map("role", "assistant", "content", content);
        Map<String, Object> choice = map("index", 0, "message", message, "finish_reason", "stop");
        Map<String, Object> usage = map(
                "prompt_tokens", promptTokens,
                "completion_tokens", completionTokens,
                "total_tokens", promptTokens + completionTokens);
        Map<String, Object> resp = map(
                "id", id, "object", "chat.completion",
                "created", created, "model", modelId,
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
     * - messages 模式：由被选中模型的 ChatTemplate 渲染（Qwen3 为 ChatML，微模型为纯文本）
     * - 纯文本模式（prompt 字段）：原样返回
     */
    @SuppressWarnings("unchecked")
    private static String extractPrompt(ChatTemplate chatTemplate, Map<String, Object> req,
                                        boolean enableThinking) {
        Object msgs = req.get("messages");
        if (msgs == null) {
            // 纯文本 prompt（学习用微模型）：不套对话模板，原样返回
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
        return chatTemplate.render(list, enableThinking);
    }

    private String readBody(HttpExchange exchange) throws IOException {
        try (InputStream is = exchange.getRequestBody()) {
            byte[] buf = new byte[READ_BUFFER_SIZE];
            int n, total = 0;
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            while ((n = is.read(buf)) > 0) {
                out.write(buf, 0, n);
                total += n;
                if (total > MAX_BODY_BYTES) {
                    // 超限直接拒绝：静默截断会拿到残缺 JSON，报错信息误导调用方
                    throw new PayloadTooLargeException(
                            "request body exceeds " + MAX_BODY_BYTES + " bytes");
                }
            }
            return new String(out.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    /** 请求体超限信号（映射为 HTTP 413） */
    private static final class PayloadTooLargeException extends RuntimeException {
        PayloadTooLargeException(String msg) {
            super(msg);
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
        return s == null ? "" : s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t");
    }
}
