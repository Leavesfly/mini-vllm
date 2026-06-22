package io.leavesfly.minivllm.api;

import com.sun.net.httpserver.HttpExchange;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/**
 * SseWriter —— 封装 HTTP 响应为 Server-Sent Events 流式输出。
 *
 * 学习要点：
 * 1. SSE 协议：每条消息以 "data: " 前缀 + 内容 + 两个换行结尾。客户端逐条读取。
 * 2. OpenAI 流式 API 用 SSE：每个 token 作为一个 chunk 推送，最后发 "data: [DONE]" 结束。
 * 3. 用 sendResponseHeaders(200, 0) 启用 chunked 传输，可边生成边写，无需预知总长度。
 * 4. write 方法 synchronized：onToken 回调在 engine 线程触发，与 handler 线程并发写需同步。
 */
public final class SseWriter {

    private final OutputStream os;
    private volatile boolean closed = false;

    public SseWriter(HttpExchange exchange) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
        exchange.getResponseHeaders().set("Cache-Control", "no-cache");
        exchange.sendResponseHeaders(200, 0); // 0 = chunked 传输
        this.os = exchange.getResponseBody();
    }

    /** 写一条 SSE data 事件 */
    public synchronized void write(String data) throws IOException {
        if (closed) return;
        os.write(("data: " + data + "\n\n").getBytes(StandardCharsets.UTF_8));
        os.flush();
    }

    /** 写结束标记并关闭 */
    public synchronized void finish() throws IOException {
        if (closed) return;
        write("[DONE]");
        closed = true;
        os.close();
    }

    public synchronized void closeQuietly() {
        if (closed) return;
        closed = true;
        try {
            os.close();
        } catch (IOException ignored) {
        }
    }
}
