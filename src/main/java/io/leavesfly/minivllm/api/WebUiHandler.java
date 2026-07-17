package io.leavesfly.minivllm.api;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/**
 * WebUiHandler —— 内置对话演示页面的静态资源处理器。
 *
 * 学习要点：
 * 1. 页面打包在 classpath（src/main/resources/web/index.html），随 jar 分发，无需外部文件。
 * 2. 启动时一次性读入内存，避免每次请求重复 IO。
 * 3. 仅响应 GET / 与 /index.html，其余路径返回 404。
 * 4. 页面本身通过 fetch 调用 /v1/chat/completions（SSE 流式），与 OpenAiHandler 解耦。
 */
public final class WebUiHandler implements HttpHandler {

    private final byte[] indexHtml;

    public WebUiHandler() throws IOException {
        try (InputStream is = WebUiHandler.class.getResourceAsStream("/web/index.html")) {
            if (is == null) {
                throw new IOException("classpath 中未找到 /web/index.html");
            }
            this.indexHtml = is.readAllBytes();
        }
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        String path = exchange.getRequestURI().getPath();
        boolean isIndex = "/".equals(path) || "/index.html".equals(path);
        if ("GET".equals(exchange.getRequestMethod()) && isIndex) {
            exchange.getResponseHeaders().set("Content-Type", "text/html; charset=utf-8");
            exchange.getResponseHeaders().set("Cache-Control", "no-cache");
            exchange.sendResponseHeaders(200, indexHtml.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(indexHtml);
            }
            return;
        }
        byte[] msg = ("{\"error\":\"not found: " + path + "\"}").getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(404, msg.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(msg);
        }
    }
}
