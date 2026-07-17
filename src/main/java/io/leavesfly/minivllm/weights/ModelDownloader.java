package io.leavesfly.minivllm.weights;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

/**
 * ModelDownloader —— 默认模型的自动解析与下载（零依赖，纯 JDK HttpClient）。
 *
 * 学习要点：
 * 1. 三级解析顺序，命中即返回：
 *      HF 缓存（~/.cache/huggingface/hub/models--org--name/snapshots/&lt;rev&gt;）
 *      → 本地缓存（~/.cache/mini-vllm/models/&lt;repo&gt;）
 *      → 在线下载（ModelScope 优先，HuggingFace 兜底）。
 * 2. 下载可靠性：.part 临时文件 + 完成后原子改名（避免半成品被当作完整文件），
 *    Range 请求头实现断点续传（1.5GB 权重中断后不必重下）。
 * 3. HF 的 resolve URL 会 302 跳转到 CDN，HttpClient 需开启 followRedirects。
 */
public final class ModelDownloader {

    /** Qwen3 模式运行所需的最小文件集 */
    private static final List<String> REQUIRED_FILES = List.of(
            "config.json", "model.safetensors", "vocab.json", "merges.txt", "tokenizer_config.json");

    private static final String HF_URL = "https://huggingface.co/%s/resolve/main/%s";
    private static final String MS_URL = "https://modelscope.cn/models/%s/resolve/master/%s";
    private static final int BUFFER = 1 << 20; // 1 MB

    private final String repo;
    private final String mirror; // auto | hf | modelscope

    public ModelDownloader(String repo, String mirror) {
        this.repo = repo;
        this.mirror = mirror == null || mirror.isBlank() ? "auto" : mirror;
    }

    /**
     * 解析模型目录，优先级：
     *   项目内 ./models/&lt;name&gt;（手动下载/指定，最优先）
     *   → HuggingFace 缓存（~/.cache/huggingface/hub）
     *   → mini-vllm 本地缓存（~/.cache/mini-vllm/models，缺失则自动下载）
     */
    public Path resolve() throws IOException, InterruptedException {
        Path project = projectLocalDir();
        if (isComplete(project)) {
            System.out.println("命中项目模型目录: " + project);
            return project;
        }
        Path hf = findInHfCache();
        if (hf != null) {
            System.out.println("命中 HuggingFace 缓存: " + hf);
            return hf;
        }
        Path local = localCacheDir();
        if (isComplete(local)) {
            System.out.println("命中本地模型缓存: " + local);
            return local;
        }
        downloadAll(local);
        return local;
    }

    /** 项目内模型目录：./models/&lt;name&gt;（取 repo 末段作为目录名，如 Qwen3-0.6B） */
    private Path projectLocalDir() {
        String name = repo.substring(repo.lastIndexOf('/') + 1);
        return Path.of("models", name);
    }

    /** 本地缓存目录：~/.cache/mini-vllm/models/&lt;org-name&gt; */
    private Path localCacheDir() {
        return Path.of(System.getProperty("user.home"),
                ".cache", "mini-vllm", "models", repo.replace('/', '-'));
    }

    /** 在 HF 缓存中查找完整快照（多个 revision 取最近修改的）。
     *  目录命名规则：models--&lt;org&gt;--&lt;name&gt;（org 与 name 之间为双横线） */
    private Path findInHfCache() {
        Path snapshots = Path.of(System.getProperty("user.home"),
                ".cache", "huggingface", "hub", "models--" + repo.replace("/", "--"), "snapshots");
        if (!Files.isDirectory(snapshots)) {
            return null;
        }
        try (Stream<Path> s = Files.list(snapshots)) {
            return s.filter(Files::isDirectory)
                    .filter(ModelDownloader::isComplete)
                    .max(Comparator.comparingLong(p -> p.toFile().lastModified()))
                    .orElse(null);
        } catch (IOException e) {
            return null;
        }
    }

    /** 目录中必需文件齐全且非空 */
    private static boolean isComplete(Path dir) {
        if (!Files.isDirectory(dir)) {
            return false;
        }
        for (String f : REQUIRED_FILES) {
            try {
                Path p = dir.resolve(f);
                if (!Files.isRegularFile(p) || Files.size(p) == 0) {
                    return false;
                }
            } catch (IOException e) {
                return false;
            }
        }
        return true;
    }

    private void downloadAll(Path dir) throws IOException, InterruptedException {
        Files.createDirectories(dir);
        System.out.println("模型未就绪，开始下载 " + repo + " → " + dir);
        for (String file : REQUIRED_FILES) {
            Path dest = dir.resolve(file);
            if (Files.isRegularFile(dest) && Files.size(dest) > 0) {
                continue;
            }
            downloadWithFallback(file, dest);
        }
        System.out.println("模型下载完成: " + dir);
    }

    /** 按镜像策略生成候选 URL（auto：ModelScope 优先，HF 兜底） */
    private List<String> mirrorUrls(String file) {
        List<String> urls = new ArrayList<>();
        switch (mirror) {
            case "hf" -> urls.add(String.format(HF_URL, repo, file));
            case "modelscope" -> urls.add(String.format(MS_URL, repo, file));
            default -> {
                urls.add(String.format(MS_URL, repo, file));
                urls.add(String.format(HF_URL, repo, file));
            }
        }
        return urls;
    }

    private void downloadWithFallback(String file, Path dest) throws IOException, InterruptedException {
        List<String> urls = mirrorUrls(file);
        // 若 .part 有来源记录，优先从原镜像续传：不同镜像的文件版本可能不一致，
        // 跨来源拼接会得到损坏文件（实测 ModelScope 与 HF 同尺寸文件内容有差异）
        String recorded = readSource(dest);
        if (recorded != null && urls.remove(recorded)) {
            urls.add(0, recorded);
        }
        IOException last = null;
        for (String url : urls) {
            try {
                downloadFile(url, dest);
                return;
            } catch (IOException e) {
                last = e;
                System.out.println("\n  " + file + " 下载失败，切换镜像重试 (" + e.getMessage() + ")");
            }
        }
        throw new IOException("所有镜像均下载失败: " + file, last);
    }

    /** 读取 .part 的来源 URL（无记录返回 null） */
    private static String readSource(Path dest) {
        try {
            Path sidecar = sourceFile(dest);
            return Files.isRegularFile(sidecar) ? Files.readString(sidecar).trim() : null;
        } catch (IOException e) {
            return null;
        }
    }

    private static Path partFile(Path dest) {
        return dest.resolveSibling(dest.getFileName() + ".part");
    }

    private static Path sourceFile(Path dest) {
        return dest.resolveSibling(dest.getFileName() + ".part.url");
    }

    /** 单文件下载：断点续传 + 进度显示 + 原子改名。
     *  .part 的字节来源必须与本次 URL 一致，否则放弃旧进度从头下载。 */
    private void downloadFile(String url, Path dest) throws IOException, InterruptedException {
        Path part = partFile(dest);
        Path sidecar = sourceFile(dest);
        long existing = Files.isRegularFile(part) ? Files.size(part) : 0;
        if (existing > 0) {
            String recorded = readSource(dest);
            if (recorded != null && !recorded.equals(url)) {
                System.out.println("\n  " + dest.getFileName() + " 镜像变更，放弃已有进度重新下载");
                Files.delete(part);
                existing = 0;
            }
        }
        if (existing == 0) {
            Files.writeString(sidecar, url);
        }

        HttpClient client = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .connectTimeout(Duration.ofSeconds(30))
                .build();
        // 注意：不要设置请求级 timeout —— HttpClient 的超时计时器在响应体流式读取期间
        // 仍然生效，会在超时后强制断开连接（大文件下载必然被误杀）。连接超时由 client 控制。
        HttpRequest.Builder req = HttpRequest.newBuilder(URI.create(url)).GET();
        if (existing > 0) {
            req.header("Range", "bytes=" + existing + "-");
        }

        HttpResponse<InputStream> resp = client.send(req.build(), HttpResponse.BodyHandlers.ofInputStream());
        int status = resp.statusCode();
        boolean append = false;
        long total = -1;
        if (status == 200) {
            total = resp.headers().firstValueAsLong("Content-Length").orElse(-1);
        } else if (status == 206 && existing > 0) {
            append = true;
            long remaining = resp.headers().firstValueAsLong("Content-Length").orElse(-1);
            total = remaining > 0 ? existing + remaining : -1;
        } else if (status == 416 && existing > 0) {
            // 服务端拒绝区间：.part 很可能已完整（上次改名前中断），直接采用
            resp.body().close();
            moveAtomically(part, dest);
            Files.deleteIfExists(sidecar);
            return;
        } else {
            resp.body().close();
            throw new IOException("HTTP " + status);
        }

        long downloaded = append ? existing : 0;
        long sessionStart = downloaded; // 速度只统计本次会话新下载的字节
        long t0 = System.currentTimeMillis();
        long lastPrint = 0;
        try (InputStream in = resp.body();
             OutputStream out = Files.newOutputStream(part, StandardOpenOption.CREATE,
                     append ? StandardOpenOption.APPEND : StandardOpenOption.TRUNCATE_EXISTING)) {
            byte[] buf = new byte[BUFFER];
            int n;
            while ((n = in.read(buf)) > 0) {
                out.write(buf, 0, n);
                downloaded += n;
                long now = System.currentTimeMillis();
                if (now - lastPrint > 500) {
                    lastPrint = now;
                    printProgress(dest.getFileName().toString(), downloaded, total, sessionStart, t0, now);
                }
            }
        }
        printProgress(dest.getFileName().toString(), downloaded, total, sessionStart, t0, System.currentTimeMillis());
        System.out.println();
        if (total > 0 && downloaded != total) {
            throw new IOException("下载不完整: " + downloaded + "/" + total);
        }
        moveAtomically(part, dest);
        Files.deleteIfExists(sidecar);
    }

    private static void moveAtomically(Path src, Path dest) throws IOException {
        try {
            Files.move(src, dest, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (java.nio.file.AtomicMoveNotSupportedException e) {
            Files.move(src, dest, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void printProgress(String name, long downloaded, long total,
                                      long sessionStart, long t0, long now) {
        double secs = Math.max(0.001, (now - t0) / 1000.0);
        double mbps = (downloaded - sessionStart) / 1e6 / secs;
        if (total > 0) {
            System.out.printf("\r  %s: %5.1f%% (%.0f/%.0f MB, %.1f MB/s)   ",
                    name, downloaded * 100.0 / total, downloaded / 1e6, total / 1e6, mbps);
        } else {
            System.out.printf("\r  %s: %.0f MB (%.1f MB/s)   ", name, downloaded / 1e6, mbps);
        }
    }
}
