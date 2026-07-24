package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.json.SimpleJson;
import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;
import io.leavesfly.minivllm.tokenizer.BpeTokenizer;
import io.leavesfly.minivllm.tokenizer.ChatTemplate;
import io.leavesfly.minivllm.tokenizer.Qwen3ChatMLTemplate;
import io.leavesfly.minivllm.weights.Qwen3Loader;
import io.leavesfly.minivllm.weights.SafetensorsLoader;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Qwen3AlignmentTest —— 与 HuggingFace Qwen3-0.6B 参考输出的严格数值对齐。
 *
 * 参考数据：src/test/resources/qwen3/reference_{0,1}.json（tools/dump_reference.py 生成，
 * HF F32 eager 前向）。对齐内容：
 *   1. ChatML 模板字符串与 apply_chat_template 完全一致
 *   2. BPE 分词 ids 完全一致
 *   3. 第 0/1/27 层 block 输出 hidden states 数值逼近
 *   4. 最后位置 logits 数值逼近 + top-5 token 一致
 *   5. greedy 生成 32 个 token 与 HF 完全一致（终极标准）
 *
 * 默认跳过：需显式 -Dqwen3.align=true 且能找到模型目录（-Dqwen3.dir 或 HF 缓存）。
 * 运行示例：mvn test -Dtest=Qwen3AlignmentTest -Dqwen3.align=true -DargLine="-Xmx6g"
 */
class Qwen3AlignmentTest {

    private static final int[] TRACE_LAYERS = {0, 1};
    private static final int GREEDY_TOKENS = 32;

    private static Path modelDir;
    private static Qwen3Model model;
    private static ModelConfig cfg;

    @BeforeAll
    static void setup() throws IOException {
        assumeTrue(Boolean.parseBoolean(System.getProperty("qwen3.align", "false")),
                "未指定 -Dqwen3.align=true，跳过对齐测试");
        modelDir = resolveModelDir();
        assumeTrue(modelDir != null && Files.exists(modelDir.resolve("model.safetensors")),
                "未找到 Qwen3-0.6B 模型目录");
        cfg = ModelConfig.fromConfigJson(SimpleJson.parseObject(
                Files.readString(modelDir.resolve("config.json"))));
        cfg.maxSeqLen(2048);
        Map<String, float[]> weights = SafetensorsLoader.load(modelDir.resolve("model.safetensors"));
        model = Qwen3Loader.load(cfg, weights);
    }

    @Test
    void alignCase0() throws IOException {
        alignWithReference("/qwen3/reference_0.json");
    }

    @Test
    void alignCase1() throws IOException {
        alignWithReference("/qwen3/reference_1.json");
    }

    private void alignWithReference(String resource) throws IOException {
        Map<String, Object> ref = parseRef(resource);
        String user = (String) ref.get("user");
        String prompt = (String) ref.get("prompt");
        int[] ids = intList(ref.get("ids"));
        float[] refLogits = floatList(ref.get("logits_last"));
        int[] greedyIds = intList(ref.get("greedy_ids"));

        // ---------- 1. ChatML 模板一致 ----------
        List<ChatTemplate.Message> msgs = new ArrayList<>();
        msgs.add(new ChatTemplate.Message("user", user));
        // HF 参考用默认 enable_thinking=true 生成，这里显式开启以对齐
        assertEquals(prompt, new Qwen3ChatMLTemplate().render(msgs, true), "ChatML 模板与 HF apply_chat_template 不一致");

        // ---------- 2. 分词 ids 一致 ----------
        BpeTokenizer tok = BpeTokenizer.fromModelDir(modelDir);
        assertArrayEquals(ids, tok.encode(prompt), "BPE 分词与 HF 不一致");

        // ---------- 3. 逐层 hidden states 对齐（layer 0/1 为 pre-norm block 输出） ----------
        Map<Integer, float[]> traced = new TreeMap<>();
        model.layerTrace = (layer, hidden) -> {
            if (traced.size() < 64) { // 防御性上限
                float[] last = new float[cfg.dModel()];
                System.arraycopy(hidden, hidden.length - cfg.dModel(), last, 0, cfg.dModel());
                traced.put(layer, last);
            }
        };
        float[] logits;
        float[] finalHidden;
        try {
            logits = model.forwardLastLogits(ids);
            finalHidden = model.forwardLastHidden(ids);
        } finally {
            model.layerTrace = null;
        }
        for (int l : TRACE_LAYERS) {
            float[] refHidden = floatList(ref.get("hidden_L" + l));
            float[] actHidden = traced.get(l);
            assertNotNull(actHidden, "缺少第 " + l + " 层 trace");
            Diff d = diff(refHidden, actHidden);
            System.out.printf("[align] layer %2d hidden: maxAbs=%.6f meanAbs=%.6f%n", l, d.max, d.mean);
            assertTrue(d.max < 2e-2, "第 " + l + " 层 hidden 偏差过大: " + d.max);
        }
        // final norm 之后的 hidden（hs[-1]）对齐
        Diff fd = diff(floatList(ref.get("hidden_final")), finalHidden);
        System.out.printf("[align] final hidden: maxAbs=%.6f meanAbs=%.6f%n", fd.max, fd.mean);
        assertTrue(fd.max < 5e-3, "final hidden 偏差过大: " + fd.max);

        // ---------- 4. logits 对齐 ----------
        Diff ld = diff(refLogits, logits);
        System.out.printf("[align] logits: maxAbs=%.6f meanAbs=%.6f%n", ld.max, ld.mean);
        assertTrue(ld.max < 0.2, "logits 偏差过大: " + ld.max);
        assertEquals(topKIds(refLogits, 5), topKIds(logits, 5), "top-5 token 不一致");

        // ---------- 5. greedy 生成 32 tokens 完全一致（终极标准） ----------
        int[] actual = greedyGenerate(ids, GREEDY_TOKENS);
        assertArrayEquals(greedyIds, actual, "greedy 生成与 HF 不一致");
        System.out.println("[align] greedy 32 tokens 完全一致: " + ref.get("greedy_text"));
    }

    // ===================== 工具 =====================

    /** 引擎路径 greedy：prefill + 逐 token decode + argmax */
    private int[] greedyGenerate(int[] promptIds, int n) {
        KVCacheManager kvMgr = new KVCacheManager(256, cfg.blockSize(), cfg.kvDim());
        BlockTable[] bts = new BlockTable[cfg.nLayer()];
        for (int i = 0; i < bts.length; i++) {
            bts[i] = new BlockTable();
        }
        int total = promptIds.length + n;
        for (BlockTable bt : bts) {
            assertTrue(kvMgr.ensureCapacity(bt, total));
        }
        int[] out = new int[n];
        float[] logits = model.prefillLogits(promptIds, kvMgr, bts, 0);
        for (int t = 0; t < n; t++) {
            int next = argmax(logits);
            out[t] = next;
            if (t < n - 1) {
                logits = model.decodeLogits(next, promptIds.length + t, kvMgr, bts);
            }
        }
        return out;
    }

    private static int argmax(float[] x) {
        int best = 0;
        for (int i = 1; i < x.length; i++) {
            if (x[i] > x[best]) {
                best = i;
            }
        }
        return best;
    }

    private static List<Integer> topKIds(float[] logits, int k) {
        Integer[] idx = new Integer[logits.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        java.util.Arrays.sort(idx, (a, b) -> Float.compare(logits[b], logits[a]));
        List<Integer> out = new ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            out.add(idx[i]);
        }
        return out;
    }

    private static final class Diff {
        float max;
        float mean;
    }

    private static Diff diff(float[] a, float[] b) {
        assertEquals(a.length, b.length, "长度不一致");
        Diff d = new Diff();
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            float dd = Math.abs(a[i] - b[i]);
            d.max = Math.max(d.max, dd);
            sum += dd;
        }
        d.mean = sum / a.length;
        return d;
    }

    private static Path resolveModelDir() {
        String prop = System.getProperty("qwen3.dir");
        if (prop != null) {
            return Path.of(prop);
        }
        Path snapshots = Path.of(System.getProperty("user.home"),
                ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
        if (Files.isDirectory(snapshots)) {
            try (var stream = Files.list(snapshots)) {
                return stream.filter(Files::isDirectory).findFirst().orElse(null);
            } catch (IOException e) {
                return null;
            }
        }
        return null;
    }

    private static Map<String, Object> parseRef(String resource) throws IOException {
        try (InputStream in = Qwen3AlignmentTest.class.getResourceAsStream(resource)) {
            assertNotNull(in, "缺少参考数据: " + resource + "（先运行 tools/dump_reference.py）");
            String json = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            return SimpleJson.parseObject(json);
        }
    }

    @SuppressWarnings("unchecked")
    private static int[] intList(Object o) {
        List<Object> list = (List<Object>) o;
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) list.get(i)).intValue();
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    private static float[] floatList(Object o) {
        List<Object> list = (List<Object>) o;
        float[] out = new float[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) list.get(i)).floatValue();
        }
        return out;
    }
}
