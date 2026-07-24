package io.leavesfly.minivllm.family;

import io.leavesfly.minivllm.json.SimpleJson;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.ServiceLoader;

/**
 * ModelRegistry —— 按架构分发到对应 ModelFamily（对照 vLLM 的 ModelRegistry）。
 *
 * 学习要点：
 * 1. 实现发现用 JDK 标准库 {@link ServiceLoader}：扫描 classpath 上
 *    META-INF/services/io.leavesfly.minivllm.family.ModelFamily 中登记的实现类。
 *    这是 Java 内建的插件机制（JDBC 驱动、JUL provider 都用它），零外部依赖。
 * 2. 分发键是模型目录 config.json 的 model_type 字段（HF 生态的架构标识），
 *    逐个询问家族 supports()，命中即用。
 * 3. 接入新模型 = 新增一个 ModelFamily 实现 + 一行 services 注册，
 *    本类与 MiniVllmServer 都无需改动。
 */
public final class ModelRegistry {

    private final List<ModelFamily> families = new ArrayList<>();

    /** 从 classpath 发现所有已注册的 ModelFamily 实现 */
    public ModelRegistry() {
        ServiceLoader.load(ModelFamily.class).forEach(families::add);
    }

    /** 测试或嵌入场景：显式给定家族列表，跳过 ServiceLoader */
    public ModelRegistry(List<ModelFamily> families) {
        this.families.addAll(families);
    }

    /**
     * 从模型目录加载：读 config.json 的 model_type，分发给支持它的家族。
     *
     * @throws IllegalArgumentException 没有任何已注册家族支持该架构
     */
    public LoadedModel load(Path modelDir, Precision precision, boolean random, int maxSeqLenCap)
            throws IOException {
        String modelType = readModelType(modelDir);
        for (ModelFamily family : families) {
            if (family.supports(modelType)) {
                return family.load(modelDir, precision, random, maxSeqLenCap);
            }
        }
        throw new IllegalArgumentException("不支持的模型架构 model_type=" + modelType
                + "（已注册家族: " + families.size() + " 个）。"
                + "接入方法：实现 ModelFamily 并在 META-INF/services 中注册。");
    }

    private static String readModelType(Path modelDir) throws IOException {
        Object type = SimpleJson.parseObject(
                Files.readString(modelDir.resolve("config.json"))).get("model_type");
        return String.valueOf(type);
    }
}
