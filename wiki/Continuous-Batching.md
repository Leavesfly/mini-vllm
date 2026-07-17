# Continuous Batching

Continuous Batching（连续批处理）是 vLLM 高吞吐的另一半（PagedAttention 是省显存，Continuous Batching 是提吞吐）。本文讲清楚它解决什么问题、mini-vllm 的 `admit → decode → sweep` 三段循环如何实现，以及请求如何随时进出。

## 问题：静态批处理的空转

传统"静态批处理"（static batching）把一批请求凑齐一起跑，必须等**整批**都生成完才能返回、才能接新请求。问题：

- 各请求生成长度不同，短的早早算完却要**空等**最长的那个。
- 批次锁定期间新请求只能排队，GPU 大量时间在处理"已经完成但还没退出"的请求，利用率低。

## 解法：以 token 为粒度动态组批

Continuous Batching 把调度粒度从"整批"降到"每一步（每生成一个 token）"：

- 每一步都重新组织当前 batch：完成的立刻退出、腾出的资源立刻接纳新请求。
- 请求可以在任意 step **随时进入、随时离开**，batch 始终尽量填满。
- 配合 PagedAttention 的按需分配，退出请求释放的 KV block 立刻能给新请求用。

## mini-vllm 的三段循环

核心在 [LLMEngine.step()](../src/main/java/io/leavesfly/minivllm/core/LLMEngine.java)，一步即一个调度周期：

```java
public void step() {
    admitNew();       // 1. 从 waiting 接纳新请求做 prefill
    decodeStep();     // 2. 对每个 running 请求各走一步 decode
    sweepFinished();  // 3. 清扫完成/中止的请求，释放 KV cache
}
```

这三步正是 continuous batching 的本质。下面逐段拆解。

### 队列管理：Scheduler

[Scheduler](../src/main/java/io/leavesfly/minivllm/core/Scheduler.java) 维护两个队列，对应 vLLM 的 waiting/running：

```java
private final ConcurrentLinkedDeque<Sequence> waiting = new ConcurrentLinkedDeque<>(); // 待 prefill
private final List<Sequence> running = new ArrayList<>();                              // 正在 decode
private final int maxNumSeqs;                                                          // 并发上限
```

- `waiting` 用**并发安全队列**：HTTP 线程随时 `add`，引擎线程消费。
- `running` 只被引擎线程读写，用普通 `ArrayList`。
- `hasWork()`（`!waiting.isEmpty() || !running.isEmpty()`）决定引擎线程是 step 还是休眠。

### 请求状态机：Sequence

[Sequence](../src/main/java/io/leavesfly/minivllm/core/Sequence.java) 是一个请求的完整运行时状态：

```java
public enum Stage { WAITING, PREFILL, DECODE, FINISHED, ABORTED }

public final int[] promptTokens;             // 输入 token
public final List<Integer> outputTokens;     // 已生成 token
public final BlockTable[] blockTables;        // 每层一张（KV cache 按层独立）
public final int maxTokens, topK; public final float temperature, topP;
public final int[] eosTokens;                 // 停止 token 集合
public final Consumer<String> onToken;        // 流式回调
public volatile Stage stage = Stage.WAITING;
```

完成判定 `isFinished()`：达到 maxTokens、命中任一 EOS、或状态为 FINISHED/ABORTED。

### 第一段：admitNew —— 接纳新请求

从 waiting 取请求做 prefill，受"并发上限"与"KV 显存"双重约束：

```java
private void admitNew() {
    while (scheduler.running().size() < scheduler.maxNumSeqs()) {
        Sequence seq = scheduler.waiting().peek();
        if (seq == null) break;
        int promptLen = seq.promptTokens.length;

        // 为每层分配 KV cache block
        boolean ok = true;
        for (BlockTable bt : seq.blockTables)
            if (!kvMgr.ensureCapacity(bt, promptLen)) { ok = false; break; }

        if (!ok) {  // 显存不足：回滚已分配 block，留在 waiting 等下次
            for (BlockTable bt : seq.blockTables)
                if (bt.numBlocks() > 0) kvMgr.free(bt);
            break;
        }
        scheduler.waiting().poll(); // 正式取出

        seq.stage = Sequence.Stage.PREFILL;
        float[] logits = model.prefillLogits(seq.promptTokens, kvMgr, seq.blockTables, 0);
        int nextToken = sampler.sample(logits);   // 生成第一个 token
        seq.outputTokens.add(nextToken);
        seq.stage = Sequence.Stage.DECODE;
        scheduler.running().add(seq);
        emitToken(seq, nextToken);
    }
}
```

要点：
- **并发上限**：`running.size() < maxNumSeqs`（Qwen3 默认 2，学习模式默认 8）。
- **显存约束 + 回滚**：任一层分配失败就**回滚已分配的 block**，请求留在 waiting，等 running 中的请求释放后下次再试。这避免了半分配导致的泄漏。
- prefill 一次性前向整段 prompt，采样出**第一个** token，然后转入 DECODE 并加入 running。

### 第二段：decodeStep —— 每个请求走一步

对 running 中每个请求各生成一个 token：

```java
private void decodeStep() {
    for (Sequence seq : scheduler.running()) {
        if (seq.stage != Sequence.Stage.DECODE) continue;
        int lastToken = seq.outputTokens.get(seq.outputTokens.size() - 1);
        int curIdx = seq.totalLen() - 1;   // 当前 token 的全局位置
        int need = curIdx + 1;

        // 按需扩容 KV cache（新 token 的 K/V 需要落位）
        boolean ok = true;
        for (BlockTable bt : seq.blockTables)
            if (!kvMgr.ensureCapacity(bt, need)) { ok = false; break; }
        if (!ok) { seq.stage = Sequence.Stage.ABORTED; continue; } // 学习版不做 preemption

        float[] logits = model.decodeLogits(lastToken, curIdx, kvMgr, seq.blockTables);
        int nextToken = sampler.sample(logits);
        seq.outputTokens.add(nextToken);
        emitToken(seq, nextToken);
    }
}
```

要点：
- decode 每步只处理一个新 token，通过 PagedAttention 复用历史 KV cache（见 [PagedAttention](PagedAttention.md)）。
- decode 途中也可能需要**新 block**（当前 block 填满时），同样 `ensureCapacity` 按需扩容。
- 扩容失败时置 `ABORTED`——学习版不做抢占（preemption）；真实 vLLM 会把某些请求的 KV cache swap 到 CPU 或重算（recompute）。

### 第三段：sweepFinished —— 清扫与释放

完成或中止的请求释放 KV cache 并移出 running：

```java
private void sweepFinished() {
    scheduler.running().removeIf(seq -> {
        if (seq.isFinished()) {
            // 冲刷增量解码器里残留的不完整 UTF-8 尾字节
            if (seq.incDecoder != null && seq.onToken != null) {
                String rest = seq.incDecoder.flush();
                if (!rest.isEmpty()) seq.onToken.accept(rest);
            }
            for (BlockTable bt : seq.blockTables) kvMgr.free(bt); // 按引用计数释放
            seq.stage = Sequence.Stage.FINISHED;
            return true;
        }
        return false;
    });
}
```

释放的 block 立刻回到 `BlockPool.freeList`，**下一个 step 的 admitNew 就能拿去用**——这就是 continuous batching 与 PagedAttention 协同的关键闭环。

## emitToken：流式输出与 EOS 处理

每生成一个 token 就通过 `onToken` 回调推送文本，但要处理两个细节：

```java
private void emitToken(Sequence seq, int token) {
    if (seq.onToken == null) return;
    // EOS/停止 token 不输出文本，否则 <|im_end|> 会泄露到响应、污染多轮 ChatML 上下文
    for (int eos : seq.eosTokens) if (token == eos) return;
    if (seq.incDecoder != null) {
        String piece = seq.incDecoder.accept(token); // BPE 增量解码，避免跨 token UTF-8 截断乱码
        if (!piece.isEmpty()) seq.onToken.accept(piece);
    } else {
        seq.onToken.accept(tokenizer.decode(new int[]{token}));
    }
}
```

- **EOS 不外泄**：命中停止 token 时直接 return，不把 `<|im_end|>` 等特殊 token 解码成文本。
- **增量解码**：BPE 分词下，中文/emoji 的多字节 UTF-8 可能被切在两个 token 里，`IncrementalDecoder` 缓冲不完整字节，只输出可完整解码的片段（见 [OpenAI-API](OpenAI-API.md) 与 BpeTokenizer）。

## 两种驱动模式

[LLMEngine](../src/main/java/io/leavesfly/minivllm/core/LLMEngine.java) 提供两种驱动方式：

### 服务模式：后台线程持续 step

```java
public void start() {
    Thread t = new Thread(() -> {
        while (running.get()) {
            if (scheduler.hasWork()) step();
            else Thread.sleep(1);   // 空闲时休眠，避免忙等
        }
    }, "mini-vllm-engine");
    t.setDaemon(true);
    t.start();
}
```

HTTP 请求通过 `addRequest` 入队后，阻塞轮询 `seq.isFinished()`，由后台引擎线程异步推进。这体现了**请求与引擎的解耦**。

### 同步模式：自己驱动到完成

用于测试与单请求：

```java
public String generate(String prompt, int maxTokens, float temperature, int topK, float topP) {
    List<String> collected = Collections.synchronizedList(new ArrayList<>());
    Sequence seq = addRequest(prompt, maxTokens, temperature, topK, topP, collected::add);
    while (!seq.isFinished()) if (scheduler.hasWork()) step();
    return String.join("", collected);
}
```

## 一个具象例子

`maxNumSeqs=2`，但提交 5 个请求：

```
step 1: admit R0,R1（并发满）; R0,R1 各 prefill 出首 token
step 2: R0,R1 各 decode 1 token; R2~R4 在 waiting 等
...
step k: R0 命中 EOS → sweep 释放其 KV block → running=[R1]
step k+1: admit R2（复用 R0 释放的 block）→ running=[R1,R2]
...
最终: 5 个请求全部完成，running 清空
```

这正是 [LLMEngineE2ETest.continuousBatchingCompletesAllRequests](../src/test/java/io/leavesfly/minivllm/core/LLMEngineE2ETest.java) 验证的场景。

## 测试验证

[LLMEngineE2ETest](../src/test/java/io/leavesfly/minivllm/core/LLMEngineE2ETest.java) 用固定 seed 的随机模型（输出无语义但过程确定）验证：

| 测试方法 | 验证点 |
|---|---|
| `singleRequestGeneratesExactlyMaxTokens` | 无 EOS 时恰好生成 maxTokens 个 |
| `greedyDecodingIsReproducibleAcrossEngines` | greedy 解码在相同权重下可复现 |
| `eosTokenStopsGenerationEarly` | 命中 EOS 提前停止 |
| `continuousBatchingCompletesAllRequests` | maxSeqs=2 提交 5 个请求全部完成、running 清空 |
| `streamingCallbackFiresOncePerGeneratedToken` | 流式回调次数 == 生成 token 数 |
| `sampledTokensAlwaysWithinVocab` | 采样 token 恒在合法词表范围 |
| `gpt3SparseModelGeneratesEndToEnd` | GPT-3 稀疏注意力模型 decode 路径可跑通 |
| `asyncServiceModeProcessesRequest` | 服务模式 start/addRequest/stop 全链路 |

## 与真实 vLLM 的差异

| 方面 | mini-vllm | vLLM |
|---|---|---|
| 调度线程 | 单线程串行 step | scheduler + 多 GPU worker |
| 显存不足 | decode 时置 ABORTED，不抢占 | preemption：swap 到 CPU 或 recompute |
| batch 执行 | 逐请求串行前向（内部行并行） | 真正的 batched GEMM |
| 前缀共享 | 内存层提供接口，引擎未接入 | RadixAttention 自动共享 |

## 延伸

- 释放的 block 如何被复用：[KV-Cache-Memory](KV-Cache-Memory.md)
- decode 一步内部发生了什么：[PagedAttention](PagedAttention.md) 与 [Transformer](Transformer.md)
- 请求如何从 HTTP 进来：[OpenAI-API](OpenAI-API.md)
