# Continuous Batching

Continuous Batching 是 vLLM 高吞吐的另一支柱，让 GPU 始终满载。本文结合 [`LLMEngine`](../src/main/java/io/leavesfly/minivllm/core/LLMEngine.java) 代码讲清调度循环。

## 1. 问题：静态 batching 浪费 GPU

GPU 处理多个请求（一个 batch）比逐个处理高效得多——这是 batching。但**静态 batching** 有个硬伤：

```
静态 batching：整批必须等最慢的请求完成才能开下一批

step:   1    2    3    4    5    6    7    8
Req A:  ✓    ✓    ✓    done -    -    -    -     ← A 早完成，槽位空转
Req B:  ✓    ✓    ✓    ✓    ✓    ✓    ✓    done  ← B 慢，全批陪它等
```

请求 A 在 step 4 完成，但它的槽位要空等到 step 8（B 完成）才能释放。这段空转就是浪费的 GPU 算力。

## 2. 核心思想：每步换入换出

Continuous batching 的解法：**每个 decode 步都检查谁完成了，完成即移出，立刻换入等待的新请求**。

```
continuous batching：完成的槽位立即被新请求填充

step:   1    2    3    4    5    6    7    8
slot1:  A    A    A    C    C    C    D    D    ← A 完成，C 顶上，C 完成后 D 顶上
slot2:  B    B    B    B    B    B    B    done
```

slot1 永远不空转——这就是 "continuous"（连续）的含义。

## 3. Scheduler：两个队列

[`Scheduler`](../src/main/java/io/leavesfly/minivllm/core/Scheduler.java) 管理两个队列：

```java
public final class Scheduler {
    private final ConcurrentLinkedDeque<Sequence> waiting;  // 待 prefill 的新请求
    private final List<Sequence> running;                    // 正在 decode 的请求
    private final int maxNumSeqs;                            // 并发上限
}
```

- `waiting`：刚进来的请求，还没 prefill。用并发安全队列，因为 HTTP 线程随时 `add`，engine 线程 `poll`。
- `running`：已 prefill、正在逐 token 生成的请求。engine 单线程操作。
- `maxNumSeqs`：running 队列上限，即同时能服务多少请求。

## 4. LLMEngine.step：三段循环

[`LLMEngine.step()`](../src/main/java/io/leavesfly/minivllm/core/LLMEngine.java) 是调度的核心，每个循环执行三段：

```java
public void step() {
    admitNew();       // 1. 接纳新请求做 prefill
    decodeStep();     // 2. running 请求各走一步 decode
    sweepFinished();  // 3. 清扫完成的请求
}
```

### admitNew：换入

从 `waiting` 取新请求做 prefill，加入 `running`，直到占满 `maxNumSeqs` 或显存不足：

```java
private void admitNew() {
    while (running.size() < maxNumSeqs) {
        Sequence seq = waiting.peek();
        if (seq == null) break;
        // 为每层分配 KV block
        boolean ok = true;
        for (BlockTable bt : seq.blockTables) {
            if (!kvMgr.ensureCapacity(bt, promptLen)) { ok = false; break; }
        }
        if (!ok) {
            // 显存不足：回滚已分配 block，留在 waiting 等下次
            for (BlockTable bt : seq.blockTables)
                if (bt.numBlocks() > 0) kvMgr.free(bt);
            break;  // 等 running 释放显存
        }
        waiting.poll();
        // prefill 一次性处理 prompt，生成第一个 token
        float[] hidden = model.prefill(seq.promptTokens, kvMgr, seq.blockTables, 0);
        int nextToken = sampler.sample(model.logits(model.lastRow(hidden, promptLen)));
        seq.outputTokens.add(nextToken);
        seq.stage = DECODE;
        running.add(seq);
        emitToken(seq, nextToken);
    }
}
```

关键点：
- **显存约束**：admit 不只看 `maxNumSeqs`，还要看 KV cache 够不够。不够就回滚、停止 admit，等 running 中请求完成后释放显存再试。
- **回滚**：若某层分配失败，已分配的层要释放，保证不留半分配状态。

### decodeStep：推进

对每个 `running` 请求各走一步 decode（生成 1 个 token）：

```java
private void decodeStep() {
    for (Sequence seq : running) {
        if (seq.stage != DECODE) continue;
        int lastToken = seq.outputTokens.get(seq.outputTokens.size() - 1);
        int curIdx = seq.totalLen() - 1;   // 当前 token 全局位置
        // 按需扩容（新 token 的 K/V 需要落位）
        for (BlockTable bt : seq.blockTables)
            kvMgr.ensureCapacity(bt, curIdx + 1);
        // PagedAttention：从散落 block 读历史 KV，生成下一个 token
        float[] hidden = model.decode(lastToken, curIdx, kvMgr, seq.blockTables);
        int nextToken = sampler.sample(model.logits(hidden));
        seq.outputTokens.add(nextToken);
        emitToken(seq, nextToken);
    }
}
```

每个请求独立前进，互不阻塞——A 早完成不影响 B 继续生成。

### sweepFinished：换出

清扫完成/中止的请求，释放 KV cache：

```java
private void sweepFinished() {
    running.removeIf(seq -> {
        if (seq.isFinished()) {
            for (BlockTable bt : seq.blockTables) kvMgr.free(bt);  // 释放 block
            return true;
        }
        return false;
    });
}
```

**释放的 block 立刻回到 freeList**，下一轮 `admitNew` 就能用——这就是 continuous batching 与 PagedAttention 的协同：PagedAttention 让释放即时、无碎片；continuous batching 让释放的显存立刻被新请求用上。

## 5. 完成判定

[`Sequence.isFinished()`](../src/main/java/io/leavesfly/minivllm/core/Sequence.java) 定义完成条件：

```java
public boolean isFinished() {
    if (outputTokens.size() >= maxTokens) return true;          // 达到长度上限
    if (eosToken >= 0 && lastToken == eosToken) return true;    // 命中 EOS
    return false;
}
```

## 6. 静态 vs 连续 对比

| | 静态 batching | Continuous batching |
|---|---|---|
| 换入时机 | 整批完成后 | 每个 decode 步 |
| 完成请求 | 空等整批 | 立即移出 |
| 显存利用 | 预留固定，浪费 | 按需分配，即时回收 |
| GPU 空转 | 多（短请求空等） | 几乎无 |
| 实现复杂度 | 简单 | 中等（需调度循环） |

## 7. 驱动模式

LLMEngine 提供两种驱动方式：

**服务模式**（配合 HTTP）：独立线程持续跑 step：

```java
public void start() {
    Thread t = new Thread(() -> {
        while (running.get()) {
            if (scheduler.hasWork()) step();
            else Thread.sleep(1);   // 空闲让出 CPU
        }
    }, "mini-vllm-engine");
    t.setDaemon(true);
    t.start();
}
```

**同步模式**（测试/单请求）：自己驱动 step 直到完成：

```java
public String generate(String prompt, int maxTokens, ...) {
    Sequence seq = addRequest(prompt, ..., collected::add);
    while (!seq.isFinished()) {
        if (scheduler.hasWork()) step();
    }
    return String.join("", collected);
}
```

## 8. 观察调度效果

启动时加 `--quiet` 关闭日志，或默认开启查看每步状态：

```
[engine] running=3 waiting=2 freeBlocks=984
[engine] running=4 waiting=1 freeBlocks=980
[engine] running=3 waiting=2 freeBlocks=988   ← 一个完成，释放 block，新请求顶上
```

可以看到 running 数动态变化、freeBlocks 随完成请求回升——这就是 continuous batching 在运转。

---

**相关**：[PagedAttention](PagedAttention.md) · [Architecture](Architecture.md)
