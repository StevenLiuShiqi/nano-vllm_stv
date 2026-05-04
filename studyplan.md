# Nano-vLLM 学习计划

> 目标：通过阅读 + 动手改进，深入理解 vLLM 核心思想的实际实现。
> 前提：已读过 vLLM 论文，了解 PagedAttention / Continuous Batching 的概念。
> 顺序：按重要性从高到低，先啃核心，再补边角。

---

## Phase 1：KV Cache 的物理管理 — BlockManager + Attention（最核心）

**为什么先看这里**：PagedAttention 是 vLLM 的灵魂。论文讲了思想，这里看它怎么落地。

### 文件
- `engine/block_manager.py`（112行）
- `layers/attention.py`（75行）

### 要搞清楚的问题

**BlockManager：**
1. `Block` 有三个字段：`block_id`、`ref_count`、`hash`。`ref_count > 1` 什么时候发生？（提示：prefix caching 共享块）
2. `allocate()` 里的逻辑分两条路：cache hit 和 cache miss。hit 的条件是什么？为什么要同时检查 hash 和 `token_ids`？
3. `may_append()` 在 `len(seq) % block_size == 0` 时做了什么？为什么这个时刻要封存（freeze）当前块？
4. `can_append()` 的条件：`len(self.free_block_ids) >= (len(seq) % block_size == 1)`。为什么只需要检查"是否需要新块"而不是检查"当前块够不够用"？

**Attention：**
5. `store_kvcache` 用 Triton kernel 写的。它的输入是当前 step 计算出的 K/V，输出写到哪里？`slot_mapping` 起什么作用？slot 和 block 的关系是什么？（`slot = block_id * block_size + offset`）
6. prefill 阶段用 `flash_attn_varlen_func`，decode 阶段用 `flash_attn_with_kvcache`。为什么要用两个不同的函数？两者接收的 KV 形状有什么区别？
7. prefill 有 prefix cache 和无 prefix cache 时，传给 flash attention 的 K/V 有什么不同？

### 动手改进
**给 BlockManager 加内存统计接口**：

```python
def stats(self) -> dict:
    """返回当前内存使用情况"""
    total = len(self.blocks)
    used = len(self.used_block_ids)
    cached = sum(1 for b in self.blocks if b.hash != -1 and b.block_id in self.used_block_ids)
    return {
        "total_blocks": total,
        "used_blocks": used,
        "free_blocks": total - used,
        "prefix_cached_blocks": cached,
        "utilization": used / total,
    }
```

在 `llm_engine.py` 的 `step()` 里每隔 N 步打印一次，观察 prefix cache 命中率的变化。

---

## Phase 2：调度器 — Continuous Batching 的实际决策

**为什么第二看这里**：Scheduler 决定哪些请求进、哪些被抢占，是 throughput 优化的关键。

### 文件
- `engine/scheduler.py`（71行）
- `engine/sequence.py`（83行）

### 要搞清楚的问题

**Scheduler：**
1. `schedule()` 返回 `(seqs, is_prefill)`。为什么 prefill 和 decode **不能**在同一次 schedule 里混合返回？（看 model_runner 的 `run()` 方法就懂了）
2. prefill 阶段的 token budget：`num_batched_tokens += len(seq) - seq.num_cached_tokens`。为什么减去 cached tokens？如果不减，会有什么问题？
3. decode 阶段的抢占逻辑（preempt）：当内存不足时，优先抢占 `self.running.pop()`（队尾，最新加入的），而不是 LRU。这是什么策略？有什么取舍？
4. `preempt()` 把 seq 放回 `waiting.appendleft()`（队头），而不是队尾。为什么？

**Sequence：**
5. `__getstate__` / `__setstate__` 做了自定义序列化。decode 阶段只序列化 `last_token` 而不是整个 `token_ids`。为什么？（提示：看 model_runner 里 SharedMemory 的用法）
6. `Sequence.block_size = 256` 是类变量。这意味着所有 seq 共享同一个 block_size。这个设计有什么限制？

### 动手改进
**加抢占计数器**，观察抢占频率与 batch size 的关系：

```python
# 在 Scheduler.__init__ 里加
self.num_preemptions = 0

# 在 preempt() 里加
self.num_preemptions += 1
```

在 `llm_engine.py` 生成结束后打印总抢占次数。用 `bench.py` 跑一遍，看在什么负载下抢占开始出现。

---

## Phase 3：ModelRunner — Prefill/Decode 的 Batch 组装

**为什么第三看这里**：这是调度器和模型之间的桥梁，搞清楚 tensor 是怎么组装的，PagedAttention 的工程细节才算真正理解。

### 文件
- `engine/model_runner.py`（251行）

### 要搞清楚的问题

**KV Cache 分配：**
1. `warmup_model()` 先跑一次假的 prefill，目的是什么？为什么要 `reset_peak_memory_stats()`？
2. `allocate_kv_cache()` 里的公式：`total * gpu_memory_utilization - used - peak + current`。`peak - current` 代表什么？为什么要减去这个值？
3. `kv_cache` 的形状：`[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`。第一维 2 代表什么？`num_blocks` 在这里是物理块数量，和 Sequence 里的逻辑块表有什么对应关系？

**Batch 组装：**
4. `prepare_prefill()` 里，`input_ids` 只取 `seq[seq.num_cached_tokens:]`（跳过已缓存的 token），但 `cu_seqlens_k` 用的是完整的 `seqlen`。为什么 Q 和 K 的长度可以不一样？Flash Attention 支持这种情况吗？
5. `slot_mapping` 是什么？在 prefill 里，一个 seq 会映射到多个 slot；在 decode 里，只映射到 1 个 slot。为什么 decode 只需要 1 个？
6. `prepare_decode()` 里的 `context_lens` 是 `len(seq)`（含已生成的 token），而不是 prompt 长度。这个值传给 flash attention 后起什么作用？

**CUDA Graph：**
7. `capture_cudagraph()` 用 `graph_bs = [1, 2, 4, 8, 16, 32, ...]` 这些离散的 batch size 捕获图。为什么不只捕获一个最大 batch size 的图？
8. `run_model()` 里，当 `bs > 512` 时退化到 eager 模式，不用 CUDA Graph。为什么？（提示：图的 overhead 相对收益）
9. Tensor Parallelism 用 SharedMemory 传参（而不是 `dist.broadcast`）。这个设计的好处是什么？`write_shm` / `read_shm` / `event` 三者是怎么协调的？

### 动手改进
**给 prefill 和 decode 阶段分别计时**，统计每个 step 的耗时构成：

在 `run()` 方法里加计时，在 `llm_engine.step()` 里分别累计 prefill_time 和 decode_time，生成结束后打印平均值。这能让你直观感受到 prefill 和 decode 的延迟差异。

---

## Phase 4：LLMEngine — 调度循环的驱动者

**为什么第四看这里**：这是把所有组件串起来的地方，但逻辑本身很薄，93行里一半是胶水代码。

### 文件
- `engine/llm_engine.py`（93行）

### 要搞清楚的问题

1. `step()` 的返回值 `num_tokens`：prefill 时是正数（新处理的 token 数），decode 时是负数（`-len(seqs)`）。为什么用正负来区分？
2. Tensor Parallelism 的初始化：主进程用 `spawn` 启动 worker 进程，rank 0 和 rank 1+ 共享同一个 ModelRunner 类但走不同分支。rank > 0 的进程会在 `__init__` 里直接进入 `loop()` 阻塞，永远不返回。为什么这样设计？
3. `atexit.register(self.exit)` — 为什么用 atexit 而不是在 `generate()` 结束时直接调用？

### 动手改进
**实现 streaming 输出**（生产环境最常见的需求）：

当前 `generate()` 要等所有请求都完成才返回结果。改成支持回调：

```python
def generate(self, prompts, sampling_params, on_token=None, use_tqdm=True):
    # on_token(seq_id, token_id) 在每个新 token 生成时被调用
```

需要修改 `step()` 返回每步新生成的 token，再在 `scheduler.postprocess()` 里暴露出来。

---

## Phase 5：模型架构与并行层（相对次要）

**为什么最后看**：这部分是标准 Transformer 实现 + tensor parallelism 切割，相对独立，不影响对核心机制的理解。

### 文件
- `models/qwen3.py`（215行）
- `layers/linear.py`（153行）—— 张量并行的核心
- `layers/attention.py` 已在 Phase 1 看过
- `layers/rotary_embedding.py`（61行）
- `layers/sampler.py`（15行）
- `layers/layernorm.py`（50行）

### 要搞清楚的问题

1. `ColumnParallelLinear` 和 `RowParallelLinear` 各自按哪个维度切割权重矩阵？输入/输出需要做什么集合通信（all-reduce）？
2. GQA（Grouped Query Attention）在 tensor parallel 下，KV head 怎么分配？如果 `num_kv_heads < world_size` 会怎样？
3. `Sampler` 只有 15 行，目前只支持温度采样（temperature scaling + argmax/multinomial）。`top_p` / `top_k` 没有实现。

### 动手改进
**给 Sampler 加 top_p（nucleus sampling）支持**：

```python
# sampling_params.py 加 top_p 字段
# sampler.py 在 temperature scaling 之后，sample 之前，做 top_p 截断
```

这是一个完整的端到端改进：从 SamplingParams → Sequence → ModelRunner.prepare_sample → Sampler，串联所有层。

---

## 学习节奏

每个 Phase 按这个流程走：
1. **读代码**，在纸上或注释里回答上面列出的问题
2. **加 assert / print**，验证你对数据形状和控制流的猜测
3. **做改进**，跑 `python example.py` 验证功能正确
4. **遇到问题直接提问**，我们一起分析

---

## 文件速查表

| 文件 | 行数 | 核心概念 | 学习优先级 |
|------|------|---------|-----------|
| `engine/block_manager.py` | 112 | PagedAttention 内存管理、Prefix Cache | ⭐⭐⭐⭐⭐ |
| `layers/attention.py` | 75 | KV Cache 写入、FlashAttention 调用 | ⭐⭐⭐⭐⭐ |
| `engine/scheduler.py` | 71 | Continuous Batching、抢占策略 | ⭐⭐⭐⭐ |
| `engine/sequence.py` | 83 | 请求的内部表示、逻辑块表 | ⭐⭐⭐⭐ |
| `engine/model_runner.py` | 251 | Batch 组装、CUDA Graph、TP 通信 | ⭐⭐⭐⭐ |
| `engine/llm_engine.py` | 93 | 调度循环、进程管理 | ⭐⭐⭐ |
| `models/qwen3.py` | 215 | Transformer 架构 | ⭐⭐ |
| `layers/linear.py` | 153 | 张量并行线性层 | ⭐⭐ |
| `layers/rotary_embedding.py` | 61 | RoPE | ⭐ |
| `layers/sampler.py` | 15 | Token 采样 | ⭐ |
