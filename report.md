# Nano-vLLM 技术报告

## 目录

1. [项目概述](#项目概述)
2. [整体架构](#整体架构)
3. [核心模块详解](#核心模块详解)
4. [执行流程分析](#执行流程分析)
5. [关键技术实现](#关键技术实现)
6. [性能优化技术](#性能优化技术)
7. [代码示例与实践](#代码示例与实践)

---

## 项目概述

### 什么是 Nano-vLLM？

Nano-vLLM 是一个从零开始构建的轻量级大语言模型推理引擎，使用约 1,314 行 Python 代码实现了与完整 vLLM 相当的推理性能。这是一个绝佳的学习项目，可以帮助你深入理解现代 LLM 推理引擎的核心技术。

### 项目特点

- **高性能**: 在 RTX 4070 上达到 1434 tokens/s，略优于 vLLM 的 1362 tokens/s
- **代码简洁**: 核心代码仅约 1,314 行，可读性强
- **完整功能**: 实现了 Prefix Caching、Tensor Parallelism、CUDA Graph、Flash Attention 等核心优化
- **易于扩展**: 模块化设计，方便添加新功能

### 技术栈

```
核心依赖：
- PyTorch: 深度学习框架
- transformers: HuggingFace 模型加载
- flash-attn: Flash Attention 优化
- xxhash: 快速哈希（用于 Prefix Caching）
- torch.distributed: 分布式通信（Tensor Parallelism）
```

---

## 整体架构

### 目录结构

```
nano-vllm/
├── nanovllm/
│   ├── __init__.py              # 导出 LLM 和 SamplingParams
│   ├── llm.py                   # 用户API入口
│   ├── config.py                # 配置管理
│   ├── sampling_params.py       # 采样参数
│   ├── engine/                  # 推理引擎核心
│   │   ├── llm_engine.py       # 引擎主控制器
│   │   ├── model_runner.py     # 模型执行器
│   │   ├── scheduler.py        # 序列调度器
│   │   ├── sequence.py         # 序列数据结构
│   │   └── block_manager.py    # KV缓存管理
│   ├── layers/                  # 神经网络层
│   │   ├── attention.py        # Flash Attention
│   │   ├── linear.py           # 张量并行线性层
│   │   ├── sampler.py          # 采样器
│   │   ├── embed_head.py       # 嵌入层和LM头
│   │   ├── rotary_embedding.py # RoPE位置编码
│   │   ├── layernorm.py        # RMSNorm
│   │   └── activation.py       # 激活函数
│   ├── models/                  # 模型实现
│   │   └── qwen3.py            # Qwen3模型
│   └── utils/                   # 工具模块
│       ├── loader.py           # 权重加载
│       └── context.py          # 全局上下文
├── example.py                   # 使用示例
└── bench.py                     # 性能测试
```

### 架构分层

```
┌─────────────────────────────────────────┐
│          用户API层 (llm.py)              │
│   LLM.generate(prompts, params)         │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│       引擎层 (engine/llm_engine.py)      │
│  - 请求管理                              │
│  - 多进程协调                            │
│  - 生成流程控制                          │
└─────────┬───────────────┬───────────────┘
          │               │
     ┌────▼────┐    ┌────▼─────┐
     │Scheduler│    │ModelRunner│
     │  调度器  │    │  执行器   │
     └────┬────┘    └────┬──────┘
          │              │
     ┌────▼────────┐     │
     │BlockManager │     │
     │ KV缓存管理   │     │
     └─────────────┘     │
                    ┌────▼────────────┐
                    │  模型层          │
                    │  - Qwen3Model   │
                    │  - Attention    │
                    │  - MLP          │
                    └─────────────────┘
```

---

## 核心模块详解

### 1. 配置系统 (config.py)

配置系统定义了推理引擎的所有参数：

```python
@dataclass
class Config:
    model: str                          # 模型路径
    max_num_batched_tokens: int = 16384 # 最大批处理token数
    max_num_seqs: int = 512             # 最大并发序列数
    max_model_len: int = 4096           # 模型最大长度
    gpu_memory_utilization: float = 0.9 # GPU内存利用率
    tensor_parallel_size: int = 1       # 张量并行大小(1-8)
    kvcache_block_size: int = 256       # KV缓存块大小
    enforce_eager: bool = False         # 是否禁用CUDA Graph
```

**关键配置说明**：

- `max_num_batched_tokens`: 控制每次前向传播的最大token数，影响prefill阶段的吞吐量
- `max_num_seqs`: 限制同时运行的序列数，影响内存占用
- `kvcache_block_size`: KV缓存的块大小，256是一个经验值，平衡内存利用率和管理开销
- `gpu_memory_utilization`: 控制KV缓存占用的GPU内存比例

### 2. 序列管理 (engine/sequence.py)

`Sequence` 类是推理过程中最基本的数据单元：

```python
class Sequence:
    def __init__(self, prompt_token_ids: list[int], sampling_params: SamplingParams):
        self.seq_id = next_seq_id()
        self.status = SequenceStatus.WAITING  # WAITING -> RUNNING -> FINISHED
        self.prompt_token_ids = prompt_token_ids
        self.completion_token_ids = []

        # KV缓存管理
        self.block_table: list[int] = []      # 分配的KV缓存块ID列表
        self.num_cached_tokens = 0            # Prefix Caching: 已缓存的token数

        # 采样参数
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```

**核心属性**：

1. **状态管理** (`status`):
   - `WAITING`: 在等待队列中
   - `RUNNING`: 正在推理
   - `FINISHED`: 已完成

2. **Token管理**:
   - `prompt_token_ids`: 输入提示
   - `completion_token_ids`: 生成的token
   - `token_ids`: 两者的组合（只读属性）

3. **KV缓存**:
   - `block_table`: 映射到物理KV缓存块的表
   - `num_cached_tokens`: 支持Prefix Caching，记录已缓存的token数

4. **块管理**:
```python
@property
def num_blocks(self) -> int:
    """计算需要的块数"""
    return (len(self) + block_size - 1) // block_size

def block(self, i: int) -> list[int]:
    """获取第i个块的token"""
    start = i * block_size
    end = min(start + block_size, len(self))
    return self.token_ids[start:end]
```

### 3. 块管理器 (engine/block_manager.py)

`BlockManager` 实现了 Paged Attention 的核心机制，使用块来管理 KV 缓存。

#### 3.1 数据结构

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0        # 引用计数（支持共享）
        self.hash = -1            # 块内容的哈希值
        self.token_ids = []       # 块中的token（用于验证）

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}  # 哈希到块ID的映射
        self.free_block_ids: deque[int] = deque()   # 空闲块队列
        self.used_block_ids: set[int] = set()       # 使用中的块集合
```

#### 3.2 Prefix Caching 实现

**哈希计算**（使用 xxhash 快速哈希）:

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    """计算token序列的哈希值，支持链式哈希"""
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))  # 包含前一个块的哈希
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

**块分配流程**（支持缓存命中）:

```python
def allocate(self, seq: Sequence):
    """为序列分配KV缓存块"""
    h = -1
    cache_miss = False

    for i in range(seq.num_blocks):
        token_ids = seq.block(i)  # 获取第i个块的token

        # 只有完整的块才计算哈希
        if len(token_ids) == self.block_size:
            h = self.compute_hash(token_ids, h)  # 链式哈希
            block_id = self.hash_to_block_id.get(h, -1)

            # 验证哈希命中（防止哈希冲突）
            if block_id != -1 and self.blocks[block_id].token_ids == token_ids:
                # 缓存命中！
                if not cache_miss:
                    seq.num_cached_tokens += self.block_size
                    if block_id in self.used_block_ids:
                        # 块正在使用，增加引用计数
                        self.blocks[block_id].ref_count += 1
                    else:
                        # 块在缓存中但未使用，分配它
                        self._allocate_block(block_id)
                else:
                    # 前面的块缓存未命中，后续块也不能复用
                    block_id = self._allocate_new_block()
            else:
                # 缓存未命中
                cache_miss = True
                block_id = self._allocate_new_block()

            # 更新块的哈希信息
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        else:
            # 非完整块，直接分配
            h = -1
            block_id = self._allocate_new_block()

        seq.block_table.append(block_id)
```

**关键设计**：
- **链式哈希**: 每个块的哈希包含前一个块的哈希，确保前缀唯一性
- **缓存验证**: 哈希命中后还要验证 token_ids，防止哈希冲突
- **引用计数**: 支持多个序列共享相同的缓存块

#### 3.3 动态块追加

在 decode 阶段，每次生成一个 token，需要动态更新块表：

```python
def may_append(self, seq: Sequence):
    """在decode阶段动态追加块"""
    block_table = seq.block_table
    last_block = self.blocks[block_table[-1]]

    if len(seq) % self.block_size == 1:
        # 上一个块刚填满，需要分配新块
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        block_table.append(block_id)

    elif len(seq) % self.block_size == 0:
        # 当前块刚填满，计算并保存哈希
        token_ids = seq.block(seq.num_blocks - 1)
        prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
        h = self.compute_hash(token_ids, prefix)
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id
```

### 4. 调度器 (engine/scheduler.py)

`Scheduler` 是推理引擎的"大脑"，负责决定何时执行哪些序列。

#### 4.1 两阶段调度策略

```python
class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.block_manager = BlockManager(...)
        self.waiting: deque[Sequence] = deque()  # 等待队列
        self.running: deque[Sequence] = deque()  # 运行队列
```

**调度流程**：

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    """调度下一批要执行的序列

    Returns:
        (scheduled_seqs, is_prefill): 调度的序列列表和是否为prefill阶段
    """
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0

    # ========== 阶段1: Prefill 调度 ==========
    # 尝试从waiting队列中调度新序列
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]

        # 检查资源限制
        new_tokens = len(seq) - seq.num_cached_tokens  # 需要计算的token数
        if num_batched_tokens + new_tokens > self.max_num_batched_tokens:
            break  # 超出批处理限制

        if not self.block_manager.can_allocate(seq):
            break  # KV缓存块不足

        # 分配资源并调度
        self.block_manager.allocate(seq)
        num_batched_tokens += new_tokens
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
        num_seqs += 1

    if scheduled_seqs:
        return scheduled_seqs, True  # Prefill阶段

    # ========== 阶段2: Decode 调度 ==========
    # 从running队列中调度序列（每个序列生成1个token）
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()

        # 检查是否有足够的块来追加新token
        while not self.block_manager.can_append(seq):
            if self.running:
                # 抢占优先级最低的序列（队尾）
                self.preempt(self.running.pop())
            else:
                # 只剩这一个序列也无法运行，抢占它
                self.preempt(seq)
                break
        else:
            # 有足够资源，调度这个序列
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
            num_seqs += 1

    # 将调度的序列放回running队列前面
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False  # Decode阶段
```

**关键机制**：

1. **Prefill 优先**: 总是先尝试调度新序列，充分利用GPU并行计算能力
2. **资源检查**: 同时检查token数量限制和KV缓存块数量
3. **抢占策略**: 当资源不足时，抢占优先级最低的序列（FIFO队列的末尾）
4. **Prefix Caching**: 通过 `seq.num_cached_tokens` 减少实际需要计算的token数

#### 4.2 后处理

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
    """处理采样结果，更新序列状态"""
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)

        # 检查是否完成
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)  # 释放KV缓存
            self.running.remove(seq)
```

### 5. 模型执行器 (engine/model_runner.py)

`ModelRunner` 负责实际的模型推理执行，是最复杂的组件。

#### 5.1 初始化流程

```python
class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.rank = rank
        self.world_size = config.tensor_parallel_size

        # 1. 初始化分布式环境
        dist.init_process_group("nccl", "tcp://localhost:2333",
                               world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        # 2. 加载模型
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()

        # 3. 预热模型（用于计算内存占用）
        self.warmup_model()

        # 4. 分配KV缓存
        self.allocate_kv_cache()

        # 5. 捕获CUDA Graph（可选）
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 6. Rank > 0 的进程进入消息循环
        if self.world_size > 1 and rank > 0:
            self.loop()
```

#### 5.2 KV缓存分配

这是一个精妙的设计，动态计算可用的KV缓存块数：

```python
def allocate_kv_cache(self):
    """根据GPU内存动态分配KV缓存"""
    # 1. 获取GPU内存信息
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

    # 2. 计算单个块的内存占用
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    # 每个块: 2(K,V) * 层数 * block_size * 头数 * head_dim * dtype大小
    block_bytes = (2 * hf_config.num_hidden_layers * self.block_size *
                   num_kv_heads * head_dim * hf_config.torch_dtype.itemsize)

    # 3. 计算可分配的块数
    # 可用内存 = 总内存 * 利用率 - 已用内存 - 峰值内存 + 当前内存
    available = total * config.gpu_memory_utilization - used - peak + current
    config.num_kvcache_blocks = int(available) // block_bytes

    # 4. 分配KV缓存张量
    # shape: [2, num_layers, num_blocks, block_size, num_heads, head_dim]
    self.kv_cache = torch.empty(
        2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
        self.block_size, num_kv_heads, head_dim
    )

    # 5. 将缓存绑定到各个attention层
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

**为什么这样计算**：
- `peak - current`: 预热时的峰值与当前的差值，表示模型推理需要的临时内存
- `total * utilization - used - (peak - current)`: 就是留给KV缓存的内存

#### 5.3 Prefill数据准备

Prefill阶段处理变长序列，需要精心准备输入数据：

```python
def prepare_prefill(self, seqs: list[Sequence]):
    """准备prefill阶段的输入

    Prefill特点：
    - 序列长度不同
    - 每个序列可能有部分token已缓存（Prefix Caching）
    - 使用Flash Attention的变长序列接口
    """
    input_ids = []
    positions = []
    cu_seqlens_q = [0]  # Query的累积序列长度
    cu_seqlens_k = [0]  # Key的累积序列长度
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []

    for seq in seqs:
        seqlen = len(seq)

        # 只计算未缓存的token
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(list(range(seq.num_cached_tokens, seqlen)))

        seqlen_q = seqlen - seq.num_cached_tokens  # 实际查询长度
        seqlen_k = seqlen                           # KV缓存长度（包含缓存）

        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)

        # 构建slot_mapping（token到KV缓存块的映射）
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            block_id = seq.block_table[i]
            start = block_id * self.block_size
            if i != seq.num_blocks - 1:
                # 完整块
                end = start + self.block_size
                slot_mapping.extend(list(range(start, end)))
            else:
                # 最后一个块（可能不完整）
                end = start + (seqlen - i * self.block_size)
                slot_mapping.extend(list(range(start, end)))

    # 转换为CUDA张量
    input_ids = torch.tensor(input_ids, dtype=torch.int64).cuda()
    positions = torch.tensor(positions, dtype=torch.int64).cuda()
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64).cuda()
    block_tables = self.prepare_block_tables(seqs)

    # 设置全局上下文（供attention层使用）
    set_context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
    )

    return input_ids, positions
```

**关键点**：
- `cu_seqlens`: 累积序列长度，Flash Attention 需要这个来处理变长序列
- `slot_mapping`: 将 token 位置映射到 KV 缓存的物理位置
- Prefix Caching: 通过 `seq.num_cached_tokens` 跳过已缓存的 token

#### 5.4 Decode数据准备

Decode阶段所有序列都只生成1个token，数据准备更简单：

```python
def prepare_decode(self, seqs: list[Sequence]):
    """准备decode阶段的输入

    Decode特点：
    - 所有序列长度相同（都是1个token）
    - 批大小固定，可以使用CUDA Graph
    """
    num_seqs = len(seqs)
    input_ids = torch.tensor([seq[-1] for seq in seqs], dtype=torch.int64)
    positions = torch.tensor([len(seq) - 1 for seq in seqs], dtype=torch.int64)

    # slot_mapping: 每个序列的最后一个token在KV缓存中的位置
    slot_mapping = []
    for seq in seqs:
        block_id = seq.block_table[-1]
        offset = (len(seq) - 1) % self.block_size
        slot_mapping.append(block_id * self.block_size + offset)

    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)
    block_tables = self.prepare_block_tables(seqs)

    set_context(
        is_prefill=False,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
    )

    return input_ids.cuda(), positions.cuda()
```

#### 5.5 CUDA Graph 优化

CUDA Graph 是 decode 阶段的关键优化，通过预先捕获计算图来消除kernel启动开销：

```python
def capture_cudagraph(self):
    """捕获不同批大小的CUDA Graph"""
    self.graphs = {}
    self.graph_pool = {}

    # 预定义的批大小（2的幂次）
    batch_sizes = [1, 2, 4] + [i * 8 for i in range(1, 33)]

    for bs in batch_sizes:
        if bs > self.config.max_num_seqs:
            break

        # 创建虚拟序列
        seqs = [Sequence([0] * self.block_size) for _ in range(bs)]
        for seq in seqs:
            self.block_manager.allocate(seq)
            self.block_manager.may_append(seq)

        # 预热
        for _ in range(3):
            input_ids, positions = self.prepare_decode(seqs)
            self.run_model(input_ids, positions, use_cudagraph=False)

        # 捕获图
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool.get(bs)):
            input_ids, positions = self.prepare_decode(seqs)
            logits = self.run_model(input_ids, positions, use_cudagraph=True)

        self.graphs[bs] = (graph, input_ids, positions, logits)

        # 清理
        for seq in seqs:
            self.block_manager.deallocate(seq)
        reset_context()
```

**使用CUDA Graph**：

```python
def run_model_with_cudagraph(self, seqs):
    bs = len(seqs)
    if bs in self.graphs:
        graph, input_ids_buf, positions_buf, logits_buf = self.graphs[bs]

        # 复制数据到预分配的buffer
        input_ids, positions = self.prepare_decode(seqs)
        input_ids_buf.copy_(input_ids)
        positions_buf.copy_(positions)

        # Replay图（非常快！）
        graph.replay()

        return logits_buf
    else:
        # 批大小不匹配，使用eager模式
        return self.run_model_eager(seqs)
```

### 6. 引擎主控制器 (engine/llm_engine.py)

`LLMEngine` 协调所有组件，提供统一的生成接口。

#### 6.1 多进程管理

对于 Tensor Parallelism，需要启动多个进程：

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        config = Config(model, **kwargs)

        # 启动worker进程（rank 1, 2, ...）
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # 用于同步
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # 主进程的ModelRunner（rank 0）
        self.model_runner = ModelRunner(config, 0, self.events)

        # 加载tokenizer和创建scheduler
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.scheduler = Scheduler(config)
```

#### 6.2 进程间通信

使用 SharedMemory 实现高效的进程间通信：

```python
# 在ModelRunner中
def __init__(self, config, rank, event):
    # ...
    if self.world_size > 1:
        if rank == 0:
            # 主进程创建共享内存
            self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
        else:
            # worker进程连接共享内存
            self.shm = SharedMemory(name="nanovllm")
            self.loop()  # 进入消息循环

def write_shm(self, method_name, *args):
    """主进程写入命令"""
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        event.set()  # 通知worker进程

def read_shm(self):
    """worker进程读取命令"""
    self.event.wait()  # 等待通知
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args

def loop(self):
    """worker进程的消息循环"""
    while True:
        method_name, args = self.read_shm()
        self.call(method_name, *args)
        if method_name == "exit":
            break
```

#### 6.3 生成流程

```python
def generate(
    self,
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
    use_tqdm: bool = True,
) -> list[dict]:
    """批量生成接口

    Returns:
        list[dict]: [{"text": str, "token_ids": list[int]}, ...]
    """
    # 1. 准备采样参数
    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)

    # 2. 添加所有请求到调度器
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)

    # 3. 逐步执行直到所有序列完成
    outputs = {}
    while not self.is_finished():
        output, num_tokens = self.step()

        # 收集完成的序列
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids

    # 4. 按照输入顺序返回结果
    outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
    outputs = [
        {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
        for token_ids in outputs
    ]
    return outputs

def step(self):
    """执行一步推理"""
    # 1. 调度
    seqs, is_prefill = self.scheduler.schedule()

    # 2. 执行模型
    token_ids = self.model_runner.call("run", seqs, is_prefill)

    # 3. 后处理
    self.scheduler.postprocess(seqs, token_ids)

    # 4. 返回完成的序列
    outputs = [(seq.seq_id, seq.completion_token_ids)
               for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

---

## 执行流程分析

### 完整执行流程图

```
用户调用 llm.generate(prompts, sampling_params)
│
├─> 1. 添加请求阶段
│   └─> for prompt in prompts:
│       └─> add_request(prompt, sampling_params)
│           ├─> tokenizer.encode(prompt) → token_ids
│           ├─> seq = Sequence(token_ids, sampling_params)
│           └─> scheduler.add(seq)  # 加入waiting队列
│
├─> 2. 迭代推理阶段 (while not is_finished())
│   │
│   └─> step():
│       │
│       ├─> 2.1 调度阶段: scheduler.schedule()
│       │   │
│       │   ├─> 尝试Prefill调度（新序列）
│       │   │   ├─> 检查资源: token数 + KV缓存块
│       │   │   ├─> block_manager.allocate(seq)
│       │   │   │   ├─> 计算哈希: hash(token_ids)
│       │   │   │   ├─> 查找缓存: hash_to_block_id[hash]
│       │   │   │   ├─> 命中缓存: 复用块，ref_count++
│       │   │   │   └─> 未命中: 分配新块
│       │   │   └─> seq.status = RUNNING
│       │   │
│       │   └─> 或Decode调度（运行中序列）
│       │       ├─> 检查资源: KV缓存块
│       │       ├─> 不足时抢占低优先级序列
│       │       └─> block_manager.may_append(seq)
│       │           └─> 块满时分配新块
│       │
│       ├─> 2.2 执行阶段: model_runner.run(seqs, is_prefill)
│       │   │
│       │   ├─> if is_prefill:
│       │   │   ├─> prepare_prefill(seqs)
│       │   │   │   ├─> 构建input_ids（跳过cached tokens）
│       │   │   │   ├─> 构建positions
│       │   │   │   ├─> 构建cu_seqlens（累积长度）
│       │   │   │   ├─> 构建slot_mapping（token→cache映射）
│       │   │   │   └─> set_context(...)
│       │   │   └─> run_model_eager(input_ids, positions)
│       │   │
│       │   └─> else:  # Decode
│       │       ├─> prepare_decode(seqs)
│       │       │   ├─> input_ids = [seq[-1] for seq in seqs]
│       │       │   ├─> positions = [len(seq)-1 for seq in seqs]
│       │       │   ├─> 构建slot_mapping
│       │       │   └─> set_context(...)
│       │       └─> if use_cudagraph and bs in graphs:
│       │           ├─> 复制数据到graph buffer
│       │           └─> graph.replay()  # 超快！
│       │           └─> else: run_model_eager(...)
│       │
│       ├─> 2.3 模型前向传播
│       │   │
│       │   └─> model(input_ids, positions)
│       │       ├─> embed_tokens(input_ids) → hidden_states
│       │       │
│       │       └─> for layer in layers:
│       │           ├─> input_layernorm(hidden_states)
│       │           │
│       │           ├─> attention(hidden_states, positions)
│       │           │   ├─> qkv_proj(hidden_states) → q, k, v
│       │           │   ├─> rotary_emb(q, k, positions)
│       │           │   ├─> store_kvcache(k, v, slot_mapping)
│       │           │   │   └─> Triton kernel写入KV cache
│       │           │   │
│       │           │   └─> if is_prefill:
│       │           │       └─> flash_attn_varlen_func(q, k, v, cu_seqlens, ...)
│       │           │       └─> else:
│       │           │           └─> flash_attn_with_kvcache(q, k_cache, v_cache, ...)
│       │           │
│       │           ├─> post_attention_layernorm(...)
│       │           │
│       │           └─> mlp(hidden_states)
│       │               ├─> gate_up_proj(x) → gate, up
│       │               ├─> silu(gate) * up
│       │               └─> down_proj(...)
│       │
│       ├─> 2.4 采样阶段: sampler(logits)
│       │   ├─> logits /= temperature
│       │   ├─> probs = softmax(logits)
│       │   └─> token_ids = sample(probs)  # Gumbel-max采样
│       │
│       └─> 2.5 后处理: scheduler.postprocess(seqs, token_ids)
│           ├─> seq.append_token(token_id)
│           └─> if is_finished:
│               ├─> seq.status = FINISHED
│               ├─> block_manager.deallocate(seq)
│               │   └─> 释放块，ref_count--
│               └─> running.remove(seq)
│
└─> 3. 返回结果
    └─> [{"text": tokenizer.decode(tokens), "token_ids": tokens}, ...]
```

### 典型场景示例

#### 场景1: 单个序列生成

```
Prompt: "Hello" (token_ids: [1, 2, 3])
max_tokens: 5
block_size: 256

时刻T0: 添加请求
  waiting: [seq0]
  running: []

时刻T1: Prefill
  schedule() → [seq0], is_prefill=True
  block_manager.allocate(seq0)
    → 分配1个块: block_table=[0]
  model.forward([1,2,3]) → logits
  sampler(logits) → token_id=4
  seq0.token_ids = [1,2,3,4]
  waiting: []
  running: [seq0]

时刻T2-T5: Decode (生成4个token)
  schedule() → [seq0], is_prefill=False
  model.forward([4]) → logits → token_id=5
  seq0.token_ids = [1,2,3,4,5]
  ... (重复3次)
  seq0.token_ids = [1,2,3,4,5,6,7,8]
  num_completion_tokens=5 → FINISHED

返回: {"text": "Hello world", "token_ids": [4,5,6,7,8]}
```

#### 场景2: Prefix Caching

```
Prompt1: "System: You are a helpful assistant.\nUser: Hello" (100 tokens)
Prompt2: "System: You are a helpful assistant.\nUser: Hi"    (99 tokens)
block_size: 32

Prompt1 执行:
  allocate() → 分配4个块
    Block0: tokens[0:32],   hash=H0
    Block1: tokens[32:64],  hash=H1 (包含H0)
    Block2: tokens[64:96],  hash=H2 (包含H1)
    Block3: tokens[96:100], hash=-1 (不完整)
  hash_to_block_id = {H0: 0, H1: 1, H2: 2}

Prompt2 执行:
  allocate() → 检查缓存
    Block0: hash=H0 → 命中! 复用block0, ref_count=2
    Block1: hash=H1 → 命中! 复用block1, ref_count=2
    Block2: hash=H2' → 未命中（第3个块内容不同）
           → 分配新块block4
    Block3: tokens[96:99], hash=-1
  num_cached_tokens = 64  # 前2个块命中
  只需计算后35个token！
```

#### 场景3: Continuous Batching

```
时间线:
T0: Req1到达 (prompt: 10 tokens, max: 50)
T1: Prefill Req1 (10 tokens)
T2: Decode Req1 (1 token), Req2到达 (prompt: 20 tokens)
T3: Prefill Req2 (20 tokens)
T4: Decode Req1 + Req2 (2 tokens, batched!)
T5: Decode Req1 + Req2 (2 tokens)
...
T20: Req1完成, Decode Req2 (1 token)
...
T50: Req2完成

关键: 新请求立即开始prefill，完成后join到decode batch
```

---

## 关键技术实现

### 1. Flash Attention (layers/attention.py)

Flash Attention 是现代LLM推理的核心优化，通过优化内存访问模式大幅加速attention计算。

```python
class Attention(nn.Module):
    def __init__(self, config):
        self.num_heads = config.num_attention_heads // world_size
        self.num_kv_heads = config.num_key_value_heads // world_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        # KV缓存（由ModelRunner分配）
        self.k_cache = None  # [num_blocks, block_size, num_kv_heads, head_dim]
        self.v_cache = None

    def forward(self, hidden_states, positions):
        ctx = get_context()

        # 1. QKV投影（使用张量并行）
        q, k, v = self.qkv_proj(hidden_states)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # 2. RoPE位置编码
        q, k = self.rotary_emb(q, k, positions)

        # 3. 存储KV到缓存
        store_kvcache(k, v, self.k_cache, self.v_cache, ctx.slot_mapping)

        # 4. Attention计算
        if ctx.is_prefill:
            # Prefill: 变长序列attention
            out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=ctx.cu_seqlens_q,
                cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                causal=True,
            )
        else:
            # Decode: 使用KV缓存
            out = flash_attn_with_kvcache(
                q, self.k_cache, self.v_cache,
                block_table=ctx.block_tables,
                cache_seqlens=None,  # 自动从block_table推导
                causal=True,
            )

        # 5. 输出投影
        out = out.view(-1, self.num_heads * self.head_dim)
        return self.o_proj(out)
```

**store_kvcache Kernel** (Triton实现):

```python
@triton.jit
def store_kvcache_kernel(
    k_ptr, v_ptr,           # 输入K,V
    k_cache_ptr, v_cache_ptr,  # KV缓存
    slot_mapping_ptr,       # token到缓存的映射
    num_tokens, num_heads, head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个program处理一个token的一个head
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # 获取缓存位置
    slot = tl.load(slot_mapping_ptr + token_id)

    # 计算指针偏移
    k_offset = (token_id * num_heads + head_id) * head_dim
    cache_offset = (slot * num_heads + head_id) * head_dim

    # 写入缓存（向量化）
    for i in range(0, head_dim, BLOCK_SIZE):
        mask = i + tl.arange(0, BLOCK_SIZE) < head_dim
        k_vec = tl.load(k_ptr + k_offset + i, mask=mask)
        v_vec = tl.load(v_ptr + k_offset + i, mask=mask)
        tl.store(k_cache_ptr + cache_offset + i, k_vec, mask=mask)
        tl.store(v_cache_ptr + cache_offset + i, v_vec, mask=mask)
```

### 2. Tensor Parallelism (layers/linear.py)

张量并行通过切分模型权重到多个GPU，实现模型并行推理。

```python
class ColumnParallelLinear(nn.Module):
    """列并行线性层

    权重切分: W [D_in, D_out] → W_i [D_in, D_out/N]
    输出: Y = X @ W → Y_i = X @ W_i (每个rank计算部分输出)
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features // world_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, x):
        return F.linear(x, self.weight)

    def weight_loader(self, param, loaded_weight, shard_id):
        """加载权重时自动切分"""
        start = shard_id * self.out_features
        end = start + self.out_features
        param.data.copy_(loaded_weight[start:end])

class RowParallelLinear(nn.Module):
    """行并行线性层

    权重切分: W [D_in, D_out] → W_i [D_in/N, D_out]
    输出: Y = X @ W → Y = sum(X_i @ W_i) (需要all-reduce)
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features // world_size
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, x):
        out = F.linear(x, self.weight)
        # All-reduce求和（跨所有GPU）
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out
```

**典型使用模式**：

```python
class TransformerMLP(nn.Module):
    def __init__(self, config):
        # gate和up投影: 列并行
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, 2 * intermediate_size
        )
        # down投影: 行并行
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size
        )

    def forward(self, x):
        # x: [batch, hidden_size]
        gate_up = self.gate_up_proj(x)  # [batch, 2*intermediate_size/N]
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.silu(gate) * up
        out = self.down_proj(x)  # [batch, hidden_size], all-reduced
        return out
```

### 3. RoPE 位置编码 (layers/rotary_embedding.py)

RoPE (Rotary Position Embedding) 通过旋转实现相对位置编码。

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_position_embeddings, base=10000):
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # 预计算cos/sin表
        t = torch.arange(max_position_embeddings)
        freqs = torch.outer(t, inv_freq)  # [max_pos, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_pos, head_dim]
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    @torch.compile
    def forward(self, q, k, positions):
        """应用RoPE到query和key

        Args:
            q, k: [num_tokens, num_heads, head_dim]
            positions: [num_tokens]
        """
        cos = self.cos_cached[positions]  # [num_tokens, head_dim]
        sin = self.sin_cached[positions]

        # 旋转变换: (x, y) → (x*cos - y*sin, x*sin + y*cos)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

def rotate_half(x):
    """将向量分成两半并旋转: [x1, x2] → [-x2, x1]"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)
```

### 4. Sampler 采样器 (layers/sampler.py)

采样器负责根据logits生成下一个token。

```python
class Sampler(nn.Module):
    @torch.compile  # 使用torch.compile加速
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """使用Gumbel-max技巧采样

        Args:
            logits: [batch_size, vocab_size]
            temperatures: [batch_size]

        Returns:
            token_ids: [batch_size]
        """
        # 1. 温度缩放
        logits = logits / temperatures.unsqueeze(-1)

        # 2. Gumbel-max采样
        # 标准方法: probs = softmax(logits); sample(probs)
        # Gumbel-max: argmax(logits + gumbel_noise)
        #           = argmax(log(probs) + gumbel)
        #           ≈ argmax(probs / exponential(1))  # 数值更稳定
        probs = torch.softmax(logits, dim=-1)
        q = torch.empty_like(probs).exponential_(1.0)
        token_ids = torch.argmax(probs / q, dim=-1)

        return token_ids
```

**为什么用Gumbel-max**：
- 避免显式采样（无需torch.multinomial，对CUDA Graph更友好）
- 数值稳定（避免log(softmax(...))的精度问题）
- 更快（一次argmax vs 多次随机采样）

### 5. 模型权重加载 (utils/loader.py)

支持从HuggingFace格式加载并自动处理张量并行切分。

```python
def load_model(model, model_path):
    """从safetensors文件加载模型权重"""
    # 1. 获取所有参数
    params_dict = {name: param for name, param in model.named_parameters()}

    # 2. 遍历safetensors文件
    for filename in os.listdir(model_path):
        if not filename.endswith('.safetensors'):
            continue

        filepath = os.path.join(model_path, filename)
        with safe_open(filepath, framework="pt") as f:
            for name in f.keys():
                loaded_weight = f.get_tensor(name)

                # 3. 映射权重名称（处理fused层）
                mapped_name = map_weight_name(name, model.packed_modules_mapping)

                # 4. 获取目标参数
                if mapped_name not in params_dict:
                    continue

                param = params_dict[mapped_name]

                # 5. 加载权重（自动处理张量并行）
                if hasattr(param, 'weight_loader'):
                    # 自定义加载器（处理权重切分）
                    param.weight_loader(param, loaded_weight, rank)
                else:
                    # 直接复制
                    param.data.copy_(loaded_weight)
```

**Packed Modules 处理**：

某些模型会将多个权重矩阵合并存储以提高效率：

```python
# HuggingFace格式:
# - q_proj.weight: [hidden_size, hidden_size]
# - k_proj.weight: [hidden_size, kv_size]
# - v_proj.weight: [hidden_size, kv_size]

# Nano-vLLM格式 (fused):
# - qkv_proj.weight: [hidden_size, hidden_size + 2*kv_size]

packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

# 加载时自动拼接
def weight_loader(param, loaded_weight, shard_id):
    # loaded_weight可能是q_proj, k_proj或v_proj之一
    # 需要知道在qkv_proj中的位置
    if name == "q_proj":
        start, end = 0, hidden_size
    elif name == "k_proj":
        start, end = hidden_size, hidden_size + kv_size
    else:  # v_proj
        start, end = hidden_size + kv_size, hidden_size + 2*kv_size

    # 同时处理张量并行切分
    shard_start = start + shard_id * (end - start) // world_size
    shard_end = start + (shard_id + 1) * (end - start) // world_size
    param.data[shard_start:shard_end].copy_(loaded_weight)
```

---

## 性能优化技术

### 1. Prefix Caching

**原理**: 相同的prompt前缀只需计算一次，后续请求可以复用KV缓存。

**实现**:
- 使用xxhash计算每个块的哈希值
- 哈希包含前缀依赖（链式哈希），确保前缀一致性
- 引用计数管理共享块

**效果**:
- 共享系统提示词的请求可节省90%+的prefill计算
- 例如: 1000个请求共享100 token的系统提示，只需计算一次

### 2. Continuous Batching

**原理**: 动态批处理，新请求立即开始，完成的请求立即释放资源。

**实现**:
- Prefill和Decode分开调度
- 新请求在下一个step立即开始prefill
- Decode阶段动态组batch

**效果**:
- GPU利用率提高30-50%
- 平均延迟降低
- 吞吐量提高

### 3. Paged Attention

**原理**: 将KV缓存分块管理，按需分配，避免内存碎片。

**实现**:
- KV缓存划分为固定大小的块（如256 tokens）
- 使用block_table映射逻辑位置到物理位置
- 动态分配和释放

**效果**:
- 内存利用率提高60-80%
- 支持更大的batch size
- 避免预留固定大小的缓存

### 4. CUDA Graph

**原理**: 预先捕获计算图，消除kernel启动开销。

**实现**:
- Decode阶段批大小固定，适合使用CUDA Graph
- 预先捕获常见batch size的图
- 运行时replay图，无需重新调度kernel

**效果**:
- Decode延迟降低20-40%
- 对小batch效果更明显
- RTX 4070上: 1个token从2ms降到1.2ms

### 5. Torch Compile

**原理**: 使用torch.compile将Python代码编译为优化的kernel。

**实现**:
```python
@torch.compile
def rotary_embedding_forward(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**效果**:
- 消除Python开销
- 算子融合
- 小kernel加速10-30%

### 6. Tensor Parallelism

**原理**: 将模型权重切分到多个GPU，并行计算。

**实现**:
- 列并行: 切分输出维度
- 行并行: 切分输入维度，需all-reduce
- Attention: 切分注意力头

**效果**:
- 支持更大模型
- 推理加速接近线性（通信开销小）
- 4卡TP: 吞吐量提升3.5-3.8x

---

## 代码示例与实践

### 基础使用

```python
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 1. 加载模型
model_path = "~/huggingface/Qwen3-0.6B/"
llm = LLM(
    model_path,
    enforce_eager=False,      # 启用CUDA Graph
    tensor_parallel_size=1,   # 单卡
    max_num_seqs=128,         # 最大并发序列
    gpu_memory_utilization=0.9,
)

# 2. 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 3. 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    ignore_eos=False,
)

# 4. 准备prompts
prompts = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about coding.",
]

# 应用chat template
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for prompt in prompts
]

# 5. 批量生成
outputs = llm.generate(prompts, sampling_params)

# 6. 输出结果
for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Response: {output['text']}")
    print(f"Tokens: {len(output['token_ids'])}")
    print("-" * 80)
```

### 多卡推理 (Tensor Parallelism)

```python
# 使用4张GPU
llm = LLM(
    model_path,
    tensor_parallel_size=4,  # 4卡并行
    gpu_memory_utilization=0.9,
)

# 其他代码相同
# 模型权重会自动切分到4张卡
# 前向传播自动协调多卡计算
```

### Prefix Caching 示例

```python
# 共享系统提示词的多个请求
system_prompt = """You are a helpful assistant specialized in Python programming.
Always provide code examples and explain your reasoning."""

user_queries = [
    "How do I read a file in Python?",
    "What's the difference between list and tuple?",
    "Explain Python decorators.",
]

prompts = []
for query in user_queries:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},  # 共享部分
            {"role": "user", "content": query},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts.append(prompt)

# 生成时，第2、3个请求会复用第1个请求的系统提示词KV缓存
outputs = llm.generate(prompts, sampling_params)

# 可以通过日志观察缓存命中率
```

### 性能测试

```python
import time
import numpy as np

# 生成测试数据
np.random.seed(42)
num_requests = 100
input_lens = np.random.randint(100, 1024, num_requests)
output_lens = np.random.randint(100, 512, num_requests)

prompts = []
sampling_params_list = []
for i in range(num_requests):
    # 生成指定长度的prompt
    prompt_tokens = [0] * input_lens[i]
    prompts.append(prompt_tokens)

    # 设置输出长度
    sp = SamplingParams(temperature=0.7, max_tokens=output_lens[i])
    sampling_params_list.append(sp)

# 开始测试
start_time = time.time()
outputs = llm.generate(prompts, sampling_params_list, use_tqdm=True)
end_time = time.time()

# 统计
total_tokens = sum(len(out['token_ids']) for out in outputs)
total_time = end_time - start_time
throughput = total_tokens / total_time

print(f"Total requests: {num_requests}")
print(f"Total tokens: {total_tokens}")
print(f"Total time: {total_time:.2f}s")
print(f"Throughput: {throughput:.2f} tokens/s")
```

---

## 总结

### Nano-vLLM 的核心设计

1. **模块化架构**: 清晰的分层设计，易于理解和扩展
2. **Paged Attention**: 块管理的KV缓存，提高内存利用率
3. **Continuous Batching**: 动态批处理，提高GPU利用率
4. **Prefix Caching**: 复用相同前缀，减少重复计算
5. **CUDA Graph**: 消除kernel启动开销
6. **Tensor Parallelism**: 支持多卡推理

### 学习路径建议

1. **第一步**: 理解基本数据流（Sequence → Scheduler → ModelRunner → Model）
2. **第二步**: 深入BlockManager，理解Paged Attention和Prefix Caching
3. **第三步**: 研究ModelRunner，理解prefill/decode的数据准备
4. **第四步**: 学习Attention实现，理解Flash Attention的使用
5. **第五步**: 探索Tensor Parallelism，理解多卡协调

### 与完整vLLM的差异

**Nano-vLLM 有的**:
- Prefix Caching
- Continuous Batching
- Paged Attention
- CUDA Graph
- Tensor Parallelism
- Flash Attention

**Nano-vLLM 没有的**:
- 推测解码 (Speculative Decoding)
- 多LoRA支持
- 量化 (AWQ, GPTQ)
- Pipeline Parallelism
- Ray分布式
- 流式输出
- 更多模型支持

### 扩展方向

学习完这个项目后，你可以尝试：
1. 添加新的采样策略（top-k, top-p, beam search）
2. 实现推测解码
3. 支持新的模型架构（Llama, Mistral等）
4. 添加量化支持
5. 实现流式输出API
6. 优化通信效率（Tensor Parallelism）

这个项目是理解现代LLM推理引擎的绝佳起点！
