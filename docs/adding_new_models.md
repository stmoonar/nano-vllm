# 如何为 nano-vllm 添加新模型支持

本文档以添加 Llama3.2 支持为例，详细说明如何为 nano-vllm 添加新的模型支持。

## 项目架构概述

nano-vllm 的模型相关代码主要分布在以下位置：

```
nanovllm/
├── models/                 # 模型实现
│   ├── qwen3.py           # Qwen3 模型
│   └── llama.py           # Llama 模型（新增）
├── engine/
│   └── model_runner.py    # 模型运行器，包含模型注册表
└── layers/                # 通用层实现
    ├── attention.py       # Flash-Attention
    ├── linear.py          # 张量并行线性层
    ├── embed_head.py      # 词表并行嵌入层
    ├── rotary_embedding.py # RoPE 位置编码
    ├── layernorm.py       # RMSNorm
    └── activation.py      # 激活函数
```

## 添加新模型的步骤

### 步骤 1：分析目标模型架构

在添加新模型前，需要了解目标模型的架构特点。以 Llama3.2 为例：

| 特性 | Llama3.2 | Qwen3 |
|------|----------|-------|
| LayerNorm | RMSNorm | RMSNorm |
| 位置编码 | RoPE | RoPE |
| 激活函数 | SiLU | SiLU |
| Attention | GQA | GQA |
| QKV Bias | 无 | 可选 |
| QK-Norm | 无 | 有 |

关键差异：
- Llama 没有 QK-Norm（Qwen3 有 `q_norm` 和 `k_norm`）
- Llama 没有 QKV bias
- 权重命名约定相同

### 步骤 2：创建模型实现文件

在 `nanovllm/models/` 目录下创建新的模型文件，例如 `llama.py`。

模型文件需要实现以下类：

```python
# nanovllm/models/llama.py

import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig  # 使用 HuggingFace 的配置类

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
```

#### 2.1 实现 Attention 层

```python
class LlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()

        # 计算张量并行后的头数
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads

        # QKV 投影（打包在一起）
        self.qkv_proj = QKVParallelLinear(
            hidden_size, self.head_dim,
            self.total_num_heads, self.total_num_kv_heads,
            bias=False,  # Llama 没有 bias
        )

        # 输出投影
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size, bias=False,
        )

        # RoPE 位置编码
        self.rotary_emb = get_rope(
            self.head_dim, rotary_dim=self.head_dim,
            max_position=max_position, base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # Flash-Attention
        self.attn = Attention(
            self.num_heads, self.head_dim,
            self.head_dim ** -0.5, self.num_kv_heads,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 注意：Llama 没有 QK-Norm，直接应用 RoPE
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))
```

#### 2.2 实现 MLP 层

```python
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        # Gate 和 Up 投影打包在一起
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        return self.down_proj(x)
```

#### 2.3 实现 Decoder Layer

```python
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        # Pre-LayerNorm 架构
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

#### 2.4 实现完整模型

```python
class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    # 权重映射：将 HuggingFace 的权重名映射到打包后的权重
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

### 步骤 3：注册模型

在 `nanovllm/engine/model_runner.py` 中注册新模型：

```python
from nanovllm.models.llama import LlamaForCausalLM

# 模型注册表
MODEL_REGISTRY = {
    "qwen3": Qwen3ForCausalLM,
    "qwen2": Qwen3ForCausalLM,
    "llama": LlamaForCausalLM,  # 新增
}
```

`get_model_class` 函数会根据 HuggingFace 配置的 `model_type` 字段自动选择正确的模型类。

### 步骤 4：测试新模型

使用以下代码测试新模型：

```python
from nanovllm import LLM, SamplingParams

# 使用 Llama 3.2 模型
llm = LLM(model="meta-llama/Llama-3.2-1B")

prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output)
```

## 关键概念说明

### packed_modules_mapping

这个字典定义了权重加载时的映射关系。nano-vllm 将多个权重矩阵打包在一起以提高效率：

- `qkv_proj`：包含 `q_proj`、`k_proj`、`v_proj`
- `gate_up_proj`：包含 `gate_proj`、`up_proj`

映射格式：
```python
{
    "原始权重名": ("打包后的权重名", 索引或标识符),
}
```

### 张量并行

所有线性层都支持张量并行：
- `ColumnParallelLinear`：按输出维度分片
- `RowParallelLinear`：按输入维度分片，需要 all_reduce
- `QKVParallelLinear`：专门用于 QKV 投影的并行层
- `MergedColumnParallelLinear`：合并多个 ColumnParallel 层

### 层实现要点

1. **Attention 层**：使用 `Attention` 类封装 Flash-Attention
2. **RoPE**：使用 `get_rope()` 获取位置编码
3. **LayerNorm**：使用 `RMSNorm`，支持融合残差加法
4. **激活函数**：使用 `SiluAndMul` 实现融合的 SiLU 激活

## 常见问题

### Q: 如何处理不同的配置参数名？

A: 使用 `getattr(config, "param_name", default_value)` 来兼容不同模型的配置命名。

### Q: 如何添加新的层类型？

A: 在 `nanovllm/layers/` 目录下创建新的层实现，并在模型中使用。

### Q: 模型加载失败怎么办？

A: 检查 `packed_modules_mapping` 是否正确，确保权重名称映射与 HuggingFace 模型一致。

## 总结

添加新模型支持的核心步骤：

1. 分析目标模型架构，确定与现有模型的差异
2. 在 `models/` 目录创建模型实现文件
3. 实现 Attention、MLP、DecoderLayer 和完整模型类
4. 定义正确的 `packed_modules_mapping`
5. 在 `model_runner.py` 的 `MODEL_REGISTRY` 中注册模型
6. 测试模型加载和推理
