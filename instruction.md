# 推测解码 (Speculative Decoding) 实现指导

## 目录

1. [什么是推测解码](#什么是推测解码)
2. [算法原理](#算法原理)
3. [实现架构设计](#实现架构设计)
4. [详细实现步骤](#详细实现步骤)
5. [代码实现指南](#代码实现指南)
6. [测试与验证](#测试与验证)
7. [性能优化](#性能优化)

---

## 什么是推测解码

### 背景

自回归语言模型（如GPT、Llama等）生成文本时，每次只能生成一个token，这个过程是串行的：

```
Prompt: "The capital of France is"

Step 1: Generate token1 = "Paris"
Step 2: Generate token2 = ","
Step 3: Generate token3 = "a"
Step 4: Generate token4 = "beautiful"
...
```

每一步都需要：
1. 完整的模型前向传播（计算密集）
2. 访问所有历史token的KV缓存（内存密集）
3. 等待前一个token生成完成（串行依赖）

**问题**: 对于大模型（如70B），单步推理可能需要几十到上百毫秒，生成速度慢。

### 推测解码的核心思想

**关键洞察**:
- 小模型（如0.6B）推理速度快（1-2ms/token），但质量略低
- 大模型（如70B）推理速度慢（50-100ms/token），但质量高
- 小模型的预测往往有一定准确性

**策略**:
1. 使用快速的小模型（draft model）**投机性地**预测接下来的 K 个token
2. 使用慢速的大模型（target model）**并行验证**这 K 个token
3. 接受正确的预测，拒绝错误的预测
4. 重复这个过程

**优势**:
- 当小模型预测准确时，一次可以生成多个token（加速）
- 当小模型预测错误时，仍然保证大模型的输出质量（不损失质量）
- 在数学上保证输出分布与原始大模型相同

### 性能提升

在典型场景下：
- **平均接受率**: 60-80%（取决于draft model质量和领域）
- **理论加速**: 如果预测K=4个token，接受率70%，则加速约 1 + 4*0.7 = 3.8x
- **实际加速**: 2-3x（考虑验证开销）

---

## 算法原理

### 基础算法

推测解码的核心是 **Modified Rejection Sampling**，确保输出分布不变。

#### 符号定义

- $p(x)$: 目标模型（大模型）的概率分布
- $q(x)$: draft模型（小模型）的概率分布
- $\gamma$: 投机步数（一次预测多少个token）
- $\alpha(x) = \min(1, \frac{p(x)}{q(x)})$: 接受概率

#### 算法流程

```python
def speculative_decoding(prompt, target_model, draft_model, gamma, max_tokens):
    """推测解码算法

    Args:
        prompt: 输入提示
        target_model: 目标模型（大模型）
        draft_model: draft模型（小模型）
        gamma: 每次投机的步数
        max_tokens: 最大生成token数
    """
    tokens = tokenize(prompt)

    while len(tokens) < max_tokens:
        # ============ 步骤1: Draft阶段 ============
        # 使用小模型自回归生成gamma个token
        draft_tokens = []
        draft_probs = []

        for i in range(gamma):
            q_dist = draft_model(tokens + draft_tokens)  # 概率分布
            x = sample(q_dist)                            # 采样
            draft_tokens.append(x)
            draft_probs.append(q_dist)

        # ============ 步骤2: Verification阶段 ============
        # 使用大模型并行验证所有draft token
        # 输入: [tokens..., draft_tokens[0], draft_tokens[1], ..., draft_tokens[gamma-1]]
        # 输出: gamma个位置的概率分布
        target_probs = target_model(tokens + draft_tokens[:-1])  # gamma个分布

        # ============ 步骤3: 接受/拒绝 ============
        accepted = []
        for i in range(gamma):
            x = draft_tokens[i]
            q = draft_probs[i][x]  # draft模型对x的概率
            p = target_probs[i][x]  # target模型对x的概率

            # 计算接受概率
            alpha = min(1.0, p / q)

            # 以概率alpha接受
            if random.random() < alpha:
                accepted.append(x)
            else:
                # 拒绝：从修正分布中采样一个新token
                p_adjusted = adjust_distribution(target_probs[i], draft_probs[i])
                x_new = sample(p_adjusted)
                accepted.append(x_new)
                break  # 停止接受后续token

        # ============ 步骤4: Bonus token ============
        # 如果所有draft token都被接受，从最后一个位置的target分布再采样一个token
        if len(accepted) == gamma:
            bonus = sample(target_probs[-1])
            accepted.append(bonus)

        tokens.extend(accepted)

    return tokens

def adjust_distribution(p_dist, q_dist):
    """计算修正分布: p'(x) = norm(max(0, p(x) - q(x)))"""
    adjusted = torch.clamp(p_dist - q_dist, min=0)
    adjusted = adjusted / adjusted.sum()  # 归一化
    return adjusted
```

### 为什么这个算法正确？

**定理**: 推测解码的输出分布与直接使用target模型采样的分布相同。

**证明思路**:

对于每个位置的token x:
- 被接受的概率: $\alpha(x) \cdot q(x) = \min(p(x), q(x))$
- 被拒绝后从修正分布采样的概率: $(1-\alpha(x)) \cdot q(x) \cdot \frac{p(x)-q(x)}{Z}$

其中修正分布:
$$p'(x) = \max(0, p(x) - q(x)) / Z$$

总概率:
$$\min(p(x), q(x)) + (1 - \min(1, \frac{p(x)}{q(x)})) \cdot q(x) \cdot \frac{p(x) - q(x)}{Z}$$

当 $p(x) \geq q(x)$ 时:
$$q(x) + (1 - \frac{q(x)}{p(x)}) \cdot q(x) \cdot \frac{p(x) - q(x)}{Z} = q(x) + \frac{p(x)-q(x)}{Z} \cdot (p(x) - q(x)) = p(x)$$

因此输出分布等于 $p(x)$！

### 实际优化

基础算法有一些低效之处，实际实现中需要优化：

#### 优化1: 树状并行验证

基础算法中，draft阶段仍然是串行的（一次生成一个token）。可以改进为：

```
Draft阶段（串行）:
  t=0: draft_model([prompt]) → token1
  t=1: draft_model([prompt, token1]) → token2
  t=2: draft_model([prompt, token1, token2]) → token3

优化：并行化draft阶段
  使用KV缓存，每次只需计算新token的attention
```

#### 优化2: Batch Verification

验证阶段可以批处理多个序列：

```python
# 同时验证多个序列的draft tokens
batch_tokens = [
    seq1_tokens + seq1_draft_tokens,
    seq2_tokens + seq2_draft_tokens,
    ...
]
batch_probs = target_model.forward_batch(batch_tokens)
```

---

## 实现架构设计

### 整体架构

在 nano-vllm 中添加推测解码，需要修改以下模块：

```
nanovllm/
├── engine/
│   ├── llm_engine.py         [修改] 添加draft model支持
│   ├── scheduler.py          [修改] 添加投机调度逻辑
│   ├── sequence.py           [修改] 添加draft token字段
│   ├── model_runner.py       [修改] 添加验证逻辑
│   └── speculative_engine.py [新增] 推测解码核心逻辑
├── models/
│   └── qwen3.py              [修改] 支持输出概率分布
└── config.py                 [修改] 添加推测解码配置
```

### 关键组件

#### 1. SpeculativeConfig

```python
@dataclass
class SpeculativeConfig:
    """推测解码配置"""
    enabled: bool = False                    # 是否启用
    draft_model: Optional[str] = None        # draft模型路径
    num_speculative_tokens: int = 4          # 投机步数gamma
    draft_tensor_parallel_size: int = 1      # draft模型并行度
```

#### 2. SpeculativeSequence

```python
class SpeculativeSequence(Sequence):
    """支持推测解码的序列"""

    def __init__(self, ...):
        super().__init__(...)
        # Draft tokens相关
        self.draft_token_ids: list[int] = []           # 预测的draft tokens
        self.draft_probs: list[torch.Tensor] = []      # draft模型的概率分布
        self.num_drafted: int = 0                      # 已预测的token数
        self.num_accepted: int = 0                     # 已接受的token数
```

#### 3. SpeculativeScheduler

```python
class SpeculativeScheduler(Scheduler):
    """支持推测解码的调度器"""

    def schedule_speculative(self):
        """推测解码调度

        Returns:
            (draft_seqs, verify_seqs, bonus_seqs)
        """
        draft_seqs = []    # 需要draft的序列
        verify_seqs = []   # 需要verify的序列
        bonus_seqs = []    # 需要生成bonus token的序列

        for seq in self.running:
            if seq.num_drafted < gamma:
                draft_seqs.append(seq)
            elif seq.num_drafted == gamma:
                verify_seqs.append(seq)

        return draft_seqs, verify_seqs, bonus_seqs
```

#### 4. SpeculativeModelRunner

```python
class SpeculativeModelRunner:
    """管理target和draft两个模型"""

    def __init__(self, config, draft_config):
        self.target_runner = ModelRunner(config)
        self.draft_runner = ModelRunner(draft_config)

    def draft_step(self, seqs):
        """Draft阶段：使用小模型预测下一个token"""
        # 使用draft model生成
        token_ids, probs = self.draft_runner.run(seqs, return_probs=True)
        return token_ids, probs

    def verify_step(self, seqs):
        """Verify阶段：使用大模型验证draft tokens"""
        # 准备验证输入
        verify_inputs = self.prepare_verify_inputs(seqs)

        # 使用target model并行验证
        target_probs = self.target_runner.run(verify_inputs, return_probs=True)

        # 执行接受/拒绝逻辑
        accepted_tokens = self.accept_reject(seqs, target_probs)

        return accepted_tokens
```

### 数据流

```
初始状态: tokens = [1, 2, 3]

==== 迭代1 ====

Draft阶段 (使用draft model):
  Step 1: draft([1,2,3]) → token=4, q_dist_1
  Step 2: draft([1,2,3,4]) → token=5, q_dist_2
  Step 3: draft([1,2,3,4,5]) → token=6, q_dist_3
  Step 4: draft([1,2,3,4,5,6]) → token=7, q_dist_4

  draft_tokens = [4, 5, 6, 7]
  draft_probs = [q_dist_1, q_dist_2, q_dist_3, q_dist_4]

Verify阶段 (使用target model，并行):
  target([1,2,3,4,5,6]) → [p_dist_1, p_dist_2, p_dist_3, p_dist_4]

  位置1: draft=4, p(4)=0.8, q(4)=0.6, alpha=1.0 → 接受
  位置2: draft=5, p(5)=0.7, q(5)=0.5, alpha=1.0 → 接受
  位置3: draft=6, p(6)=0.3, q(6)=0.6, alpha=0.5 → 拒绝（概率）
    → 从adjust_dist(p_dist_3, q_dist_3)采样 → token=8

  accepted = [4, 5, 8]

更新: tokens = [1, 2, 3, 4, 5, 8]

==== 迭代2 ====
...
```

---

## 详细实现步骤

### 步骤1: 扩展配置系统

**文件**: `nanovllm/config.py`

```python
@dataclass
class Config:
    # ... 现有配置 ...

    # 推测解码配置
    use_speculative_decoding: bool = False
    draft_model_path: Optional[str] = None
    num_speculative_tokens: int = 4  # gamma

    def __post_init__(self):
        # ... 现有初始化 ...

        # 验证推测解码配置
        if self.use_speculative_decoding:
            assert self.draft_model_path is not None, \
                "draft_model_path必须指定"
            assert self.num_speculative_tokens > 0, \
                "num_speculative_tokens必须大于0"
```

### 步骤2: 扩展Sequence类

**文件**: `nanovllm/engine/sequence.py`

```python
class Sequence:
    def __init__(self, prompt_token_ids, sampling_params=None):
        # ... 现有字段 ...

        # 推测解码字段
        self.draft_token_ids: list[int] = []
        self.draft_probs: list[torch.Tensor] = []

    def add_draft_token(self, token_id: int, prob_dist: torch.Tensor):
        """添加一个draft token"""
        self.draft_token_ids.append(token_id)
        self.draft_probs.append(prob_dist)

    def clear_drafts(self):
        """清空draft tokens"""
        self.draft_token_ids.clear()
        self.draft_probs.clear()

    @property
    def num_draft_tokens(self) -> int:
        return len(self.draft_token_ids)

    @property
    def has_pending_drafts(self) -> bool:
        """是否有待验证的draft tokens"""
        return len(self.draft_token_ids) > 0
```

### 步骤3: 创建SpeculativeEngine

**文件**: `nanovllm/engine/speculative_engine.py` (新建)

```python
import torch
from typing import List, Tuple
from nanovllm.engine.sequence import Sequence

class SpeculativeEngine:
    """推测解码引擎"""

    def __init__(self, config):
        self.gamma = config.num_speculative_tokens

    def accept_reject(
        self,
        seqs: List[Sequence],
        target_probs: torch.Tensor,  # [num_seqs, gamma, vocab_size]
    ) -> List[List[int]]:
        """执行接受-拒绝采样

        Args:
            seqs: 序列列表
            target_probs: target模型的概率分布

        Returns:
            accepted_tokens: 每个序列接受的token列表
        """
        batch_size = len(seqs)
        accepted_tokens = []

        for i in range(batch_size):
            seq = seqs[i]
            accepted = []

            for j in range(seq.num_draft_tokens):
                draft_token = seq.draft_token_ids[j]
                q_dist = seq.draft_probs[j]
                p_dist = target_probs[i, j]

                # 计算接受概率
                q_prob = q_dist[draft_token].item()
                p_prob = p_dist[draft_token].item()
                alpha = min(1.0, p_prob / (q_prob + 1e-10))

                # 采样决定是否接受
                if torch.rand(1).item() < alpha:
                    # 接受
                    accepted.append(draft_token)
                else:
                    # 拒绝：从修正分布采样
                    adjusted_dist = self._adjust_distribution(p_dist, q_dist)
                    new_token = torch.multinomial(adjusted_dist, 1).item()
                    accepted.append(new_token)
                    break  # 停止验证后续token

            # Bonus token: 如果所有draft都被接受
            if len(accepted) == seq.num_draft_tokens:
                # 从最后一个位置的target分布采样
                bonus_dist = target_probs[i, seq.num_draft_tokens - 1]
                bonus_token = torch.multinomial(bonus_dist, 1).item()
                accepted.append(bonus_token)

            accepted_tokens.append(accepted)

        return accepted_tokens

    def _adjust_distribution(
        self,
        p_dist: torch.Tensor,  # target分布
        q_dist: torch.Tensor,  # draft分布
    ) -> torch.Tensor:
        """计算修正分布: p'(x) = norm(max(0, p(x) - q(x)))"""
        adjusted = torch.clamp(p_dist - q_dist, min=0.0)
        adjusted = adjusted / (adjusted.sum() + 1e-10)
        return adjusted
```

### 步骤4: 修改Sampler支持返回概率

**文件**: `nanovllm/layers/sampler.py`

```python
class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        return_probs: bool = False,
    ):
        """采样

        Args:
            logits: [batch_size, vocab_size]
            temperatures: [batch_size]
            return_probs: 是否返回概率分布

        Returns:
            token_ids: [batch_size]
            probs (optional): [batch_size, vocab_size]
        """
        # 温度缩放
        logits = logits / temperatures.unsqueeze(-1)

        # 计算概率
        probs = torch.softmax(logits, dim=-1)

        # Gumbel-max采样
        q = torch.empty_like(probs).exponential_(1.0)
        token_ids = torch.argmax(probs / q, dim=-1)

        if return_probs:
            return token_ids, probs
        else:
            return token_ids
```

### 步骤5: 修改ModelRunner支持推测解码

**文件**: `nanovllm/engine/model_runner.py`

添加方法：

```python
class ModelRunner:
    # ... 现有代码 ...

    def run_with_probs(self, seqs: list[Sequence], is_prefill: bool):
        """运行模型并返回概率分布（用于推测解码）

        Returns:
            token_ids: [batch_size]
            probs: [batch_size, vocab_size]
        """
        # 准备输入
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)

        # 运行模型
        logits = self.run_model(input_ids, positions)

        # 采样并返回概率
        temperatures = torch.tensor(
            [seq.temperature for seq in seqs],
            dtype=torch.float32,
        ).cuda()

        token_ids, probs = self.sampler(logits, temperatures, return_probs=True)

        return token_ids.tolist(), probs

    def verify_draft_tokens(
        self,
        seqs: list[Sequence],
    ) -> torch.Tensor:
        """验证draft tokens

        Args:
            seqs: 包含draft_token_ids的序列列表

        Returns:
            target_probs: [batch_size, num_draft_tokens, vocab_size]
        """
        batch_size = len(seqs)
        gamma = seqs[0].num_draft_tokens

        # 准备验证输入
        # 对于每个序列，输入是: [original_tokens..., draft_token_0, ..., draft_token_{gamma-1}]
        # 输出gamma个位置的概率分布

        all_input_ids = []
        all_positions = []

        for seq in seqs:
            # 包含所有draft token（除了最后一个）
            tokens = seq.token_ids + seq.draft_token_ids[:-1]
            all_input_ids.extend(tokens[-gamma:])

            base_pos = len(seq.token_ids) - 1
            all_positions.extend([base_pos + i for i in range(gamma)])

        input_ids = torch.tensor(all_input_ids, dtype=torch.int64).cuda()
        positions = torch.tensor(all_positions, dtype=torch.int64).cuda()

        # 使用target model进行前向传播
        logits = self.run_model(input_ids, positions)
        # logits: [batch_size * gamma, vocab_size]

        # Reshape
        logits = logits.view(batch_size, gamma, -1)

        # 转换为概率
        probs = torch.softmax(logits, dim=-1)

        return probs
```

### 步骤6: 修改Scheduler支持推测解码

**文件**: `nanovllm/engine/scheduler.py`

```python
class Scheduler:
    # ... 现有代码 ...

    def schedule_speculative(self, gamma: int):
        """推测解码调度

        Returns:
            (draft_seqs, verify_seqs): 需要draft的序列和需要verify的序列
        """
        draft_seqs = []
        verify_seqs = []

        for seq in self.running:
            if not seq.has_pending_drafts:
                # 没有待验证的draft，需要生成draft tokens
                draft_seqs.append(seq)
            else:
                # 有待验证的draft tokens
                verify_seqs.append(seq)

        return draft_seqs, verify_seqs
```

### 步骤7: 修改LLMEngine整合推测解码

**文件**: `nanovllm/engine/llm_engine.py`

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        # ... 现有初始化 ...

        # 检查是否启用推测解码
        self.use_speculative = config.use_speculative_decoding

        if self.use_speculative:
            # 创建draft model runner
            draft_config = Config(
                config.draft_model_path,
                max_num_seqs=config.max_num_seqs,
                tensor_parallel_size=1,  # draft模型通常用单卡
            )
            self.draft_runner = ModelRunner(draft_config, 0, None)

            # 创建推测解码引擎
            from nanovllm.engine.speculative_engine import SpeculativeEngine
            self.spec_engine = SpeculativeEngine(config)
            self.gamma = config.num_speculative_tokens

    def step_speculative(self):
        """推测解码的一步"""
        # 1. 调度
        draft_seqs, verify_seqs = self.scheduler.schedule_speculative(self.gamma)

        # 2. Draft阶段：为需要draft的序列生成draft tokens
        if draft_seqs:
            for _ in range(self.gamma):
                # 使用draft model生成一个token
                token_ids, probs = self.draft_runner.call(
                    "run_with_probs", draft_seqs, False
                )

                # 保存draft token和概率
                for seq, token_id, prob_dist in zip(draft_seqs, token_ids, probs):
                    seq.add_draft_token(token_id, prob_dist)

        # 3. Verify阶段：验证draft tokens
        if verify_seqs:
            # 使用target model并行验证
            target_probs = self.model_runner.call(
                "verify_draft_tokens", verify_seqs
            )

            # 执行接受-拒绝采样
            accepted_tokens = self.spec_engine.accept_reject(
                verify_seqs, target_probs
            )

            # 更新序列
            for seq, tokens in zip(verify_seqs, accepted_tokens):
                for token_id in tokens:
                    seq.append_token(token_id)
                seq.clear_drafts()

        # 4. 后处理
        outputs = []
        for seq in verify_seqs:
            if seq.is_finished:
                self.scheduler.postprocess([seq], [])
                outputs.append((seq.seq_id, seq.completion_token_ids))

        return outputs

    def step(self):
        """执行一步（自动选择普通或推测解码）"""
        if self.use_speculative:
            return self.step_speculative()
        else:
            # ... 现有step逻辑 ...
            pass
```

---

## 代码实现指南

### 完整的Draft阶段实现

```python
def draft_phase(
    self,
    draft_seqs: List[Sequence],
    draft_runner: ModelRunner,
    gamma: int,
):
    """Draft阶段：使用draft model生成gamma个token

    关键：利用KV缓存，每次只计算新token
    """
    for step in range(gamma):
        # 准备输入：每个序列的最后一个token
        input_ids = []
        positions = []

        for seq in draft_seqs:
            if step == 0:
                # 第一步：使用原始序列的最后一个token
                input_ids.append(seq.token_ids[-1])
                positions.append(len(seq.token_ids) - 1)
            else:
                # 后续步骤：使用上一步生成的draft token
                input_ids.append(seq.draft_token_ids[-1])
                positions.append(len(seq.token_ids) + step - 1)

        input_ids = torch.tensor(input_ids).cuda()
        positions = torch.tensor(positions).cuda()

        # 运行draft model
        logits = draft_runner.run_model(input_ids, positions)

        # 采样
        token_ids, probs = draft_runner.sampler(
            logits,
            temperatures=torch.tensor([seq.temperature for seq in draft_seqs]).cuda(),
            return_probs=True,
        )

        # 保存结果
        for i, seq in enumerate(draft_seqs):
            seq.add_draft_token(token_ids[i].item(), probs[i])
```

### 完整的Verify阶段实现

```python
def verify_phase(
    self,
    verify_seqs: List[Sequence],
    target_runner: ModelRunner,
):
    """Verify阶段：使用target model并行验证所有draft tokens

    关键：构造特殊的输入，一次前向传播验证所有draft tokens
    """
    batch_size = len(verify_seqs)
    gamma = verify_seqs[0].num_draft_tokens

    # ===== 方法1: 简单但低效的实现 =====
    # 对每个draft token位置单独做一次前向传播

    all_target_probs = []

    for seq in verify_seqs:
        seq_probs = []

        for i in range(gamma):
            # 输入：原始tokens + 前i个draft tokens
            input_tokens = seq.token_ids + seq.draft_token_ids[:i]

            # 前向传播
            input_ids = torch.tensor([input_tokens[-1]]).cuda()
            positions = torch.tensor([len(input_tokens) - 1]).cuda()
            logits = target_runner.run_model(input_ids, positions)
            probs = torch.softmax(logits[0], dim=-1)

            seq_probs.append(probs)

        all_target_probs.append(torch.stack(seq_probs))

    target_probs = torch.stack(all_target_probs)
    # shape: [batch_size, gamma, vocab_size]

    # ===== 方法2: 高效的批处理实现 =====
    # 利用Flash Attention的变长序列支持，一次处理所有位置

    # 构造批处理输入
    all_input_ids = []
    all_positions = []
    cu_seqlens = [0]

    for seq in verify_seqs:
        # 输入序列：[original_tokens..., draft_0, draft_1, ..., draft_{gamma-1}]
        # 我们需要在draft_0, draft_1, ..., draft_{gamma-1}位置之前的logits

        # 实际上需要：
        # - 位置0的输入: [original_tokens]
        # - 位置1的输入: [original_tokens, draft_0]
        # - 位置2的输入: [original_tokens, draft_0, draft_1]
        # ...

        # 优化：使用KV缓存，只计算新添加的token
        base_len = len(seq.token_ids)

        for i in range(gamma):
            all_input_ids.append(seq.draft_token_ids[i])
            all_positions.append(base_len + i)

        cu_seqlens.append(cu_seqlens[-1] + gamma)

    # 准备输入
    input_ids = torch.tensor(all_input_ids).cuda()
    positions = torch.tensor(all_positions).cuda()

    # 设置上下文（用于attention）
    set_context(
        is_prefill=False,
        cu_seqlens=cu_seqlens,
        slot_mapping=...,  # 需要构造slot mapping
        block_tables=...,
    )

    # 前向传播
    logits = target_runner.run_model(input_ids, positions)
    # shape: [batch_size * gamma, vocab_size]

    # Reshape并转换为概率
    logits = logits.view(batch_size, gamma, -1)
    target_probs = torch.softmax(logits, dim=-1)

    return target_probs
```

### 接受-拒绝采样的详细实现

```python
def accept_reject_sampling(
    draft_tokens: List[int],      # [gamma]
    draft_probs: List[torch.Tensor],  # [gamma, vocab_size]
    target_probs: torch.Tensor,   # [gamma, vocab_size]
    temperature: float = 1.0,
) -> List[int]:
    """对单个序列执行接受-拒绝采样

    Returns:
        accepted_tokens: 接受的token列表（长度1到gamma+1）
    """
    accepted = []

    for i in range(len(draft_tokens)):
        draft_token = draft_tokens[i]
        q_dist = draft_probs[i]
        p_dist = target_probs[i]

        # 获取draft token的概率
        q_prob = q_dist[draft_token].item()
        p_prob = p_dist[draft_token].item()

        # 计算接受概率
        alpha = min(1.0, p_prob / (q_prob + 1e-10))

        # 采样决定
        if torch.rand(1).item() < alpha:
            # 接受draft token
            accepted.append(draft_token)
        else:
            # 拒绝：从修正分布采样新token
            # 修正分布: p'(x) = norm(max(0, p(x) - q(x)))
            adjusted_probs = torch.clamp(p_dist - q_dist, min=0.0)
            adjusted_probs = adjusted_probs / (adjusted_probs.sum() + 1e-10)

            # 从修正分布采样
            new_token = torch.multinomial(adjusted_probs, 1).item()
            accepted.append(new_token)

            # 拒绝后停止验证
            break

    # Bonus token：如果所有draft都被接受
    if len(accepted) == len(draft_tokens):
        # 从最后一个位置的target分布采样
        last_p_dist = target_probs[-1]
        bonus_token = torch.multinomial(last_p_dist, 1).item()
        accepted.append(bonus_token)

    return accepted
```

### KV缓存管理

推测解码对KV缓存的要求：

```python
# 问题：draft阶段生成的token也需要KV缓存
# 但这些token可能被拒绝，需要能够回滚

class SpeculativeBlockManager(BlockManager):
    """支持推测解码的块管理器"""

    def allocate_speculative(self, seq: Sequence):
        """为draft tokens预分配块（可回滚）"""
        # 保存当前状态
        checkpoint = {
            'block_table': seq.block_table.copy(),
            'num_cached_tokens': seq.num_cached_tokens,
        }

        # 为draft tokens分配块
        draft_blocks = []
        for i in range(seq.num_draft_tokens):
            if self.need_new_block(seq, i):
                block_id = self.allocate_block()
                draft_blocks.append(block_id)
                seq.block_table.append(block_id)

        return checkpoint, draft_blocks

    def rollback_speculative(self, seq: Sequence, checkpoint):
        """回滚draft token的块分配"""
        # 释放新分配的块
        while len(seq.block_table) > len(checkpoint['block_table']):
            block_id = seq.block_table.pop()
            self.deallocate_block(block_id)

        # 恢复状态
        seq.num_cached_tokens = checkpoint['num_cached_tokens']

    def commit_speculative(self, seq: Sequence, num_accepted: int):
        """提交被接受的draft tokens"""
        # 释放未被接受的draft tokens对应的块
        total_tokens = len(seq.token_ids) + num_accepted
        needed_blocks = (total_tokens + self.block_size - 1) // self.block_size

        while len(seq.block_table) > needed_blocks:
            block_id = seq.block_table.pop()
            self.deallocate_block(block_id)
```

---

## 测试与验证

### 单元测试

#### 测试1: 接受-拒绝采样的正确性

```python
import torch
from nanovllm.engine.speculative_engine import SpeculativeEngine

def test_accept_reject_distribution():
    """验证接受-拒绝采样不改变输出分布"""

    vocab_size = 100
    num_samples = 10000

    # 创建target和draft分布
    p_dist = torch.softmax(torch.randn(vocab_size), dim=0)
    q_dist = torch.softmax(torch.randn(vocab_size), dim=0)

    spec_engine = SpeculativeEngine(config)

    # 采样多次
    samples = []
    for _ in range(num_samples):
        # Draft: 从q分布采样
        draft_token = torch.multinomial(q_dist, 1).item()

        # Accept-Reject
        q_prob = q_dist[draft_token]
        p_prob = p_dist[draft_token]
        alpha = min(1.0, p_prob / q_prob)

        if torch.rand(1).item() < alpha:
            token = draft_token
        else:
            adjusted = torch.clamp(p_dist - q_dist, min=0)
            adjusted = adjusted / adjusted.sum()
            token = torch.multinomial(adjusted, 1).item()

        samples.append(token)

    # 统计采样分布
    empirical_dist = torch.zeros(vocab_size)
    for token in samples:
        empirical_dist[token] += 1
    empirical_dist /= num_samples

    # 验证：empirical_dist应该接近p_dist
    error = torch.abs(empirical_dist - p_dist).mean()
    assert error < 0.01, f"分布误差过大: {error}"
    print(f"✓ 分布误差: {error:.4f}")
```

#### 测试2: Draft-Verify流程

```python
def test_draft_verify_pipeline():
    """测试完整的draft-verify流程"""

    # 创建模型
    target_model = load_model("Qwen3-0.6B")
    draft_model = load_model("Qwen3-0.1B")  # 更小的模型

    # 创建序列
    prompt = "The capital of France is"
    seq = Sequence(tokenizer.encode(prompt))

    # Draft阶段
    gamma = 4
    for _ in range(gamma):
        logits = draft_model(seq.token_ids + seq.draft_token_ids)
        probs = torch.softmax(logits[-1], dim=0)
        token = torch.multinomial(probs, 1).item()
        seq.add_draft_token(token, probs)

    print(f"Draft tokens: {seq.draft_token_ids}")

    # Verify阶段
    target_probs = []
    for i in range(gamma):
        tokens = seq.token_ids + seq.draft_token_ids[:i]
        logits = target_model(tokens)
        probs = torch.softmax(logits[-1], dim=0)
        target_probs.append(probs)

    target_probs = torch.stack(target_probs)

    # Accept-Reject
    spec_engine = SpeculativeEngine(config)
    accepted = accept_reject_sampling(
        seq.draft_token_ids,
        seq.draft_probs,
        target_probs,
    )

    print(f"Accepted tokens: {accepted}")
    print(f"Acceptance rate: {len(accepted) / gamma:.2%}")
```

### 集成测试

#### 测试3: 端到端生成测试

```python
def test_end_to_end_generation():
    """测试推测解码的端到端生成"""

    # 创建LLM（启用推测解码）
    llm = LLM(
        "Qwen3-0.6B",
        use_speculative_decoding=True,
        draft_model_path="Qwen3-0.1B",
        num_speculative_tokens=4,
    )

    # 生成
    prompt = "Write a short story about a robot:"
    sampling_params = SamplingParams(temperature=0.8, max_tokens=100)

    output = llm.generate([prompt], sampling_params)[0]

    print(f"Generated text: {output['text']}")
    print(f"Num tokens: {len(output['token_ids'])}")
```

### 正确性验证

最重要的测试：验证推测解码的输出与普通解码相同（在采样确定的情况下）。

```python
def test_output_equivalence():
    """验证推测解码的输出分布与普通解码相同"""

    # 设置随机种子
    torch.manual_seed(42)

    # 普通解码
    llm_normal = LLM("Qwen3-0.6B", use_speculative_decoding=False)
    output_normal = llm_normal.generate(
        ["Hello, world!"],
        SamplingParams(temperature=0.0, max_tokens=50)  # temperature=0确定性生成
    )[0]

    # 推测解码
    torch.manual_seed(42)  # 相同随机种子
    llm_spec = LLM(
        "Qwen3-0.6B",
        use_speculative_decoding=True,
        draft_model_path="Qwen3-0.1B",
        num_speculative_tokens=4,
    )
    output_spec = llm_spec.generate(
        ["Hello, world!"],
        SamplingParams(temperature=0.0, max_tokens=50)
    )[0]

    # 验证输出相同
    assert output_normal['token_ids'] == output_spec['token_ids'], \
        "推测解码输出与普通解码不一致！"

    print("✓ 输出一致性验证通过")
```

### 性能测试

```python
def benchmark_speculative_decoding():
    """性能基准测试"""

    import time

    prompts = ["Tell me a story"] * 100
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

    # 测试普通解码
    llm_normal = LLM("Qwen3-0.6B")
    start = time.time()
    outputs_normal = llm_normal.generate(prompts, sampling_params)
    time_normal = time.time() - start
    tokens_normal = sum(len(out['token_ids']) for out in outputs_normal)

    # 测试推测解码
    llm_spec = LLM(
        "Qwen3-0.6B",
        use_speculative_decoding=True,
        draft_model_path="Qwen3-0.1B",
        num_speculative_tokens=4,
    )
    start = time.time()
    outputs_spec = llm_spec.generate(prompts, sampling_params)
    time_spec = time.time() - start
    tokens_spec = sum(len(out['token_ids']) for out in outputs_spec)

    # 统计
    print(f"普通解码:")
    print(f"  时间: {time_normal:.2f}s")
    print(f"  Tokens: {tokens_normal}")
    print(f"  吞吐量: {tokens_normal / time_normal:.2f} tokens/s")

    print(f"推测解码:")
    print(f"  时间: {time_spec:.2f}s")
    print(f"  Tokens: {tokens_spec}")
    print(f"  吞吐量: {tokens_spec / time_spec:.2f} tokens/s")

    print(f"加速比: {time_normal / time_spec:.2f}x")
```

---

## 性能优化

### 优化1: Draft Model的选择

Draft model应该满足：
1. **速度快**: 至少比target model快5-10倍
2. **质量可接受**: 接受率至少50%以上
3. **词表兼容**: 与target model使用相同的tokenizer

**推荐配置**:
- Target: Qwen3-7B → Draft: Qwen3-0.6B (10-15x faster, 60-70% acceptance)
- Target: Llama-70B → Draft: Llama-7B (8-10x faster, 50-60% acceptance)

### 优化2: 动态调整gamma

根据接受率动态调整投机步数：

```python
class AdaptiveSpeculativeEngine(SpeculativeEngine):
    def __init__(self, config):
        super().__init__(config)
        self.gamma_min = 2
        self.gamma_max = 6
        self.gamma = config.num_speculative_tokens
        self.acceptance_history = []

    def update_gamma(self, acceptance_rate: float):
        """根据接受率调整gamma"""
        self.acceptance_history.append(acceptance_rate)

        # 每100步调整一次
        if len(self.acceptance_history) >= 100:
            avg_acceptance = sum(self.acceptance_history) / len(self.acceptance_history)

            if avg_acceptance > 0.7:
                # 接受率高，增加gamma
                self.gamma = min(self.gamma + 1, self.gamma_max)
            elif avg_acceptance < 0.4:
                # 接受率低，减少gamma
                self.gamma = max(self.gamma - 1, self.gamma_min)

            self.acceptance_history.clear()
```

### 优化3: 批处理优化

同时处理多个序列的draft和verify：

```python
def batch_draft_verify(
    seqs: List[Sequence],
    draft_runner: ModelRunner,
    target_runner: ModelRunner,
    gamma: int,
):
    """批处理draft和verify"""

    # ==== Draft阶段：批处理 ====
    for step in range(gamma):
        # 所有序列一起draft
        input_ids = torch.tensor([
            seq.draft_token_ids[-1] if step > 0 else seq.token_ids[-1]
            for seq in seqs
        ]).cuda()

        logits = draft_runner.run_model(input_ids, ...)
        token_ids, probs = draft_runner.sampler(logits, ..., return_probs=True)

        for i, seq in enumerate(seqs):
            seq.add_draft_token(token_ids[i].item(), probs[i])

    # ==== Verify阶段：批处理 ====
    # 构造批处理输入：[batch_size * gamma, ...]
    all_inputs = []
    for seq in seqs:
        for i in range(gamma):
            all_inputs.append(seq.token_ids + seq.draft_token_ids[:i+1])

    # 一次前向传播验证所有
    target_probs = target_runner.forward_batch(all_inputs)
    target_probs = target_probs.view(len(seqs), gamma, -1)

    return target_probs
```

### 优化4: CUDA Graph支持

Verify阶段可以使用CUDA Graph（批大小固定）：

```python
def capture_verify_cudagraph(self, batch_size: int, gamma: int):
    """捕获verify阶段的CUDA Graph"""

    # 创建虚拟输入
    dummy_input = torch.zeros(batch_size * gamma, dtype=torch.int64).cuda()
    dummy_positions = torch.arange(batch_size * gamma).cuda()

    # 预热
    for _ in range(3):
        self.model(dummy_input, dummy_positions)

    # 捕获
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = self.model(dummy_input, dummy_positions)

    self.verify_graphs[(batch_size, gamma)] = (graph, dummy_input, dummy_positions, logits)
```

---

## 实现检查清单

完成以下任务来实现推测解码：

- [ ] **配置系统**
  - [ ] 在`config.py`中添加推测解码配置
  - [ ] 添加draft model路径配置
  - [ ] 添加gamma参数

- [ ] **数据结构**
  - [ ] 扩展`Sequence`类支持draft tokens
  - [ ] 添加`draft_token_ids`和`draft_probs`字段
  - [ ] 实现`add_draft_token()`和`clear_drafts()`方法

- [ ] **核心引擎**
  - [ ] 创建`SpeculativeEngine`类
  - [ ] 实现`accept_reject()`方法
  - [ ] 实现`_adjust_distribution()`方法

- [ ] **模型执行**
  - [ ] 修改`Sampler`支持`return_probs`参数
  - [ ] 在`ModelRunner`中添加`run_with_probs()`方法
  - [ ] 在`ModelRunner`中添加`verify_draft_tokens()`方法

- [ ] **调度系统**
  - [ ] 在`Scheduler`中添加`schedule_speculative()`方法
  - [ ] 支持draft和verify两种调度模式

- [ ] **引擎集成**
  - [ ] 在`LLMEngine`中添加draft model加载
  - [ ] 实现`step_speculative()`方法
  - [ ] 整合draft和verify流程

- [ ] **测试**
  - [ ] 单元测试：接受-拒绝采样
  - [ ] 单元测试：draft-verify流程
  - [ ] 集成测试：端到端生成
  - [ ] 正确性测试：与普通解码对比
  - [ ] 性能测试：加速比

- [ ] **优化**
  - [ ] 批处理优化
  - [ ] CUDA Graph支持
  - [ ] 动态gamma调整
  - [ ] KV缓存管理优化

---

## 调试建议

### 常见问题

1. **接受率过低（<30%）**
   - 检查：draft model是否太小
   - 检查：温度参数设置
   - 解决：使用更大的draft model或减小gamma

2. **速度反而变慢**
   - 检查：draft model是否真的快
   - 检查：gamma是否过大（验证开销）
   - 解决：测量draft和target的实际速度比

3. **输出分布不一致**
   - 检查：接受-拒绝采样实现
   - 检查：修正分布计算
   - 解决：添加数值稳定性处理（+1e-10）

4. **KV缓存错误**
   - 检查：draft tokens的KV缓存管理
   - 检查：回滚机制
   - 解决：仔细追踪block分配和释放

### 调试工具

```python
class SpeculativeDebugger:
    """推测解码调试工具"""

    def __init__(self):
        self.stats = {
            'total_steps': 0,
            'total_draft_tokens': 0,
            'total_accepted_tokens': 0,
            'acceptance_rates': [],
        }

    def log_step(self, num_draft: int, num_accepted: int):
        """记录一步的统计"""
        self.stats['total_steps'] += 1
        self.stats['total_draft_tokens'] += num_draft
        self.stats['total_accepted_tokens'] += num_accepted
        rate = num_accepted / num_draft if num_draft > 0 else 0
        self.stats['acceptance_rates'].append(rate)

    def print_summary(self):
        """打印统计摘要"""
        avg_acceptance = (
            self.stats['total_accepted_tokens'] /
            self.stats['total_draft_tokens']
        )
        print(f"推测解码统计:")
        print(f"  总步数: {self.stats['total_steps']}")
        print(f"  Draft tokens: {self.stats['total_draft_tokens']}")
        print(f"  Accepted tokens: {self.stats['total_accepted_tokens']}")
        print(f"  平均接受率: {avg_acceptance:.2%}")
```

---

## 总结

### 学习路径

1. **理解算法**: 先理解推测解码的数学原理，特别是为什么输出分布不变
2. **实现基础版本**: 先实现简单的单序列版本，验证正确性
3. **批处理优化**: 然后添加批处理支持，提高效率
4. **集成到引擎**: 最后整合到完整的LLM引擎中
5. **性能调优**: 通过profiling找到瓶颈并优化

### 关键要点

1. **数学正确性**: 接受-拒绝采样保证输出分布
2. **工程实现**: 批处理和并行化是性能关键
3. **KV缓存管理**: 需要支持draft tokens的临时缓存
4. **Draft model选择**: 速度和质量的平衡

### 扩展方向

完成基础实现后，可以探索：
1. **Tree-based Speculative Decoding**: 使用树结构同时验证多个分支
2. **Multi-model Collaboration**: 使用多个draft model
3. **Adaptive Drafting**: 根据输入动态选择draft策略
4. **Hardware-aware Optimization**: 针对特定硬件优化

祝你实现顺利！这是一个很好的学习机会，能够深入理解LLM推理优化的核心技术。
