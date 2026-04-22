# IFEval 实验结果表（按 revision.md 指标）

> **说明**：Taboo + JSON 两类任务的结果已合并（取平均）。`user_interact` 列对自动化方法为空（N/A）。
> **注意**：Regenerate 方法在 DeepSeek Taboo 上表现极差是因为模型本身对 Taboo 约束遵守能力极弱（Vanilla 0%），与 self-correction 策略无关。
> **acc@K (K=5)**：5 次内通过率，包含重试纠错的效果。

---

## 表1: Automated Baselines（Model × Method，Taboo+JSON 平均）

### 指标说明
- **acc@1**: 首次通过率
- **acc@K**: K次内通过率（K=5）
- **wall_clock**: 任务总耗时（秒）
- **gen**: 模型生成时间（秒）；自动化实验中 gen ≈ wall_clock
- **inspect**: 用户检查/决策时间（秒）；自动化实验为 N/A
- **clicks**: 平均点击次数（仅人类实验）
- **avg_type**: 平均打字字符数（仅人类实验）
- **token_eff%**: Token节省率（相对于该模型 Vanilla 基线）

### DeepSeek-R1-Distill-Llama-8B

| Method | acc@1 | acc@K | wall_clock | gen | inspect | clicks | avg_type | token_eff% | api_calls |
|--------|-------|-------|------------|-----|---------|--------|----------|------------|-----------|
| Vanilla | 0.08 | 0.08 | 15.0 | 15.0 | N/A | N/A | N/A | 0.0 | 1.00 |
| Constrained Decoding | 0.72 | 0.72 | 4.9 | 4.9 | N/A | N/A | N/A | 72.4 | 1.00 |
| Regenerate | 0.37 | 0.50 | 45.4 | 45.4 | N/A | N/A | N/A | -224.5 | 3.13 |
| Auto Local Repair | 0.12 | 0.23 | 39.9 | 39.9 | N/A | N/A | N/A | -143.1 | 4.38 |

### Llama-3.1-8B

| Method | acc@1 | acc@K | wall_clock | gen | inspect | clicks | avg_type | token_eff% | api_calls |
|--------|-------|-------|------------|-----|---------|--------|----------|------------|-----------|
| Vanilla | 0.60 | 0.60 | 2.3 | 2.3 | N/A | N/A | N/A | 0.0 | 1.00 |
| Constrained Decoding | 0.93 | 0.93 | 1.7 | 1.7 | N/A | N/A | N/A | 48.3 | 1.00 |
| Regenerate | 0.60 | 0.95 | 4.9 | 4.9 | N/A | N/A | N/A | -98.8 | 1.60 |
| Auto Local Repair | 0.62 | 0.97 | 13.9 | 13.9 | N/A | N/A | N/A | -36.6 | 1.57 |

### Qwen3-8B

| Method | acc@1 | acc@K | wall_clock | gen | inspect | clicks | avg_type | token_eff% | api_calls |
|--------|-------|-------|------------|-----|---------|--------|----------|------------|-----------|
| Vanilla | 0.52 | 0.52 | 15.5 | 15.5 | N/A | N/A | N/A | 0.0 | 1.00 |
| Constrained Decoding | 1.00 | 1.00 | 5.3 | 5.3 | N/A | N/A | N/A | 72.2 | 1.00 |
| Regenerate | 0.40 | 0.77 | 21.1 | 21.1 | N/A | N/A | N/A | -150.5 | 2.52 |
| Auto Local Repair | 0.60 | 0.92 | 26.2 | 26.2 | N/A | N/A | N/A | -2.2 | 1.92 |

### Qwen2.5-14B-Instruct

| Method | acc@1 | acc@K | wall_clock | gen | inspect | clicks | avg_type | token_eff% | api_calls |
|--------|-------|-------|------------|-----|---------|--------|----------|------------|-----------|
| Vanilla | 0.75 | 0.75 | 7.6 | 7.6 | N/A | N/A | N/A | 0.0 | 1.00 |
| Constrained Decoding | 0.97 | 0.97 | 1.9 | 1.9 | N/A | N/A | N/A | 37.3 | 1.00 |
| Regenerate | 0.75 | 0.88 | 12.1 | 12.1 | N/A | N/A | N/A | -88.8 | 1.65 |
| Auto Local Repair | 0.75 | 0.98 | 9.6 | 9.6 | N/A | N/A | N/A | -24.9 | 1.33 |

### Qwen2.5-1.5B

| Method | acc@1 | acc@K | wall_clock | gen | inspect | clicks | avg_type | token_eff% | api_calls |
|--------|-------|-------|------------|-----|---------|--------|----------|------------|-----------|
| Vanilla | 0.25 | 0.25 | 1.7 | 1.7 | N/A | N/A | N/A | 0.0 | 1.00 |
| Constrained Decoding | 0.92 | 0.92 | 1.5 | 1.5 | N/A | N/A | N/A | 53.8 | 1.00 |
| Regenerate | 0.25 | 0.30 | 6.9 | 6.9 | N/A | N/A | N/A | 299.3 | 3.87 |
| Auto Local Repair | 0.25 | 0.50 | 14.6 | 14.6 | N/A | N/A | N/A | -265.9 | 3.53 |

### DeepSeek-R1-Distill-Qwen-1.5B

| Method | acc@1 | acc@K | wall_clock | gen | inspect | clicks | avg_type | token_eff% | api_calls |
|--------|-------|-------|------------|-----|---------|--------|----------|------------|-----------|
| Vanilla | 0.00 | 0.00 | 5.6 | 5.6 | N/A | N/A | N/A | 0.0 | 1.00 |
| Constrained Decoding | 0.65 | 0.65 | 2.2 | 2.2 | N/A | N/A | N/A | 71.7 | 1.00 |
| Regenerate | 0.00 | 0.03 | 26.2 | 26.2 | N/A | N/A | N/A | 386.3 | 4.97 |
| Auto Local Repair | 0.03 | 0.07 | 34.8 | 34.8 | N/A | N/A | N/A | -156.3 | 4.82 |

---

## 分Dataset结果详情

### DeepSeek-R1-Distill-Llama-8B

**Taboo:**

| Method | acc@1 | acc@K | wall_clock | token_eff% |
|--------|-------|-------|------------|------------|
| Vanilla | 0.00 | 0.00 | 16.1s | 0.0 |
| Constrained Decoding | 0.43 | 0.43 | 9.5s | 48.1 |
| Regenerate | 0.07 | 0.20 | 68.1s | -342.4 |
| Auto Local Repair | 0.00 | 0.03 | 62.3s | -157.3 |

**JSON:**

| Method | acc@1 | acc@K | wall_clock | token_eff% |
|--------|-------|-------|------------|------------|
| Vanilla | 0.17 | 0.17 | 13.9s | 0.0 |
| Constrained Decoding | 1.00 | 1.00 | 0.3s | 96.7 |
| Regenerate | 0.67 | 0.80 | 22.7s | -106.7 |
| Auto Local Repair | 0.23 | 0.43 | 17.6s | -128.8 |

### Llama-3.1-8B

**Taboo:**

| Method | acc@1 | acc@K | wall_clock | token_eff% |
|--------|-------|-------|------------|------------|
| Vanilla | 0.43 | 0.43 | 3.9s | 0.0 |
| Constrained Decoding | 0.87 | 0.87 | 3.1s | 27.3 |
| Regenerate | 0.43 | 0.90 | 8.5s | -116.0 |
| Auto Local Repair | 0.47 | 0.93 | 24.0s | -38.9 |

**JSON:**

| Method | acc@1 | acc@K | wall_clock | token_eff% |
|--------|-------|-------|------------|------------|
| Vanilla | 0.77 | 0.77 | 0.7s | 0.0 |
| Constrained Decoding | 1.00 | 1.00 | 0.2s | 69.3 |
| Regenerate | 0.77 | 1.00 | 1.2s | -81.5 |
| Auto Local Repair | 0.77 | 1.00 | 3.9s | -34.3 |

### Qwen3-8B

**Taboo:**

| Method | acc@1 | acc@K | wall_clock | token_eff% |
|--------|-------|-------|------------|------------|
| Vanilla | 0.73 | 0.73 | 15.7s | 0.0 |
| Constrained Decoding | 1.00 | 1.00 | 10.2s | 47.5 |
| Regenerate | 0.33 | 0.53 | 28.3s | -230.7 |
| Auto Local Repair | 0.97 | 1.00 | 17.5s | 45.8 |

**JSON:**

| Method | acc@1 | acc@K | wall_clock | token_eff% |
|--------|-------|-------|------------|------------|
| Vanilla | 0.30 | 0.30 | 15.3s | 0.0 |
| Constrained Decoding | 1.00 | 1.00 | 0.3s | 97.0 |
| Regenerate | 0.47 | 1.00 | 13.8s | -70.3 |
| Auto Local Repair | 0.23 | 0.83 | 34.9s | -50.2 |

### Qwen2.5-14B-Instruct

**Taboo:**

| Method | acc@1 | acc@K | wall_clock | token_eff% |
|--------|-------|-------|------------|------------|
| Vanilla | 0.50 | 0.50 | 12.0s | 0.0 |
| Constrained Decoding | 0.93 | 0.93 | 3.5s | 12.4 |
| Regenerate | 0.50 | 0.77 | 23.2s | -177.6 |
| Auto Local Repair | 0.50 | 0.97 | 17.2s | -49.8 |

**JSON:**

| Method | acc@1 | acc@K | wall_clock | token_eff% |
|--------|-------|-------|------------|------------|
| Vanilla | 1.00 | 1.00 | 3.1s | 0.0 |
| Constrained Decoding | 1.00 | 1.00 | 0.4s | 62.2 |
| Regenerate | 1.00 | 1.00 | 1.0s | 0.0 |
| Auto Local Repair | 1.00 | 1.00 | 2.1s | 0.0 |

### Qwen2.5-1.5B

**Taboo:**

| Method | acc@1 | acc@K | wall_clock | token_eff% |
|--------|-------|-------|------------|------------|
| Vanilla | 0.30 | 0.30 | 2.2s | 0.0 |
| Constrained Decoding | 0.83 | 0.83 | 2.8s | 23.8 |
| Regenerate | 0.30 | 0.40 | 8.6s | 281.8 |
| Auto Local Repair | 0.30 | 0.80 | 15.2s | -167.5 |

**JSON:**

| Method | acc@1 | acc@K | wall_clock | token_eff% |
|--------|-------|-------|------------|------------|
| Vanilla | 0.20 | 0.20 | 1.2s | 0.0 |
| Constrained Decoding | 1.00 | 1.00 | 0.2s | 83.7 |
| Regenerate | 0.20 | 0.20 | 5.1s | 332.1 |
| Auto Local Repair | 0.20 | 0.20 | 14.0s | -364.4 |

### DeepSeek-R1-Distill-Qwen-1.5B

**Taboo:**

| Method | acc@1 | acc@K | wall_clock | token_eff% |
|--------|-------|-------|------------|------------|
| Vanilla | 0.00 | 0.00 | 5.8s | 0.0 |
| Constrained Decoding | 0.30 | 0.30 | 4.3s | 46.3 |
| Regenerate | 0.00 | 0.00 | 25.0s | 399.7 |
| Auto Local Repair | 0.00 | 0.03 | 49.2s | -168.0 |

**JSON:**

| Method | acc@1 | acc@K | wall_clock | token_eff% |
|--------|-------|-------|------------|------------|
| Vanilla | 0.00 | 0.00 | 5.3s | 0.0 |
| Constrained Decoding | 1.00 | 1.00 | 0.2s | 97.1 |
| Regenerate | 0.00 | 0.07 | 27.4s | 373.2 |
| Auto Local Repair | 0.07 | 0.10 | 20.4s | -144.7 |

---

## 关键发现

1. **Constrained Decoding (CD)** 在 JSON 上全面满分（1.00），在 Taboo 上对 Qwen3/14B 满分但对 DeepSeek 仅 0.43
2. **Auto Local Repair** 在 Llama3-8B 上表现最佳（acc@K=0.97），其次 Qwen3-8B（acc@K=0.92）；但对 DeepSeek 系列极差
3. **DeepSeek Taboo 问题**：模型本身 Vanilla 0% 成功率，Regenerate/CD/AutoLR 均无法有效解决
4. **negative token_eff%**：Regenerate 和 Auto Local Repair 的 token 消耗均超过 Vanilla，因为多轮重生成导致
5. **acc@K vs acc@1 差异**：Regenerate 和 Auto Local Repair 通过重试显著提升 acc@K（Llama3 AutoLR: 0.62→0.97, Qwen3-8B AutoLR: 0.60→0.92）

---

## 待补充：人类实验数据（Chat / Steer）

### Llama-3.1-8B

| Method | acc@1 | acc@K | wall_clock | gen | inspect | clicks | avg_type | token_eff% |
|--------|-------|-------|------------|-----|---------|--------|----------|------------|
| Chat | — | — | — | — | — | — | — | — |
| Steer | — | — | — | — | — | — | — | — |

### DeepSeek-R1-Distill-Llama-8B

| Method | acc@1 | acc@K | wall_clock | gen | inspect | clicks | avg_type | token_eff% |
|--------|-------|-------|------------|-----|---------|--------|----------|------------|
| Chat | — | — | — | — | — | — | — | — |
| Steer | — | — | — | — | — | — | — | — |

### Qwen3-8B

| Method | acc@1 | acc@K | wall_clock | gen | inspect | clicks | avg_type | token_eff% |
|--------|-------|-------|------------|-----|---------|--------|----------|------------|
| Chat | — | — | — | — | — | — | — | — |
| Steer | — | — | — | — | — | — | — | — |

---

## Ablation Study 数据（待补充）

| Model | Condition | acc@K | Avg Turns | Completion Time | Interventions |
|-------|-----------|-------|-----------|-----------------|---------------|
| Llama-3.1 | Steer-lite (Signals Off) | — | — | — | — |
| | Steer+SOS (Signals On) | — | — | — | — |
| DeepSeek | Steer-lite | — | — | — | — |
| | Steer+SOS | — | — | — | — |
| Qwen3 | Steer-lite | — | — | — | — |
| | Steer+SOS | — | — | — | — |
