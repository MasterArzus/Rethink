# IFEval 实验结果表（v3 - compute_stats.py 统计，DeepSeek 修复）

> **说明**：Taboo + JSON 两类任务的结果已合并（各30条，共60条）。
> **统计方法**：C0/C1 Acc@K 从 final_response 反推，S.R = C0 AND C1
> **DeepSeek 修复**：使用 `</think>` 分隔符提取实际回答
> **更新日期**：2026/04/27

---

## 表1: Automated Baselines（Model × Method）

### DeepSeek-R1-Distill-Llama-8B

| Method | K0 Acc@1 | K0 Acc@K | C1 Acc@K | S.R |
|--------|----------|----------|----------|-----|
| Regenerate | 0.0 | 85.0 | 36.7 | 36.7 |
| Auto LR | 21.7 | 90.0 | 43.3 | 43.3 |
| CD | 71.7 | 96.7 | 71.7 | 71.7 |

### Llama-3.1-8B

| Method | K0 Acc@1 | K0 Acc@K | C1 Acc@K | S.R |
|--------|----------|----------|----------|-----|
| Regenerate | 28.3 | 95.0 | 91.7 | 90.0 |
| Auto LR | 46.7 | 95.0 | 90.0 | 88.3 |
| CD | 93.3 | 98.3 | 81.7 | 75.0 |

### Qwen3-8B

| Method | K0 Acc@1 | K0 Acc@K | C1 Acc@K | S.R |
|--------|----------|----------|----------|-----|
| Regenerate | 56.7 | 85.0 | 58.3 | 58.3 |
| Auto LR | 63.3 | 85.0 | 63.3 | 63.3 |
| CD | 95.0 | 95.0 | 100.0 | 95.0 |

### Qwen2.5-14B-Instruct

| Method | K0 Acc@1 | K0 Acc@K | C1 Acc@K | S.R |
|--------|----------|----------|----------|-----|
| Regenerate | 48.3 | 96.7 | 95.0 | 93.3 |
| Auto LR | 61.7 | 98.3 | 100.0 | 98.3 |
| CD | 96.7 | 98.3 | 96.7 | 96.7 |

### Qwen2.5-1.5B

| Method | K0 Acc@1 | K0 Acc@K | C1 Acc@K | S.R |
|--------|----------|----------|----------|-----|
| Regenerate | 15.0 | 80.0 | 33.3 | 31.7 |
| Auto LR | 31.7 | 88.3 | 48.3 | 46.7 |
| CD | 81.7 | 95.0 | 50.0 | 40.8 |

### DeepSeek-R1-Distill-Qwen-1.5B

| Method | K0 Acc@1 | K0 Acc@K | C1 Acc@K | S.R |
|--------|----------|----------|----------|-----|
| Regenerate | 0.0 | 53.3 | 21.7 | 20.0 |
| Auto LR | 15.0 | 61.7 | 23.3 | 23.3 |
| CD | 65.0 | 98.3 | 65.0 | 65.0 |

---

## 表2: LLM Actor Simulation（Chat / Steer）

### DeepSeek-R1-Distill-Llama-8B

| Method | K0 Acc@1 | K0 Acc@K | C1 Acc@K | S.R |
|--------|----------|----------|----------|-----|
| Chat | 1.7 | 80.0 | 50.0 | 48.3 |
| Steer | 1.7 | 83.3 | 73.3 | 71.7 |

### Llama-3.1-8B

| Method | K0 Acc@1 | K0 Acc@K | C1 Acc@K | S.R |
|--------|----------|----------|----------|-----|
| Chat | 48.3 | 95.0 | 91.7 | 90.0 |
| Steer | 58.3 | 95.2 | 96.4 | 94.0 |

### Qwen3-8B

| Method | K0 Acc@1 | K0 Acc@K | C1 Acc@K | S.R |
|--------|----------|----------|----------|-----|
| Chat | 60.0 | 90.0 | 60.0 | 60.0 |
| Steer | 62.5 | 87.5 | 60.0 | 59.2 |

### Qwen2.5-14B-Instruct

| Method | K0 Acc@1 | K0 Acc@K | C1 Acc@K | S.R |
|--------|----------|----------|----------|-----|
| Chat | 63.3 | 98.3 | 95.0 | 93.3 |
| Steer | 65.0 | 96.7 | 98.3 | 96.7 |

### Qwen2.5-1.5B

| Method | K0 Acc@1 | K0 Acc@K | C1 Acc@K | S.R |
|--------|----------|----------|----------|-----|
| Chat | 18.3 | 93.3 | 36.7 | 36.7 |
| Steer | 28.3 | 91.7 | 60.0 | 58.3 |

### DeepSeek-R1-Distill-Qwen-1.5B

| Method | K0 Acc@1 | K0 Acc@K | C1 Acc@K | S.R |
|--------|----------|----------|----------|-----|
| Chat | 1.7 | 70.0 | 20.0 | 20.0 |
| Steer | 1.6 | 60.3 | 50.8 | 47.6 |

---

## 关键发现

1. **DeepSeek 修复有效**：使用 `』』` 分隔符后，C0 Acc@K 从 ~36% 提升到 85%（deepseek_r1 regenerate）
2. **Steer > Chat**：DeepSeek Steer S.R = 71.7% vs Chat 48.3%
3. **CD 在 C0 上优秀**：但 C1 需要新设计（CD 无法处理动态约束变化）
4. **模型排序**：Qwen2.5-14B > Llama3-8B > Qwen3-8B > DeepSeek > Qwen2.5-1.5B

---

## 待办项

### Phase 5: CD 两轮实验
- CD 第一轮：C0 → C0 Acc@1/K
- CD 第二轮：C0+C1 → C1 Acc@2/K
