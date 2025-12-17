# ACL 2026 Sketch for rethink
## Title
Rethink: An Interactive, Process-Oriented Debugging Framework for Large Language Model Reasoning
Opening the Black Box of Reasoning: Fine-Grained Interactive Intervention via Token-Level Rethinking

## Introduction

The prevailing paradigm in human-computer interaction for Large Language Models (LLMs) is conversational dialogue (e.g., OpenAI, 2023). Within this paradigm, extensive research has been conducted on long-context understanding, multi-turn iteration, and in-context learning to align model outputs with user needs and human values (Ouyang et al., 2022; Bai et al., 2022). While these automated alignment strategies provide valuable algorithms and metrics, they often lack granular human involvement, making it difficult to align with the diverse and specific preferences of individual users.

In both academic testing and real-world applications, users frequently encounter scenarios where they must correct specific parts of a generated response (e.g., "This part is wrong, please modify it"). However, current conversational models typically treat the entire previous dialogue—including the erroneous segments—as context. This leads to two critical issues during re-generation: **Error Propagation** and **Ambiguity in Feedback**. When a user simply states that an output is "incorrect," the model often fails to discern the precise source of the error—whether it is a logical fallacy, a calculation error, or a factual hallucination. Consequently, the model may continue to hallucinate or fail to attend to the specific correction required, leading to **Hallucination Accumulation**. Furthermore, re-generating the entire response to fix a local error results in significant computational inefficiency and token wastage.

To address these challenges, we propose **Rethink**, an interactive, process-oriented debugging framework for LLM reasoning. Rethink shifts the interaction paradigm from "black-box dialogue" to "glass-box debugging." By visualizing internal model signals—such as token-level entropy, logit distributions, and hidden state evolution—Rethink empowers users to understand the model's uncertainty and decision-making process. Crucially, it enables **fine-grained, inference-time intervention**, allowing users to pinpoint the exact token where reasoning diverges and trigger a "rewind and regenerate" operation. This "point-and-click" capability resolves the ambiguity in natural language feedback and effectively blocks error propagation by physically truncating the erroneous context from the Key-Value (KV) cache.

In summary, our contributions are as follows:
1.  We identify the limitations of conversational correction, specifically error propagation and feedback ambiguity.
2.  We introduce Rethink, a modular framework that combines mechanistic interpretability with human-in-the-loop control for precise reasoning debugging.
3.  We propose diagnostic metrics (e.g., Suspicion Score) to guide user intervention and demonstrate the framework's effectiveness in improving reasoning accuracy and efficiency.

---
### Original Chinese Introduction (for reference)
在目前语言模型领域，人机交互范式主要为人机对话（引用chat gpt等 chat model）。在这种交互方式下，有多种研究有进行，例如长文本、长上下文对话（），多轮对话迭代（），in context learning（不一定合适），对话构建用户偏好、用户画像等（），这些对话交互的目的是为了使模型的输出更贴切用户需求，在某种程度上与人类价值对齐（）。然而，上述的自动化对齐方式虽然提供了一定的算法和指标（），但缺乏具体人的参与，同一算法不一定能够很好的对齐不同人类（用户）的偏好。在学界实验测试及业界用户使用中都会不可避免的出现“这个不对，请修改某一部分”的现象（）。这种情况下模型会把之前的对话当作上下文历史，在重新生成中，错误的token会造成错误传播 Error Propagation”和“反馈歧义 Ambiguity in Feedback”，过长的上下文模型使模型不一定能很好的捕捉用户需要修改的部分的信息（attention机制），同时又重复生成了用户不需要修改的部分，造成了token的浪费（）。

**核心痛点：** 当用户说“不对”时，模型往往不知道具体是哪里不对（是逻辑？是计算？还是事实？），导致模型在错误的道路上越走越远（Hallucination accumulation）。Rethink 解决了“指哪打哪”的问题。

**本文贡献：** 本文提出了 Rethink，一个用于大型语言模型（LLM）推理的交互式、过程导向调试框架。核心思想是通过揭示内部信号如 logits、熵和隐藏状态演化，实现代币级、推理时间的人工干预 ，使用户能够精准定位并纠正自回归生成过程中的错误。这项工作的动机源于现有基于对话的纠正和自我完善方法的局限性，特别是错误传播和反馈中的歧义。论文提出了模块化框架、诊断指标及一组拟议的实验评估。

补充点：需要调研细粒度的人机交互人类偏好对齐（）-别人的做法；自动化方便，但缺乏可解释性；我们采用human in the loop，给人类比较大的权限，具有非常好的可解释性和实用价值。

## Related Work

**Interactive Generation & Human-in-the-Loop Debugging**
Interactive text generation has long been explored to enhance human-AI collaboration, aiming to align model outputs with user intent through iterative feedback. Early works focused on co-authoring tools like Wordcraft (Coenen et al., 2021) and CoAuthor (Lee et al., 2022), which allow users to edit text or request infilling at the sentence level. In the domain of code generation, tools like Debug-gym (Gero et al., 2023) and I-Code-Viz facilitate interactive debugging via test cases. However, these approaches typically operate at the surface level of text or code, treating the model as a black box. Users can modify the *output*, but they lack visibility into the *process* that led to errors. Unlike these methods, Rethink provides a "glass-box" interface, exposing internal signals (logits, entropy) to enable fine-grained, token-level intervention during the generation process.

**Inference-Time Alignment & Backtracking**
Recent advancements in inference-time techniques aim to steer model outputs without expensive parameter fine-tuning. Self-correction methods like Reflexion (Shinn et al., 2023) and Self-Refine (Madaan et al., 2023) prompt models to critique and revise their own outputs. However, these methods rely on the model's self-evaluation capabilities, which can be prone to hallucination and error propagation. To address this, search-based methods such as Tree of Thoughts (ToT) (Yao et al., 2023) and RAIN (Li et al., 2023) introduce backtracking mechanisms, allowing the model to discard low-quality generation paths and "rewind" to previous states. While RAIN automates this process via self-evaluation or heuristic verifiers, it still suffers from the model's inherent calibration errors. Rethink adopts the technical mechanism of backtracking (KV cache rewinding) but shifts the control to the human user, resolving the ambiguity of automated verifiers with precise human judgment.

**Mechanistic Interpretability for Steering**
Understanding the internal dynamics of LLMs is crucial for reliable control. Techniques like Logit Lens (Nostalgebraist, 2020) and DoLa (Chuang et al., 2024) project hidden states into the vocabulary space to reveal the model's evolving confidence and potential hallucinations across layers. Similarly, inference-time intervention (ITI) (Li et al., 2024) steers model behavior by shifting activation vectors. While these works primarily focus on post-hoc analysis or automated steering, Rethink operationalizes these interpretability tools into a real-time debugging interface. We utilize token-level entropy and logit evolution not just for analysis, but as actionable signals to guide human intervention.

## Preliminary

**Autoregressive Language Modeling**
Given a sequence of input tokens $x = (x_1, ..., x_m)$, a Large Language Model (LLM) generates a response $y = (y_1, ..., y_n)$ autoregressively. The joint probability of the sequence is factorized as the product of conditional probabilities:
$$ P(y|x) = \prod_{t=1}^{n} P(y_t | y_{<t}, x) $$
At each time step $t$, the model computes the probability distribution over the vocabulary $V$ via a softmax function applied to the logits $z_t$: $P(y_t | y_{<t}, x) = \text{softmax}(z_t)$. The next token $y_t$ is then selected using a decoding strategy such as greedy decoding (selecting $\arg\max P(y_t)$) or nucleus sampling (Holtzman et al., 2020).

**KV Cache and State Rollback**
To ensure efficient inference, Transformer-based LLMs utilize a Key-Value (KV) Cache mechanism. Instead of recomputing the attention maps for the entire prefix at every step, the model caches the Key and Value matrices of previous tokens. Let $K_{<t}$ and $V_{<t}$ denote the cached states up to step $t-1$. The computation for step $t$ only requires the current input embedding and the cached states:
$$ h_t = \text{Attention}(q_t, K_{<t} \cup k_t, V_{<t} \cup v_t) $$
In the Rethink framework, we leverage this mechanism for efficient backtracking. To "rewind" the generation to time step $t^* < t$, we perform a **KV Cache Slicing** operation, truncating the cache to retain only $K_{\le t^*}$ and $V_{\le t^*}$. This physically restores the model's internal state to exactly how it was at step $t^*$, allowing for clean regeneration without the interference of subsequent erroneous tokens.


## methods
本节将工程实现形式化 (Formalize) 为学术定义，避免写成单纯的代码文档。

### 3.1 Problem Formulation (问题定义)
定义语言模型生成过程为 $P(y_t | y_{<t}, x)$。
定义调试任务的目标：给定一个包含错误的轨迹 $Y_{err}$，找到错误发生的时间步 $t^*$，并通过干预 $Action(t^*)$ 引导模型生成修正后的轨迹 $Y_{corr}$，使得 $Reward(Y_{corr}) > Reward(Y_{err})$。

### 3.2 The Rethink Framework (框架架构)
介绍 Rethink 的三个核心模块及其数据流：
1.  **Generator & Recorder:** 负责自回归生成并捕获中间状态（Hidden States, Logits）。
2.  **Analyzer (Diagnostic Module):** 负责计算诊断指标。
    *   *Entropy Analysis:* 定义 $H(y_t) = -\sum p(y_t) \log p(y_t)$ 作为不确定性指标。
    *   *Semantic Evolution (Logit Lens):* 定义层间预测差异 $D(L_i, L_j)$ 来量化语义漂移。
3.  **Controller (Intervention Module):** 提供交互接口。
    *   *Truncate & Resample:* $y_{>t} \leftarrow \text{Generate}(y_{\le t})$。
    *   *Alternative Selection:* 用户手动选择 $y'_t \in \text{TopK}(P(y_t))$ 替换原 Token。

### 3.3 Diagnostic Metrics (诊断算法)
详细描述如何利用 Token-level 指标自动发现潜在错误点（即你代码中的 `token_analysis.py` 逻辑）。
*   定义 **Suspicion Score (怀疑分数)**：结合熵值和层间一致性，自动高亮可能的错误 Token。
    $$ S(t) = \alpha \cdot \text{Norm}(H(t)) + \beta \cdot \text{Drift}(t) $$

### 3.4 Interactive Steering Mechanism (交互引导机制)
描述用户交互的具体流程：
1.  **Visualization:** 系统如何将 $S(t)$ 映射为热力图。
2.  **Action:** 用户点击 $y_t$ 后，系统如何回滚 KV Cache 并重置生成状态（技术细节：KV Cache Slicing）。
（需要检查软对齐的效果）

## experiment
（不知道）
试试多个数据集mmlu，math500，gsm8k和多个操作base，rethink，rethink with choice，rethink with instruct

为了全面评估 Rethink 框架的有效性、效率以及在对齐任务上的潜力，我们设计了以下四个研究问题 (Research Questions, RQs) 进行实验验证。

### RQ1: Effectiveness (有效性)
**Rethink 能否有效提升推理任务的准确率？**
旨在证明“人机协作”模式在解决复杂推理问题上优于“纯模型生成”或“粗粒度重试”。

*   **Datasets:** GSM8K (Mathematics), Math500 (Harder Math), HumanEval (Code Generation - Optional).
*   **Baselines:**
    1.  **Standard Generation (Zero-shot):** 模型直接贪婪解码或采样生成。
    2.  **Self-Correction (Prompting):** 模型生成错误后，通过 Prompt 提示“答案错误，请重试”进行自我修正。
    3.  **Best-of-N:** 生成 N 条轨迹，通过多数投票或打分模型选择最佳答案（模拟暴力搜索）。
*   **Ours (Rethink):**
    *   **Human-Guided Intervention:** 在模型推理轨迹出现错误的第一个步骤 (Step) 或 Token 处，由人类专家（或模拟策略）点击截断并触发重生成。
*   **Metrics:**
    *   **Pass@1 Accuracy:** 最终答案的准确率。
    *   **Correction Success Rate (CSR):** 在 Baseline 做错的题目集合中，经 Rethink 干预后成功修正的比例。
*   **Hypothesis:** Rethink 的 CSR 显著高于 Self-Correction，证明“定点清除错误”比“盲目重试”更能阻断错误传播。

### RQ2: Efficiency (效率)
**Rethink 是否比传统交互更节省计算资源？**
旨在回应 Introduction 中提到的“Token 浪费”问题，证明细粒度干预的高效性。

*   **Setup:** 记录修正一个错误样本直至正确所需的 Token 总生成量。
*   **Comparison:**
    *   **Method A (Full Regeneration):** 用户发现错误 -> 输入指令 -> 模型重新生成全文。
    *   **Method B (Rethink):** 用户发现错误 -> 点击 Token -> 模型仅生成后续部分 (Completion)。
*   **Metrics:**
    *   **Token Saving Rate (TSR):** $TSR = \frac{Tokens_{Regen} - Tokens_{Rethink}}{Tokens_{Regen}} \times 100\%$。
    *   **Interaction Turns:** 达成目标所需的交互轮数。
*   **Hypothesis:** Rethink 能够节省 30%-50% 的 Token 开销，尤其是在长思维链 (CoT) 场景下。

### RQ3: Interpretability & Utility (可解释性与实用性)
**可视化指标（熵/Logits）能否辅助人类更精准地定位错误？**
旨在验证系统提供的“白盒”信息（Entropy Heatmap, Logit Lens）是否具有实际指导意义。

*   **Ablation Study (消融实验):**
    *   **Setting 1 (Blind Rethink):** 界面仅显示文本，不显示 Entropy 热力图或 Logit Lens 候选词。用户仅凭语义直觉进行干预。
    *   **Setting 2 (Full Rethink):** 界面显示完整可视化信息（高熵区域高亮、Logit Lens 备选词展示）。
*   **Task:** 给定一组包含错误的推理轨迹，要求用户（或标注员）尽快修正。
*   **Metrics:**
    *   **Localization Accuracy:** 用户首次点击的位置与真实错误源头（Root Cause）的距离。
    *   **Trials to Success:** 用户平均需要尝试几次点击才能修好一个 Case。
*   **Hypothesis:** 开启可视化辅助后，用户能更快找到错误的“病灶”，点击次数显著减少。

### RQ4: Alignment & Safety (对齐与安全 - 扩展实验)
**Rethink 在“价值观对齐”任务上表现如何？**
旨在提升论文立意，展示框架在 Safety/Alignment 领域的通用性。

*   **Datasets:** TruthfulQA (Hallucination), HH-RLHF (Harmlessness).
*   **Scenario:** 诱导攻击 (Jailbreak) 或 事实性错误修正。
*   **Operation:** 当模型开始输出有害内容（如“制造炸弹的第一步...”）或幻觉时，利用 Rethink 在第一个有害 Token 处进行干预（选择另一个概率较低但安全的 Token，或截断重写）。
*   **Metrics:**
    *   **Harmlessness Rate:** 干预后输出的安全性比例。
    *   **Truthfulness Score:** 干预后输出的事实准确性得分。
*   **Case Study:** 展示 Logit Lens 截图，对比模型 Top-1（有害/错误）与 Top-2（安全/正确）在层间的演化，证明 Rethink 可以作为“Inference-time Alignment”的有效工具。



## conclusion
Rethink框架的构建基于一个简单的程序员直觉-->*改为“Software Engineering for AI” 或者 “White-box Steering”*：我们在进行大模型生成的时候，能不能像在Ide里面一样加断点，测试逐步生成，了解里面的数据结构？基于这个思想，我们构造Rethink框架来降低用户debug模型生成内容的成本，这种对话型人机交互范式给予用户深入了解的能力，具有更高的可控生成能力与可解释性。

limitation
由于时间及人员原因，该工作并没有像llama factory（）一样构建一个庞大完备的体系，而是做了一个可以随时复现的demo。该工作还有一系列新模型未做适配，未构建适合多种环境的运行框架。该工作由于进行了layer/logits数据结构的存储，对运行内存及并行计算内存的需求较大，计算资源需要增加。该工作是human in the loop，相较于模型自动化的方法具有一定的人工成本。

future work
该工作的交互范式可以作为模型提供给用户的一种选择模式，那些需求较高，对推理过程，偏好对齐可以选择这种模式进行debug。该工作可以进一步完善判断、推荐算法的能力，给用户提供细粒度，个性化的分析及建议。

---

我这里是否能够考虑局部最优和全局最优？（目前算是贪心吗？）
错误传播（居然和我这篇有一点像）
插一帧（token）告诉模型这次生成是否存在问题