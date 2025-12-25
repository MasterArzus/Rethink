# ACL 2026 Sketch for Rethink
## Title
Rethink: Bridging the Agency Gap in LLM Reasoning via Interactive Activation Steering
*Subtitle: Empowering Users with the "Right to Know" and "Right to Choose" in Generative AI*

## Introduction

The prevailing paradigm in Human-Computer Interaction (HCI) for Large Language Models (LLMs) is **"Black-box Prompting"** (e.g., ChatGPT, Claude). In this mode, users act as "prompters" who submit natural language queries and passively await the model's output. While techniques like Chain-of-Thought (CoT) and Multi-turn Prompt Engineering (e.g., Reflexion) have improved model performance, they fundamentally limit **User Agency**. When a model hallucinates or deviates from user intent, the user's only recourse is to provide external feedback (e.g., "You are wrong, try again") and hope the model's internal "black box" self-corrects.

This interaction model suffers from a critical **"Agency Gap"**:
1.  **Lack of Transparency ("Right to Know"):** Users are blind to the model's internal decision-making process. They cannot see *why* the model chose a specific path, nor can they see the alternatives the model discarded (e.g., a correct answer that had slightly lower probability).
2.  **Lack of Control ("Right to Choose"):** Users cannot intervene in the generation process. They can only reject the *outcome*, not the *process*. This leads to **inefficient correction loops**, where users struggle to "prompt" the model out of a stubborn error mode (e.g., Sycophancy or Logical Loops).

We argue that the next generation of AI interaction must shift from **"Prompting"** to **"Steering."** To truly align LLMs with complex human intent, we must restore the user's **Right to Know** (by visualizing internal uncertainty and alternatives) and **Right to Choose** (by allowing direct intervention in the generation trajectory).

To this end, we propose **Rethink**, a framework for **Interactive Activation Steering**. Unlike static steering methods (e.g., Inference-Time Intervention) that apply a global vector, Rethink enables **fine-grained, human-in-the-loop steering**. It opens the "black box" by visualizing token-level entropy and logit distributions, allowing users to identify "high-leverage points" where the model is uncertain or biased. Users can then act as a "co-pilot," surgically truncating erroneous paths in the KV cache and selecting alternative directions.

Our experiments demonstrate that this "Glass-box" approach is not only more **precise** than multi-turn prompting (avoiding error propagation) but also unlocks the latent potential of base models, allowing them to solve complex reasoning tasks (e.g., GSM8K) that typically require extensive fine-tuning.

In summary, our contributions are:
1.  **Concept:** We redefine the HCI for LLMs around **User Agency**, proposing the "Right to Know" (Transparency) and "Right to Choose" (Steerability) as core principles for alignment.
2.  **Framework:** We introduce Rethink, which operationalizes **Interactive Activation Steering**, bridging the gap between mechanistic interpretability and user interaction.
3.  **Validation:** We show that giving users "steering rights" significantly outperforms traditional "prompting" in both reasoning accuracy and token efficiency, effectively mitigating hallucinations and logic errors.

## Related Work

**Prompt Engineering vs. Activation Steering**
Current approaches to controlling LLMs fall into two categories. **Prompt Engineering** (e.g., *Reflexion* [Shinn et al., 2023], *Self-Refine* [Madaan et al., 2023]) treats the model as a black box, using natural language to guide behavior. While accessible, it is often "brittle" and imprecise—a single prompt change can lead to unpredictable outputs. In contrast, **Activation Steering** (or *Representation Engineering* [Zou et al., 2023], *Inference-Time Intervention* [Li et al., 2023]) directly manipulates the model's internal representations (activations) to enforce behaviors like truthfulness or harmlessness.
**Rethink's Position:** We argue that Prompt Engineering lacks **precision** (it cannot fix a specific neuron/token), while current Activation Steering lacks **flexibility** (it usually applies a static vector). Rethink combines the best of both: the **flexibility** of human interaction with the **precision** of white-box steering.

**Human-AI Collaboration & Agency**
Early tools like *Wordcraft* (Coenen et al., 2021) allowed for "Co-writing," but users were limited to editing text. They could not see *why* the model wrote what it wrote. Recent work in **Mechanistic Interpretability** (e.g., *Logit Lens*) has developed tools to inspect the "mind" of the LLM, but these are typically reserved for researchers, not end-users.
**Rethink's Contribution:** We democratize these interpretability tools, turning them into **interaction primitives**. By giving users the "Right to Know" (visualizing Entropy/Logits), we empower them to make informed decisions, moving beyond "Human-in-the-loop" to **"Human-on-the-steering-wheel."**

## Preliminary

**Autoregressive Generation & State Space**
We model the generation process as a trajectory through a state space. At time step $t$, the model state $S_t$ consists of the Key-Value (KV) cache of all preceding tokens: $S_t = \{(K_i, V_i)\}_{i=1}^{t-1}$. The probability of the next token $y_t$ is conditioned on this state and the current input $x$:
$$ P(y_t | y_{<t}, x) = \text{softmax}(W_U \cdot f(S_t, y_{t-1})) $$
where $f$ is the Transformer forward pass and $W_U$ is the unembedding matrix.

**Intervention Operators**
Standard generation is a linear extension of the trajectory: $S_{t+1} = S_t \cup \{(K_t, V_t)\}$.
In Rethink, we define an **Intervention Operator** $\mathcal{I}(S_t, a)$ that modifies the trajectory based on user action $a$:
1.  **Truncation (Rewind):** If the user rejects token $y_t$, we perform a **KV Cache Slicing** operation. The state is reset to $S_t$, physically discarding all subsequent computations ($S_{>t} \leftarrow \emptyset$).
    $$ \mathcal{I}(S_{t+k}, \text{truncate}(t)) = S_t $$
2.  **Steering (Resample):** The user can force the selection of an alternative token $y'_t$ (e.g., from the top-k logits). This branches the trajectory into a new path:
    $$ S'_{t+1} = S_t \cup \{(K'_t, V'_t)\} \quad \text{where } y_t = y'_t $$

This formulation highlights that Rethink operates directly on the **memory state (KV Cache)** of the model, ensuring that interventions are mathematically exact and computationally efficient (avoiding re-computation of the prefix).


## Methods
We formalize Rethink as a framework for **Interactive Activation Steering**, designed to operationalize the principles of "Right to Know" and "Right to Choose."

### 3.1 Problem Formulation
Let the generation process be $P(y_t | y_{<t}, x)$.
We define the **Steering Task**: Given a trajectory $Y_{err}$ where the model deviates from user intent (e.g., hallucination, logic error), the goal is to identify the **Bifurcation Point** $t^*$ (the exact moment the model "thought" wrong) and apply a steering action $A(t^*)$ to guide the model to a target trajectory $Y_{target}$, such that $Utility(Y_{target}) > Utility(Y_{err})$.
Unlike Prompt Engineering which appends a correction $c$ to the context ($x, Y_{err}, c$), Rethink modifies the state $S_{t^*}$ directly.

### 3.2 The Rethink Framework
Rethink consists of three modules corresponding to the user's rights:

1.  **Generator & Recorder:** Handles autoregressive generation and captures the "Mental State" (Hidden States, Logits, Attention).
2.  **Transparency Module ("Right to Know"):** Visualizes the internal state to expose uncertainty.
    *   *Entropy Map:* $H(y_t) = -\sum p(y_t) \log p(y_t)$. High entropy indicates the model is "confused."
    *   *Logit Lens:* Displays the top-k candidates at each step, revealing "what the model almost said."
3.  **Steering Module ("Right to Choose"):** Provides primitives for intervention.
    *   *Truncate (Veto):* Physically removes $S_{>t}$ from the KV Cache.
    *   *Resample (Steer):* Forces the selection of $y'_t \in \text{TopK}$ or injects a custom token, branching the trajectory.

### 3.3 Steering Opportunity Detection
To assist users in finding the "Bifurcation Point," we propose a mathematically grounded **Steering Opportunity Score (SOS)**. Unlike simple entropy which is sensitive to trivial synonym variations (e.g., "happy" vs "glad"), our metric combines **Internal Conflict** and **Semantic Ambiguity**:
$$ SOS(t) = \underbrace{D_{KL}(P_{final} || P_{mid})}_{\text{Internal Conflict}} \times \underbrace{(1 - \text{SemanticSim}(TopK))}_{\text{Semantic Ambiguity}} $$
*   **Internal Conflict (Layer-wise Divergence):** Measured by the KL Divergence (or Jensen-Shannon Divergence) between the logits of an intermediate layer ($P_{mid}$) and the final layer ($P_{final}$). High divergence suggests the model is "suppressing" its initial intuition (potential hallucination or sycophancy).
*   **Semantic Ambiguity (Semantic Entropy):** Measures whether the top-k candidates belong to different semantic clusters (using embedding cosine similarity). This filters out false positives where the model is uncertain only about surface phrasing.
High SOS indicates a **"High-Leverage Point"** where the model is internally conflicted about the *meaning* of the output, making it an ideal candidate for steering.

### 3.4 Interactive Steering Mechanism
The interaction loop is defined as:
1.  **Observe:** User views the generated text overlaid with the SOS heatmap.
2.  **Diagnose:** User clicks a high-SOS token to inspect the Logit Lens (seeing alternatives).
3.  **Act:** User selects a better alternative or truncates the sequence.
4.  **Update:** The system rewinds the KV cache to $t^*$ and continues generation from the new state.

### 3.5 Implementation Details
We implemented the Rethink framework with the following components:
*   **Metric Calculation (`rethink/analysis/token_analysis.py`):** 
    *   **Internal Conflict:** Computed via `compute_kl_divergence` between the logits of an intermediate layer (e.g., layer 15) and the final layer (e.g., layer 20/32).
    *   **Semantic Ambiguity:** Computed via `compute_semantic_similarity` using the cosine similarity of the input embeddings of the top-k tokens.
    *   **SOS Metric:** $SOS = \tanh(D_{KL}) \times (1 - \text{Sim})$.
*   **Simulation Engine (`run_rethink_simulation.py`):** To evaluate the framework at scale without human intervention, we implemented a simulation script that automatically triggers interventions when the SOS score exceeds a threshold (e.g., 0.3).
*   **Oracle Baseline (`run_oracle_baseline.py`):** A baseline system that uses an external judge to detect errors and truncate the generation, simulating a perfect "Right to Know" but limited "Right to Choose" (prompting only).

## Experiments

To validate the "Agency" paradigm, we compare Rethink against state-of-the-art Multi-turn Prompting methods.

### RQ1: Precision & Effectiveness (Steering vs. Prompting)
**Can "White-box Steering" outperform "Black-box Prompting" in correcting reasoning errors?**
*   **Hypothesis:** Prompting suffers from "Sycophancy" (agreeing with the user but not fixing the root cause) and "Error Propagation." Steering fixes the root cause.
*   **Datasets:** GSM8K (Reasoning), TruthfulQA (Hallucination).
*   **Baselines:**
    1.  **Standard Generation:** Zero-shot CoT.
    2.  **Reflexion (Shinn et al., 2023):** Automated self-correction via prompting ("You are wrong, fix it").
    3.  **Oracle Prompting (Strong Baseline):** A theoretical upper bound where the user identifies the exact error location $t^*$ immediately but is restricted to natural language feedback (e.g., "You made a mistake at step $t^*$, please rewrite from there"). This isolates the benefit of *Intervention Mechanism* (KV manipulation) from *Information Advantage*.
*   **Ours (Rethink Simulation):** Automatically trigger KV-Cache truncation at the first error step (detected by Ground Truth or SOS).
*   **Metrics:**
    *   **Correction Success Rate (CSR):** % of errors fixed.
    *   **Faithfulness:** Does the model actually change its internal logic, or just the output surface form?

### RQ2: Efficiency (The Cost of Correction)
**Does "Right to Choose" (Intervention) save computational resources?**
*   **Hypothesis:** Rethink avoids re-generating the correct prefix and the erroneous suffix, saving significant tokens.
*   **Metrics:**
    *   **Token Saving Rate (TSR):** $1 - \frac{Tokens_{Rethink}}{Tokens_{Prompt}}$.
    *   **Interaction Turns:** Number of turns to reach the correct answer.

### RQ3: The Value of "Right to Know" (Ablation Study)
**Does visualizing internal states (Entropy/Logits) actually help users steer?**
*   **Setup:** User study with two conditions:
    *   **Blind Steering:** Users can truncate/edit, but see no Entropy/Logits.
    *   **Informed Steering (Rethink):** Users see the full dashboard (SOS Heatmap).
*   **Task:** Fix errors in Math500.
*   **Metrics:** Time to fix, Number of trials, User confidence ratings, Localization Error.

### RQ4: Alignment & Safety (Robustness to Sycophancy)
**Can Rethink enforce safety constraints where Prompting fails?**
*   **Hypothesis:** Prompting is vulnerable to user bias (Sycophancy); Rethink (Internal State) is robust.
*   **Setup:** Attack the model by appending "I think the answer is [Wrong Answer]. Are you sure?" to correctly answered questions.
*   **Method:** Check **Internal Confidence** (Entropy/Logit Gap). If internal confidence is high, **refuse to flip**.
*   **Metric:** **Flip Rate** (Lower is better).

上下文召回率


## Conclusion
The construction of the Rethink framework is grounded in the philosophy of **"Software Engineering for AI"** and **"White-box Steering."** Just as developers use breakpoints and step-by-step debugging in an IDE to understand code execution, we ask: can we apply similar principles to LLM generation? Rethink answers this by providing a framework that lowers the cost of debugging model outputs. This conversational, human-in-the-loop paradigm empowers users with deep insight into the model's internal states, offering higher controllability and interpretability than traditional black-box methods.

## Limitations
Due to time and resource constraints, this work is presented as a reproducible demo rather than a comprehensive system like LLaMA Factory. We have not yet adapted the framework to a wide range of new models or built a runtime environment suitable for diverse deployment scenarios. Additionally, storing layer-wise logits and hidden states requires significant memory and computational resources. Finally, as a human-in-the-loop approach, Rethink incurs a higher labor cost compared to fully automated methods, though it offers greater precision.

机器占主导地位的话效率提高，人占主导地位的话质量提高

## Future Work
The interaction paradigm introduced by Rethink can serve as an advanced "Debug Mode" for users with high requirements for reasoning accuracy and preference alignment. Future work will focus on improving the **Steering Opportunity Score (SOS)** and recommendation algorithms to provide users with more fine-grained, personalized analysis and suggestions. We also aim to explore **automated steering** strategies that can learn from human interventions to reduce the need for manual oversight.
