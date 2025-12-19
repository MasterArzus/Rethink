# Advanced Uncertainty & Hallucination Metrics

Based on the research into "Trustworthy LLMs", "Semantic Uncertainty", and "Internal State Analysis", here are robust mathematical metrics to replace or augment simple Entropy/Drift.

## 1. Semantic Entropy (SE)
**Source:** Kuhn et al., "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in LLMs"

**Concept:** Simple entropy ($H$) is high if the model is uncertain about the *exact phrasing*, even if the *meaning* is the same (e.g., "Paris" vs. "The city of Paris"). **Semantic Entropy** measures uncertainty over *meanings*, not tokens.

**Algorithm:**
1.  Generate $M$ different sequences $\mathbf{y}^{(1)}, \dots, \mathbf{y}^{(M)}$ from the same prompt $x$.
2.  Group sequences into semantic clusters $C_1, \dots, C_k$ based on bidirectional entailment (does $A \implies B$ and $B \implies A$?).
3.  Sum the probabilities of sequences in each cluster to get cluster probabilities $P(C_j|x)$.
4.  Compute entropy over clusters:
    $$ SE(x) = - \sum_{j=1}^k P(C_j|x) \log P(C_j|x) $$

**Why it's better:** It ignores trivial phrasing variations. High SE strongly correlates with hallucinations (confabulations).

## 2. Layer-wise Divergence (JS Divergence / KL Divergence)
**Source:** "Logit Lens" analysis, "Early Exit" literature.

**Concept:** Measures the disagreement between the model's "subconscious" (early/middle layers) and its "conscious" output (final layer). A sudden shift in distribution often indicates the model is "forcing" a conclusion or hallucinating.

**Metric:** Jensen-Shannon Divergence (JSD) between Layer $L_i$ and Final Layer $L_{final}$.
$$ JSD(P_i || P_{final}) = \frac{1}{2} D_{KL}(P_i || M) + \frac{1}{2} D_{KL}(P_{final} || M) $$
where $M = \frac{1}{2}(P_i + P_{final})$.

**Implementation:**
- For each token $t$, compute the distribution $P_i$ projected from hidden state $h_i$ (using the final LM head).
- Calculate $JSD(P_i, P_{final})$ for $i \in [L_{start}, L_{end}]$.
- **Steering Opportunity:** If JSD is high at late layers (e.g., layer 28/32), the model changed its mind at the last minute.

## 3. Eigengap (Differential Entropy / Spectral Metric)
**Source:** Spectral analysis of internal representations.

**Concept:** If the model is confident, the hidden state $h$ should lie firmly within a specific "meaning manifold". If it's hallucinating, $h$ might be in a "diffuse" region.

**Metric (Simple Proxy - Probability Margin):**
$$ \text{Margin} = P(w_{top1}) - P(w_{top2}) $$
Small margin = High uncertainty.

**Metric (Advanced - Local Intrinsic Dimension / Eigengap):**
Analyze the covariance of the attention value vectors or hidden states across heads.
$$ \text{Eigengap} = \lambda_1 - \lambda_2 $$
(Difference between top two eigenvalues of the local covariance).
*Note: This is computationally expensive for real-time steering.*

## 4. Token Probability Ratios (Log-Prob Drop)
**Source:** "Do Large Language Models Know What They Don't Know?"

**Concept:** Hallucinations often start with a token that has a significantly lower probability than the preceding context would suggest, or a sharp drop in probability compared to the average.

**Metric:**
$$ \text{Ratio} = \frac{P(t_i)}{\text{Avg}(P(t_{i-k} \dots t_{i-1}))} $$
Or simply the **Min-k% Prob**: The minimum probability among the top $k$% of tokens in the generated sequence.

## Recommendation for "Steering Opportunities"

Combine **Semantic Entropy** (if compute allows multiple samples) or **Layer-wise JSD** (single pass) with **Probability Margin**.

**Proposed Composite Metric:**
$$ S_{steering} = \alpha \cdot \text{Entropy}_{\text{semantic}} + \beta \cdot \text{JSD}(L_{mid}, L_{final}) + \gamma \cdot (1 - \text{Margin}) $$
