# Rethink Project To-Do List & Challenges

## 🚀 To-Do List (Engineering Phase)

### 1. Core Algorithm Upgrade (`rethink/analysis/token_analysis.py`)
- [ ] **Implement KL Divergence:** Create `compute_kl_divergence(logits_mid, logits_final)` to measure internal conflict.
- [ ] **Implement Semantic Similarity:** Create `compute_semantic_similarity(top_k_tokens)` using the model's own input embedding matrix (Zero-cost approach).
- [ ] **Implement Hybrid SOS Metric:** 
    - Combine Internal Conflict and Semantic Ambiguity.
    - **Crucial:** Apply normalization (e.g., Sigmoid or Percentile) to the KL term to ensure it doesn't dominate the score.
    - Formula: $SOS = \text{Normalize}(D_{KL}) \times (1 - \text{Sim})$.

### 2. Configuration System Update (`configs/`)
- [ ] **Update Model Configs:** Add `reference_layer_idx` (e.g., layer 20 for Llama-3-8B) to `configs/models/*.yaml`.
- [ ] **Add Thresholds:** Add `sos_threshold` and `entropy_threshold` parameters for automated steering experiments.

### 3. Automated Evaluation Pipeline (Experiment Infrastructure)
- [ ] **Build "Automated Judge":** Integrate a strong judge model (e.g., GPT-4o or DeepSeek-V3) to identify the *first error step* in a reasoning trace.
- [ ] **Implement Oracle Baseline Script:** 
    - Create `run_oracle_baseline.py`.
    - Logic: When Judge detects error at step $t$, truncate context and append prompt "You made a mistake at step $t$, please rewrite...".
- [ ] **Implement Simulation Script:**
    - Create `run_rethink_simulation.py`.
    - Logic: When SOS > Threshold, automatically trigger intervention (e.g., beam search or rejection sampling) to simulate "User Agency".

### 4. Data Recording & Visualization
- [ ] **Snapshot Recorder:** Ensure `recorder/` can save full Logits/Entropy states for specific interesting steps.
- [ ] **Case Study Visualization:** Prepare scripts to plot "Logit Lens Evolution" for the paper's "Aha!" moment figure (showing how steering fixes internal conflict).

---

## ⚠️ Challenges & Risks

### 1. SOS Metric Stability (Normalization)
- **Risk:** KL Divergence values vary wildly across different models and layers.
- **Mitigation:** Need to run a calibration pass on a small dataset to determine appropriate scaling factors ($\alpha, \beta$) or normalization functions for each model family.

### 2. "Oracle" Implementation Feasibility
- **Risk:** Using GPT-4 as a judge for thousands of GSM8K steps is expensive and slow.
- **Mitigation:** 
    - Use a cheaper but strong model (e.g., DeepSeek-V3 API or a local Llama-3-70B) as the Judge.
    - Or, use a "Known-Answer" approach where we check intermediate equation values against ground truth (if available in the dataset).

### 3. Latency vs. Interactivity
- **Risk:** Calculating Semantic Similarity for Top-K tokens at every step might slow down generation.
- **Mitigation:** 
    - Use the model's input embeddings (lookup table) instead of full forward passes.
    - Optimize tensor operations to run in batch.

### 4. Narrative Consistency
- **Risk:** The paper claims "User Agency," but experiments rely on "Simulation."
- **Mitigation:** Explicitly distinguish between "Simulation Mode" (for measuring theoretical upper bounds in RQ1/RQ2) and "User Study Mode" (for measuring human experience in RQ3).
