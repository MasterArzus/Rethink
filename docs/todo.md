# Rethink Project To-Do List & Challenges

## 🚀 To-Do List (Engineering Phase)

### 1. Core Algorithm Upgrade (`rethink/analysis/token_analysis.py`)
- [x] **Implement KL Divergence:** Create `compute_kl_divergence(logits_mid, logits_final)` to measure internal conflict.
- [x] **Implement Semantic Similarity:** Create `compute_semantic_similarity(top_k_tokens)` using the model's own input embedding matrix (Zero-cost approach).
- [x] **Implement Hybrid SOS Metric:** 
    - Combine Internal Conflict and Semantic Ambiguity.
    - **Crucial:** Apply normalization (e.g., Sigmoid or Percentile) to the KL term to ensure it doesn't dominate the score.
    - Formula: $SOS = \text{Normalize}(D_{KL}) \times (1 - \text{Sim})$.

### 2. Configuration System Update (`configs/`)
- [x] **Update Model Configs:** Add `reference_layer_idx` (e.g., layer 20 for Llama-3-8B) to `configs/models/*.yaml`.
- [x] **Add Thresholds:** Add `sos_threshold` and `entropy_threshold` parameters for automated steering experiments.

### 3. Automated Evaluation Pipeline (Experiment Infrastructure)
- [x] **Build "Automated Judge":** Integrate a strong judge model (e.g., GPT-4o or DeepSeek-V3) to identify the *first error step* in a reasoning trace. (Infrastructure ready in `run_oracle_baseline.py`)
- [x] **Implement Oracle Baseline Script:** 
    - Create `run_oracle_baseline.py`.
    - Logic: When Judge detects error at step $t$, truncate context and append prompt "You made a mistake at step $t$, please rewrite...".
- [x] **Implement Simulation Script (Glass-box User):**
    - Create `run_rethink_simulation.py`.
    - Logic: When SOS > Threshold, trigger LLM Judge.
    - **New:** Provide Judge with Logit Lens + Self-Explanation to simulate "Glass-box" visibility.
- [ ] **Implement Traditional Interaction Baseline (Black-box User):**
    - Create `run_dialogue_baseline.py` (or extend `run_oracle_baseline.py`).
    - Logic: LLM Judge sees *only* the text output. It must use natural language prompts to correct the model ("I think you are wrong because...").
    - Goal: Compare "Token Selection" (Rethink) vs "Prompting" (Traditional) efficiency.

### 4. Data Recording & Visualization
- [ ] **Snapshot Recorder:** Ensure `recorder/` can save full Logits/Entropy states for specific interesting steps.
- [x] **Case Study Visualization:** Prepare scripts to plot "Logit Lens Evolution" for the paper's "Aha!" moment figure (showing how steering fixes internal conflict).
- [ ] **Ablation Study:** Verify if "Self-Explanation" actually helps the LLM Judge make better decisions compared to just seeing the Logit Lens or just the text.

### 5. Experiments Execution (New)
- [ ] **Run Exp 1.1 (Precision/Efficiency):** Execute `run_rethink_simulation.py` on GSM8K and TruthfulQA.
- [ ] **Run Exp 1.2 (Sycophancy):** Execute `run_sycophancy_test.py` and calculate Flip Rate.
- [ ] **Run Exp 2.1 (User Study):** 
    - [ ] Verify `app.py` heatmap visualization (ensure `is_critical` logic is visible).
    - [ ] Recruit N=10 participants.
    - [ ] Collect logs from `interactive_sessions/`.

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

### 5. User Study Robustness (HCI Track)
- **Risk:** N=10 is small for a full ACL paper.
- **Mitigation:** Frame it as a "Pilot Study" or "Qualitative Analysis" if N cannot be increased. Focus on the *depth* of insight (e.g., "Why did users intervene here?") rather than just p-values.
