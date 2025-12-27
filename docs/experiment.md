# Rethink Experimental Design: Steerability & Constraint Satisfaction

This document outlines the revised experimental plan for the Rethink paper, pivoting from general reasoning (GSM8K) to **Steerability and Constraint Satisfaction**. This shift addresses the "Knowledge Asymmetry" problem by focusing on tasks where users can objectively verify success (Instruction Following).

## 🏗️ Infrastructure & Setup

### Base Models
*   **Llama-3-8B-Instruct** (Primary Backbone)
*   **Qwen2.5-7B-Instruct** (Secondary/Validation)

### Datasets: IFEval (Instruction Following Evaluation)
We utilize a curated subset of IFEval, supplemented with custom templates, to create a balanced **120-item Task Set**.
*   **Source:** `dataset/ifeval/taskset_120.json`
*   **Categories:**
    1.  **Forbidden Words (Taboo):** Negative constraints. The model must generate text without using specific forbidden words.
        *   *Why:* Tests the model's ability to suppress high-probability tokens.
        *   *Mechanism:* Users can use **Force** to select alternative synonyms or **Truncate** to rewrite paths.
    2.  **JSON Format:** Structural constraints. The model must generate valid JSON, often with specific keys.
        *   *Why:* Tests syntax and schema adherence.
        *   *Mechanism:* Users can use **Force** to correct syntax errors (e.g., missing quotes, wrong brackets) or **Truncate** to retry generation from a valid state.

---

## 🤖 Part 1: Automated Simulation (RQ1)
*Goal: Demonstrate that "Glass-box Steering" (Intervention) is more efficient than "Black-box Prompting" (Reflexion) for satisfying constraints.*

### Experimental Logic
We simulate a user trying to fix a model's constraint violation.
*   **Task:** The model generates a response. A deterministic **Checker** verifies the constraint.
*   **Baselines (Black-box):**
    1.  **Standard Generation:** Zero-shot Pass@1.
    2.  **Reflexion (Self-Correction):** If Checker fails, the error message is fed back as a prompt: *"You failed the constraint: [Error Details]. Please regenerate."*
*   **Ours (Rethink Simulation):**
    *   **Mechanism:** If Checker fails at token $t$ (or if the final output fails), the system performs a **Local Repair**.
    *   **Action:**
        *   **Truncate:** Rollback to the point of violation (or slightly before).
        *   **Force:** Ban the violating token and sample the next best valid token (or regenerate with a penalty).
    *   **Script:** `experiments/ifeval/run_rethink_simulation.py`

### 📉 Metrics
We focus on three key dimensions to evaluate the "Glass-box" advantage:

1.  **Effectiveness & Efficiency:**
    *   **Constraint Satisfaction Rate (CSR):** % of tasks successfully completed within $N$ turns.
    *   **Token Saving Rate (TSR):** $1 - \frac{\text{Tokens}_{\text{Rethink}}}{\text{Tokens}_{\text{Reflexion}}}$. Measures computational efficiency.
    *   **Effective Token Ratio (ETR):** $\frac{\text{Length of Final Correct Answer}}{\text{Total Token Consumption}}$. Measures generation quality per cost.

2.  **Interaction Process:**
    *   **Turns-to-Success (TTS):** Average number of correction rounds (Prompts vs. Actions).
    *   **Intervention Position:** Relative position of the first error (Early vs. Late).

3.  **Error Dynamics:**
    *   **Error Persistence (Stubbornness):** How many times the *same* error repeats despite feedback. Tests if Rethink breaks "sycophancy loops."
    *   **New Error Introduction Rate:** Probability of introducing new errors during correction.

---

## 👥 Part 2: Human-in-the-Loop Study (RQ2)
*Goal: Validate that Rethink empowers users to fix errors faster and with a greater sense of agency.*

### Study Design
*   **Participants:** N=12-20.
*   **Task:** Each participant performs 10 tasks (5 Taboo, 5 JSON) in each condition.
*   **Conditions (Within-Subject):**
    1.  **Baseline (Chat Mode):** Users see the output. If it fails the constraint (indicated by the Checker), they must prompt the model to fix it (e.g., "You used the word 'apple', remove it").
    2.  **Rethink (Steering Mode):** Users see the output. They can click on tokens to **Truncate** (rewind) or **Force** (select alternatives) to fix violations directly.

### Interface Features
*   **Live Checker:** The app displays a "Pass/Fail" status in real-time (or after generation).
*   **Visual Feedback:** In Rethink mode, violating tokens (e.g., forbidden words) could be highlighted (optional, but helpful).

### 📉 Metrics
**Objective:**
*   **Time-to-Success (sec):** Total time from task start to passing output.
*   **Correction Turns:** Number of prompts (Baseline) vs. Number of Actions (Rethink).
*   **CSR:** Success rate given a fixed time limit (e.g., 5 mins per task).

**Subjective (Questionnaire):**
*   **Agency:** "I felt in control of the model's output."
*   **Frustration:** "How frustrated were you during the task?"
*   **Transparency:** "I understood why the model made the mistake."

---

## 🛠️ Implementation Roadmap

### 1. Data & Checkers
*   [x] **Task Set:** `dataset/ifeval/taskset_120.json` generated.
*   [ ] **Checkers:** Implement `dataset/ifeval/checkers.py` containing:
    *   `TabooChecker`: Checks for forbidden words (word boundary aware).
    *   `JsonChecker`: Checks for valid JSON and required keys.

### 2. App Integration
*   [ ] **Task Loader:** Update `app.py` to load tasks from `taskset_120.json`.
*   [ ] **Checker Integration:** Display "Pass/Fail" status in the UI.
*   [ ] **Logging:** Ensure all Truncate/Force actions and Chat messages are logged with timestamps.

### 3. Simulation Scripts
*   [ ] **Reflexion Baseline:** Script to run the task set with feedback loops.
*   [ ] **Rethink Simulation:** Script to run the task set with automated rollback/forcing.
