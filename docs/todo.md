# Rethink Project To-Do List

## 🚀 To-Do List (Steerability Pivot)

### 1. Data & Infrastructure (IFEval/Taboo)
- [x] **Task Set Generation:** Created `dataset/ifeval/taskset_120.json` with 60 Taboo and 60 JSON tasks.
- [ ] **Implement Checkers:** Create `dataset/ifeval/checkers.py`.
    - `TabooChecker`: Regex-based word boundary check.
    - `JsonChecker`: `json.loads` + key existence check.
- [ ] **Unit Test Checkers:** Verify checkers against known pass/fail examples.

### 2. App Integration (Human Study)
- [ ] **Task Loader:** Modify `app.py` to read from `taskset_120.json`.
- [ ] **State Management:** Add session state for `current_task_index`, `task_start_time`, `attempts`.
- [ ] **Checker UI:**
    - Add a visual indicator (Green Check / Red X) for the current output.
    - Show specific error messages (e.g., "Found forbidden word: 'apple'", "Invalid JSON: missing key 'response'").
- [ ] **Logging Update:** Ensure `outputs/interactive_sessions/` logs include:
    - Task ID.
    - Condition (Chat vs. Rethink).
    - Time stamps for every action.
    - Final Pass/Fail status.

### 3. Automated Simulation (RQ1)
- [ ] **Reflexion Baseline Script:** `experiments/ifeval/run_reflexion.py`.
    - Loop: Generate -> Check -> If Fail, Prompt with Error -> Retry (Max K times).
- [ ] **Rethink Simulation Script:** `experiments/ifeval/run_rethink_simulation.py`.
    - Loop: Generate -> Check -> If Fail, Truncate to error index -> Force/Ban token -> Continue.
- [ ] **Metric Calculation:** Script to parse logs and compute CSR and TSR.

### 4. Human Study Execution (RQ2)
- [ ] **Pilot Run:** Test the app with 1-2 internal users to verify flow.
- [ ] **Recruitment:** Gather N=12 participants.
- [ ] **Data Collection:** Run sessions and collect logs.
- [ ] **Analysis:** Compare Time-to-Success and CSR between conditions.

### 5. Paper Writing
- [ ] **Methodology:** Describe the "Glass-box Steering" vs "Black-box Prompting" paradigm.
- [ ] **Results:** Fill in the tables defined in `experiment.md`.
