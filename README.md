# Rethink: Bridging the Agency Gap in LLM Reasoning

**Rethink** is a framework for **Interactive Activation Steering**, designed to empower users with the "Right to Know" (transparency via Logit Lens & Entropy) and the "Right to Choose" (steerability via KV Cache manipulation).

This project implements the "Steering Opportunity Score" (SOS) to automatically detect internal conflicts in LLMs and allows for surgical interventions during the generation process.

## 🚀 Quick Start

### 1. Installation

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Interactive Framework (Streamlit App)

The best way to understand Rethink is to visualize the generation process.

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```
*   **Features**:
    *   Load models (Llama-3, DeepSeek, etc.).
    *   Visualize Token-level Entropy and Logit Lens.
    *   Manually intervene (truncate/rewrite) at specific steps.

### 3. Automated Simulation (RQ1 & RQ2)

To evaluate the efficacy of the **Steering Opportunity Score (SOS)** at scale, use the simulation script. This script automatically triggers an intervention (e.g., "Reject Top-1") when the SOS metric exceeds a threshold.

```bash
python run_rethink_simulation.py \
    --model-path /root/autodl-fs/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --sos-threshold 0.3 \
    --prompt "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:"
```

### 4. Oracle Baseline (Prompting Comparison)

Run the "Oracle Prompting" baseline to compare Steering against traditional Prompt Engineering. This script uses a Judge to detect errors and re-prompt the model.

```bash
python run_oracle_baseline.py \
    --model-path /root/autodl-fs/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --dataset gsm8k
```

## ⚙️ Configuration

### Model Configs (`configs/models/`)
You can define model-specific parameters in YAML files.
*   `reference_layer_idx`: The layer index used as the "truth" reference for calculating Internal Conflict (KL Divergence).
    *   Llama-3-8B: `20`
    *   DeepSeek-R1: `20`

### Metrics (`rethink/analysis/token_analysis.py`)
*   **Internal Conflict**: KL Divergence between intermediate and final layer logits.
*   **Semantic Ambiguity**: Cosine similarity of top-k token embeddings.
*   **SOS**: $\tanh(D_{KL}) \times (1 - \text{Sim})$.

## 📂 Project Structure

*   `app.py`: Streamlit web application for interactive debugging.
*   `run_rethink_simulation.py`: Script for automated steering experiments (RQ1/RQ2).
*   `run_oracle_baseline.py`: Script for prompting baselines.
*   `rethink/`: Core library.
    *   `analysis/`: Metric calculations (SOS, Entropy).
    *   `engine/`: Model wrappers and generation controllers.
    *   `recorder/`: Data structures for capturing hidden states.
*   `configs/`: Configuration files.
*   `docs/`: Experimental guides and research notes.

## 📝 Notes
*   **Cached Datasets**: The runner tries to use cached datasets first. Use `HF_DATASETS_OFFLINE=1` to force offline mode.
*   **Git Config**: If you encounter connection issues, try `git config --global http2.maxrequests 0`.
