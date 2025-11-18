# GSM8K Hidden-State Rethink Analysis

This repository now focuses on probing **token-level hidden states** while a Meta-Llama-3 8B Instruct checkpoint solves GSM8K questions. The refreshed `rethink_run.py` workflow:

1. Fetches a GSM8K test sample (defaults to the very first item) and renders the prompt template in `prompts/gsm8k_prompt.txt`.
2. Runs the model autoregressively and records every newly generated token's probability, top-k alternatives, and per-layer hidden-state norms.
3. Performs a teacher-forced pass over the ground-truth answer to compute token probabilities, entropy, and similarity between answer states and the prompt context.
4. Saves a structured JSON report plus optional visualizations (heatmaps/curves) so you can inspect when the model stays on a “trustworthy” trajectory.

The intent is to validate that the model can still reason end-to-end while exposing enough intermediate signals to decide whether a token (or group of tokens) helps the solution, mimicking a reflection or debugging loop.

## Prerequisites

```bash
pip install -r requirements.txt
```

> **Note:** You must have accepted the GSM8K dataset terms on Hugging Face. If the official endpoint is slow, export `HF_ENDPOINT=https://hf-mirror.com` (already the default in the helper script).

## Running the analysis

The easiest entrypoint is the shell helper:

```bash
# Optional overrides before running
export MODEL_PATH=/root/autodl-fs/LLM-Research/Meta-Llama-3___1-8B-Instruct
export DATASET_INDEX=0            # focus on gsm8k test[0]
export OUTPUT_DIR=outputs/analysis/gsm8k_0

bash run_rethink.sh              # adds --visualize by default
```

Or call the Python CLI directly:

```bash
python rethink_run.py \
  --model-path /root/autodl-fs/LLM-Research/Meta-Llama-3___1-8B-Instruct \
  --prompt-file prompts/gsm8k_prompt.txt \
  --dataset-name gsm8k --dataset-config main --dataset-split test --dataset-index 0 \
  --max-new-tokens 96 --top-k 5 --visualize
```

Key switches:

- `--dataset-index` – pick which GSM8K test item to inspect.
- `--max-new-tokens` – how many tokens to let the model produce before stopping.
- `--top-k` – number of alternative tokens to log per decoding step.
- `--visualize` – save `layer_norm_heatmap.png` and `reference_probabilities.png` next to the JSON report.

All logs land in `<output_dir>/analysis.log`, the JSON report in `<output_dir>/token_analysis.json`, and optional PNGs in the same directory.

## Output anatomy

`token_analysis.json` includes:

- `generated_tokens`: probability, entropy, per-layer norm vector, and competing tokens for every generated token.
- `reference_tokens`: teacher-forced probabilities plus cosine similarity between each answer token's hidden state and the averaged prompt state (useful for gauging how much the answer “sticks” to the question context).

These stats align with the four guiding goals:

1. **Normal reasoning** – you can still read the final answer text.
2. **Per-token probabilities** – mirrors the `entropy.py` idea but bundled in the report.
3. **Hidden-state correlations** – norms and cosine similarities highlight when layers diverge.
4. **Visualization-ready cache** – heatmaps make it easy to eyeball the layers or steps worth “restarting” from if you later build a controller.

## Relationship to the `rethink` package

The lower-level `rethink/` modules are unchanged: adapters cache arbitrary layers, expose detokenization utilities, and contain scoring hooks for future confidence-driven rewind policies. The new CLI simply assembles these pieces into a GSM8K-focused probe so you can iterate on reflection strategies quickly.

### Quick smoke tests

```bash
python -m unittest tests/test_rethink.py
```

Use the tests after editing any core components to ensure the cache/controller plumbing remains intact.