# GSM8K Prompted Evaluation

This folder contains a lightweight evaluation harness for running the local Meta-Llama-3 8B Instruct checkpoint on the GSM8K math reasoning benchmark while keeping the prompt template editable.

## Prerequisites

1. Activate your Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you can access the GSM8K dataset on Hugging Face (it requires accepting the terms of use). If needed, create an access token at https://huggingface.co/settings/tokens.

## Prompt template

Edit `prompts/gsm8k_prompt.txt` (or point to your own file with `--prompt-file`). The file must contain the `{question}` placeholder, which is replaced with each GSM8K problem statement. After the reasoning trace and the `#### <number>` final answer line, the template now asks the model to report two extra lines:

```
Confidence: <0-100>%
SeenBefore: <true or false>
```

This enables the evaluator to parse the model's self-reported certainty and whether it believes the item was memorized versus solved on the spot.

## Quick validation

Before running full inference you can verify the configuration and prompt without loading the model:

```bash
python eval.py --setup-only
```

Logs are captured with the standard `logging` module. By default they go to `outputs/gsm8k_eval.log` **and** the console; override with `--log-file /path/to/run.log` if you want a different location.

## Running evaluation

```bash
python eval.py \
  --model-path /root/autodl-fs/LLM-Research/Meta-Llama-3___1-8B-Instruct \
  --dataset-name gsm8k \
  --dataset-config main \
  --split test \
  --max-samples 50 \
  --batch-size 1 \
  --output-jsonl outputs/gsm8k_llama3_run.jsonl \
  --use-auth-token <HF_TOKEN>
```

If you prefer a single reusable command, export your Hugging Face token and run the provided helper script (which accepts the same extra flags as `eval.py`):

```bash
export HF_TOKEN=hf_xxx            # required once per shell; must have GSM8K access
./run_gsm8k.sh                    # override defaults via env vars or pass extra CLI args
# Example with overrides:
# MODEL_PATH=/custom/model OUTPUT_JSONL=outputs/custom.jsonl ./run_gsm8k.sh --max-samples 100
```

### Dealing with network or token issues

- GSM8K requires an authenticated download. Pass your token via `--use-auth-token <HF_TOKEN>` or export `HF_TOKEN` before calling `run_gsm8k.sh` (the script forwards it automatically).
- Recent versions of `datasets` renamed `use_auth_token` to `token`; the evaluator automatically detects whichever API your installation provides, so no additional changes are needed.
- If the official hub is slow from your region, point the tooling to a mirror such as `hf-mirror.com` by exporting `HF_ENDPOINT`:

```bash
export HF_ENDPOINT=https://hf-mirror.com
./run_gsm8k.sh --max-samples 10
```

Any `python eval.py ...` invocation in the same shell will reuse that endpoint.

Key options:
- `--prompt-file` or `--prompt-template` to provide a custom prompt.
- `--max-samples` controls how many GSM8K questions to evaluate (set `None` to run the full split).
- `--temperature`, `--top-p`, and `--max-new-tokens` tune generation behavior.

Each sample is saved as JSONL with the rendered prompt, raw prediction, gold answer, and extracted final numbers. A running accuracy is printed by comparing the `#### final` numbers when both predictions and references follow the GSM8K format.

The JSON objects also expose `confidence` (float percentage) and `seen_before` (boolean) derived from the model's structured footer, so you can filter or analyze responses based on self-reported certainty and familiarity.
