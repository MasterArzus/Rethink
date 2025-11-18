# Rethink Debugger

Prototype toolkit for capturing per-token statistics from reasoning models and contrasting teacher-forced traces with free-form generations.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run.sh
```

- `run.sh` now defaults to the local `/root/autodl-fs/LLM-Research/Meta-Llama-3.1-8B-Instruct` checkpoint, so no model download is required on this machine. Override it with `MODEL_NAME=... bash run.sh` if you want a different repo or path.
- `run_rethink.py` exposes the full set of knobs (dataset slice, token limits, logging, etc.). Use `python run_rethink.py --help` to discover every option.

## Cached datasets & offline-friendly runs

- GSM8K already lives at `~/.cache/huggingface/datasets/gsm8k/main/0.0.0/...`. The runner always tries the cache first and only hits the network if that fails. Pass `--local-files-only` (or export `HF_DATASETS_OFFLINE=1`) to enforce cache-only behavior:

```bash
HF_DATASETS_OFFLINE=1 python run_rethink.py --local-files-only --limit 2 --setup-only
```

- If the cache is missing, the script automatically retries online; provide `--use-auth-token <HF_TOKEN>` when GSM8K access requires authentication.
- To route online fetches through a mirror, export `HF_ENDPOINT=https://hf-mirror.com` before invoking `run.sh` or `run_rethink.py`.
- Additional dataset knobs (`--dataset-name`, `--dataset-config`, `--split`) let you point at alternative corpora without touching the code.

## Logs and structured outputs

- Every run produces a stem of the form `<model_name>_<dataset_name>_run` (slashes stripped). Logs and summaries are emitted to `outputs/<stem>.log` and `outputs/<stem>.json` by default. Override the stem with `--run-name custom_tag`, or override individual paths via `--log-file` / `--output`.
- The JSON file contains metadata (model, dataset, token caps, device, timestamp) plus the per-example traces so you can diff experiments offline.
- Use `--setup-only` to quickly verify dataset/cache/log paths without instantiating the model—handy when testing configuration changes.

Example manual invocation with explicit naming:

```bash
python run_rethink.py \
	--model-name /root/autodl-fs/LLM-Research/Meta-Llama-3.1-8B-Instruct \
	--limit 5 \
	--run-name llama31_gsm8k_debug \
	--local-files-only
```
