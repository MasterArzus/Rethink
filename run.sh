
#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-"/root/autodl-fs/LLM-Research/Meta-Llama-3.1-8B-Instruct"}
LIMIT=${LIMIT:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
GEN_MAX_TOKENS=${GEN_MAX_TOKENS:-128}

python run_rethink.py \
	--model-name "$MODEL_NAME" \
	--limit "$LIMIT" \
	--max-new-tokens "$MAX_NEW_TOKENS" \
	--generation-max-new-tokens "$GEN_MAX_TOKENS"
