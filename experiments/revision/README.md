# Staged Revision Experiments

This directory contains the revised K=8 experiment suite.

## Dataset Design

Each case has four active requirement stages:

- `K=1`: base writing request with length bounds.
- `K=2`: hard lexical or JSON constraints that constrained decoding can address.
- `K=3`: soft tone constraint, plus a weaker constrained-decoding proxy.
- `K=4`: dynamic half-answer constraint that requires inspecting the generated content.
- `K=5..8`: repair turns until the checker passes or the run reaches the final answer.

## Main Entrypoints

- `reflexion.py`: full self-repair after checker feedback.
- `autoLR.py`: local self-repair using an estimated first failure position.
- `constraint_decoding.py`: native CD for K=1/2, proxy for K=3, marked not-applicable for K=4 dynamics.
- `chat.py`: LLM-as-Actor chat feedback with simulated human inspection time.
- `steer.py`: LLM-as-Actor steering with rewind/local continuation prompts.
- `steer_lite.py`: steering ablation with rewind-only information.
- `score.py`: LLM-as-judge scoring from five perspectives.
- `run_exp.sh`: unified runner.
- `run_detached.sh`: nohup runner for server-side jobs.
- `summarize.py`: aggregate per-case JSONL files into table-friendly summaries.

## Output Schema

Each method writes:

- `{model}_{method}.csv`: one row per case and per `k`, with escaped prompt, answer, checker text, actor instruction, timing, and token counts.

Important fields:

- `k`: interaction turn, from 1 to 8.
- `prompt_text`: the exact prompt/user instruction used for this generation.
- `answer`: generated content at this turn.
- `model_time_seconds`: wall-clock model generation time.
- `inspect_time_seconds`: simulated human inspection time for LLM-as-Actor methods.
- `actor_instruction`, `actor_time_seconds`, `actor_tokens`: LLM-as-Actor action record when applicable.
- `generated_tokens`, `prompt_tokens`, `total_tokens`, `actor_tokens`.
- `passed`, `checker_message`, `failed_stage`, `failure_type`.

## Quick Checks

```bash
cd /root/Rethink/experiments/revision
/root/miniconda3/bin/python dataset.py --output data/staged_cases.json
/root/miniconda3/bin/python reflexion.py --dry-run --limit 2
```

Run all methods on the default 1.5B model:

```bash
CASE_LIMIT=4 ./run_exp.sh
```

Defaults:

- `MAX_NEW_TOKENS=1024`.
- Automated methods use greedy decoding (`do_sample=false`).
- LLM-as-Actor methods use sampling (`do_sample=true`, `temperature=0.7`, `top_p=0.9`).
- `RESUME_ARGS=--resume` is enabled by default.

Run one method on one model:

```bash
/root/miniconda3/bin/python steer.py \
  --dataset-path data/staged_cases.json \
  --model qwen2_5_1_5b \
  --output-dir outputs \
  --max-new-tokens 512
```

Use LLM judge for dynamic checks:

```bash
MINIMAX_API_KEY=... /root/miniconda3/bin/python steer.py --use-llm-judge
```

Detached server run:

```bash
ACTOR_API_KEY=... MODEL_ARGS="--model qwen2_5_1_5b --model qwen3_8b" ./run_detached.sh
tail -f logs/*.log
```

Summarize existing outputs:

```bash
/root/miniconda3/bin/python summarize.py --input-dir outputs --output outputs/summary.csv
```
