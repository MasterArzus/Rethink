# Revision Experiment Design

The current experiment focuses on staged constraint following rather than the old static IFEval setup.

## Core Setup

Each case has up to `K=8` interaction turns.

- `K=1`: base writing request with length bounds.
- `K=2`: hard lexical or JSON constraint that constrained decoding can handle.
- `K=3`: soft tone requirement; constrained decoding only has a weak proxy.
- `K=4`: dynamic/exploratory constraint that depends on inspecting the generated answer.
- `K=5..8`: repair turns until the checker passes or the run reaches the final answer.

The active code lives in:

```text
/root/Rethink/experiments/revision
```

## Methods

- `reflexion.py`: full answer regeneration after checker feedback.
- `autoLR.py`: local repair using an estimated first failure position.
- `constraint_decoding.py`: CD for K=1/2, proxy for K=3, not applicable for K=4 dynamic constraints.
- `chat.py`: LLM-as-Actor as a chat collaborator.
- `steer.py`: LLM-as-Actor using steering actions and local continuation.
- `steer_lite.py`: steering ablation with rewind-only information.

## Metrics

Each method records:

- Stage pass/fail and checker message for `K=1/2/3/4/final`.
- Generated content at each stage.
- Model generation time.
- Simulated inspection time for LLM-as-Actor methods.
- Prompt, generated, total, and actor token counts.
- Final success and pass turn.

## Running

```bash
cd /root/Rethink/experiments/revision
./run_exp.sh
```

For smoke tests:

```bash
/root/miniconda3/bin/python reflexion.py --dry-run --limit 2
```

For scoring:

```bash
/root/miniconda3/bin/python score.py \
  --input outputs/qwen2_5_1_5b_steer.jsonl \
  --samples 10
```

