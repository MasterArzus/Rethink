# Rethink: Interactive Steering for Staged Constraint Following

Rethink is a framework for interactive activation steering and staged constraint-following experiments. The current paper revision centers on settings where users receive new requirements over multiple turns and must efficiently repair model outputs.

## Quick Start

Install dependencies in the active server environment:

```bash
/root/miniconda3/bin/python -m pip install -r requirements.txt
```

Run the Streamlit interface:

```bash
cd /root/Rethink
/root/miniconda3/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

The app supports chat and steering modes. `Fast generation` is enabled by default so normal generation does not precompute hidden states for every token; token-level analysis is computed lazily after selecting a token.

## Active Experiments

The old IFEval experiment scripts have been removed. The active experiment suite is:

```text
experiments/revision/
```

Main entrypoints:

- `reflexion.py`: self-repair baseline.
- `autoLR.py`: local self-repair baseline.
- `constraint_decoding.py`: native CD for K=1/2, proxy for K=3, not applicable for K=4 dynamics.
- `chat.py`: LLM-as-Actor chat baseline.
- `steer.py`: LLM-as-Actor steering baseline.
- `steer_lite.py`: rewind-only steering ablation.
- `score.py`: LLM-as-judge scoring from five user perspectives.
- `run_exp.sh`: unified runner.

Run a quick check:

```bash
cd /root/Rethink/experiments/revision
/root/miniconda3/bin/python dataset.py --output data/staged_cases.json
/root/miniconda3/bin/python reflexion.py --dry-run --limit 2
```

Run all methods:

```bash
cd /root/Rethink/experiments/revision
LIMIT_ARGS="--limit 4" ./run_exp.sh
```

Remove `LIMIT_ARGS` for a full run.

## Project Structure

- `app.py`: Streamlit interface for human interaction and steering.
- `rethink/`: Core model wrappers, controllers, recorders, analysis utilities, and logging.
- `dataset/`: Dataset utilities retained for the app and legacy-compatible task loading.
- `experiments/revision/`: Current staged K=8 experiment suite.
- `configs/`: Model, generation, dataset, and prompt configs.
- `docs/`: Notes for the current experimental design.

