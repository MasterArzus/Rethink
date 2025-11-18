# Rethink Debugger Gameplan

## Goals
- Capture per-token probabilities and hidden states for both *teacher-forced* (reference) and *free-form* (model) generations.
- Compare traces to highlight spans where reasoning diverges.
- Provide hooks to intervene mid-generation instead of restarting from scratch.

## Package Layout
```
rethink/
  config.py                  # Shared dataclasses for dataset/model instrumentation config
  data/                      # Benchmark preparation utilities
  instrumentation/           # HF model overrides + hook management
  analysis/                  # Divergence metrics + visualization stubs
  pipeline/                  # Controllers orchestrating the workflow
```

### Dataset Layer
- `BenchmarkExample`: pairs a reasoning question with one correct answer and curated incorrect answers where only intermediate reasoning is flawed.
- `gsm8k.load_gsm8k_slice`: converts the first *n* rows into the unified format; placeholder creates trivial distractors until annotated ones exist.
- Extend with loaders for MMLU, Humaneval, etc. by returning the same dataclasses.

### Model Layer
- `InstrumentedLlamaForCausalLM`: thin wrapper around HF `LlamaForCausalLM`.
  - `collect_forced_trace`: consumes a prompt + known answer to capture teacher-forced probabilities.
  - `generate_autoregressive_trace`: mirrors greedy/sampled decoding while storing traces.
  - Hooks rely on `HiddenStateRecorder`, which registers per-layer forward hooks.
  - `intervene_from_span`: stub for future interventions (logit biasing, KV cache surgery, constrained decoding, etc.).
- To support Qwen / other models, add peer modules under `rethink/instrumentation` that implement the same trace API.

### Analysis Layer
- `compare_traces`: aligns token probability gaps and hidden-state cosine distances, emitting `DivergenceReport` with suspicious spans.
- Visualization functions remain stubs (`render_prob_trajectory`, `render_hidden_state_heatmap`) until UX requirements solidify.

### Pipeline Layer
- `RethinkController`: wires dataset examples, instrumented models, and analysis. Returns both raw traces and divergence reports for notebooks or dashboards.
- Future: implement `intervene` to call `model.intervene_from_span` once confident heuristics are available.

## Open Questions / Next Steps
1. **Better incorrect answers**: need semi-automatic perturbation tools that tweak intermediate reasoning instead of naive suffix edits.
2. **Efficient hidden-state storage**: current recorder clones tensors to CPU; consider quantization or chunk-wise persistence for long generations.
3. **Visualization spec**: decide between lightweight Matplotlib vs. interactive dashboards (Plotly, Streamlit) before implementing stubs.
4. **Reflection strategies**: define how to modify hidden states/logits when a suspicious span is detected (e.g., re-encoding corrected sub-plan, mixing gold states, or rerunning selected layers only).
