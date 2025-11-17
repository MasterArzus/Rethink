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

## Rethink instrumentation

The `rethink/` package is now split into a model-agnostic core and model-specific adapters:

- `rethink/core/`: cache, metrics, controller, options, and `RethinkEngine` (no HF deps).
- `rethink/adapters/`: plug-ins per foundation model (`llama.py` today, `qwen.py` next).
- `rethink/core/base.py`: interface every adapter implements so experiments can swap models freely.

This lets you:
- capture per-layer hidden states during generation and store them in a `HiddenStateCache`
- detokenize cached states via `HiddenStateDecoder` to inspect what a layer “wanted” to say
- compute same-layer and cross-layer similarities with `ConfidenceScorer`
- plug the scores into `RethinkController` or custom policies via `RethinkEngine`

### Quickstart

```bash
python examples/rethink_demo.py
```

The script instantiates a toy 2-layer Llama adapter, builds a `RethinkEngine`, runs generation, prints cache stats, and reports controller decisions. Swap in future adapters (e.g., Qwen) by changing the import only.

### Tests

Run the lightweight unit tests (all CPU-friendly):

```bash
python -m unittest tests/test_rethink.py
```


在这个rethink_modeling.py文件中，我想实现一些修改transformers中的modeling_llama.py中的某些功能：1.我可以自定义提取在推理时第几层生成的隐状态（隐向量），然后给每一层设置一个cache用于存储这些隐向量（隐状态）用于后续比对；2，我想实现可以用户选择对特定层的cache中的隐状态直接解码（detokenize）看看表达什么；3. 我想实现某些概率论或统计理论的计算方式（待定），计算同一层不同的隐状态的差距，计算不同层隐状态的差距，实现某种置信度，可信度推理的估计，希望模型能够在目前层选择可信度较高的隐状态；4，我想用户可以根据3中的置信度，选择模型从那一层开始重新推理，开始“反思”（rethink），而不是根据最后的标签和正确率（acc）才进行从头开始的推理。这种思维类似于判定模型中哪些隐状态是走在对的道路上的，而不用重新在推理一遍，而是反思错误的状态，从某些置信度高的状态再出发。我想重载我目前使用的modeling_llama.py添加实现上述的功能，不用全部结构重载，你可以分析需要重载哪些部分，需要添加哪些部分，给出一个方案。