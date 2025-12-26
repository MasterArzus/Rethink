# IFEval data + Rethink task set

This folder contains:

- `input_data.jsonl`: the original IFEval prompt set downloaded from the official Google Research repository.
- `taskset_120.json`: a derived IFEval-style task set (≈120 tasks) focused on **Taboo (Forbidden Words)** and **JSON constraints**, intended for steerability / truncate+force human evaluation.
- `build_taskset_120.py`: script that builds `taskset_120.json` from `input_data.jsonl`.

## 1) Download source

The raw prompts are taken from:

- https://github.com/google-research/google-research/tree/master/instruction_following_eval

Specifically:

- `instruction_following_eval/data/input_data.jsonl`

## 2) Raw IFEval format (input_data.jsonl)

Each line is a JSON object with (at least) these fields:

- `key`: integer id
- `prompt`: the instruction text
- `instruction_id_list`: list of verifiable instruction ids
- `kwargs`: list of per-instruction arguments

Example keys:

```json
{"key": 1000, "prompt": "...", "instruction_id_list": ["punctuation:no_comma", "keywords:forbidden_words"], "kwargs": [...]}
```

## 3) Derived task set format (taskset_120.json)

Top-level:

- `meta`: metadata (counts, seed, provenance)
- `tasks`: list of tasks, each with:
  - `id`: unique string id
  - `source`: `ifeval` or `curated`
  - `type`: `taboo` or `json`
  - `prompt`: instruction text
  - `constraints`: checker-oriented config

### 3.1 Taboo tasks

Constraints contain:

- `forbidden_words`: list of forbidden words
- `match`: matching rules (currently `casefold=true`, `word_boundary=true`)

### 3.2 JSON tasks

Constraints contain:

- `json.strict`: whether to require strict JSON-only output
- `json.allow_code_fence`: whether to allow code fences
- `json.schema`: optional simple schema config

## 4) Build

```bash
python dataset/ifeval/build_taskset_120.py \
  --raw dataset/ifeval/input_data.jsonl \
  --out dataset/ifeval/taskset_120.json
```
