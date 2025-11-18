"""Token-level hidden-state analysis workflow for GSM8K."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from rethink.adapters import RethinkLlamaForCausalLM

DEFAULT_MODEL_PATH = "/root/autodl-fs/LLM-Research/Meta-Llama-3___1-8B-Instruct"
DEFAULT_PROMPT_FILE = "prompts/gsm8k_prompt.txt"


@dataclass
class GeneratedTokenStat:
    index: int
    absolute_position: int
    token: str
    token_id: int
    probability: float
    neg_log_prob: float
    entropy_bits: float
    cumulative_entropy_bits: float
    top_k: List[Dict[str, float]]
    layer_norms: List[float]


@dataclass
class ReferenceTokenStat:
    index: int
    absolute_position: int
    token: str
    token_id: int
    probability: float
    neg_log_prob: float
    cumulative_entropy_bits: float
    similarity_to_prompt: Optional[float]


def configure_hf_endpoint(endpoint: Optional[str]) -> None:
    if not endpoint:
        return
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = endpoint
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_DATASETS_ENDPOINT"] = endpoint
    logging.info("Using Hugging Face endpoint: %s", endpoint)


def parse_args() -> argparse.Namespace:
    """Build the CLI so analysis settings remain reproducible."""

    parser = argparse.ArgumentParser(description="Run GSM8K hidden-state diagnostics")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--dataset-name", default="gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def resolve_dtype(value: Optional[str]) -> torch.dtype:
    if value is None:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    value = value.lower()
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )


def load_prompt_template(path: str) -> str:
    template_path = Path(path)
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return template_path.read_text(encoding="utf-8")


def load_components(args: argparse.Namespace):
    """Load tokenizer/model with padding + dtype tweaks for analysis."""

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left",
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.pad_token is not None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    dtype = resolve_dtype(args.torch_dtype)
    model = RethinkLlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if args.device is None else {"": args.device},
    )
    return tokenizer, model


def fetch_gsm8k_sample(args: argparse.Namespace) -> Dict[str, str]:
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    if args.dataset_index >= len(dataset):
        raise IndexError(f"Dataset index {args.dataset_index} out of range (size={len(dataset)})")
    record = dataset[int(args.dataset_index)]
    return {
        "question": record[args.question_field].strip(),
        "answer": record[args.answer_field].strip(),
    }


def format_prompt(template: str, question: str) -> str:
    """Fill the prompt template with the GSM8K question text."""

    return template.format(question=question.strip())


def run_autoregressive_trace(model, tokenizer, prompt: str, max_new_tokens: int, top_k: int) -> Dict[str, object]:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True, padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[0, prompt_len:]
    if len(generated_ids) == 0:
        return {"generated_text": "", "token_stats": []}

    token_stats: List[GeneratedTokenStat] = []
    entropy_accumulator = 0.0
    num_steps = len(outputs.scores)
    hidden_traces = list(outputs.hidden_states)[-num_steps:]

    for idx, (token_id, logits, layer_states) in enumerate(zip(generated_ids, outputs.scores, hidden_traces)):
        probs = torch.nn.functional.softmax(logits[0], dim=-1)
        prob = max(float(probs[token_id]), 1e-12)
        neg_log_prob = -math.log(prob)
        entropy_bits = -math.log2(prob)
        entropy_accumulator += entropy_bits
        layer_norms = [float(state[0, -1, :].detach().norm().cpu()) for state in layer_states[1:]]
        top_scores = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        top_items = [
            {"token": tokenizer.decode([int(tok_id)], skip_special_tokens=False), "prob": float(score)}
            for tok_id, score in zip(top_scores.indices, top_scores.values)
        ]
        token_stats.append(
            GeneratedTokenStat(
                index=idx,
                absolute_position=prompt_len + idx,
                token=tokenizer.decode([token_id], skip_special_tokens=False),
                token_id=int(token_id),
                probability=prob,
                neg_log_prob=neg_log_prob,
                entropy_bits=entropy_bits,
                cumulative_entropy_bits=entropy_accumulator,
                top_k=top_items,
                layer_norms=layer_norms,
            )
        )

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {"generated_text": generated_text.strip(), "token_stats": token_stats}


def analyze_reference_answer(model, tokenizer, prompt: str, answer: str) -> Sequence[ReferenceTokenStat]:
    model.eval()
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
    answer_trim = answer_ids["input_ids"][:, 1:] if tokenizer.bos_token_id in answer_ids["input_ids"][0] else answer_ids["input_ids"]
    input_ids = torch.cat([prompt_ids["input_ids"], answer_trim], dim=-1)
    attention_mask = torch.ones_like(input_ids)

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    logits = outputs.logits
    final_layer = outputs.hidden_states[-1][0]
    prompt_len = prompt_ids["input_ids"].shape[1]
    total_len = input_ids.shape[1]
    prompt_vector = final_layer[:prompt_len].mean(dim=0)
    stats: List[ReferenceTokenStat] = []
    entropy_accumulator = 0.0

    for rel_idx, pos in enumerate(range(prompt_len, total_len)):
        token_id = int(input_ids[0, pos])
        step_logits = logits[0, pos - 1]
        probs = torch.nn.functional.softmax(step_logits, dim=-1)
        prob = max(float(probs[token_id]), 1e-12)
        neg_log_prob = -math.log(prob)
        entropy_bits = -math.log2(prob)
        entropy_accumulator += entropy_bits
        token_hidden = final_layer[pos]
        similarity = torch.nn.functional.cosine_similarity(token_hidden, prompt_vector, dim=0).item()
        stats.append(
            ReferenceTokenStat(
                index=rel_idx,
                absolute_position=pos,
                token=tokenizer.decode([token_id], skip_special_tokens=False),
                token_id=token_id,
                probability=prob,
                neg_log_prob=neg_log_prob,
                cumulative_entropy_bits=entropy_accumulator,
                similarity_to_prompt=similarity,
            )
        )

    return stats


def save_json_report(
    output_dir: Path,
    question: str,
    answer: str,
    prompt: str,
    generation_trace: Dict[str, object],
    reference_stats: Sequence[ReferenceTokenStat],
) -> Path:
    payload = {
        "question": question,
        "reference_answer": answer,
        "prompt": prompt,
        "generated_text": generation_trace.get("generated_text"),
        "generated_tokens": [asdict(stat) for stat in generation_trace.get("token_stats", [])],
        "reference_tokens": [asdict(stat) for stat in reference_stats],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "token_analysis.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Saved JSON report to %s", json_path)
    return json_path


def plot_layer_norm_heatmap(token_stats: Sequence[GeneratedTokenStat], output_dir: Path) -> Optional[Path]:
    if not token_stats:
        return None
    matrix = np.array([stat.layer_norms for stat in token_stats])
    plt.figure(figsize=(12, 6))
    plt.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Hidden state L2 norm")
    plt.xlabel("Layer index")
    plt.ylabel("Generated token index")
    plt.title("Layer-wise hidden state norms during generation")
    heatmap_path = output_dir / "layer_norm_heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=200)
    plt.close()
    logging.info("Saved layer norm heatmap to %s", heatmap_path)
    return heatmap_path


def plot_reference_probabilities(reference_stats: Sequence[ReferenceTokenStat], output_dir: Path) -> Optional[Path]:
    if not reference_stats:
        return None
    indices = [stat.index for stat in reference_stats]
    probs = [stat.probability for stat in reference_stats]
    similarities = [stat.similarity_to_prompt for stat in reference_stats]
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(indices, probs, marker="o", label="Token probability", color="#1f77b4")
    ax1.set_xlabel("Answer token index")
    ax1.set_ylabel("Probability", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(indices, similarities, marker="x", label="Cosine similarity", color="#ff7f0e")
    ax2.set_ylabel("Similarity to prompt", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    fig.suptitle("Reference token probabilities vs. prompt similarity")
    fig.tight_layout()
    plot_path = output_dir / "reference_probabilities.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    logging.info("Saved probability plot to %s", plot_path)
    return plot_path


def summarize_console_output(generation_trace: Dict[str, object], reference_stats: Sequence[ReferenceTokenStat]) -> None:
    generated_text = generation_trace.get("generated_text", "")
    logging.info("Generated answer preview: %s", generated_text[:200].strip())
    if reference_stats:
        avg_prob = np.mean([stat.probability for stat in reference_stats])
        total_entropy = reference_stats[-1].cumulative_entropy_bits
        logging.info("Reference tokens | avg prob=%.4f | cumulative entropy=%.2f bits", avg_prob, total_entropy)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    configure_hf_endpoint(args.hf_endpoint)
    logging.info(
        "Loading dataset %s/%s (%s) index=%s",
        args.dataset_name,
        args.dataset_config,
        args.dataset_split,
        args.dataset_index,
    )
    sample = fetch_gsm8k_sample(args)
    prompt_template = load_prompt_template(args.prompt_file)
    prompt = format_prompt(prompt_template, sample["question"])
    tokenizer, model = load_components(args)
    logging.info("Model and tokenizer ready. Beginning autoregressive analysis…")
    generation_trace = run_autoregressive_trace(
        model,
        tokenizer,
        prompt,
        args.max_new_tokens,
        args.top_k,
    )
    logging.info("Running teacher-forced probability analysis against the reference answer…")
    reference_stats = analyze_reference_answer(model, tokenizer, prompt, sample["answer"])
    save_json_report(output_dir, sample["question"], sample["answer"], prompt, generation_trace, reference_stats)
    if args.visualize:
        plot_layer_norm_heatmap(generation_trace.get("token_stats", []), output_dir)
        plot_reference_probabilities(reference_stats, output_dir)
    summarize_console_output(generation_trace, reference_stats)


if __name__ == "__main__":
    main()