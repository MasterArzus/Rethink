"""Simple CLI to run rethink-enabled generation on a single prompt."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoTokenizer

from rethink import HiddenStateCache, RethinkController, RethinkEngine
from rethink.adapters import RethinkLlamaForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single rethink generation prompt")
    parser.add_argument("prompt", help="Text prompt to feed into the model")
    parser.add_argument("--model-path", default="/root/autodl-fs/LLM-Research/Meta-Llama-3___1-8B-Instruct", help="Model checkpoint or repo ID")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--capture-layers", default="", help="Comma-separated layers to cache (e.g., 8,16,24)")
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--log-file", default="outputs/rethink_run.log")
    parser.add_argument("--output-json", default="outputs/rethink_run.json")
    parser.add_argument("--device", default=None, help="Force device (e.g., cuda:0)")
    parser.add_argument("--torch-dtype", default=None, help="torch dtype (float16,float32)")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def parse_layers(value: str):
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def resolve_dtype(value: str | None):
    if value is None:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    value = value.lower()
    if value == "float16" or value == "fp16":
        return torch.float16
    if value == "float32" or value == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def setup_logging(path: str) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )


def run_prompt(args: argparse.Namespace) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left",
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(args.torch_dtype)
    model = RethinkLlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto" if args.device is None else {"": args.device},
        trust_remote_code=args.trust_remote_code,
        rethink_options={
            "capture_layers": parse_layers(args.capture_layers),
            "confidence_threshold": args.confidence_threshold,
        },
    )

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    cache = HiddenStateCache(max_steps=2048)
    controller = RethinkController(confidence_threshold=args.confidence_threshold)
    engine = RethinkEngine(model, cache=cache, controller=controller)

    outputs, cache, scores, action = engine.generate(
        inputs["input_ids"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    completion = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return {
        "prompt": args.prompt,
        "completion": completion,
        "scores": [score.__dict__ for score in scores],
        "action": action.__dict__,
        "cache_summary": cache.summary(),
    }


def save_output(payload: Dict[str, Any], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    logging.info("Running rethink prompt: %s", args.prompt)
    result = run_prompt(args)
    save_output(result, args.output_json)
    logging.info("Completion: %s", result["completion"])
    logging.info("Scores: %s", result["scores"])
    logging.info("Cache summary: %s", result["cache_summary"])


if __name__ == "__main__":
    main()