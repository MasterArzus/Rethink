"""Run Meta-Llama-3 8B Instruct on GSM8K with editable prompts."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from datasets import Dataset, DownloadConfig, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = "/root/autodl-fs/LLM-Research/Meta-Llama-3___1-8B-Instruct"
DEFAULT_PROMPT_FILE = REPO_ROOT / "prompts" / "gsm8k_prompt.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
ANSWER_PATTERN = re.compile(r"####\s*([-+]?\d[\d,\.\/-]*)")
CONFIDENCE_PATTERN = re.compile(r"confidence\s*[:=-]\s*([\d\.]+)\s*%?", re.IGNORECASE)
SEEN_BEFORE_PATTERN = re.compile(r"seen[_\s-]*before\s*[:=-]\s*(true|false)", re.IGNORECASE)
LOAD_DATASET_PARAMS = inspect.signature(load_dataset).parameters


@dataclass
class SampleResult:
	question: str
	gold_answer: str
	prediction: str
	prompt: str
	gold_final: str | None
	pred_final: str | None
	confidence: float | None
	seen_before: bool | None


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate Meta-Llama-3 on GSM8K with a configurable prompt",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local path or repo id for the model")
	parser.add_argument("--dataset-name", default="gsm8k", help="HF dataset to use")
	parser.add_argument("--dataset-config", default="main", help="HF dataset config name")
	parser.add_argument("--split", default="test", help="Dataset split to evaluate")
	parser.add_argument("--max-samples", type=int, default=50, help="Limit number of samples (None uses full split)")
	parser.add_argument("--batch-size", type=int, default=1, help="Number of prompts to generate per batch")
	parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum tokens to generate")
	parser.add_argument("--temperature", type=float, default=0.2, help="Softmax temperature for sampling")
	parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus value")
	parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE), help="Path to a prompt template that contains {question}")
	parser.add_argument("--prompt-template", default=None, help="Inline prompt template (overrides --prompt-file)")
	parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_DIR / "gsm8k_predictions.jsonl"), help="Where to save per-sample outputs")
	parser.add_argument("--log-file", default=str(DEFAULT_OUTPUT_DIR / "gsm8k_eval.log"), help="Path to append experiment logs")
	parser.add_argument("--use-auth-token", default=None, help="Optional HF token for private datasets")
	parser.add_argument("--setup-only", action="store_true", help="Validate arguments and prompt without running inference")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for torch")
	parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote code when loading the model")
	return parser.parse_args()


def set_seed(seed: int) -> None:
	if seed is None:
		return
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def load_prompt(args: argparse.Namespace) -> str:
	if args.prompt_template:
		return args.prompt_template
	if args.prompt_file:
		prompt_path = Path(args.prompt_file)
		if not prompt_path.exists():
			raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
		return prompt_path.read_text(encoding="utf-8")
	raise ValueError("No prompt template provided")


def format_prompt(template: str, question: str) -> str:
	return template.format(question=question.strip())


def build_download_config(token: str | None) -> DownloadConfig | None:
	if not token:
		return None
	config = DownloadConfig()
	if hasattr(config, "token"):
		setattr(config, "token", token)
	if hasattr(config, "use_auth_token"):
		setattr(config, "use_auth_token", token)
	return config


def load_split(name: str, config: str, split: str, token: str | None) -> Dataset:
	extra_kwargs: dict[str, object] = {}
	if token:
		if "token" in LOAD_DATASET_PARAMS:
			extra_kwargs["token"] = token
		elif "use_auth_token" in LOAD_DATASET_PARAMS:
			extra_kwargs["use_auth_token"] = token
		download_config = build_download_config(token)
		if download_config is not None:
			extra_kwargs["download_config"] = download_config
	return load_dataset(name, config, split=split, **extra_kwargs)


def prepare_model(args: argparse.Namespace):
	tokenizer = AutoTokenizer.from_pretrained(
		args.model_path,
		padding_side="left",
		use_fast=True,
		trust_remote_code=args.trust_remote_code,
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
	model = AutoModelForCausalLM.from_pretrained(
		args.model_path,
		torch_dtype=torch_dtype,
		device_map="auto" if torch.cuda.is_available() else None,
		trust_remote_code=args.trust_remote_code,
	)
	return tokenizer, model


def chunk(iterable: Sequence, batch_size: int) -> Iterable[Sequence]:
	if batch_size <= 0:
		raise ValueError("batch_size must be positive")
	for start in range(0, len(iterable), batch_size):
		yield iterable[start : start + batch_size]


def extract_final_answer(text: str) -> str | None:
	match = ANSWER_PATTERN.search(text)
	if match:
		return match.group(1).replace(",", "").strip()
	return None


def extract_confidence(text: str) -> float | None:
	match = CONFIDENCE_PATTERN.search(text)
	if not match:
		return None
	try:
		value = float(match.group(1))
	except ValueError:
		return None
	return max(0.0, min(100.0, value))


def extract_seen_before(text: str) -> bool | None:
	match = SEEN_BEFORE_PATTERN.search(text)
	if not match:
		return None
	return match.group(1).strip().lower() == "true"


def generate_batch(prompts: List[str], tokenizer, model, args) -> List[str]:
	inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
	input_lengths = inputs.attention_mask.sum(dim=1)
	with torch.no_grad():
		outputs = model.generate(
			**inputs,
			do_sample=args.temperature > 0,
			temperature=args.temperature,
			top_p=args.top_p,
			max_new_tokens=args.max_new_tokens,
			eos_token_id=tokenizer.eos_token_id,
			pad_token_id=tokenizer.pad_token_id,
		)
	generations: List[str] = []
	for idx in range(outputs.size(0)):
		prompt_len = int(input_lengths[idx].item())
		completion_ids = outputs[idx, prompt_len:]
		text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
		generations.append(text)
	return generations


def ensure_output_dir(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def setup_logging(log_path: Path) -> None:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	handlers = [logging.StreamHandler()]
	try:
		file_handler = logging.FileHandler(log_path, encoding="utf-8")
		handlers.append(file_handler)
	except OSError:
		logging.warning("Unable to write logs to %s", log_path)

	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(message)s",
		handlers=handlers,
	)


def evaluate(args: argparse.Namespace) -> None:
	setup_logging(Path(args.log_file))
	logging.info("Starting evaluation with model=%s", args.model_path)
	logging.info(
		"Dataset=%s/%s split=%s max_samples=%s batch_size=%s",
		args.dataset_name,
		args.dataset_config,
		args.split,
		args.max_samples,
		args.batch_size,
	)
	prompt_template = load_prompt(args)
	if args.setup_only:
		logging.info("Setup verified. Exiting because --setup-only was set.")
		return

	set_seed(args.seed)
	dataset = load_split(args.dataset_name, args.dataset_config, args.split, args.use_auth_token)
	total = len(dataset)
	if args.max_samples is not None:
		max_samples = min(args.max_samples, total)
		data = dataset.select(range(max_samples))
	else:
		data = dataset

	tokenizer, model = prepare_model(args)
	results: List[SampleResult] = []
	correct = 0
	count = 0

	total_batches = math.ceil(len(data) / max(1, args.batch_size))
	for batch_indices in tqdm(
		chunk(range(len(data)), max(1, args.batch_size)),
		total=total_batches,
		desc="Evaluating",
	):
		indices = list(batch_indices)
		prompts = [format_prompt(prompt_template, data[i]["question"]) for i in indices]
		generations = generate_batch(prompts, tokenizer, model, args)
		for idx, generation in zip(indices, generations):
			example = data[idx]
			gold = example["answer"].strip()
			pred_final = extract_final_answer(generation)
			gold_final = extract_final_answer(gold)
			pred_confidence = extract_confidence(generation)
			pred_seen_before = extract_seen_before(generation)
			is_correct = pred_final is not None and gold_final is not None and pred_final == gold_final
			results.append(
				SampleResult(
					question=example["question"].strip(),
					gold_answer=gold,
					prediction=generation,
					prompt=format_prompt(prompt_template, example["question"]),
					gold_final=gold_final,
					pred_final=pred_final,
					confidence=pred_confidence,
					seen_before=pred_seen_before,
				)
			)
			if gold_final is not None and pred_final is not None:
				count += 1
				if is_correct:
					correct += 1

	accuracy = correct / count if count else math.nan
	ensure_output_dir(Path(args.output_jsonl))
	with open(args.output_jsonl, "w", encoding="utf-8") as fh:
		for item in results:
			fh.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")

	logging.info(
		"Evaluated %s samples. Comparable answers: %s. Accuracy: %.2f%%.",
		len(results),
		count,
		accuracy * 100 if not math.isnan(accuracy) else float("nan"),
	)


def main():
	args = parse_args()
	evaluate(args)


if __name__ == "__main__":
	main()
