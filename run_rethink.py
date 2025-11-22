"""Entry point that runs the rethink debugger on a GSM8K example."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import torch
from datasets import DownloadConfig, load_dataset, load_from_disk
from transformers import AutoTokenizer

from rethink.utils.config import DatasetSlice, InstrumentationConfig, RethinkConfig
from dataset.benchmark import BenchmarkExample
from dataset.gsm8k import load_gsm8k_slice
from rethink.engine.llama import RethinkLlamaForCausalLM
from rethink.engine.controller import RethinkController


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "outputs"
LOAD_DATASET_PARAMS = inspect.signature(load_dataset).parameters


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run rethink debugger over GSM8K")
	parser.add_argument(
		"--model-name",
		default="hf-internal-testing/tiny-random-LlamaForCausalLM",
		help="HF repo id or local path to the base model",
	)
	parser.add_argument("--device", default=None, help="Override torch device (cpu|cuda)")
	parser.add_argument("--limit", type=int, default=1, help="How many GSM8K samples to inspect")
	parser.add_argument("--max-new-tokens", type=int, default=64, help="Decode cap for teacher forcing")
	parser.add_argument("--generation-max-new-tokens", type=int, default=64, help="Decode cap for free generation")
	parser.add_argument("--output", default=None, help="Optional path to dump run summaries as JSON")
	parser.add_argument("--dataset-name", default="gsm8k", help="HF dataset identifier")
	parser.add_argument("--dataset-config", default="main", help="HF dataset config name")
	parser.add_argument("--split", default="test", help="Dataset split to load")
	parser.add_argument("--use-auth-token", default=None, help="Optional HF token for private datasets")
	parser.add_argument(
		"--local-files-only",
		action="store_true",
		help="Disallow network calls and reuse cached datasets/models only",
	)
	parser.add_argument("--run-name", default=None, help="Override the auto-generated run name for outputs")
	parser.add_argument("--log-file", default=None, help="Optional path to write logs (defaults to outputs/<run>.log)")
	parser.add_argument("--setup-only", action="store_true", help="Validate Dataset/model paths without running inference")
	return parser.parse_args()


def build_download_config(token: str | None, local_files_only: bool) -> DownloadConfig:
	config = DownloadConfig(local_files_only=local_files_only)
	if token:
		if hasattr(config, "token"):
			setattr(config, "token", token)
		if hasattr(config, "use_auth_token"):
			setattr(config, "use_auth_token", token)
	return config


def build_dataset_kwargs(token: str | None, local_files_only: bool) -> dict[str, object]:
	kwargs: dict[str, object] = {}
	if token:
		if "token" in LOAD_DATASET_PARAMS:
			kwargs["token"] = token
		elif "use_auth_token" in LOAD_DATASET_PARAMS:
			kwargs["use_auth_token"] = token
	kwargs["download_config"] = build_download_config(token, local_files_only)
	if local_files_only:
		kwargs["download_mode"] = "reuse_dataset_if_exists"
	return kwargs


def load_benchmark_examples(
	name: str,
	config: str,
	split: str,
	limit: int,
	token: str | None,
	local_files_only: bool,
) -> List[BenchmarkExample]:
	# 1. Try loading from local disk (priority)
	local_dataset_path = REPO_ROOT / "dataset" / name
	if local_dataset_path.exists():
		logging.info(f"Found local dataset at {local_dataset_path}, loading from disk.")
		try:
			dataset = load_from_disk(str(local_dataset_path))
			if split in dataset:
				dataset = dataset[split]
			logging.info("Loaded dataset from local disk.")
			return load_gsm8k_slice(dataset, limit=limit)
		except Exception as e:
			logging.warning(f"Failed to load from disk: {e}. Falling back to standard loading.")

	# 2. Fallback to load_dataset (Hugging Face Hub or cache)
	def _load(local_only: bool):
		kwargs = build_dataset_kwargs(token, local_only)
		return load_dataset(name, config, split=split, **kwargs)

	if local_files_only:
		logging.info("Attempting to load %s/%s from cache only.", name, config)
		dataset = _load(True)
		logging.info("Loaded dataset from local cache.")
	else:
		try:
			logging.info("Trying cached %s/%s before hitting the network.", name, config)
			dataset = _load(True)
			logging.info("Using cached dataset for %s/%s.", name, config)
		except Exception as cache_err:
			logging.warning("Cache load failed (%s). Falling back to online download.", cache_err)
			dataset = _load(False)
	return load_gsm8k_slice(dataset, limit=limit)


def prepare_model(model_name: str, device: torch.device) -> tuple:
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
	tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
	tokenizer.padding_side = "left"

	model = RethinkLlamaForCausalLM.from_pretrained(model_name)
	model.to(device)
	model.eval()
	return tokenizer, model


def summarize_run(example: BenchmarkExample, controller_artifacts) -> dict:
	ref = controller_artifacts.benchmark_result.reference_trace
	hyp = controller_artifacts.benchmark_result.model_trace
	report = controller_artifacts.divergence_report
	
	# Convert Interval objects to dicts for JSON serialization
	critical_intervals = []
	if hasattr(report, 'critical_intervals'):
		for interval in report.critical_intervals:
			critical_intervals.append({
				"start": interval.start,
				"end": interval.end,
				"type": interval.type,
				"score": interval.score,
				"description": interval.description
			})
	
	summary = {
		"question": example.question,
		"reference_answer": example.correct_answer,
		"model_answer": hyp.answer,
		"teacher_forced_tokens": [t.token for t in ref.tokenlist],
		"model_tokens": [t.token for t in hyp.tokenlist],
		"critical_intervals": critical_intervals,
		"divergence_index": report.divergence_index if hasattr(report, 'divergence_index') else None,
	}
	return summary


def slugify(value: str) -> str:
	value = value.replace(os.sep, "_")
	value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
	value = value.strip("_")
	return value or "run"


def derive_run_name(model_name: str, dataset_name: str, override: str | None) -> str:
	if override:
		return slugify(override)
	model_leaf = model_name.rstrip("/").split("/")[-1] or model_name
	model_slug = slugify(model_leaf)
	dataset_slug = slugify(dataset_name)
	return f"{model_slug}_{dataset_slug}_run"


def ensure_parent(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def setup_logging(log_path: Path) -> None:
	ensure_parent(log_path)
	handlers = [logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")]
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers, force=True)


def build_run_report(
	args: argparse.Namespace,
	device: str,
	run_name: str,
	log_path: Path,
	output_path: Path,
	summaries: List[dict],
 ) -> dict:
	return {
		"run": {
			"name": run_name,
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"model_name": args.model_name,
			"dataset": {
				"name": args.dataset_name,
				"config": args.dataset_config,
				"split": args.split,
			},
			"limit": args.limit,
			"max_new_tokens": args.max_new_tokens,
			"generation_max_new_tokens": args.generation_max_new_tokens,
			"device": device,
			"log_file": str(log_path),
			"output_file": str(output_path),
			"local_files_only": args.local_files_only,
		},
		"metrics": {
			"num_examples": len(summaries),
		},
		"examples": summaries,
	}


def write_run_report(output_path: Path, payload: dict) -> None:
	ensure_parent(output_path)
	with open(output_path, "w", encoding="utf-8") as fh:
		json.dump(payload, fh, ensure_ascii=False, indent=2)
	logging.info("Saved run summary to %s", output_path)


def main() -> None:
	args = parse_args()
	device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device(device_str)
	run_name = derive_run_name(args.model_name, args.dataset_name, args.run_name)
	log_path = Path(args.log_file) if args.log_file else OUTPUT_DIR / f"{run_name}.log"
	output_path = Path(args.output) if args.output else OUTPUT_DIR / f"{run_name}.json"
	setup_logging(log_path)
	logging.info("Starting run '%s' on device %s", run_name, device_str)

	examples = load_benchmark_examples(
		name=args.dataset_name,
		config=args.dataset_config,
		split=args.split,
		limit=args.limit,
		token=args.use_auth_token,
		local_files_only=args.local_files_only,
	)
	dataset_slice = DatasetSlice(name=args.dataset_name, split=args.split, max_examples=args.limit)
	instrumentation_cfg = InstrumentationConfig(max_tokens=args.max_new_tokens)
	cfg = RethinkConfig(dataset=dataset_slice, instrumentation=instrumentation_cfg, device=device_str)
	logging.info("Loaded %d examples for inspection", len(examples))

	if args.setup_only:
		logging.info("Setup-only flag detected; skipping model load and inference")
		payload = build_run_report(args, device_str, run_name, log_path, output_path, summaries=[])
		write_run_report(output_path, payload)
		return

	tokenizer, model = prepare_model(args.model_name, device)
	controller = RethinkController(model=model, tokenizer=tokenizer, cfg=cfg)

	summaries = []
	generation_kwargs = {"max_new_tokens": args.generation_max_new_tokens, "eos_token_id": tokenizer.eos_token_id}
	for example in examples:
		artifacts = controller.run_single_example(example, generation_kwargs=generation_kwargs)
		summary = summarize_run(example, artifacts)
		summaries.append(summary)
		print("\n=== Question ===")
		print(example.question)
		print("\n--- Reference Answer ---")
		print(example.correct_answer)
		print("\n--- Teacher-Forced Tokens ---")
		print(summary["teacher_forced_tokens"][:20], "...")
		print("\n--- Model Generated Tokens ---")
		print(summary["model_tokens"][:20], "...")
		print("\n--- Critical Intervals ---")
		for interval in summary["critical_intervals"]:
			print(f"  [{interval['start']}:{interval['end']}] {interval['type']} - {interval['description']}")

	payload = build_run_report(args, device_str, run_name, log_path, output_path, summaries)
	write_run_report(output_path, payload)


if __name__ == "__main__":
	main()
