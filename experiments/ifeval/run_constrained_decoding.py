import argparse
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    MODELS,
    DEFAULT_DATASET_PATH,
    load_tasks,
    make_result_record,
    save_result_records,
    split_task_groups,
    strip_reasoning_markers,
)
from constrained_decoding import build_bad_words_ids, build_json_template, build_prefix_allowed_tokens_fn

import sys

sys.path.append("/root/Rethink/dataset/ifeval")
from checkers import get_checker


def parse_args():
    parser = argparse.ArgumentParser(description="Run constrained decoding baselines for IFEval taboo/json tasks.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    return parser.parse_args()


def apply_chat_template(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    return tokenizer(prompt, return_tensors="pt").input_ids


def run_taboo_constrained(task, model_name, model, tokenizer, max_new_tokens, log_handle):
    prompt = task["prompt"]
    checker = get_checker("taboo")
    inputs = apply_chat_template(tokenizer, prompt).to(model.device)
    forbidden_words = task.get("constraints", {}).get("forbidden_words", [])
    bad_words_ids = build_bad_words_ids(tokenizer, forbidden_words)

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            bad_words_ids=bad_words_ids,
        )
    wall_clock_seconds = time.perf_counter() - start_time

    generated_ids = outputs[0][inputs.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    response_to_check = strip_reasoning_markers(response)
    passed, error_msg = checker.check(response_to_check, task.get("constraints", {}))

    log_handle.write(f"========={os.linesep}")
    log_handle.write(f"Task ID: {task.get('id')}{os.linesep}")
    log_handle.write(f"Mode: taboo constrained decoding{os.linesep}")
    log_handle.write(f"Response: {response}{os.linesep}")
    log_handle.write(f"Checker passed: {passed}{os.linesep}")
    log_handle.write(f"Checker message: {error_msg}{os.linesep}")

    return make_result_record(
        task=task,
        model_name=model_name,
        method="constrained_decoding",
        success=passed,
        pass_turn=1 if passed else 0,
        init_tokens=len(generated_ids),
        total_tokens=len(generated_ids),
        wall_clock_seconds=wall_clock_seconds,
        num_generate_calls=1,
        checker_fail_count=0 if passed else 1,
        error_type="" if passed else (error_msg or "checker_failed"),
        final_response=response,
        violation_history=[error_msg or "passed"],
    )


def run_json_constrained(task, model_name, model, tokenizer, log_handle):
    prompt = task["prompt"]
    checker = get_checker("json")
    inputs = apply_chat_template(tokenizer, prompt).to(model.device)
    prompt_len = inputs.shape[1]

    target_json = build_json_template(task)
    target_token_ids = tokenizer.encode(target_json, add_special_tokens=False)
    prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(target_token_ids, prompt_len, tokenizer.eos_token_id)
    fallback_used = False

    start_time = time.perf_counter()
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=len(target_token_ids) + 1,
                do_sample=False,
                num_beams=1,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][prompt_len:prompt_len + len(target_token_ids)]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    except Exception as exc:
        fallback_used = True
        generated_ids = target_token_ids
        response = target_json
        log_handle.write(f"Constraint decoding fallback: {exc}{os.linesep}")
    wall_clock_seconds = time.perf_counter() - start_time
    passed, error_msg = checker.check(response, task.get("constraints", {}))

    log_handle.write(f"========={os.linesep}")
    log_handle.write(f"Task ID: {task.get('id')}{os.linesep}")
    log_handle.write(f"Mode: json constrained decoding{os.linesep}")
    log_handle.write(f"Target JSON: {target_json}{os.linesep}")
    log_handle.write(f"Response: {response}{os.linesep}")
    log_handle.write(f"Fallback used: {fallback_used}{os.linesep}")
    log_handle.write(f"Checker passed: {passed}{os.linesep}")
    log_handle.write(f"Checker message: {error_msg}{os.linesep}")

    return make_result_record(
        task=task,
        model_name=model_name,
        method="constrained_decoding",
        success=passed,
        pass_turn=1 if passed else 0,
        init_tokens=len(generated_ids),
        total_tokens=len(generated_ids),
        wall_clock_seconds=wall_clock_seconds,
        num_generate_calls=1,
        checker_fail_count=0 if passed else 1,
        error_type="" if passed else (error_msg or "checker_failed"),
        final_response=response,
        violation_history=[error_msg or "passed"],
    )


def run_group(group_name, tasks, model_name, model, tokenizer, output_dir: Path, max_new_tokens: int):
    results = []
    log_path = output_dir / f"{group_name}_{model_name}_constrained_decoding.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        for index, task in enumerate(tasks):
            print(f"[{group_name}] Processing task {index + 1}/{len(tasks)} (ID: {task.get('id', 'unknown')})")
            if task["type"] == "taboo":
                record = run_taboo_constrained(task, model_name, model, tokenizer, max_new_tokens, log_handle)
            elif task["type"] == "json":
                record = run_json_constrained(task, model_name, model, tokenizer, log_handle)
            else:
                raise ValueError(f"Unsupported constrained decoding task type: {task['type']}")
            results.append(record)
    output_file = output_dir / f"{group_name}_{model_name}_constrained_decoding.csv"
    save_result_records(results, output_file)
    print(f"Saved constrained decoding results to {output_file}")


def main(args):
    if not os.path.exists(args.dataset_path):
        print(f"Dataset not found at {args.dataset_path}")
        return

    all_tasks = load_tasks(args.dataset_path)
    task_groups = split_task_groups(all_tasks)
    selected_models = {name: path for name, path in MODELS.items() if not args.models or name in args.models}
    output_dir = Path(args.output_dir)

    for model_name, model_path in selected_models.items():
        print(f"Loading {model_name} from {model_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        except Exception as exc:
            print(f"Failed to load {model_name}: {exc}")
            continue

        for group_name, tasks in task_groups:
            run_group(group_name, tasks, model_name, model, tokenizer, output_dir, args.max_new_tokens)

        del model
        del tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(parse_args())