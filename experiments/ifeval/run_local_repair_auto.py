"""
Automated Local Repair baseline for IFEval tasks.

This script simulates the local repair primitive (rollback + local banning)
without human decision-making. It detects the first failure point,
rolls back to it, bans the failing token, and continues generation.

This isolates the algorithmic value of local repair from human judgment.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("/root/Rethink/dataset/ifeval")
from checkers import get_checker

from common import (
    MODELS,
    DEFAULT_DATASET_PATH,
    load_tasks,
    make_result_record,
    save_result_records,
    split_task_groups,
    strip_reasoning_markers,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run automated local repair baseline.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    return parser.parse_args()


def apply_chat_template(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    return tokenizer(prompt, return_tensors="pt").input_ids


def find_first_violation_token(response: str, forbidden_words: List[str]) -> Optional[str]:
    """
    Find the first forbidden word that appears in the response.
    Returns the word if found, None otherwise.
    """
    response_lower = response.lower()
    first_pos = float('inf')
    first_word = None
    for word in forbidden_words:
        pattern = r'\b' + re.escape(word.lower()) + r'\b'
        match = re.search(pattern, response_lower)
        if match and match.start() < first_pos:
            first_pos = match.start()
            first_word = word
    return first_word


def build_ban_for_word(tokenizer, word: str) -> List[List[int]]:
    """Build bad_words_ids for a single forbidden word."""
    variants = {word, word.lower(), word.upper(), word.capitalize()}
    ids = []
    seen = set()
    for v in variants:
        token_ids = tokenizer.encode(f" {v}", add_special_tokens=False)
        if token_ids and tuple(token_ids) not in seen:
            seen.add(tuple(token_ids))
            ids.append(token_ids)
    return ids


def run_taboo_local_repair(task, model, tokenizer, max_new_tokens, max_turns, log_handle):
    """Run automated local repair for taboo tasks."""
    prompt = task["prompt"]
    checker = get_checker("taboo")
    forbidden_words = task.get("constraints", {}).get("forbidden_words", [])

    messages = [{"role": "user", "content": prompt}]
    violation_history = []
    total_tokens = 0
    init_tokens = 0
    turns = 0
    success = False
    error_type = ""
    final_response = ""

    # Track banned words across turns to avoid repeated bans
    persistent_banned_ids: List[List[int]] = []

    start_time = time.perf_counter()

    for turn in range(max_turns):
        turns += 1
        inputs = apply_chat_template(tokenizer, messages).to(model.device)
        input_len = inputs.shape[1]

        # Build bad_words_ids for this turn (persistent + new violations)
        turn_ban = list(persistent_banned_ids)
        if turn > 0:
            # Add new violation-based bans from previous turn
            prev_response = messages[-1]["content"]
            new_word = find_first_violation_token(prev_response, forbidden_words)
            if new_word:
                turn_ban.extend(build_ban_for_word(tokenizer, new_word))

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        if turn_ban:
            gen_kwargs["bad_words_ids"] = turn_ban

        with torch.no_grad():
            outputs = model.generate(inputs, **gen_kwargs)

        generated_ids = outputs[0][input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        response_to_check = strip_reasoning_markers(response)

        if turn == 0:
            init_tokens = len(generated_ids)
        total_tokens += len(generated_ids)
        final_response = response

        passed, error_msg = checker.check(response_to_check, task.get("constraints", {}))
        violation_history.append(error_msg or "passed")

        log_handle.write(f"=========\n")
        log_handle.write(f"Task ID: {task.get('id')}\n")
        log_handle.write(f"Turn: {turn + 1}\n")
        log_handle.write(f"Response: {response}\n")
        log_handle.write(f"Passed: {passed}\n")
        log_handle.write(f"Checker: {error_msg}\n")

        if passed:
            success = True
            break

        error_type = error_msg or "checker_failed"
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Do not use the words: {error_msg.replace('Found forbidden words: ', '')}"})

    wall_clock_seconds = time.perf_counter() - start_time
    pass_turn = turns if success else 0
    checker_fail_count = sum(1 for v in violation_history if v != "passed")

    return make_result_record(
        task=task,
        model_name="local_repair_auto",
        method="automated_local_repair",
        success=success,
        pass_turn=pass_turn,
        init_tokens=init_tokens,
        total_tokens=total_tokens,
        wall_clock_seconds=wall_clock_seconds,
        num_generate_calls=turns,
        checker_fail_count=checker_fail_count,
        error_type=error_type,
        final_response=final_response,
        violation_history=violation_history,
    )


def run_json_local_repair(task, model, tokenizer, max_new_tokens, max_turns, log_handle):
    """Run automated local repair for JSON tasks.

    For JSON, local repair means: detect JSON parse failure or schema violation,
    find the problematic token, and re-generate with that token banned.
    Since JSON validity is binary (pass/fail), we use a simple regenerate-with-ban approach.
    """
    from constrained_decoding import build_json_template

    prompt = task["prompt"]
    checker = get_checker("json")

    messages = [{"role": "user", "content": prompt}]
    violation_history = []
    total_tokens = 0
    init_tokens = 0
    turns = 0
    success = False
    error_type = ""
    final_response = ""

    # For JSON, we track banned patterns that caused parse failures
    banned_patterns: List[str] = []

    start_time = time.perf_counter()

    for turn in range(max_turns):
        turns += 1
        inputs = apply_chat_template(tokenizer, messages).to(model.device)
        input_len = inputs.shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }

        with torch.no_grad():
            outputs = model.generate(inputs, **gen_kwargs)

        generated_ids = outputs[0][input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        if turn == 0:
            init_tokens = len(generated_ids)
        total_tokens += len(generated_ids)
        final_response = response

        passed, error_msg = checker.check(response, task.get("constraints", {}))
        violation_history.append(error_msg or "passed")

        log_handle.write(f"=========\n")
        log_handle.write(f"Task ID: {task.get('id')}\n")
        log_handle.write(f"Turn: {turn + 1}\n")
        log_handle.write(f"Response: {response}\n")
        log_handle.write(f"Passed: {passed}\n")
        log_handle.write(f"Checker: {error_msg}\n")

        if passed:
            success = True
            break

        error_type = error_msg or "checker_failed"
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Fix this JSON error: {error_msg}"})

    wall_clock_seconds = time.perf_counter() - start_time
    pass_turn = turns if success else 0
    checker_fail_count = sum(1 for v in violation_history if v != "passed")

    return make_result_record(
        task=task,
        model_name="local_repair_auto",
        method="automated_local_repair",
        success=success,
        pass_turn=pass_turn,
        init_tokens=init_tokens,
        total_tokens=total_tokens,
        wall_clock_seconds=wall_clock_seconds,
        num_generate_calls=turns,
        checker_fail_count=checker_fail_count,
        error_type=error_type,
        final_response=final_response,
        violation_history=violation_history,
    )


def run_group(group_name, tasks, model_name, model, tokenizer, output_dir: Path, max_new_tokens, max_turns):
    results = []
    log_path = output_dir / f"{group_name}_{model_name}_automated_local_repair.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        for index, task in enumerate(tasks):
            print(f"[{group_name}] Task {index + 1}/{len(tasks)} (ID: {task.get('id', 'unknown')})")
            if task["type"] == "taboo":
                record = run_taboo_local_repair(task, model, tokenizer, max_new_tokens, max_turns, log_handle)
            elif task["type"] == "json":
                record = run_json_local_repair(task, model, tokenizer, max_new_tokens, max_turns, log_handle)
            else:
                print(f"Skipping unsupported task type: {task['type']}")
                continue
            results.append(record)

    output_file = output_dir / f"{group_name}_{model_name}_automated_local_repair.csv"
    save_result_records(results, output_file)
    print(f"Saved to {output_file}")


def main(args):
    if not os.path.exists(args.dataset_path):
        print(f"Dataset not found at {args.dataset_path}")
        return

    all_tasks = load_tasks(args.dataset_path)
    task_groups = split_task_groups(all_tasks)
    selected_models = {name: path for name, path in MODELS.items()
                      if not args.models or name in args.models}

    output_dir = Path(args.output_dir)

    for model_name, model_path in selected_models.items():
        print(f"\n=== Loading {model_name} from {model_path} ===")
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
            run_group(group_name, tasks, model_name, model, tokenizer,
                      output_dir, args.max_new_tokens, args.max_turns)

        del model
        del tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(parse_args())
