"""
Run IFEval with C0/C1 constraint level measurement.

C0 = base constraints (before [HARD CONSTRAINT])
C1 = C0 + hard constraints (full prompt, simulating user adding requirements)

Methods:
- vanilla: C0 only, 1 round, no C1
- regenerate: C0 multi-turn, then C1 multi-turn (regardless of C0 success)
- automated_local_repair: C0 multi-turn with local repair, then C1 multi-turn
- constrained_decoding: C0 only (1 round), C1 = N/A

Logic:
1. Extract base_prompt and hard_constraint from task["prompt"]
2. Run C0 with base constraints for K=0..max_turns-1
3. Run C1 with full constraints for K=0..max_turns-1 (regardless of C0)
"""
import os
import re
import sys
import time
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("/root/Rethink/dataset/ifeval")
from checkers import get_checker

from common import (
    MODELS,
    DEFAULT_DATASET_PATH,
    build_correction_prompt,
    load_tasks,
    make_result_record,
    save_result_records,
    split_task_groups,
    strip_reasoning_markers,
    extract_base_constraint,
    extract_hard_constraint,
)

# For constrained decoding
from constrained_decoding import build_bad_words_ids, build_json_template, build_prefix_allowed_tokens_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Run IFEval C0/C1 measurement.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--method", required=True,
                        choices=["regenerate", "automated_local_repair", "constrained_decoding"],
                        help="Method to evaluate")
    parser.add_argument("--max-turns", type=int, default=7,
                        help="Total max turns: C0=1 round, C1=K=1..max_turns-1")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--checkpoint-dir", default=None, help="Directory to save/load checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")
    return parser.parse_args()


def apply_chat_template(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    return tokenizer(prompt, return_tensors="pt").input_ids


def generate_response(model, tokenizer, messages, max_new_tokens):
    inputs = apply_chat_template(tokenizer, messages).to(model.device)
    input_len = inputs.shape[1]
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response, len(generated_ids)


def generate_with_bad_words(model, tokenizer, messages, bad_words_ids, max_new_tokens):
    inputs = apply_chat_template(tokenizer, messages).to(model.device)
    input_len = inputs.shape[1]
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if bad_words_ids:
        gen_kwargs["bad_words_ids"] = bad_words_ids
    with torch.no_grad():
        outputs = model.generate(inputs, **gen_kwargs)
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response, len(generated_ids)


def generate_with_prefix_constraint(model, tokenizer, messages, target_token_ids, prompt_len, max_new_tokens):
    inputs = apply_chat_template(tokenizer, messages).to(model.device)
    input_len = inputs.shape[1]
    target_len = len(target_token_ids)
    prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(target_token_ids, input_len, tokenizer.eos_token_id)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=target_len + 1,
            do_sample=False,
            num_beams=1,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][input_len:input_len + target_len]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response, len(generated_ids)


def extract_hard_words(hard_constraint_text: str) -> List[str]:
    """Extract the list of hard constraint forbidden words."""
    if not hard_constraint_text:
        return []
    pattern = r"AVOID using the following[^:]*:\s*['\"]?([^'\"]+)['\"]?"
    match = re.search(pattern, hard_constraint_text, re.IGNORECASE)
    if match:
        words_str = match.group(1)
        words = [w.strip().strip("'\"") for w in words_str.split(",")]
        return [w for w in words if w]
    quoted_words = re.findall(r"'(\w+)'", hard_constraint_text)
    return list(quoted_words)


def find_first_violation_token(response: str, forbidden_words: List[str]) -> Optional[str]:
    """Find the first forbidden word in response."""
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


def run_task_c0_c1(task, model_name, model, tokenizer, method, max_turns, max_new_tokens, log_handle):
    """
    Run task with C0/C1 constraint level measurement.

    C0 = 1 round (K=0), base constraints only
    C1 = K=1..max_turns-1 rounds, base + hard constraints

    Logic:
    1. C0: Run 1 round with base constraints (K=0)
    2. C1: Always run K=1..max_turns-1 with base + hard constraints (regardless of C0)
    """
    full_prompt = task["prompt"]
    full_constraints = task["constraints"]
    task_type = task["type"]

    # Extract base and hard constraints from prompt
    base_prompt = extract_base_constraint(full_prompt)
    hard_constraint = extract_hard_constraint(full_prompt)

    # Extract hard words for base constraints filtering
    hard_words = extract_hard_words(hard_constraint)

    checker = get_checker(task_type)

    # Determine which forbidden words to use for each phase
    all_forbidden = full_constraints.get("forbidden_words", [])
    base_forbidden = [w for w in all_forbidden if w not in hard_words] if hard_words else all_forbidden

    # C0 results (1 round only)
    c0_success = False
    c0_pass_turn = 0
    c0_final_response = ""
    c0_total_tokens = 0
    c0_init_tokens = 0

    # C1 results (K=1..max_turns-1)
    c1_success = False
    c1_pass_turn = 0
    c1_final_response = ""
    c1_total_tokens = 0

    start_time = time.perf_counter()

    # ============ C0: Base constraints only (1 round) ============
    if method == "constrained_decoding":
        # CD: C0 only (1 round for taboo/json), C1 = N/A
        if task_type == "taboo":
            bad_words_ids = build_bad_words_ids(tokenizer, base_forbidden) if base_forbidden else []
            response, gen_tokens = generate_with_bad_words(
                model, tokenizer, [{"role": "user", "content": base_prompt}],
                bad_words_ids, max_new_tokens
            )
        elif task_type == "json":
            target_json = build_json_template(task)
            response, gen_tokens = generate_with_prefix_constraint(
                model, tokenizer, [{"role": "user", "content": base_prompt}],
                tokenizer.encode(target_json, add_special_tokens=False),
                len(base_prompt), max_new_tokens
            )
        else:
            response, gen_tokens = generate_response(
                model, tokenizer, [{"role": "user", "content": base_prompt}], max_new_tokens
            )

        c0_init_tokens = gen_tokens
        c0_total_tokens = gen_tokens
        c0_final_response = response

        response_to_check = strip_reasoning_markers(response)
        passed, error_msg = checker.check(response_to_check, {"forbidden_words": base_forbidden})
        c0_success = passed
        c0_pass_turn = 1 if passed else 0

        log_handle.write(f"========= CD C0 =========\n")
        log_handle.write(f"Task ID: {task.get('id', 'unknown')}\n")
        log_handle.write(f"Response: {response[:300]}...\n")
        log_handle.write(f"Checker passed: {passed}\n")
        log_handle.write(f"Error: {error_msg}\n")

        # C1: Run with full constraints (simulating user adding requirements)
        # Only run if hard_constraint exists
        if hard_constraint:
            c1_prompt = f"{base_prompt}\n\n[HARD CONSTRAINT] {hard_constraint}"
            if task_type == "taboo":
                # Rebuild bad_words_ids for C1 with full forbidden words
                full_forbidden = all_forbidden
                bad_words_ids_c1 = build_bad_words_ids(tokenizer, full_forbidden) if full_forbidden else []
                c1_response, c1_gen_tokens = generate_with_bad_words(
                    model, tokenizer, [{"role": "user", "content": c1_prompt}],
                    bad_words_ids_c1, max_new_tokens
                )
            elif task_type == "json":
                # For JSON C1, use regular generate (prefix constraint too complex for hard constraints)
                c1_response, c1_gen_tokens = generate_response(
                    model, tokenizer, [{"role": "user", "content": c1_prompt}], max_new_tokens
                )
            else:
                c1_response, c1_gen_tokens = generate_response(
                    model, tokenizer, [{"role": "user", "content": c1_prompt}], max_new_tokens
                )

            c1_total_tokens = c1_gen_tokens
            c1_final_response = c1_response

            c1_response_to_check = strip_reasoning_markers(c1_response)
            c1_passed, c1_error_msg = checker.check(c1_response_to_check, full_constraints)
            c1_success = c1_passed
            c1_pass_turn = 2 if c1_passed else 0  # First attempt at C1

            log_handle.write(f"========= CD C1 =========\n")
            log_handle.write(f"Hard Constraint: {hard_constraint}\n")
            log_handle.write(f"Response: {c1_response[:300]}...\n")
            log_handle.write(f"Checker passed: {c1_passed}\n")
            log_handle.write(f"Error: {c1_error_msg}\n")
        else:
            c1_success = False
            c1_pass_turn = 0
            c1_final_response = ""
            log_handle.write(f"========= CD C1: No hard constraint =========\n")

        k1_acc1 = c1_success and c1_pass_turn == 2
        k1_acc_k = c1_success
        k1_pass_turn_val = c1_pass_turn

    else:  # regenerate or automated_local_repair
        # ============ C0: 1 round with base constraints ============
        if method == "regenerate":
            response, gen_tokens = generate_response(
                model, tokenizer, [{"role": "user", "content": base_prompt}], max_new_tokens
            )
        else:  # automated_local_repair
            forbidden_words = base_forbidden if base_forbidden else all_forbidden
            bad_words_ids = build_bad_words_ids(tokenizer, forbidden_words) if forbidden_words else []
            response, gen_tokens = generate_with_bad_words(
                model, tokenizer, [{"role": "user", "content": base_prompt}],
                bad_words_ids, max_new_tokens
            )

        c0_init_tokens = gen_tokens
        c0_total_tokens = gen_tokens
        c0_final_response = response

        response_to_check = strip_reasoning_markers(response)
        base_constraints_dict = {"forbidden_words": base_forbidden} if base_forbidden else full_constraints
        passed, error_msg = checker.check(response_to_check, base_constraints_dict)
        c0_success = passed
        c0_pass_turn = 1 if passed else 0

        log_handle.write(f"========= {method} C0 =========\n")
        log_handle.write(f"Task ID: {task.get('id', 'unknown')}\n")
        log_handle.write(f"Base Prompt: {base_prompt[:200]}...\n")
        log_handle.write(f"Response: {response[:300]}...\n")
        log_handle.write(f"Checker passed: {passed}\n")
        log_handle.write(f"Error: {error_msg}\n")

        # ============ C1: K=1..max_turns-1 with base + hard constraints ============
        # Always run C1 when there are hard constraints (C1 = new requirements from user)
        if hard_constraint:
            # C1 prompt: base + additional constraints
            c1_prompt = f"{base_prompt}\n\nAdditional constraints: {hard_constraint}"

            if method == "regenerate":
                messages = [{"role": "user", "content": c1_prompt}]
            else:  # automated_local_repair
                messages = [{"role": "user", "content": c1_prompt}]
                persistent_banned_ids = []

            c1_max_turns = max_turns - 1  # K=1..max_turns-1

            for turn in range(c1_max_turns):
                if method == "regenerate":
                    response, gen_tokens = generate_response(model, tokenizer, messages, max_new_tokens)
                else:  # automated_local_repair
                    turn_ban = list(persistent_banned_ids)
                    if turn > 0:
                        prev_response = messages[-1]["content"]
                        new_word = find_first_violation_token(prev_response, all_forbidden)
                        if new_word:
                            turn_ban.extend(build_ban_for_word(tokenizer, new_word))
                    response, gen_tokens = generate_with_bad_words(
                        model, tokenizer, messages, turn_ban if turn_ban else None, max_new_tokens
                    )

                c1_total_tokens += gen_tokens
                c1_final_response = response

                response_to_check = strip_reasoning_markers(response)
                passed, error_msg = checker.check(response_to_check, full_constraints)

                log_handle.write(f"========= {method} C1 Turn {turn+1} =========\n")
                log_handle.write(f"Hard Constraint: {hard_constraint}\n")
                log_handle.write(f"Response: {response[:300]}...\n")
                log_handle.write(f"Checker passed: {passed}\n")

                if passed:
                    c1_success = True
                    c1_pass_turn = turn + 2  # +2 because C0 was K=0, C1 starts at K=1
                    break

                if turn < c1_max_turns - 1:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": build_correction_prompt(task_type, error_msg)})

        k1_acc1 = c1_success and c1_pass_turn == 2  # C1 pass at first attempt (K=1)
        k1_acc_k = c1_success
        k1_pass_turn_val = c1_pass_turn

    wall_clock_seconds = time.perf_counter() - start_time

    # Determine final response and overall success (C0 AND C1 must both pass)
    if method == "constrained_decoding":
        overall_success = c0_success
        overall_pass_turn = c0_pass_turn
        overall_total_tokens = c0_total_tokens
        final_resp = c0_final_response
    else:
        # Both C0 and C1 must pass for overall success
        overall_success = c0_success and c1_success
        overall_pass_turn = c0_pass_turn if c0_success else (c1_pass_turn if c1_success else 0)
        overall_total_tokens = c0_total_tokens + c1_total_tokens
        final_resp = c1_final_response if c1_success else c0_final_response

    return make_result_record(
        task=task,
        model_name=model_name,
        method=method,
        success=overall_success,
        pass_turn=overall_pass_turn,
        init_tokens=c0_init_tokens,
        total_tokens=overall_total_tokens,
        wall_clock_seconds=wall_clock_seconds,
        num_generate_calls=1,
        checker_fail_count=0,
        error_type="",
        final_response=final_resp,
        violation_history=[],
        # C0/C1 measurement fields
        k0_acc1=c0_success,
        k1_acc1=k1_acc1,
        k0_acc_k=c0_success,
        k1_acc_k=k1_acc_k,
        k0_pass_turn=c0_pass_turn,
        k1_pass_turn=k1_pass_turn_val,
    )


def get_checkpoint_path(output_dir: Path, model_name: str, method: str) -> Path:
    return output_dir / f"{model_name}_{method}_c0c1_checkpoint.json"


def load_checkpoint(checkpoint_path: Path) -> Tuple[List[Dict], int]:
    """Load checkpoint and return results dict and next index."""
    if not checkpoint_path.exists():
        return [], 0
    with checkpoint_path.open() as f:
        data = json.load(f)
    results = data.get("results", [])
    next_index = data.get("next_index", 0)
    return results, next_index


def save_checkpoint(checkpoint_path: Path, results: List[Dict], next_index: int):
    with checkpoint_path.open("w") as f:
        json.dump({"results": results, "next_index": next_index}, f)


def run_group(tasks, model_name, model, tokenizer, output_dir: Path,
              method: str, max_turns: int, max_new_tokens: int, resume: bool):
    checkpoint_path = get_checkpoint_path(output_dir, model_name, method)

    # Load checkpoint if resume=True
    if resume and checkpoint_path.exists():
        results, start_index = load_checkpoint(checkpoint_path)
        print(f"[{model_name}] Resuming from checkpoint at index {start_index}")
    else:
        results = []
        start_index = 0

    log_path = output_dir / f"{model_name}_{method}_c0c1.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        for i in range(start_index, len(tasks)):
            task = tasks[i]
            print(f"[{model_name}] Processing task {i+1}/{len(tasks)} (ID: {task.get('id', 'unknown')})")
            try:
                record = run_task_c0_c1(task, model_name, model, tokenizer, method,
                                        max_turns, max_new_tokens, log_handle)
                results.append(record)

                # Save checkpoint every 10 tasks
                if (i + 1) % 10 == 0:
                    save_checkpoint(checkpoint_path, results, i + 1)
                    print(f"  Checkpoint saved at {i+1}")

            except Exception as exc:
                print(f"Error on task {task.get('id', 'unknown')}: {exc}")
                continue

    # Final save
    save_checkpoint(checkpoint_path, results, len(tasks))

    output_file = output_dir / f"{model_name}_{method}_c0c1.csv"
    save_result_records(results, output_file)
    print(f"Saved results to {output_file}")

    return results


def run_evaluation(args):
    if not os.path.exists(args.dataset_path):
        print(f"Dataset not found at {args.dataset_path}")
        return

    all_tasks = load_tasks(args.dataset_path)
    task_groups = split_task_groups(all_tasks)
    selected_models = {name: path for name, path in MODELS.items()
                       if not args.models or name in args.models}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        # Combine all tasks (taboo + json)
        all_tasks_combined = []
        for group_name, tasks in task_groups:
            all_tasks_combined.extend(tasks)

        print(f"Processing {len(all_tasks_combined)} tasks (combined taboo + json)...")
        run_group(all_tasks_combined, model_name, model, tokenizer, output_dir,
                 args.method, args.max_turns, args.max_new_tokens, args.resume)

        del model
        del tokenizer
        torch.cuda.empty_cache()

    print("\n=== All done! ===")


if __name__ == "__main__":
    run_evaluation(parse_args())