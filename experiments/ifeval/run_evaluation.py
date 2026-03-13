import os
import sys
import time
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add dataset/ifeval to path to import checkers
sys.path.append("/root/Rethink/dataset/ifeval")
try:
    from checkers import get_checker
except ImportError:
    print("Error: Could not import checkers.py. Make sure /root/Rethink/dataset/ifeval/checkers.py exists.")
    sys.exit(1)

from common import (
    MODELS,
    DEFAULT_DATASET_PATH,
    build_correction_prompt,
    load_tasks,
    make_result_record,
    save_result_records,
    split_task_groups,
    strip_reasoning_markers,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run IFEval baselines with a unified result schema.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--method", choices=["vanilla", "regenerate"], default="regenerate")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    return parser.parse_args()


def generate_response(model, tokenizer, messages, max_new_tokens):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    input_len = inputs.shape[1]
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response, len(generated_ids)


def run_task(task, model_name, model, tokenizer, method, max_turns, max_new_tokens, log_handle):
    prompt = task["prompt"]
    constraints = task["constraints"]
    task_type = task["type"]
    checker = get_checker(task_type)

    messages = [{"role": "user", "content": prompt}]
    violation_history = []
    init_tokens = 0
    total_tokens = 0
    turns = 0
    success = False
    error_type = ""
    final_response = ""

    start_time = time.perf_counter()
    turn_budget = 1 if method == "vanilla" else max_turns
    for turn in range(turn_budget):
        turns += 1
        response, gen_tokens = generate_response(model, tokenizer, messages, max_new_tokens=max_new_tokens)
        total_tokens += gen_tokens
        final_response = response

        if turn == 0:
            init_tokens = gen_tokens

        response_to_check = strip_reasoning_markers(response)
        passed, error_msg = checker.check(response_to_check, constraints)
        violation_history.append(error_msg or "passed")

        log_handle.write(f"========={os.linesep}")
        log_handle.write(f"Task ID: {task.get('id', 'unknown')}{os.linesep}")
        log_handle.write(f"Turn: {turn + 1}{os.linesep}")
        log_handle.write(f"Response: {response}{os.linesep}")
        log_handle.write(f"Checker passed: {passed}{os.linesep}")
        log_handle.write(f"Checker message: {error_msg}{os.linesep}")

        if passed:
            success = True
            break

        error_type = error_msg or "checker_failed"
        if method == "regenerate" and turn < turn_budget - 1:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": build_correction_prompt(task_type, error_msg)})

    wall_clock_seconds = time.perf_counter() - start_time
    pass_turn = turns if success else 0
    checker_fail_count = sum(1 for item in violation_history if item != "passed")

    return make_result_record(
        task=task,
        model_name=model_name,
        method=method,
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


def run_evaluation(args):
    if not os.path.exists(args.dataset_path):
        print(f"Dataset not found at {args.dataset_path}")
        return

    all_tasks = load_tasks(args.dataset_path)
    task_groups = split_task_groups(all_tasks)
    selected_models = {name: path for name, path in MODELS.items() if not args.models or name in args.models}

    for model_name, model_path in selected_models.items():
        print(f"Loading {model_name} from {model_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="auto", 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
            
        for group_name, tasks in task_groups:
            print(f"Processing {group_name} with {len(tasks)} tasks...")
            results = []

            output_dir = Path(args.output_dir)
            log_path = output_dir / f"{group_name}_{model_name}_{args.method}.log"
            with log_path.open("w", encoding="utf-8") as log_file:
                for i, task in enumerate(tasks):
                    print(f"[{group_name}] Processing task {i+1}/{len(tasks)} (ID: {task.get('id', 'unknown')})")
                    try:
                        record = run_task(
                            task,
                            model_name,
                            model,
                            tokenizer,
                            method=args.method,
                            max_turns=args.max_turns,
                            max_new_tokens=args.max_new_tokens,
                            log_handle=log_file,
                        )
                    except ValueError as exc:
                        print(f"Skipping task {task.get('id', 'unknown')}: {exc}")
                        continue
                    results.append(record)

                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(tasks)} tasks for {group_name}")

            output_file = output_dir / f"{group_name}_{model_name}_{args.method}.csv"
            save_result_records(results, output_file)
            print(f"Saved results to {output_file}")
        
        # Unload model to free memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_evaluation(parse_args())
