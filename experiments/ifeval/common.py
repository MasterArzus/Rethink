import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


MODELS = {
    "deepseek_r1": "/root/autodl-fs/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llama3_8b": "/root/autodl-fs/LLM-Research/Meta-Llama-3.1-8B-Instruct",
    "qwen3_8b": "/root/autodl-fs/Qwen/Qwen3-8B",
    "deepseek_r1_qwen_1_5b": "/root/autodl-fs/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "qwen2_5_1_5b": "/root/autodl-fs/Qwen/Qwen2.5-1.5B-Instruct",
    "llama2_13b_chat": "/root/autodl-fs/LLM-Research/Llama-2-13b-chat-hf",
    "qwen2_5_14b_instruct": "/root/autodl-fs/Qwen/Qwen2.5-14B-Instruct",
}

DEFAULT_DATASET_PATH = "/root/Rethink/dataset/ifeval/taskset_60_hard.json"


def load_tasks(dataset_path: str) -> List[Dict]:
    with open(dataset_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["tasks"]


def split_task_groups(tasks: Iterable[Dict]) -> List[Tuple[str, List[Dict]]]:
    task_list = list(tasks)
    taboo_tasks = [task for task in task_list if task["type"] == "taboo"]
    json_tasks = [task for task in task_list if task["type"] == "json"]
    return [("taboo_hard", taboo_tasks), ("json_hard", json_tasks)]


def strip_reasoning_markers(response: str) -> str:
    """
    Remove reasoning tags from various model formats:
    - DeepSeek: <think>...</think> (standard)
    - Qwen: <|im_start|>...<|im_end|> or <im>...</im>
    - Others: similar patterns
    """
    # DeepSeek style: remove think blocks
    # Format can be: <think>...</think> output  OR  content...</think> output
    if "</think>" in response:
        # Check if there's a matching <think> before </think>
        think_start = response.find("<think>")
        think_end = response.find("</think>")

        if think_start != -1 and think_start < think_end:
            # Standard format: <think>...</think> - remove complete blocks
            text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
            # Check for any trailing incomplete think
            trailing_open = text.find("<think>")
            if trailing_open != -1:
                text = text[:trailing_open]
        else:
            # Only </think> exists (no <think> before it)
            # This means content before </think> is thinking, extract after
            text = response[think_end + len("</think>"):]
    elif "<think>" in response:
        # Only <think> exists (truncated), extract content after it
        idx = response.find("<think>")
        text = response[idx + len("<think>"):]
    else:
        text = response

    # Qwen style: <|im_start|>...<|im_end|> or <im>...</im>
    text = re.sub(r"<\|im_start\|>.*?\|im_end\|>", "", text, flags=re.DOTALL)
    trailing_open = text.find("<|im_start|>")
    if trailing_open != -1:
        text = text[:trailing_open]

    text = re.sub(r"<im>.*?</im>", "", text, flags=re.DOTALL)
    trailing_open = text.find("<im>")
    if trailing_open != -1:
        text = text[:trailing_open]

    return text.strip()


def build_correction_prompt(task_type: str, error_msg: str) -> str:
    if task_type == "taboo" and error_msg and "Found forbidden words:" in error_msg:
        words = error_msg.replace("Found forbidden words:", "").strip()
        return f"Don't use the following words in the subsequent answer: {words}"
    return f"Your answer does not satisfy the constraints: {error_msg}. Please correct it."


def extract_base_constraint(prompt: str) -> str:
    """Extract the base constraint part (before [HARD CONSTRAINT])"""
    if "[HARD CONSTRAINT]" in prompt:
        return prompt.split("[HARD CONSTRAINT]")[0].strip()
    return prompt.strip()


def extract_hard_constraint(prompt: str) -> str:
    """Extract the hard constraint part (after [HARD CONSTRAINT])"""
    if "[HARD CONSTRAINT]" in prompt:
        return prompt.split("[HARD CONSTRAINT]")[1].strip()
    return ""


def extract_base_constraints(constraints: Dict) -> Dict:
    """
    Extract base constraints by removing hard constraint words.
    Hard constraints are the extra banned words added during task hardening.
    """
    # The full constraints already contain both base and hard words.
    # For K=0 measurement, we need to identify which words were the hard additions.
    # This requires knowing which words were in the original vs hard constraint.
    # A simple approach: assume hard words are high-frequency stop words not in original IFEval.
    # For now, return a copy of constraints - the actual filtering happens at checker level.
    return constraints.copy()


def make_result_record(
    *,
    task: Dict,
    model_name: str,
    method: str,
    success: bool,
    pass_turn: int,
    init_tokens: int,
    total_tokens: int,
    wall_clock_seconds: float,
    num_generate_calls: int,
    checker_fail_count: int,
    error_type: str,
    final_response: str,
    violation_history: List[str],
    inspect_time_seconds: float = 0.0,
    total_llm_actor_tokens: int = 0,
    total_clicks: int = 0,
    gen_time_seconds: float = 0.0,
    vanilla_total_tokens: int = 0,
    # K=0/K=1 staged measurement fields
    k0_acc1: bool = False,
    k1_acc1: bool = False,
    k0_acc_k: bool = False,
    k1_acc_k: bool = False,
    k0_pass_turn: int = 0,
    k1_pass_turn: int = 0,
) -> Dict:
    # gen_time_s = wall_clock_s - inspect_time_s (for human methods) or = wall_clock_s (for automated)
    if gen_time_seconds == 0.0:
        gen_time_seconds = wall_clock_seconds - inspect_time_seconds

    # token_eff% = (vanilla_tokens - method_tokens) / vanilla_tokens * 100
    if vanilla_total_tokens > 0 and total_tokens > 0:
        token_eff_percent = (vanilla_total_tokens - total_tokens) / vanilla_total_tokens * 100
    else:
        token_eff_percent = 0.0

    return {
        "task_id": task.get("id"),
        "task_type": task.get("type"),
        "model": model_name,
        "method": method,
        "success": success,
        "pass_turn": pass_turn,
        "init_tokens": init_tokens,
        "total_tokens": total_tokens,
        "wall_clock_seconds": wall_clock_seconds,
        "num_generate_calls": num_generate_calls,
        "checker_fail_count": checker_fail_count,
        "error_type": error_type,
        "final_response": final_response,
        "violation_history": json.dumps(violation_history, ensure_ascii=False),
        "prompt": task.get("prompt", ""),
        "constraints": json.dumps(task.get("constraints", {}), ensure_ascii=False),
        "inspect_time_seconds": inspect_time_seconds,
        "total_llm_actor_tokens": total_llm_actor_tokens,
        "total_clicks": total_clicks,
        "gen_time_seconds": gen_time_seconds,
        "vanilla_total_tokens": vanilla_total_tokens,
        "token_eff_percent": token_eff_percent,
        # K=0/K=1 staged measurement
        "k0_acc1": k0_acc1,
        "k1_acc1": k1_acc1,
        "k0_acc_k": k0_acc_k,
        "k1_acc_k": k1_acc_k,
        "k0_pass_turn": k0_pass_turn,
        "k1_pass_turn": k1_pass_turn,
    }


def save_result_records(records: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    jsonl_path = output_path.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")