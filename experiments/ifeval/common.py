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
    # Remove complete think blocks and hide trailing unfinished think segments.
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    trailing_open = text.find("<think>")
    if trailing_open != -1:
        text = text[:trailing_open]
    return text.strip()


def build_correction_prompt(task_type: str, error_msg: str) -> str:
    if task_type == "taboo" and error_msg and "Found forbidden words:" in error_msg:
        words = error_msg.replace("Found forbidden words:", "").strip()
        return f"Don't use the following words in the subsequent answer: {words}"
    return f"Your answer does not satisfy the constraints: {error_msg}. Please correct it."


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
) -> Dict:
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
    }


def save_result_records(records: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    jsonl_path = output_path.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")