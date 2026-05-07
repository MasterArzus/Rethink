"""LLM-as-judge scoring for staged experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from prompt import build_score_prompt


PERSPECTIVES = ["student", "engineer", "literary_worker", "model_researcher", "teacher"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL output from an experiment method")
    parser.add_argument("--output", default=None)
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "MiniMax-M2.7"))
    parser.add_argument("--api-base", default=os.environ.get("MINIMAX_API_BASE", "https://api.minimaxi.com/anthropic"))
    parser.add_argument("--samples", type=int, default=10, help="Number of score groups per perspective")
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_client(api_base: str):
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY or MINIMAX_API_KEY before scoring.")
    import anthropic

    return anthropic.Anthropic(api_key=api_key, base_url=api_base) if api_base else anthropic.Anthropic(api_key=api_key)


def call_judge(client, model: str, prompt: str) -> Dict[str, Any]:
    msg = client.messages.create(
        model=model,
        max_tokens=300,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
    try:
        return json.loads(text[text.find("{") : text.rfind("}") + 1])
    except Exception:
        return {"constraint_score": 0, "quality_score": 0, "effort_score": 0, "comments": text[:300]}


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_scores.csv")
    rows = load_rows(input_path)
    rows = rows[: args.samples]
    client = make_client(args.api_base)
    scored = []
    for row in rows:
        compact = {
            "case_id": row.get("case_id"),
            "method": row.get("method"),
            "success": row.get("success"),
            "pass_k": row.get("pass_k"),
            "final_answer": row.get("final_answer"),
            "total_tokens": row.get("total_tokens"),
            "total_model_time_seconds": row.get("total_model_time_seconds"),
            "total_inspect_time_seconds": row.get("total_inspect_time_seconds"),
        }
        for perspective in PERSPECTIVES:
            result = call_judge(client, args.judge_model, build_score_prompt(compact, perspective))
            scored.append({"case_id": row.get("case_id"), "method": row.get("method"), "perspective": perspective, **result})
            time.sleep(0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)
    print(f"Wrote {len(scored)} scores to {output_path}")


if __name__ == "__main__":
    main()

