"""Summarize staged experiment outputs into paper-table friendly rows.

Main table grain: one row per (model, method), merging taboo and JSON cases.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


KEY_STAGES = [1, 2, 3, 4]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs")
    parser.add_argument("--output", default=None)
    parser.add_argument("--detail-output", default=None)
    return parser.parse_args()


def answer_words(text: str) -> int:
    return len((text or "").split())


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def stage_by_k(row: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(stage["k"]): stage for stage in row.get("stage_records", [])}


def stage_value(stages: Dict[int, Dict[str, Any]], k: int, key: str, default: Any = "") -> Any:
    return stages.get(k, {}).get(key, default)


def final_stage(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    records = row.get("stage_records", [])
    return records[-1] if records else None


def load_case_rows(path: Path) -> Iterable[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["case_id"], []).append(row)
    for case_id, records in grouped.items():
        records = sorted(records, key=lambda item: int(item["k"]))
        first = records[0]
        final = records[-1]
        pass_records = [r for r in records if bool_value(r.get("passed")) and int(r["k"]) >= 4]
        pass_k = int(pass_records[0]["k"]) if pass_records else 0
        yield {
            "model": first["model"],
            "method": first["method"],
            "case_id": case_id,
            "case_type": first["case_type"],
            "success": bool(pass_records),
            "pass_k": pass_k,
            "final_answer": final.get("answer", ""),
            "total_model_time_seconds": sum(float(r.get("model_time_seconds") or 0) for r in records),
            "total_inspect_time_seconds": sum(float(r.get("inspect_time_seconds") or 0) for r in records),
            "total_generated_tokens": sum(int(float(r.get("generated_tokens") or 0)) for r in records),
            "total_prompt_tokens": sum(int(float(r.get("prompt_tokens") or 0)) for r in records),
            "total_tokens": sum(int(float(r.get("total_tokens") or 0)) for r in records),
            "total_actor_tokens": sum(int(float(r.get("actor_tokens") or 0)) for r in records),
            "stage_records": records,
        }


def passed_by_or_at(row: Dict[str, Any], k: int) -> bool:
    pass_k = int(row.get("pass_k") or 0)
    if pass_k and pass_k <= k:
        return True
    stages = stage_by_k(row)
    return bool_value(stages.get(k, {}).get("passed", False))


def detail_row(row: Dict[str, Any]) -> Dict[str, Any]:
    stages = stage_by_k(row)
    final = final_stage(row) or {}
    attempts = len(row.get("stage_records", []))
    actor_api_calls = sum(1 for stage in row.get("stage_records", []) if int(float(stage.get("actor_tokens", 0) or 0)) > 0)
    out: Dict[str, Any] = {
        "model": row.get("model"),
        "method": row.get("method"),
        "case_id": row.get("case_id"),
        "case_type": row.get("case_type"),
        "success": row.get("success"),
        "pass_k": row.get("pass_k"),
        "K": final.get("k", ""),
        "attempts": attempts,
        "model_api_calls": attempts,
        "actor_api_calls": actor_api_calls,
        "gen_T": row.get("total_model_time_seconds", 0.0),
        "ins_T": row.get("total_inspect_time_seconds", 0.0),
        "tok": row.get("total_tokens", 0),
        "gen_tok": row.get("total_generated_tokens", 0),
        "prompt_tok": row.get("total_prompt_tokens", 0),
        "actor_tok": row.get("total_actor_tokens", 0),
        "final_words": answer_words(row.get("final_answer", "")),
    }
    total_tokens = float(out["tok"] or 0)
    out["tok_eff"] = (out["final_words"] / total_tokens) if total_tokens else 0.0
    for k in KEY_STAGES:
        out[f"ACC@{k}"] = passed_by_or_at(row, k)
        out[f"k{k}_gen_T"] = float(stage_value(stages, k, "model_time_seconds", 0.0) or 0.0)
        out[f"k{k}_ins_T"] = float(stage_value(stages, k, "inspect_time_seconds", 0.0) or 0.0)
        out[f"k{k}_tok"] = int(stage_value(stages, k, "total_tokens", 0) or 0)
    out["ACC@K"] = bool_value(row.get("success", False))
    return out


def avg(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0) or 0) for r in rows]
    return mean(vals) if vals else 0.0


def aggregate_rows(details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in details:
        groups.setdefault((row["model"], row["method"]), []).append(row)
    output = []
    for (model, method), rows in sorted(groups.items()):
        n = len(rows)
        item: Dict[str, Any] = {
            "model": model,
            "method": method,
            "n": n,
            "ACC@1": sum(1 for r in rows if r["ACC@1"]) / n if n else 0.0,
            "ACC@2": sum(1 for r in rows if r["ACC@2"]) / n if n else 0.0,
            "ACC@3": sum(1 for r in rows if r["ACC@3"]) / n if n else 0.0,
            "ACC@4": sum(1 for r in rows if r["ACC@4"]) / n if n else 0.0,
            "ACC@K": sum(1 for r in rows if r["ACC@K"]) / n if n else 0.0,
            "gen_T": avg(rows, "gen_T"),
            "ins_T": avg(rows, "ins_T"),
            "attempts": avg(rows, "attempts"),
            "model_api_calls": avg(rows, "model_api_calls"),
            "actor_api_calls": avg(rows, "actor_api_calls"),
            "tok": avg(rows, "tok"),
            "tok_eff": avg(rows, "tok_eff"),
        }
        output.append(item)
    return output


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    details: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.csv")):
        if path.name.startswith("summary") or path.name.endswith("_scores.csv"):
            continue
        for row in load_case_rows(path):
            details.append(detail_row(row))
    output = Path(args.output) if args.output else input_dir / "summary.csv"
    detail_output = Path(args.detail_output) if args.detail_output else input_dir / "summary_detail.csv"
    write_csv(detail_output, details)
    write_csv(output, aggregate_rows(details))
    print(f"Wrote {output} and {detail_output}")


if __name__ == "__main__":
    main()
