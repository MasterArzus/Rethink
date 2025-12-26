import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RAW_DEFAULT = Path(__file__).parent / "input_data.jsonl"
OUT_DEFAULT = Path(__file__).parent / "taskset_120.json"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_forbidden_words(kwargs_list: Any) -> Optional[List[str]]:
    if not isinstance(kwargs_list, list):
        return None
    for kw in kwargs_list:
        if isinstance(kw, dict) and kw.get("forbidden_words"):
            fw = kw.get("forbidden_words")
            if isinstance(fw, list) and all(isinstance(x, str) for x in fw):
                return fw
    return None


def _curated_taboo_tasks(n: int) -> List[Dict[str, Any]]:
    # Keep the prompts non-knowledge-based and easy to verify.
    templates: List[Tuple[str, List[str]]] = [
        (
            "Write a short product description for a new note-taking app. Do not use the words {w1} or {w2}.",
            ["notes", "app"],
        ),
        (
            "Give three tips for staying focused while studying. Do not use the words {w1} or {w2}.",
            ["focus", "study"],
        ),
        (
            "Describe a peaceful morning routine in 4 sentences. Do not use the words {w1}, {w2}, or {w3}.",
            ["coffee", "sun", "music"],
        ),
        (
            "Write a short email to reschedule a meeting. Do not use the words {w1} or {w2}.",
            ["meeting", "schedule"],
        ),
        (
            "Explain what a habit is in simple terms. Do not use the words {w1} or {w2}.",
            ["habit", "routine"],
        ),
        (
            "Write a friendly welcome message for a community forum. Do not use the words {w1} or {w2}.",
            ["welcome", "community"],
        ),
        (
            "Write a two-sentence summary of why sleep matters. Do not use the words {w1}, {w2}, or {w3}.",
            ["sleep", "health", "brain"],
        ),
        (
            "Describe a rainy day in a poetic style (3-5 sentences). Do not use the words {w1} or {w2}.",
            ["rain", "cloud"],
        ),
        (
            "Write a short apology for replying late. Do not use the words {w1} or {w2}.",
            ["sorry", "late"],
        ),
        (
            "Describe a simple recipe for a sandwich. Do not use the words {w1} or {w2}.",
            ["bread", "sandwich"],
        ),
        (
            "Write a motivational quote (one sentence). Do not use the words {w1} or {w2}.",
            ["success", "dream"],
        ),
        (
            "Write a short description of a city park. Do not use the words {w1}, {w2}, or {w3}.",
            ["trees", "bench", "grass"],
        ),
    ]

    out: List[Dict[str, Any]] = []
    for i in range(n):
        tpl, words = templates[i % len(templates)]
        fmt = {"w1": words[0], "w2": words[1] if len(words) > 1 else words[0], "w3": words[2] if len(words) > 2 else words[0]}
        prompt = tpl.format(**fmt)
        out.append(
            {
                "id": f"curated_taboo_{i+1:03d}",
                "source": "curated",
                "type": "taboo",
                "prompt": prompt,
                "constraints": {
                    "forbidden_words": words,
                    "match": {"casefold": True, "word_boundary": True},
                },
            }
        )
    return out


def _curated_json_tasks(n: int) -> List[Dict[str, Any]]:
    # Keep schemas simple. Avoid requiring factual knowledge.
    schemas: List[Dict[str, Any]] = [
        {
            "keys": {"title": "string", "summary": "string", "tags": "list[string]"},
            "no_extra_keys": True,
        },
        {
            "keys": {"name": "string", "age": "int", "city": "string"},
            "no_extra_keys": True,
        },
        {
            "keys": {"steps": "list[string]", "duration_minutes": "int"},
            "no_extra_keys": True,
        },
        {
            "keys": {"question": "string", "answer": "string", "confidence": "float"},
            "no_extra_keys": True,
        },
        {
            "keys": {"items": "list[string]", "priority": "string"},
            "no_extra_keys": True,
        },
    ]

    prompts: List[str] = [
        "Return ONLY a valid JSON object with keys: {keys}. Do not include any extra keys.",
        "Output must be strictly valid JSON (no markdown). Use keys: {keys}. No extra keys.",
        "Provide a JSON object with exactly these keys: {keys}. No additional text.",
        "Respond in strict JSON with keys: {keys}. No code fences.",
    ]

    out: List[Dict[str, Any]] = []
    for i in range(n):
        schema = schemas[i % len(schemas)]
        keys = ", ".join(schema["keys"].keys())
        prompt = prompts[i % len(prompts)].format(keys=keys)
        out.append(
            {
                "id": f"curated_json_{i+1:03d}",
                "source": "curated",
                "type": "json",
                "prompt": prompt,
                "constraints": {
                    "json": {
                        "strict": True,
                        "allow_code_fence": False,
                        "schema": schema,
                    }
                },
            }
        )
    return out


def build_taskset(raw_rows: List[Dict[str, Any]], seed: int = 7) -> Dict[str, Any]:
    rng = random.Random(seed)

    taboo: List[Dict[str, Any]] = []
    json_tasks: List[Dict[str, Any]] = []

    for row in raw_rows:
        instruction_ids = row.get("instruction_id_list", [])
        if not isinstance(instruction_ids, list):
            continue

        if "keywords:forbidden_words" in instruction_ids:
            forbidden = _extract_forbidden_words(row.get("kwargs"))
            if forbidden:
                taboo.append(
                    {
                        "id": f"ifeval_taboo_{row.get('key')}",
                        "source": "ifeval",
                        "source_key": row.get("key"),
                        "type": "taboo",
                        "prompt": row.get("prompt", ""),
                        "instruction_id_list": instruction_ids,
                        "constraints": {
                            "forbidden_words": forbidden,
                            "match": {"casefold": True, "word_boundary": True},
                        },
                    }
                )

        if "detectable_format:json_format" in instruction_ids:
            json_tasks.append(
                {
                    "id": f"ifeval_json_{row.get('key')}",
                    "source": "ifeval",
                    "source_key": row.get("key"),
                    "type": "json",
                    "prompt": row.get("prompt", ""),
                    "instruction_id_list": instruction_ids,
                    "constraints": {
                        "json": {"strict": False, "allow_code_fence": True, "schema": None}
                    },
                }
            )

    # Target: ~120 tasks, balanced Taboo + JSON.
    # Use all IFEval taboo tasks and all IFEval json_format tasks, then top up with curated templates.
    target_total = 120
    target_each = target_total // 2  # 60/60

    taboo_selected = taboo[:]
    json_selected = json_tasks[:]

    taboo_needed = max(0, target_each - len(taboo_selected))
    json_needed = max(0, target_each - len(json_selected))

    taboo_selected.extend(_curated_taboo_tasks(taboo_needed))
    json_selected.extend(_curated_json_tasks(json_needed))

    # If totals are still off (odd targets), pad with curated JSON.
    tasks = taboo_selected + json_selected
    if len(tasks) < target_total:
        tasks.extend(_curated_json_tasks(target_total - len(tasks)))

    rng.shuffle(tasks)

    return {
        "meta": {
            "name": "rethink_ifeval_style_120",
            "version": "0.1",
            "seed": seed,
            "counts": {
                "taboo": sum(1 for t in tasks if t.get("type") == "taboo"),
                "json": sum(1 for t in tasks if t.get("type") == "json"),
                "total": len(tasks),
            },
            "sources": {
                "ifeval_raw": "https://github.com/google-research/google-research/tree/master/instruction_following_eval",
                "note": "Task set includes all IFEval forbidden-words and json_format prompts, topped up with curated IFEval-style templates to reach ~120 tasks.",
            },
        },
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default=str(RAW_DEFAULT))
    parser.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_path = Path(args.out)

    rows = _read_jsonl(raw_path)
    taskset = build_taskset(rows, seed=args.seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(taskset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {taskset['meta']['counts']} to {out_path}")


if __name__ == "__main__":
    main()
