import json
from typing import Dict, Iterable, List, Optional


def build_bad_words_ids(tokenizer, forbidden_words: Iterable[str]) -> List[List[int]]:
    variants = set()
    for word in forbidden_words:
        if not word:
            continue
        normalized = str(word)
        variants.update(
            {
                normalized,
                normalized.lower(),
                normalized.upper(),
                normalized.capitalize(),
                f" {normalized}",
                f" {normalized.lower()}",
                f"\n{normalized}",
                f'"{normalized}"',
                f'"{normalized.lower()}"',
            }
        )

    bad_words_ids: List[List[int]] = []
    seen = set()
    for variant in variants:
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        if not token_ids:
            continue
        key = tuple(token_ids)
        if key in seen:
            continue
        seen.add(key)
        bad_words_ids.append(token_ids)
    return bad_words_ids


def build_prefix_allowed_tokens_fn(target_token_ids: List[int], prompt_len: int, eos_token_id: Optional[int]):
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        generated_len = input_ids.shape[-1] - prompt_len
        if generated_len < len(target_token_ids):
            return [target_token_ids[generated_len]]
        if eos_token_id is not None:
            return [eos_token_id]
        return [target_token_ids[-1]]

    return prefix_allowed_tokens_fn


def build_json_template(task: Dict) -> str:
    json_constraints = task.get("constraints", {}).get("json", {})
    schema = json_constraints.get("schema")
    if not schema:
        payload = {"response": infer_freeform_value(task.get("prompt", ""))}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    values = {}
    for key, type_name in schema.get("keys", {}).items():
        values[key] = placeholder_value(key, type_name, task.get("prompt", ""))

    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def placeholder_value(key: str, type_name: str, prompt: str):
    normalized = (type_name or "string").strip().lower()
    if normalized == "string":
        return infer_string_value(key, prompt)
    if normalized == "int":
        return 1
    if normalized == "float":
        return 0.5
    if normalized == "bool":
        return True
    if normalized == "list[string]":
        return [infer_string_value(key, prompt)]
    if normalized == "list[int]":
        return [1]
    if normalized == "list[float]":
        return [0.5]
    if normalized == "object":
        return {"value": infer_string_value(key, prompt)}
    return infer_string_value(key, prompt)


def infer_string_value(key: str, prompt: str) -> str:
    key_lower = key.lower()
    if "title" in key_lower:
        return "generated title"
    if "summary" in key_lower:
        return "generated summary"
    if "question" in key_lower:
        return "generated question"
    if "answer" in key_lower:
        return "generated answer"
    if "city" in key_lower:
        return "generated city"
    if "name" in key_lower:
        return "generated name"
    if "priority" in key_lower:
        return "medium"
    if "tags" in key_lower:
        return "general"
    if "steps" in key_lower:
        return "step one"
    if "items" in key_lower:
        return "item one"
    if "response" in key_lower:
        return infer_freeform_value(prompt)
    return f"{key_lower} value"


def infer_freeform_value(prompt: str) -> str:
    text = " ".join(prompt.strip().split())
    text = text.replace('"', "'")
    if len(text) > 60:
        text = text[:57].rstrip() + "..."
    return text or "generated response"