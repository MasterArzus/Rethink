"""Stage-aware checkers for the revision experiments."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from prompt import build_judge_prompt


WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
POLITE_MARKERS = {
    "please", "could", "would", "appreciate", "kindly", "consider",
    "sorry", "thank", "thanks", "may", "might", "perhaps",
}
HARSH_MARKERS = {"must", "obviously", "wrong", "stupid", "ridiculous", "never"}


@dataclass
class CheckResult:
    passed: bool
    message: str
    failed_stage: Optional[int] = None
    failure_type: str = ""


def normalize_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def find_forbidden_words(text: str, words: List[str]) -> List[str]:
    lowered = text.lower()
    hits = []
    for word in words:
        pattern = r"\b" + re.escape(str(word).lower()) + r"\b"
        if re.search(pattern, lowered):
            hits.append(str(word))
    return hits


def extract_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    raw = normalize_text(text)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1)
    else:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end >= start:
            raw = raw[start : end + 1]
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        return None, f"Invalid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "JSON must be an object"
    return parsed, None


def split_halves(text: str) -> Tuple[str, str]:
    sentences = re.split(r"(?<=[.!?])\s+", normalize_text(text))
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) >= 2:
        mid = max(1, len(sentences) // 2)
        return " ".join(sentences[:mid]), " ".join(sentences[mid:])
    words = text.split()
    mid = max(1, len(words) // 2)
    return " ".join(words[:mid]), " ".join(words[mid:])


def heuristic_dynamic_check(case: Dict[str, Any], answer: str) -> Tuple[bool, str]:
    stage4 = next((s for s in case.get("stages", []) if int(s.get("k", 0)) == 4), {})
    dyn = stage4.get("dynamic", {})
    front, back = split_halves(answer)
    anchor_terms = dyn.get("anchor_terms", []) or dyn.get("back_required_terms", [])
    banned_front = dyn.get("front_forbidden_terms", [])
    banned_back = dyn.get("back_forbidden_terms", [])

    misses = []
    # K=4 is meant to test dynamic revision, not exact keyword placement.
    # Keep the front-half check permissive: only ensure the newly introduced
    # exploratory term does not leak into the preserved prefix.
    for term in anchor_terms:
        variants = [str(term).lower()]
        if not any(re.search(r"\b" + re.escape(v) + r"\b", back.lower()) for v in variants):
            misses.append(f"back half does not clearly include the new angle '{term}'")
    front_hits = find_forbidden_words(front, banned_front)
    back_hits = find_forbidden_words(back, banned_back)
    if front_hits:
        misses.append(f"front half contains forbidden terms: {', '.join(front_hits)}")
    if back_hits:
        misses.append(f"back half contains forbidden terms: {', '.join(back_hits)}")
    if misses:
        return False, "; ".join(misses)
    return True, "dynamic half constraint passed"


def llm_judge(case: Dict[str, Any], upto_k: int, answer: str, judge_model: Optional[str] = None) -> Optional[CheckResult]:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MINIMAX_API_KEY")
    api_base = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("MINIMAX_API_BASE") or "https://api.minimaxi.com/anthropic"
    if not api_key:
        raise RuntimeError("K=4 LLM-as-Judge requires MINIMAX_API_KEY or ANTHROPIC_API_KEY.")
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, base_url=api_base) if api_base else anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=judge_model or os.environ.get("JUDGE_MODEL", "MiniMax-M2.7"),
            max_tokens=256,
            temperature=0,
            system="Return compact JSON only. Do not include thinking or analysis.",
            messages=[{"role": "user", "content": build_judge_prompt(case, upto_k, answer)}],
        )
        blocks = msg.content or []
        text_blocks = [block.text for block in blocks if getattr(block, "type", "") == "text"]
        text = "".join(text_blocks).strip()
        if not text:
            thinking_blocks = [block.thinking for block in blocks if getattr(block, "type", "") == "thinking"]
            fallback_text = "".join(thinking_blocks).strip()
            if fallback_text:
                text = fallback_text
            else:
                block_types = [getattr(block, "type", None) for block in blocks]
                stop_reason = getattr(msg, "stop_reason", None)
                raise ValueError(f"LLM judge returned no text blocks (types={block_types}, stop_reason={stop_reason})")
        data, err = extract_json_object(text)
        if err:
            preview = text[:200].replace("\n", " ").strip()
            ok, msg = heuristic_dynamic_check(case, answer)
            return CheckResult(ok, f"LLM judge raw: {preview} | heuristic: {msg}", 4 if not ok else None, "llm_judge_raw")
        passed = bool(data.get("pass"))
        reason = str(data.get("reason", "")).strip()
        return CheckResult(passed, f"LLM judge: {reason}", 4 if not passed else None, "llm_judge")
    except Exception as exc:
        return CheckResult(False, f"LLM judge failed: {exc}", 4, "llm_judge_error")


class RevisionChecker:
    def __init__(self, use_llm_judge: bool = False, judge_model: Optional[str] = None):
        self.use_llm_judge = use_llm_judge
        self.judge_model = judge_model

    def check(self, case: Dict[str, Any], answer: str, upto_k: int) -> CheckResult:
        answer = normalize_text(answer)
        if not answer:
            return CheckResult(False, "Empty answer", 1, "empty")

        for stage in case.get("stages", []):
            k = int(stage.get("k", 0))
            if k > upto_k:
                continue
            kind = stage.get("type")
            constraints = stage.get("constraints", {})
            if kind == "base":
                continue
            elif kind == "taboo":
                hits = find_forbidden_words(answer, constraints.get("forbidden_words", []))
                if hits:
                    return CheckResult(False, f"Found forbidden words: {', '.join(hits)}", k, "taboo")
            elif kind == "json":
                obj, err = extract_json_object(answer)
                if err:
                    return CheckResult(False, err, k, "json")
                keys = constraints.get("keys", [])
                missing = [key for key in keys if key not in obj]
                if missing:
                    return CheckResult(False, f"Missing JSON keys: {', '.join(missing)}", k, "json")
                if constraints.get("no_extra_keys"):
                    extra = [key for key in obj if key not in keys]
                    if extra:
                        return CheckResult(False, f"Extra JSON keys: {', '.join(extra)}", k, "json")
            elif kind == "tone":
                lowered = answer.lower()
                if constraints.get("polite", True):
                    if not any(re.search(r"\b" + marker + r"\b", lowered) for marker in POLITE_MARKERS):
                        return CheckResult(False, "Tone proxy failed: no polite marker found", k, "tone")
                harsh_hits = find_forbidden_words(answer, list(HARSH_MARKERS))
                if harsh_hits:
                    return CheckResult(False, f"Tone proxy failed: harsh terms {', '.join(harsh_hits)}", k, "tone")
            elif kind == "dynamic":
                ok, msg = heuristic_dynamic_check(case, answer)
                judged = llm_judge(case, upto_k, answer, self.judge_model)
                judged.message = f"{judged.message} | heuristic: {msg}"
                return judged
        return CheckResult(True, "passed", None, "")


def first_repair_position(answer: str, result) -> int:
    failure_type = getattr(result, "failure_type", "")
    message = getattr(result, "message", None)
    if message is None:
        message = getattr(result, "checker_message", "")
    if failure_type == "taboo":
        words = message.replace("Found forbidden words:", "").split(",")
        lowered = answer.lower()
        positions = [lowered.find(w.strip().lower()) for w in words if w.strip()]
        positions = [p for p in positions if p >= 0]
        return min(positions) if positions else max(0, len(answer) // 2)
    if failure_type in {"json", "dynamic", "tone"}:
        return max(0, len(answer) // 2)
    return 0
