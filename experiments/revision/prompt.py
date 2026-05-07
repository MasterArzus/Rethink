"""Prompt builders for the staged K=8 revision experiments.

Stages:
K=1 base writing request
K=2 hard lexical/JSON constraint that constrained decoding can handle
K=3 tone/style constraint plus a weaker constrained-decoding proxy
K=4 dynamic exploratory constraint that refers to part of the existing answer
K=5..8 repair/continuation turns using checker feedback
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional


MAX_TURNS = 8


def get_stage(case: Dict[str, Any], k: int) -> Dict[str, Any]:
    for stage in case.get("stages", []):
        if int(stage.get("k", 0)) == k:
            return stage
    return {}


def active_stages(case: Dict[str, Any], upto_k: int) -> List[Dict[str, Any]]:
    return [s for s in case.get("stages", []) if int(s.get("k", 0)) <= upto_k]


def render_requirement(stage: Dict[str, Any]) -> str:
    text = stage.get("instruction") or stage.get("description") or ""
    return str(text).strip()


def build_task_prompt(case: Dict[str, Any], upto_k: int, previous_answer: Optional[str] = None) -> str:
    lines = [
        "Complete the user's writing task while satisfying every active requirement.",
        "Return only the answer, with no analysis or explanation.",
        "",
        "Active requirements:",
    ]
    for stage in active_stages(case, upto_k):
        req = render_requirement(stage)
        if req:
            lines.append(f"- K={stage.get('k')}: {req}")
    if previous_answer:
        lines.extend(
            [
                "",
                "Previous answer to revise:",
                previous_answer.strip(),
                "",
                "Revise the previous answer only as much as needed to satisfy the active requirements.",
            ]
        )
    return "\n".join(lines).strip()


def build_reflexion_prompt(
    case: Dict[str, Any],
    upto_k: int,
    previous_answer: str,
    checker_message: str,
) -> str:
    return "\n".join(
        [
            build_task_prompt(case, upto_k, previous_answer),
            "",
            "Checker feedback:",
            checker_message,
            "",
            "Think about the failure internally, then output a corrected final answer only.",
        ]
    )


def build_local_repair_prompt(
    case: Dict[str, Any],
    upto_k: int,
    preserved_prefix: str,
    checker_message: str,
) -> str:
    return "\n".join(
        [
            build_task_prompt(case, upto_k),
            "",
            "The following prefix has already been accepted and must be preserved verbatim:",
            preserved_prefix,
            "",
            "Continue from that prefix and fix the remaining problem.",
            f"Checker feedback: {checker_message}",
        ]
    )


def build_chat_actor_prompt(
    case: Dict[str, Any],
    upto_k: int,
    answer: str,
    checker_message: str,
) -> str:
    return "\n".join(
        [
            "You are simulating a human collaborator in a chat interface.",
            "Give one short instruction that helps the model satisfy the newly active constraints.",
            "Do not solve the task yourself.",
            "",
            build_task_prompt(case, upto_k),
            "",
            "Model answer:",
            answer,
            "",
            "Checker feedback:",
            checker_message,
        ]
    )


def build_steer_actor_prompt(
    case: Dict[str, Any],
    upto_k: int,
    answer: str,
    checker_message: str,
    candidates: Optional[Iterable[str]] = None,
    lite: bool = False,
) -> str:
    options = "rewind only" if lite else "rewind, preserve-prefix continuation, and token/logit-lens hints"
    candidate_text = ", ".join(candidates or [])
    return "\n".join(
        [
            "You are simulating a human using a steering interface.",
            f"Available controls: {options}.",
            "Choose the smallest useful rewind point and one concise steering instruction.",
            "",
            build_task_prompt(case, upto_k),
            "",
            "Current answer:",
            answer,
            "",
            "Checker feedback:",
            checker_message,
            f"Token/logit candidates: {candidate_text}" if candidate_text else "",
            "",
            "Respond as JSON with keys: action, rewind_fraction, instruction.",
        ]
    ).strip()


def build_cd_proxy_prompt(case: Dict[str, Any], upto_k: int) -> str:
    """Prompt used when true constrained decoding is unavailable for soft constraints."""
    prompt = build_task_prompt(case, upto_k)
    if upto_k >= 3:
        prompt += "\n\nUse a gentle, tactful wording style throughout."
    if upto_k >= 4:
        prompt += "\n\nNote: dynamic constraints are checked after generation and cannot be guaranteed by constrained decoding."
    return prompt


def build_judge_prompt(case: Dict[str, Any], upto_k: int, answer: str) -> str:
    stage4 = get_stage(case, 4)
    dynamic = stage4.get("dynamic", {})
    rubric = dynamic.get("judge_rubric", [])
    rubric_text = "\n".join(f"- {item}" for item in rubric)
    return "\n".join(
        [
            "You are a lenient, good-faith judge for a staged writing-constraint experiment.",
            "Return compact JSON only: {\"pass\": boolean, \"reason\": string}.",
            "The K=4 requirement is dynamic and semantic. Pass unless there is a clear, material violation; do not require exact wording.",
            "",
            "Active case summary:",
            json.dumps(
                {
                    "id": case.get("id"),
                    "type": case.get("type"),
                    "topic": case.get("topic"),
                    "requirements": [
                        {"k": s.get("k"), "type": s.get("type"), "instruction": s.get("instruction"), "constraints": s.get("constraints", {})}
                        for s in case.get("stages", [])
                        if int(s.get("k", 0)) <= upto_k
                    ],
                    "dynamic": dynamic,
                },
                ensure_ascii=False,
            ),
            "",
            f"Check all requirements up to K={upto_k}.",
            "K=4 rubric:",
            rubric_text or "- No extra rubric provided.",
            "",
            "Answer:",
            answer,
        ]
    )


def build_score_prompt(row: Dict[str, Any], perspective: str) -> str:
    return "\n".join(
        [
            f"You are scoring an experiment result from the perspective of a {perspective}.",
            "Give compact JSON only with keys: constraint_score, quality_score, effort_score, comments.",
            "Scores are integers from 1 to 10.",
            "",
            json.dumps(row, ensure_ascii=False),
        ]
    )
