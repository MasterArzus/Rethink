"""Dataset builder for staged taboo/JSON experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def taboo_case(idx: int, topic: str, banned: List[str], front: str, back: str) -> Dict[str, Any]:
    dynamic_goal = (
        f"Revise only the latter part so it becomes more exploratory and constructive, "
        f"adding a concrete future-facing angle about {back} while preserving the earlier compliant content."
    )
    return {
        "id": f"taboo_stage_{idx:03d}",
        "type": "taboo",
        "topic": topic,
        "stages": [
            {
                "k": 1,
                "type": "base",
                "instruction": f"Write a short paragraph about {topic}. Aim for 50-100 words and stay under 200 words.",
                "constraints": {"guidance_only": True},
            },
            {
                "k": 2,
                "type": "taboo",
                "instruction": f"Do not use these words: {', '.join(banned)}.",
                "constraints": {"forbidden_words": banned},
            },
            {
                "k": 3,
                "type": "tone",
                "instruction": "Make the tone gentle and tactful, as if giving careful advice.",
                "constraints": {"polite": True},
                "cd_proxy": "Prefer polite markers such as please/could/would.",
            },
            {
                "k": 4,
                "type": "dynamic",
                "instruction": dynamic_goal,
                "dynamic": {
                    "preserve_scope": "first_half",
                    "modify_scope": "second_half",
                    "dynamic_goal": dynamic_goal,
                    "anchor_terms": [back],
                    "judge_rubric": [
                        "The first half remains broadly consistent with the earlier answer and does not introduce forbidden words.",
                        "The second half meaningfully addresses the new dynamic goal rather than merely appending a keyword.",
                        "The revision is local: it should not unnecessarily rewrite the whole answer.",
                        "The answer still satisfies the active k=2 forbidden-word constraint and k=3 polite/tactful tone constraint.",
                    ],
                },
            },
        ],
    }


def json_case(idx: int, topic: str, keys: List[str], banned: List[str], front: str, back: str) -> Dict[str, Any]:
    dynamic_goal = (
        f"Revise only the latter part of the JSON values so they add a constructive exploratory detail "
        f"about {back}, while preserving the required JSON keys and earlier compliant content."
    )
    return {
        "id": f"json_stage_{idx:03d}",
        "type": "json",
        "topic": topic,
        "stages": [
            {
                "k": 1,
                "type": "base",
                "instruction": f"Write a compact JSON object summarizing {topic}. Aim for 50-100 words total and stay under 200 words.",
                "constraints": {"guidance_only": True},
            },
            {
                "k": 2,
                "type": "json",
                "instruction": f"Return only valid JSON with exactly these keys: {', '.join(keys)}.",
                "constraints": {"keys": keys, "no_extra_keys": True},
            },
            {
                "k": 2,
                "type": "taboo",
                "instruction": f"Do not use these words anywhere in the JSON values: {', '.join(banned)}.",
                "constraints": {"forbidden_words": banned},
            },
            {
                "k": 3,
                "type": "tone",
                "instruction": "Make every string value polite and considerate.",
                "constraints": {"polite": True},
                "cd_proxy": "Constrained decoding can only bias toward polite tokens, not guarantee tone.",
            },
            {
                "k": 4,
                "type": "dynamic",
                "instruction": dynamic_goal,
                "dynamic": {
                    "preserve_scope": "earlier_json_fields",
                    "modify_scope": "later_json_values",
                    "dynamic_goal": dynamic_goal,
                    "anchor_terms": [back],
                    "judge_rubric": [
                        "The output remains valid JSON with exactly the required keys.",
                        "Earlier fields remain broadly compliant instead of being unnecessarily rewritten.",
                        "Later JSON values meaningfully address the new dynamic goal rather than merely appending a keyword.",
                        "The answer still satisfies the active k=2 forbidden-word constraint and k=3 polite/considerate tone constraint.",
                    ],
                },
            },
        ],
    }


def build_cases() -> List[Dict[str, Any]]:
    topics = [
        ("a campus recycling notice", ["trash", "waste"], "campus", "pilot"),
        ("a lab meeting update", ["delay", "problem"], "meeting", "experiment"),
        ("a museum visitor reminder", ["ban", "rule"], "visitor", "interactive"),
        ("a community garden invitation", ["cheap", "free"], "garden", "seasonal"),
        ("a software release note", ["bug", "failure"], "release", "telemetry"),
        ("a teacher feedback note", ["wrong", "bad"], "student", "reflection"),
        ("a product research memo", ["market", "sales"], "prototype", "fieldwork"),
        ("a library event announcement", ["quiet", "book"], "library", "workshop"),
        ("a clinic appointment reminder", ["late", "urgent"], "clinic", "followup"),
        ("a neighborhood safety update", ["danger", "crime"], "neighborhood", "lighting"),
        ("a conference volunteer note", ["busy", "mistake"], "volunteer", "schedule"),
        ("a restaurant allergy notice", ["risk", "avoid"], "guest", "ingredient"),
        ("a gym class announcement", ["sweat", "pain"], "class", "recovery"),
        ("a remote work guideline", ["mandatory", "monitor"], "team", "timezone"),
        ("a parent teacher meeting invite", ["grade", "fail"], "family", "growth"),
        ("a city park cleanup message", ["dirty", "trash"], "park", "habitat"),
        ("a product onboarding tip", ["confusing", "error"], "onboarding", "shortcut"),
        ("a student club recruitment post", ["join", "free"], "club", "mentorship"),
        ("a patient wellness reminder", ["illness", "doctor"], "wellness", "routine"),
        ("a finance team deadline note", ["cost", "penalty"], "finance", "forecast"),
        ("a travel advisory update", ["cancel", "delay"], "traveler", "route"),
        ("a housing maintenance notice", ["broken", "complaint"], "resident", "inspection"),
        ("a research consent summary", ["risk", "harm"], "participant", "privacy"),
        ("a workshop feedback request", ["bad", "wrong"], "workshop", "iteration"),
        ("a customer support reply", ["refund", "angry"], "customer", "option"),
        ("a hiring panel reminder", ["reject", "bias"], "candidate", "rubric"),
        ("a cafeteria menu update", ["cheap", "diet"], "cafeteria", "seasonal"),
        ("a data privacy announcement", ["breach", "leak"], "privacy", "audit"),
        ("a lab equipment booking note", ["damage", "miss"], "equipment", "calibration"),
        ("a community arts invitation", ["ticket", "sell"], "artist", "collaboration"),
    ]
    cases: List[Dict[str, Any]] = []
    for i, (topic, banned, front, back) in enumerate(topics, 1):
        cases.append(taboo_case(i, topic, banned, front, back))
    json_specs = [
        ("a study plan", ["title", "summary", "next_step"], ["hard", "fail"], "plan", "revision"),
        ("a design review", ["title", "strength", "risk"], ["ugly", "broken"], "design", "prototype"),
        ("a field report", ["title", "finding", "follow_up"], ["error", "bad"], "field", "survey"),
        ("a reading list", ["title", "audience", "recommendation"], ["boring", "easy"], "reader", "annotation"),
        ("a lab safety note", ["title", "reminder", "support"], ["danger", "must"], "safety", "practice"),
        ("a workshop plan", ["title", "goal", "activity"], ["mandatory", "simple"], "workshop", "feedback"),
        ("a grant idea", ["title", "rationale", "next_step"], ["money", "guarantee"], "grant", "pilot"),
        ("a model evaluation note", ["title", "metric", "concern"], ["score", "win"], "metric", "robustness"),
        ("a clinic reminder", ["title", "message", "next_step"], ["urgent", "late"], "clinic", "followup"),
        ("a safety update", ["title", "focus", "action"], ["danger", "crime"], "neighborhood", "lighting"),
        ("a volunteer brief", ["title", "role", "support"], ["busy", "mistake"], "volunteer", "schedule"),
        ("an allergy notice", ["title", "guest_note", "staff_action"], ["risk", "avoid"], "guest", "ingredient"),
        ("a fitness class note", ["title", "benefit", "reminder"], ["sweat", "pain"], "class", "recovery"),
        ("a remote work note", ["title", "practice", "consideration"], ["mandatory", "monitor"], "team", "timezone"),
        ("a family meeting note", ["title", "purpose", "encouragement"], ["grade", "fail"], "family", "growth"),
        ("a park cleanup brief", ["title", "task", "impact"], ["dirty", "trash"], "park", "habitat"),
        ("an onboarding hint", ["title", "tip", "example"], ["confusing", "error"], "onboarding", "shortcut"),
        ("a club outreach note", ["title", "audience", "invitation"], ["join", "free"], "club", "mentorship"),
        ("a wellness reminder", ["title", "habit", "support"], ["illness", "doctor"], "wellness", "routine"),
        ("a finance deadline note", ["title", "deadline", "preparation"], ["cost", "penalty"], "finance", "forecast"),
        ("a travel update", ["title", "status", "suggestion"], ["cancel", "delay"], "traveler", "route"),
        ("a maintenance notice", ["title", "scope", "resident_action"], ["broken", "complaint"], "resident", "inspection"),
        ("a consent summary", ["title", "participant_note", "privacy_note"], ["risk", "harm"], "participant", "privacy"),
        ("a feedback request", ["title", "question", "next_step"], ["bad", "wrong"], "workshop", "iteration"),
        ("a support response", ["title", "acknowledgment", "option"], ["refund", "angry"], "customer", "option"),
        ("a hiring reminder", ["title", "criterion", "process_note"], ["reject", "bias"], "candidate", "rubric"),
        ("a menu update", ["title", "highlight", "note"], ["cheap", "diet"], "cafeteria", "seasonal"),
        ("a privacy announcement", ["title", "practice", "review"], ["breach", "leak"], "privacy", "audit"),
        ("an equipment booking note", ["title", "usage", "care"], ["damage", "miss"], "equipment", "calibration"),
        ("an arts invitation", ["title", "audience", "collaboration"], ["ticket", "sell"], "artist", "collaboration"),
    ]
    for i, spec in enumerate(json_specs, 1):
        cases.append(json_case(i, *spec))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/staged_cases.json")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"max_turns": 8, "description": "staged K=8 taboo/json constraint dataset"}, "cases": build_cases()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(payload['cases'])} cases to {path}")


if __name__ == "__main__":
    main()
