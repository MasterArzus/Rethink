"""Minimal GSM8K sampling utilities."""

from typing import Iterable, List

from .data_class import DataExample


def load_gsm8k_slice(raw_samples: Iterable[dict], limit: int = 10) -> List[DataExample]:
    """Convert the first ``limit`` GSM8K rows into ``BenchmarkExample`` records.

    Parameters
    ----------
    raw_samples:
        Iterable rows containing ``question`` and ``answer`` keys.
    limit:
        Maximum number of examples to include for a quick sanity benchmark.
    """

    examples: List[DataExample] = []
    for idx, row in enumerate(raw_samples):
        if idx >= limit:
            break
        question = row.get("question", "")
        answer = row.get("answer", "")
        incorrect = row.get("distractors", [])
        if not incorrect:
            # Placeholder: create a trivial incorrect variant for debugging
            incorrect = [f"{answer} (corrupted)"]
        examples.append(
            DataExample(
                question=question,
                correct_answer=answer,
                incorrect_answers=list(incorrect),
                metadata={"source": "gsm8k", "row_id": idx},
            )
        )
    return examples
