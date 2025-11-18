"""Shared dataclasses for the Rethink benchmark format."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BenchmarkExample:
    """Represent a single reasoning question and candidate answers."""

    question: str
    correct_answer: str
    incorrect_answers: List[str]
    metadata: Optional[dict] = None


@dataclass
class TokenTrace:
    """Lightweight container for per-token statistics."""

    tokens: List[str]
    log_probs: List[float]
    hidden_states_path: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Pair traces for later comparison and visualization."""

    example: BenchmarkExample
    reference_trace: Optional[TokenTrace]
    model_trace: Optional[TokenTrace]
