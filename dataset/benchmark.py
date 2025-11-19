"""Shared dataclasses for the Rethink benchmark format."""

from dataclasses import dataclass
from typing import List, Optional
from rethink.recorder.trace_recorder import TraceRecorder


@dataclass
class BenchmarkExample:
    """Represent a single reasoning question and candidate answers."""

    question: str
    correct_answer: str
    incorrect_answers: List[str]
    metadata: Optional[dict] = None


@dataclass
class BenchmarkResult:
    """Pair traces for later comparison and visualization."""

    example: BenchmarkExample
    reference_trace: Optional[TraceRecorder]
    model_trace: Optional[TraceRecorder]
