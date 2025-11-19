"""Controllers that orchestrate the rethink workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rethink.analysis.trace_analysis import compare_traces
from rethink.utils.config import RethinkConfig
from dataset.benchmark import BenchmarkExample, BenchmarkResult
from rethink.recorder.trace_recorder import TraceRecorder


@dataclass
class ControllerArtifacts:
    """Artifacts surfaced to downstream dashboards/notebooks."""

    benchmark_result: BenchmarkResult
    divergence_report: Optional[object]


class RethinkController:
    """Glue class wiring datasets, models, and analysis modules together."""

    def __init__(self, model, tokenizer, cfg: RethinkConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg

    def _tracepack_to_trace_recorder(self, tracepack, question: str, answer: str) -> TraceRecorder:
        return TraceRecorder(
            question=question,
            answer=answer,
            tokenlist=tracepack.token_logprobs
        )

    def run_single_example(self, example: BenchmarkExample, generation_kwargs: Optional[dict] = None) -> ControllerArtifacts:
        """Execute both teacher-forced and free-form traces for one example."""

        reference_pack = self.model.collect_forced_trace(
            tokenizer=self.tokenizer,
            prompt=example.question,
            target=example.correct_answer,
            max_new_tokens=self.cfg.instrumentation.max_tokens,
        )
        hypothesis_pack = self.model.generate_autoregressive_trace(
            tokenizer=self.tokenizer,
            prompt=example.question,
            generation_kwargs=generation_kwargs,
        )

        reference_trace = self._tracepack_to_trace_recorder(reference_pack, example.question, example.correct_answer)
        hypothesis_trace = self._tracepack_to_trace_recorder(hypothesis_pack, example.question, example.correct_answer) # Answer might be different for hypothesis, but TraceRecorder structure assumes 'answer' field. Maybe it means 'target answer' or 'generated answer'?
        # The user's TraceRecorder has 'answer'. I'll assume it's the generated answer for hypothesis.
        # But hypothesis_pack doesn't explicitly store the full generated text as a string, only tokens.
        # I'll reconstruct it.
        hypothesis_answer = "".join([t.token for t in hypothesis_pack.token_logprobs])
        hypothesis_trace.answer = hypothesis_answer

        benchmark_result = BenchmarkResult(
            example=example,
            reference_trace=reference_trace,
            model_trace=hypothesis_trace,
        )

        divergence_report = compare_traces(
            reference_trace,
            hypothesis_trace,
            hidden_states=hypothesis_pack.hidden_states,
        )

        return ControllerArtifacts(
            benchmark_result=benchmark_result,
            divergence_report=divergence_report,
        )

    def intervene(self, divergence_report, strategy: str = "reset"):
        """Placeholder: use divergence report to call model interventions."""

        raise NotImplementedError("Intervention logic will be defined in a follow-up iteration")
