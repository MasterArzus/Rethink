"""Controllers that orchestrate the rethink workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..analysis import compare_traces
from ..config import RethinkConfig
from ..data import BenchmarkExample, BenchmarkResult, TokenTrace


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

    def _tracepack_to_token_trace(self, tracepack) -> TokenTrace:
        tokens = [log.token for log in tracepack.token_logprobs]
        log_probs = [log.log_prob for log in tracepack.token_logprobs]
        return TokenTrace(tokens=tokens, log_probs=log_probs, hidden_states_path=None)

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

        reference_trace = self._tracepack_to_token_trace(reference_pack)
        hypothesis_trace = self._tracepack_to_token_trace(hypothesis_pack)

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
