"""Controllers that orchestrate the rethink workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rethink.analysis.trace_analysis import TraceAnalysis
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
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load system prompt from rethink/prompt/{dataset_name}.md"""
        from pathlib import Path
        
        # rethink/engine/controller.py -> rethink/prompt/
        current_dir = Path(__file__).resolve().parent
        rethink_dir = current_dir.parent
        prompt_path = rethink_dir / "prompt" / f"{self.cfg.dataset.name}.md"
        
        default_prompt = "You are a helpful assistant. Solve the following math problem step by step."
        
        if prompt_path.exists():
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return content
            except Exception as e:
                print(f"Warning: Failed to load prompt file {prompt_path}: {e}")
        
        return default_prompt

    def _tracepack_to_trace_recorder(self, tracepack, question: str, answer: str) -> TraceRecorder:
        return TraceRecorder(
            question=question,
            answer=answer,
            tokenlist=tracepack.token_logprobs,
            metadata={"extra": tracepack.extra}
        )

    def _format_prompt(self, question: str) -> str:
        """Apply chat template to the question."""
        # Check if tokenizer has a chat template
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback for models without chat template
            return f"{self.system_prompt}\n\nQuestion: {question}\nAnswer:"

    def run_single_example(self, example: BenchmarkExample, generation_kwargs: Optional[dict] = None) -> ControllerArtifacts:
        """Execute both teacher-forced and free-form traces for one example."""

        # Apply prompt engineering
        formatted_prompt = self._format_prompt(example.question)

        reference_pack = self.model.collect_forced_trace(
            tokenizer=self.tokenizer,
            prompt=formatted_prompt,
            target=example.correct_answer,
            max_new_tokens=self.cfg.instrumentation.max_tokens,
        )
        hypothesis_pack = self.model.generate_autoregressive_trace(
            tokenizer=self.tokenizer,
            prompt=formatted_prompt,
            generation_kwargs=generation_kwargs,
        )

        reference_trace = self._tracepack_to_trace_recorder(reference_pack, example.question, example.correct_answer)
        hypothesis_trace = self._tracepack_to_trace_recorder(hypothesis_pack, example.question, example.correct_answer)
        
        hypothesis_answer = "".join([t.token for t in hypothesis_pack.token_logprobs])
        hypothesis_trace.answer = hypothesis_answer

        benchmark_result = BenchmarkResult(
            example=example,
            reference_trace=reference_trace,
            model_trace=hypothesis_trace,
        )

        # Use the new TraceAnalysis class
        analyzer = TraceAnalysis(
            trace=hypothesis_trace,
            model=self.model,
            tokenizer=self.tokenizer,
            reference_trace=reference_trace
        )
        
        # Generate the report
        divergence_report = analyzer.generate_report()

        return ControllerArtifacts(
            benchmark_result=benchmark_result,
            divergence_report=divergence_report,
        )

    def intervene(self, divergence_report, strategy: str = "reset"):
        """Placeholder: use divergence report to call model interventions."""

        raise NotImplementedError("Intervention logic will be defined in a follow-up iteration")
