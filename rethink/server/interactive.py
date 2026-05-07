import torch
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from rethink.engine.controller import RethinkController
from rethink.recorder.trace_recorder import TraceRecorder
from rethink.recorder.token_recorder import TokenRecorder
from rethink.analysis.trace_analysis import TraceAnalysis
from rethink.utils.config import RethinkConfig, DatasetSlice, InstrumentationConfig
from rethink.server.experiment_logger import ExperimentLogger

class InteractiveSession:
    def __init__(self, model, tokenizer, cfg: RethinkConfig = None, experiment_logger: Optional[ExperimentLogger] = None):
        if cfg is None:
            cfg = RethinkConfig(
                dataset=DatasetSlice(name="gsm8k", split="test"),
                instrumentation=InstrumentationConfig(
                    track_hidden_states=True,
                    layers_to_capture=[-1],
                    max_tokens=512
                ),
                output_dir="outputs/interactive_sessions"
            )

        self.cfg = cfg
        self.controller = RethinkController(model, tokenizer, cfg)
        self.current_trace = None
        self.analysis_results = None
        self.experiment_logger = experiment_logger

        os.makedirs(self.cfg.output_dir, exist_ok=True)

    def set_experiment_logger(self, experiment_logger: ExperimentLogger):
        self.experiment_logger = experiment_logger

    def start_task(self, **kwargs):
        if self.experiment_logger:
            return self.experiment_logger.start_task(**kwargs)
        return None

    def log_event(self, event_type: str, **kwargs):
        if self.experiment_logger:
            return self.experiment_logger.log_event(event_type, **kwargs)
        return None

    def record_generation(self, generated_tokens: int, **kwargs):
        if self.experiment_logger:
            self.experiment_logger.record_generation(generated_tokens, **kwargs)

    def record_checker_result(self, passed: bool, message: Optional[str], **kwargs):
        if self.experiment_logger:
            self.experiment_logger.record_checker_result(passed, message, **kwargs)

    def finish_task(self, **kwargs):
        if self.experiment_logger:
            return self.experiment_logger.finish_task(**kwargs)
        return None

    def start_generation(self) -> None:
        if self.experiment_logger:
            self.experiment_logger.start_generation()

    def save_session(self, filename=None):
        if not self.current_trace:
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"

        filepath = os.path.join(self.cfg.output_dir, filename)

        data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": self.current_trace.question if hasattr(self.current_trace, 'question') else "",
            "generated_text": self.current_trace.get_full_text(),
            "tokens": [
                {
                    "token": t.token,
                    "prob": t.prob,
                    "step": t.step
                } for t in self.current_trace.tokenlist
            ],
            "analysis": [
                {
                    "start": interval.start,
                    "end": interval.end,
                    "type": interval.type,
                    "description": interval.description
                } for interval in (self.analysis_results or [])
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return filepath

    def _get_terminators(self):
        terminators = [self.controller.tokenizer.eos_token_id]
        vocab = self.controller.tokenizer.get_vocab()
        if "<|eot_id|>" in vocab:
            terminators.append(self.controller.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        if "<|im_end|>" in vocab:
            terminators.append(self.controller.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        return terminators

    def run_initial_inference(self, prompt_text, use_template=True, max_new_tokens=128, stream_callback=None, track_hidden=True, **kwargs):
        """
        Run the initial inference to get the baseline trace.

        All generation parameters must be provided explicitly via kwargs;
        no hardcoded defaults are applied here.
        """
        if use_template:
            final_prompt = self.controller._format_prompt(prompt_text)
        else:
            final_prompt = prompt_text

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self._get_terminators(),
        }
        gen_kwargs.update(kwargs)

        trace_pack = self.controller.model.generate_autoregressive_trace(
            tokenizer=self.controller.tokenizer,
            prompt=final_prompt,
            generation_kwargs=gen_kwargs,
            stream_callback=stream_callback,
            track_hidden=track_hidden,
        )

        generated_text = "".join([t.token for t in trace_pack.token_logprobs])
        self.current_trace = self.controller._tracepack_to_trace_recorder(trace_pack, final_prompt, generated_text)

        self._analyze_trace(deep=track_hidden)
        return self.current_trace, self.analysis_results

    def rethink_from_step(self, trace_recorder, step_idx, max_new_tokens=128, force_token=None, steering_prompt: Optional[str] = None):
        """
        Truncate the trace at step_idx and regenerate from there.
        If force_token is provided, replace the token at step_idx with it.
        """
        base_prompt = trace_recorder.question

        prefix_tokens = trace_recorder.tokenlist[:step_idx]
        prefix_text = "".join([t.token for t in prefix_tokens])

        if force_token is not None:
            current_token_text = force_token
            ids = self.controller.tokenizer.encode(force_token, add_special_tokens=False)
            token_id = ids[0] if ids else -1

            forced_recorder = TokenRecorder(
                idx=token_id,
                step=len(prefix_tokens),
                token=force_token,
                prob=1.0,
                log_prob=0.0,
                hidden_states={}
            )
            middle_token_list = [forced_recorder]
        else:
            current_token_text = trace_recorder.tokenlist[step_idx].token
            middle_token_list = [trace_recorder.tokenlist[step_idx]]

        guide_text = steering_prompt or ""
        new_full_prompt = base_prompt + prefix_text + current_token_text + guide_text

        new_trace, analysis = self.run_initial_inference(new_full_prompt, use_template=False, max_new_tokens=max_new_tokens)

        full_token_list = prefix_tokens + middle_token_list + new_trace.tokenlist

        for i, t in enumerate(full_token_list):
            t.step = i

        combined_trace = TraceRecorder(
            question=trace_recorder.question,
            answer=prefix_text + current_token_text + new_trace.get_full_text(),
            tokenlist=full_token_list,
            metadata=new_trace.metadata
        )

        analyzer = TraceAnalysis(combined_trace, self.controller.model, self.controller.tokenizer)
        combined_analysis = analyzer.locate_critical_intervals()

        return combined_trace, combined_analysis

    @dataclass
    class BranchResult:
        trace: TraceRecorder
        analysis: List[object]
        label: str
        meta: dict

    def branch_from_step(
        self,
        trace_recorder: TraceRecorder,
        step_idx: int,
        k: int = 3,
        strategy: str = "sample",
        max_new_tokens: int = 128,
        steering_prompt: Optional[str] = None,
    ) -> List["InteractiveSession.BranchResult"]:
        """
        Generate K alternative continuations from a given step, optionally with a steering prompt.
        """
        k = max(1, min(k, 8))

        base_prompt = trace_recorder.question
        prefix_tokens = trace_recorder.tokenlist[: step_idx + 1]
        prefix_text = "".join([t.token for t in prefix_tokens])
        guide_text = steering_prompt or ""
        prompt = base_prompt + prefix_text + guide_text
        terminators = self._get_terminators()

        def _gen_kwargs(seed_variation: int):
            if strategy == "beam":
                return {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "seed": None,
                }
            # default: sample/high-temp
            return {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 1.1,
                "top_p": 0.9,
                "seed": None,
            }

        results: List[InteractiveSession.BranchResult] = []
        for i in range(k):
            gen_kwargs = _gen_kwargs(i)
            gen_kwargs["eos_token_id"] = terminators
            trace_pack = self.controller.model.generate_autoregressive_trace(
                tokenizer=self.controller.tokenizer,
                prompt=prompt,
                generation_kwargs=gen_kwargs,
                stream_callback=None,
                track_hidden=True,
            )

            generated_text = "".join([t.token for t in trace_pack.token_logprobs])
            new_tokens = trace_pack.token_logprobs

            merged_tokens = prefix_tokens + new_tokens
            for idx, tok in enumerate(merged_tokens):
                tok.step = idx

            combined_trace = TraceRecorder(
                question=trace_recorder.question,
                answer=prefix_text + generated_text,
                tokenlist=merged_tokens,
                metadata=trace_pack.extra,
            )

            analyzer = TraceAnalysis(combined_trace, self.controller.model, self.controller.tokenizer)
            combined_analysis = analyzer.locate_critical_intervals()

            results.append(
                InteractiveSession.BranchResult(
                    trace=combined_trace,
                    analysis=combined_analysis,
                    label=f"Branch {i+1}",
                    meta={"strategy": strategy},
                )
            )

        return results

    def run_intervention(self, modified_tokens, start_index):
        """
        Run inference starting from a specific point with modified tokens.
        """
        if not self.current_trace:
            return None

        prefix_tokens = [t.token for t in self.current_trace.tokenlist[:start_index]]
        prefix_text = "".join(prefix_tokens)

        if isinstance(modified_tokens, list):
            intervention_text = "".join(modified_tokens)
        else:
            intervention_text = modified_tokens

        new_prompt = prefix_text + intervention_text
        terminators = self._get_terminators()

        new_trace_pack = self.controller.model.generate_autoregressive_trace(
            tokenizer=self.controller.tokenizer,
            prompt=new_prompt,
            generation_kwargs={
                "max_new_tokens": 512,
                "eos_token_id": terminators
            },
            track_hidden=True,
        )

        generated_text = "".join([t.token for t in new_trace_pack.token_logprobs])
        self.current_trace = self.controller._tracepack_to_trace_recorder(new_trace_pack, new_prompt, generated_text)

        self._analyze_trace()
        return self.current_trace, self.analysis_results

    def _analyze_trace(self, deep=True):
        if self.current_trace:
            if not deep:
                self.analysis_results = []
                return
            analyzer = TraceAnalysis(
                trace=self.current_trace,
                model=self.controller.model,
                tokenizer=self.controller.tokenizer,
                reference_trace=None
            )
            self.analysis_results = analyzer.locate_critical_intervals()
