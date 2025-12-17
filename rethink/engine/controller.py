"""Controllers that orchestrate the rethink workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from rethink.analysis.trace_analysis import TraceAnalysis
from rethink.utils.config import RethinkConfig
from dataset.data_class import DataExample, DataResult
from rethink.recorder.trace_recorder import TraceRecorder
from rethink.recorder.hiddenstate_recorder import HiddenStateRecorder, HiddenState
from rethink.recorder.token_recorder import TokenRecorder
from rethink.recorder.trajectory import Trajectory


@dataclass
class ControllerArtifacts:
    """Artifacts surfaced to downstream dashboards/notebooks."""

    benchmark_result: DataResult
    divergence_report: Optional[object]


class RethinkController:
    """Glue class wiring datasets, models, and analysis modules together."""

    def __init__(self, model, tokenizer, cfg: RethinkConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load system prompt from config or fallback to file/default."""
        # Priority 1: Config
        if self.cfg.prompt and self.cfg.prompt.system_prompt:
            return self.cfg.prompt.system_prompt

        # Priority 2: File (Legacy)
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

    @torch.no_grad()
    def _project_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Project a single hidden state through the final normalization + LM head to obtain logits.
        Assumes hidden_state has shape (1, hidden_dim).
        """
        state = hidden_state.to(self.model.device)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        normalized = self.model.model.norm(state)
        return self.model.lm_head(normalized)

    @torch.no_grad()
    def _decode_hidden_state(self, hidden_state: torch.Tensor, top_k: int = 10) -> List[Tuple[str, float]]:
        logits = self._project_hidden_state(hidden_state)
        probs = torch.nn.functional.softmax(logits[0, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k)
        return [
            (self.tokenizer.decode([idx.item()]), prob.item())
            for prob, idx in zip(top_probs, top_indices)
        ]

    @torch.no_grad()
    def compute_hidden_states_for_step(
        self,
        trace: TraceRecorder,
        step_idx: int,
        layers: Optional[List[int]] = None,
    ) -> TokenRecorder:
        """
        Re-compute hidden states for a specific generation step by teacher-forcing
        the generated tokens up to that step. Captures all requested layers.
        """

        if step_idx < 0 or step_idx >= len(trace.tokenlist):
            raise IndexError(f"step_idx {step_idx} out of range for trace of length {len(trace.tokenlist)}")

        device = self.model.device
        prompt_ids = self.tokenizer(trace.question, return_tensors="pt").input_ids.to(device)

        target_tokens = trace.tokenlist[: step_idx + 1]
        target_ids = torch.tensor([[t.idx for t in target_tokens]], device=device)

        recorder = HiddenStateRecorder(layers=layers)
        generated_ids = prompt_ids
        past_key_values = None
        token_logs: List[TokenRecorder] = []

        with recorder.attach(self.model):
            for step, token_id in enumerate(target_ids[0].tolist()):
                outputs = self.model.forward(
                    input_ids=generated_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                    return_dict=True,
                    output_attentions=True,
                )
                logits = outputs.logits[:, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                prob = torch.exp(log_probs[0, token_id]).item()

                past_key_values = outputs.past_key_values

                current_states: Dict[int, HiddenState] = {}
                if recorder.storage:
                    for layer_idx, states in recorder.storage.items():
                        if states:
                            current_states[layer_idx] = states[-1]

                token_logs.append(
                    TokenRecorder(
                        idx=token_id,
                        step=step,
                        token=self.tokenizer.decode([token_id]),
                        prob=prob,
                        log_prob=log_probs[0, token_id].item(),
                        hidden_states=current_states,
                        input_ids=generated_ids.cpu(),
                    )
                )

                next_token = torch.tensor([[token_id]], device=device)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        return token_logs[-1]

    @torch.no_grad()
    def probe_state(
        self,
        trace: TraceRecorder,
        step_idx: int,
        layer_idx: Optional[int] = None,
        max_new_tokens: int = 256,
        cached_state: Optional[HiddenState] = None,
        cached_trajectory: Optional[Trajectory] = None,
    ) -> Dict[str, object]:
        """
        Run a lightweight explanatory probe over a token's hidden state.
        Returns natural language explanation and the logit-lens distribution for the chosen layer.
        """

        if cached_state is not None:
            state_obj = cached_state
            trajectory = cached_trajectory
            layer_ids = sorted(trajectory.to_dict().keys()) if trajectory else [layer_idx or 0]
            target_layer = layer_idx if layer_idx is not None else layer_ids[-1]
        else:
            token_state = self.compute_hidden_states_for_step(trace, step_idx, layers=None)
            trajectory: Trajectory = token_state.trajectory
            layer_ids = sorted(trajectory.to_dict().keys())
            target_layer = layer_idx if layer_idx is not None else layer_ids[-1]
            state_obj = trajectory.get_by_layer(target_layer)
            if state_obj is None:
                raise RuntimeError(f"No hidden state captured for layer {target_layer}")

        logits_top = self._decode_hidden_state(state_obj.get_value(), top_k=10)

        # Build probe prompt
        context_text = trace.question + "".join([t.token for t in trace.tokenlist[: step_idx + 1]])
        top_tokens_str = ", ".join([f"{tok} ({prob:.3f})" for tok, prob in logits_top[:5]])

        system_msg = "You are an expert in interpreting language model internal states. Analyze the provided context and the current layer's token predictions to explain the model's reasoning."
        user_msg = (
            f"Context: {context_text}\n\n"
            f"At the current step, the model's internal state at Layer {target_layer} is most strongly predicting these tokens: {top_tokens_str}.\n"
            "Explain why the model is focusing on these tokens given the context. Keep the explanation concise."
        )

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            probe_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            probe_prompt = (
                f"{system_msg}\n\n"
                f"{user_msg}\n\n"
                "Explanation:"
            )

        inputs = self.tokenizer(probe_prompt, return_tensors="pt").to(self.model.device)
        probe_output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_text = self.tokenizer.decode(probe_output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return {
            "layer": target_layer,
            "logit_lens": logits_top,
            "explanation": generated_text.strip(),
        }

    def _format_prompt(self, question: str) -> str:
        """Apply chat template to the question."""
        # Check if tokenizer has a chat template
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": self.cfg.prompt.system_role, "content": self.system_prompt},
                {"role": self.cfg.prompt.user_role, "content": question}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback for models without chat template
            return f"{self.system_prompt}\n\nQuestion: {question}\nAnswer:"

    def run_single_example(self, example: DataExample, generation_kwargs: Optional[dict] = None) -> ControllerArtifacts:
        """Execute both teacher-forced and free-form traces for one example."""

        # Apply prompt engineering
        formatted_prompt = self._format_prompt(example.question)

        # Merge config with runtime kwargs
        gen_config = self.cfg.generation.to_dict()
        if generation_kwargs:
            gen_config.update(generation_kwargs)

        reference_pack = self.model.collect_forced_trace(
            tokenizer=self.tokenizer,
            prompt=formatted_prompt,
            target=example.correct_answer,
            max_new_tokens=self.cfg.instrumentation.max_tokens,
        )
        hypothesis_pack = self.model.generate_autoregressive_trace(
            tokenizer=self.tokenizer,
            prompt=formatted_prompt,
            generation_kwargs=gen_config,
        )

        reference_trace = self._tracepack_to_trace_recorder(reference_pack, example.question, example.correct_answer)
        hypothesis_trace = self._tracepack_to_trace_recorder(hypothesis_pack, example.question, example.correct_answer)
        
        hypothesis_answer = "".join([t.token for t in hypothesis_pack.token_logprobs])
        hypothesis_trace.answer = hypothesis_answer

        benchmark_result = DataResult(
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
