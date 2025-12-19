"""Instrumented LLaMA model that exposes trace-friendly APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from rethink.utils.config import InstrumentationConfig
from rethink.recorder.hiddenstate_recorder import HiddenStateRecorder, HiddenState
from rethink.recorder.token_recorder import TokenRecorder



@dataclass
class TracePack:

    """Aggregate token-wise statistics and raw hidden states."""

    token_logprobs: List[TokenRecorder]
    hidden_states: Dict[int, List[torch.Tensor]]
    extra: Dict[str, torch.Tensor]


class RethinkLlamaForCausalLM(LlamaForCausalLM):
    """Thin extension that records per-token metadata during decoding."""

    def __init__(self, config, instrumentation_cfg: Optional[InstrumentationConfig] = None):
        super().__init__(config)
        self.instrumentation_cfg = instrumentation_cfg or InstrumentationConfig()
        self._recorder = HiddenStateRecorder(layers=self.instrumentation_cfg.layers_to_capture)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def collect_forced_trace(
        self,
        tokenizer,
        prompt: str,
        target: str,
        max_new_tokens: Optional[int] = None,
    ) -> TracePack:
        """Teacher-force the ``target`` continuation and store intermediate stats."""

        device = self.device
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        target_ids = tokenizer(target, return_tensors="pt").input_ids.to(device)[0]
        generated_ids = prompt_ids.clone()

        token_logs: List[TokenRecorder] = []

        if not self.instrumentation_cfg.track_hidden_states:
            self._recorder.storage.clear()

        recorder_ctx = self._recorder.attach(self) if self.instrumentation_cfg.track_hidden_states else None
        ctx_manager = recorder_ctx if recorder_ctx is not None else torch.no_grad()

        with ctx_manager:
            for step, token_id in enumerate(target_ids.tolist()):
                outputs = super().forward(
                    input_ids=generated_ids, 
                    use_cache=True, 
                    return_dict=True, 
                    output_attentions=self.instrumentation_cfg.track_attentions
                )
                logits = outputs.logits[:, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                prob = torch.exp(log_probs[0, token_id]).item()
                
                # Extract hidden states for this step
                current_states = {}
                if self.instrumentation_cfg.track_hidden_states and self._recorder.storage:
                    for l, states in self._recorder.storage.items():
                        if states:
                            # states is now List[HiddenState]
                            current_states[l] = states[-1]

                token_logs.append(
                    TokenRecorder(
                        idx=token_id,
                        step=step,
                        token=tokenizer.decode([token_id]),
                        prob=prob,
                        log_prob=log_probs[0, token_id].item(),
                        hidden_states=current_states,
                    )
                )
                next_token = torch.tensor([[token_id]], device=device)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                if max_new_tokens and step + 1 >= max_new_tokens:
                    break

        return TracePack(
            token_logprobs=token_logs,
            hidden_states=dict(self._recorder.storage) if self.instrumentation_cfg.track_hidden_states else {},
            extra={"prompt_ids": prompt_ids.cpu()},
        )

    @torch.no_grad()
    def generate_autoregressive_trace(
        self,
        tokenizer,
        prompt: str,
        generation_kwargs: Optional[dict] = None,
        stream_callback: Optional[Any] = None,
    ) -> TracePack:
        """Run open-ended decoding while storing statistics for each emitted token."""

        generation_kwargs = generation_kwargs or {"max_new_tokens": 128}
        device = self.device
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        token_logs: List[TokenRecorder] = []

        # Setup Logits Processors
        from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, RepetitionPenaltyLogitsProcessor
        
        logits_processor = LogitsProcessorList()
        if generation_kwargs.get("repetition_penalty", 1.0) != 1.0:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=generation_kwargs["repetition_penalty"]))
        
        logits_warper = LogitsProcessorList()
        if generation_kwargs.get("temperature", 1.0) != 1.0:
            logits_warper.append(TemperatureLogitsWarper(temperature=generation_kwargs["temperature"]))
        if generation_kwargs.get("top_k", 0) > 0:
            logits_warper.append(TopKLogitsWarper(top_k=generation_kwargs["top_k"]))
        if generation_kwargs.get("top_p", 1.0) < 1.0:
            logits_warper.append(TopPLogitsWarper(top_p=generation_kwargs["top_p"]))

        recorder_ctx = self._recorder.attach(self) if self.instrumentation_cfg.track_hidden_states else None
        ctx_manager = recorder_ctx if recorder_ctx is not None else torch.no_grad()

        with ctx_manager:
            past_key_values = None
            generated_ids = input_ids
            for step in range(generation_kwargs.get("max_new_tokens", 128)):
                outputs = super().forward(
                    input_ids=generated_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                    return_dict=True,
                    output_attentions=self.instrumentation_cfg.track_attentions,
                )
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                # Apply processors (repetition penalty, etc)
                next_token_logits = logits_processor(generated_ids, next_token_logits)
                
                # Apply warpers (sampling)
                next_token_scores = logits_warper(generated_ids, next_token_logits)
                
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                
                if generation_kwargs.get("do_sample", True):
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
                
                token_id = next_token.item()
                
                # Extract hidden states for this step
                current_states = {}
                if self.instrumentation_cfg.track_hidden_states and self._recorder.storage:
                    for l, states in self._recorder.storage.items():
                        if states:
                            # states is now List[HiddenState]
                            current_states[l] = states[-1]

                token_str = tokenizer.decode([token_id])
                if stream_callback:
                    stream_callback(token_str)

                token_logs.append(
                    TokenRecorder(
                        idx=token_id,
                        step=step,
                        token=token_str,
                        prob=probs[0, token_id].item(),
                        log_prob=torch.log(probs[0, token_id]).item(),
                        hidden_states=current_states,
                    )
                )
                generated_ids = torch.cat([generated_ids, next_token.to(device)], dim=-1)
                
                # Check for stop conditions
                eos_token_id = generation_kwargs.get("eos_token_id")
                if eos_token_id is not None:
                    if isinstance(eos_token_id, int) and token_id == eos_token_id:
                        break
                    elif isinstance(eos_token_id, (list, tuple)) and token_id in eos_token_id:
                        break

        return TracePack(
            token_logprobs=token_logs,
            hidden_states=dict(self._recorder.storage) if self.instrumentation_cfg.track_hidden_states else {},
            extra={"prompt_ids": input_ids.cpu()},
        )

    # Placeholder for rethink-specific intervention API
    def intervene_from_span(self, start_token: int, strategy: str = "reset"):
        """Adjust internal cache from a problematic token span.

        This is intentionally left as a stub: we need a concrete definition of
        "rethink" actions (e.g., bias logits, rewrite KV cache, replace hidden
        states). The method signature is provided so downstream code can attach
        new strategies without modifying the core model class.
        """

        raise NotImplementedError("Rethink intervention strategies are not defined yet")

