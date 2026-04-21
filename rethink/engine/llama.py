"""Instrumented LLaMA model that exposes trace-friendly APIs."""

from __future__ import annotations

from typing import Dict, List, Optional, Any

import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from rethink.engine.base import TracePack
from rethink.utils.config import InstrumentationConfig
from rethink.recorder.hiddenstate_recorder import HiddenStateRecorder, HiddenState
from rethink.recorder.token_recorder import TokenRecorder


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
        """
        Run generation and record trace.

        Strategy:
        - Use model.generate() for correct, reliable generation
        - Re-run step-by-step only if hidden state tracking is needed
        - Otherwise, just decode tokens efficiently
        """
        generation_kwargs = generation_kwargs or {"max_new_tokens": 128}
        device = self.device

        # Prepare input
        if isinstance(prompt, str):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        else:
            input_ids = prompt

        input_len = input_ids.shape[1]
        max_new_tokens = generation_kwargs.get("max_new_tokens", 128)

        # Decide whether to track hidden states
        track_hidden = self.instrumentation_cfg.track_hidden_states

        # Use model.generate() for generation - this ensures correct behavior
        # Pass output_hidden_states only if we need it
        # Filter out model-specific kwargs that generate() doesn't support
        filtered_kwargs = {
            k: v for k, v in generation_kwargs.items()
            if k not in ["max_new_tokens", "frequency_penalty", "presence_penalty"]
        }
        gen_output = self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            output_hidden_states=track_hidden,
            return_dict_in_generate=True,
            **filtered_kwargs,
        )

        # Extract generated token IDs
        if hasattr(gen_output, "sequences"):
            generated_ids = gen_output.sequences[0][input_len:]
        else:
            generated_ids = gen_output[0][input_len:]

        token_logs: List[TokenRecorder] = []

        if track_hidden and hasattr(gen_output, "hidden_states"):
            # Re-run step-by-step to record hidden states properly
            past_kv = None
            cumulative_ids = input_ids

            for step, token_id in enumerate(generated_ids):
                input_t = token_id.unsqueeze(0).unsqueeze(0).to(device)

                outputs = self.forward(
                    input_ids=input_t,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                )

                # Get hidden states for this step
                if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    current_states = {}
                    for layer_idx, hidden_state in enumerate(outputs.hidden_states):
                        if hidden_state is not None:
                            current_states[layer_idx] = HiddenState(
                                layer_idx=layer_idx,
                                value=hidden_state[0, -1].cpu()
                            )
                else:
                    current_states = {}

                # Compute log prob for this token
                logits = outputs.logits[0, -1, :]
                probs = torch.nn.functional.softmax(logits.float(), dim=-1)
                token_prob = probs[token_id].item()

                token_str = tokenizer.decode([token_id], skip_special_tokens=False).replace("\uFFFD", "")

                token_logs.append(TokenRecorder(
                    idx=token_id.item() if hasattr(token_id, "item") else int(token_id),
                    step=step,
                    token=token_str,
                    prob=token_prob,
                    log_prob=torch.log(torch.tensor(token_prob)).item(),
                    hidden_states=current_states if track_hidden else {},
                    input_ids=cumulative_ids.cpu(),
                ))

                cumulative_ids = torch.cat([cumulative_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=-1)
                past_kv = outputs.past_key_values

                if stream_callback:
                    stream_callback(token_str)
        else:
            # Decode tokens and compute probs via forward pass
            past_kv = None
            cumulative_ids = input_ids

            for step, token_id in enumerate(generated_ids):
                input_t = token_id.unsqueeze(0).unsqueeze(0).to(device)

                outputs = self.forward(
                    input_ids=input_t,
                    past_key_values=past_kv,
                    use_cache=True,
                )

                # Compute probability for this token from logits
                logits = outputs.logits[0, -1, :]
                probs = torch.nn.functional.softmax(logits.float(), dim=-1)
                token_prob = probs[token_id.item() if hasattr(token_id, "item") else token_id].item()
                token_log_prob = torch.log(torch.clamp(torch.tensor(token_prob), min=1e-12)).item()

                token_str = tokenizer.decode([token_id], skip_special_tokens=False).replace("\uFFFD", "")

                token_logs.append(TokenRecorder(
                    idx=token_id.item() if hasattr(token_id, "item") else int(token_id),
                    step=step,
                    token=token_str,
                    prob=token_prob,
                    log_prob=token_log_prob,
                    hidden_states={},
                    input_ids=cumulative_ids.cpu(),
                ))

                cumulative_ids = torch.cat([cumulative_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=-1)
                past_kv = outputs.past_key_values

                if stream_callback:
                    stream_callback(token_str)

        return TracePack(
            token_logprobs=token_logs,
            hidden_states=dict(self._recorder.storage) if track_hidden else {},
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

