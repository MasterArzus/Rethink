"""Instrumented Qwen model that exposes trace-friendly APIs."""

from __future__ import annotations

from typing import Dict, List, Optional, Any

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from rethink.engine.base import TracePack
from rethink.utils.config import InstrumentationConfig
from rethink.recorder.hiddenstate_recorder import HiddenStateRecorder, HiddenState
from rethink.recorder.token_recorder import TokenRecorder


def _generate_autoregressive_trace_common(
    model_instance,
    tokenizer,
    prompt: str,
    generation_kwargs: Optional[dict] = None,
    stream_callback: Optional[Any] = None,
    track_hidden: Optional[bool] = None,
) -> TracePack:
    """Shared implementation for generate_autoregressive_trace across Qwen2/Qwen3."""
    generation_kwargs = generation_kwargs or {"max_new_tokens": 128}
    device = model_instance.device

    if isinstance(prompt, str):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    else:
        input_ids = prompt

    input_len = input_ids.shape[1]
    max_new_tokens = generation_kwargs.get("max_new_tokens", 128)

    # Caller can override; otherwise fall back to instrumentation config.
    if track_hidden is None:
        track_hidden = model_instance.instrumentation_cfg.track_hidden_states

    filtered_kwargs = {
        k: v for k, v in generation_kwargs.items()
        if k not in ["max_new_tokens", "frequency_penalty", "presence_penalty"]
    }
    gen_output = model_instance.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        output_hidden_states=track_hidden,
        return_dict_in_generate=True,
        **filtered_kwargs,
    )

    if hasattr(gen_output, "sequences"):
        generated_ids = gen_output.sequences[0][input_len:]
    else:
        generated_ids = gen_output[0][input_len:]

    past_kv = None
    cumulative_ids = input_ids
    token_logs: List[TokenRecorder] = []

    for step, token_id in enumerate(generated_ids):
        input_t = token_id.unsqueeze(0).unsqueeze(0).to(device)

        outputs = model_instance.forward(
            input_ids=input_t,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True,
        )

        current_states = {}
        if track_hidden and hasattr(outputs, "hidden_states") and outputs.hidden_states:
            for layer_idx, hidden_state in enumerate(outputs.hidden_states):
                if hidden_state is not None:
                    current_states[layer_idx] = HiddenState(
                        layer_idx=layer_idx,
                        value=hidden_state[0, -1].cpu()
                    )

        logits = outputs.logits[0, -1, :]
        probs = torch.nn.functional.softmax(logits.float(), dim=-1)
        token_prob = probs[token_id.item() if hasattr(token_id, "item") else token_id].item()

        token_str = tokenizer.decode([token_id], skip_special_tokens=False).replace("�", "")

        token_logs.append(TokenRecorder(
            idx=token_id.item() if hasattr(token_id, "item") else int(token_id),
            step=step,
            token=token_str,
            prob=token_prob,
            log_prob=torch.log(torch.clamp(torch.tensor(token_prob), min=1e-12)).item(),
            hidden_states=current_states,
            input_ids=cumulative_ids.cpu(),
        ))

        cumulative_ids = torch.cat([cumulative_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=-1)
        past_kv = outputs.past_key_values

        if stream_callback:
            stream_callback(token_str)

    return TracePack(
        token_logprobs=token_logs,
        hidden_states=dict(model_instance._recorder.storage) if track_hidden else {},
        extra={"prompt_ids": input_ids.cpu()},
    )


class RethinkQwenForCausalLM(Qwen2ForCausalLM):
    """Thin extension that records per-token metadata during decoding."""

    def __init__(self, config, instrumentation_cfg: Optional[InstrumentationConfig] = None):
        super().__init__(config)
        self.instrumentation_cfg = instrumentation_cfg or InstrumentationConfig()
        self._recorder = HiddenStateRecorder(layers=self.instrumentation_cfg.layers_to_capture)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def generate_autoregressive_trace(
        self,
        tokenizer,
        prompt: str,
        generation_kwargs: Optional[dict] = None,
        stream_callback: Optional[Any] = None,
        track_hidden: Optional[bool] = None,
    ) -> TracePack:
        """Run generation and record trace."""
        return _generate_autoregressive_trace_common(
            self, tokenizer, prompt, generation_kwargs, stream_callback, track_hidden
        )


class RethinkQwen3ForCausalLM(Qwen3ForCausalLM):
    """Qwen3 support - delegates to shared implementation."""

    def __init__(self, config, instrumentation_cfg: Optional[InstrumentationConfig] = None):
        super().__init__(config)
        self.instrumentation_cfg = instrumentation_cfg or InstrumentationConfig()
        self._recorder = HiddenStateRecorder(layers=self.instrumentation_cfg.layers_to_capture)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def generate_autoregressive_trace(
        self,
        tokenizer,
        prompt: str,
        generation_kwargs: Optional[dict] = None,
        stream_callback: Optional[Any] = None,
        track_hidden: Optional[bool] = None,
    ) -> TracePack:
        """Run generation and record trace."""
        return _generate_autoregressive_trace_common(
            self, tokenizer, prompt, generation_kwargs, stream_callback, track_hidden
        )
