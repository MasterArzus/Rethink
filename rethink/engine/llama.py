"""Instrumented LLaMA model that exposes trace-friendly APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from rethink.utils.config import InstrumentationConfig
from rethink.recorder.hiddenstate_recorder import HiddenStateRecorder, HiddenState
from rethink.recorder.token_recorder import TokenRecorder


from rethink.engine.debug_strategy import DebugStrategy
from rethink.recorder.trajectory import Trajectory

class LlamaDebugStrategy(DebugStrategy):
    """
    Implementation of DebugStrategy for Llama models.
    """
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.past_key_values = None
        self.current_hidden_states = None
        self.current_attention_mask = None
        self.current_position_ids = None
        self.current_position_embeddings = None
        self.current_logits = None

    def start_generation(self, prompt: str):
        self.full_text = prompt
        self.current_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        
        # Initialize PKV container
        num_layers = self._get_total_layers()
        self.past_key_values = [None] * num_layers
        
        # Initial Forward Start
        self.current_hidden_states, self.current_attention_mask, self.current_position_ids, self.current_position_embeddings = \
            self.model.debug_forward_start(self.current_input_ids, None)
        
        self.current_trajectory = Trajectory()
        self.status = "running"

    def step_layer(self) -> Dict[str, Any]:
        if self.status != "running":
            return self.get_state()

        layer_idx = self.current_trajectory.current_layer_count()
        num_layers = self._get_total_layers()

        if layer_idx < num_layers:
            # Run one layer
            self.current_hidden_states, new_pkv, attn_weights = self.model.debug_forward_layer(
                layer_idx,
                self.current_hidden_states,
                self.current_attention_mask,
                self.current_position_ids,
                self.past_key_values,
                position_embeddings=self.current_position_embeddings
            )
            
            # Update PKV for this layer
            self.past_key_values[layer_idx] = new_pkv
            
            # Record state
            # Note: We record the state AFTER the layer execution
            attn_data = {}
            if attn_weights is not None:
                attn_data["attn_weights"] = attn_weights.clone().detach().cpu()
                
            state = HiddenState(
                layer_idx=layer_idx, 
                value=self.current_hidden_states.clone().detach().cpu(),
                attention_data=attn_data
            )
            self.current_trajectory.add(state)
        
        if self.current_trajectory.current_layer_count() >= num_layers:
            # Finished all layers, run head
            self.current_logits = self.model.debug_forward_end(self.current_hidden_states)
            pass

        return self.get_state()

    def finish_token(self) -> Dict[str, Any]:
        while self.status == "running" and self.current_trajectory.current_layer_count() < self._get_total_layers():
            self.step_layer()
        return self.get_state()

    def sample_next_token(self) -> Dict[str, Any]:
        if self.current_logits is None:
            self.finish_token()
            if self.current_logits is None:
                return self.get_state()
            
        # Simple greedy for debug
        next_token_logits = self.current_logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()
        
        token_str = self.tokenizer.decode([next_token_id])
        
        # Calculate probs for recording
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        prob = probs[0, next_token_id].item()
        log_prob = torch.log(probs[0, next_token_id]).item()

        # Archive current step
        step_idx = len(self.history)
        recorder = TokenRecorder(
            idx=next_token_id,
            step=step_idx,
            token=token_str,
            prob=prob,
            log_prob=log_prob,
            hidden_states=self.current_trajectory,
            input_ids=self.current_input_ids.cpu()
        )
        self.history.append(recorder)
        self.full_text += token_str
        
        # Prepare for next token
        next_input_ids = torch.tensor([[next_token_id]], device=self.model.device)
        self.current_input_ids = next_input_ids
        
        # Reset for next pass
        self.current_hidden_states, self.current_attention_mask, self.current_position_ids, self.current_position_embeddings = \
            self.model.debug_forward_start(self.current_input_ids, tuple(self.past_key_values))
            
        self.current_trajectory = Trajectory()
        self.current_logits = None
        
        # Check EOS
        if next_token_id == self.tokenizer.eos_token_id:
            self.status = "finished"
            
        return self.get_state()

    def _update_internal_hidden_state(self, new_value: torch.Tensor):
        # Ensure device match
        self.current_hidden_states = new_value.to(self.model.device)

    def _get_total_layers(self) -> int:
        return len(self.model.model.layers)

    def _preview_current_token(self) -> str:
        if self.current_hidden_states is None:
            return ""
        # Try to project current state
        try:
            logits = self.model.debug_forward_end(self.current_hidden_states)
            tid = torch.argmax(logits[0, -1]).item()
            return self.tokenizer.decode([tid])
        except:
            return ""


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
                outputs = super().forward(input_ids=generated_ids, use_cache=True, return_dict=True, output_attentions=True)
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
                    output_attentions=True,
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

    @torch.no_grad()
    def debug_forward_start(self, input_ids, past_key_values=None):
        """
        Initialize the forward pass for debugging.
        Returns initial hidden_states and prepared masks/position_ids.
        """
        inputs_embeds = self.model.embed_tokens(input_ids)
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        past_length = 0
        if past_key_values is not None:
            # past_key_values[0] is (key, value) or None
            if len(past_key_values) > 0 and past_key_values[0] is not None:
                # past_key_values[0][0] shape: (batch, num_heads, seq_len, head_dim)
                past_length = past_key_values[0][0].shape[2]
            
        position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)
        
        # Prepare attention mask
        # For simplicity, we assume batch_size=1 and no padding
        attention_mask = torch.ones((batch_size, past_length + seq_length), device=device)
        
        # Use the model's internal helper to create the 4D mask
        extended_attention_mask = None
        if hasattr(self.model, "_prepare_decoder_attention_mask"):
            # Signature: (attention_mask, input_shape, inputs_embeds, past_key_values_length)
            extended_attention_mask = self.model._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_length
            )
            
        # Compute position embeddings (RoPE)
        position_embeddings = None
        
        # Try to find rotary_emb module
        rotary_emb = getattr(self.model, "rotary_emb", None)
        if rotary_emb is None and len(self.model.layers) > 0:
            # Fallback: try to find it in the first layer's attention
            if hasattr(self.model.layers[0].self_attn, "rotary_emb"):
                rotary_emb = self.model.layers[0].self_attn.rotary_emb
        
        if rotary_emb is not None:
             # LlamaRotaryEmbedding forward returns cos, sin
             # It takes (value_states, position_ids) usually.
             # We need to provide a tensor with the correct shape for head_dim inference if needed.
             # shape: (batch, num_heads, seq_len, head_dim)
             num_heads = self.model.config.num_attention_heads
             head_dim = self.model.config.hidden_size // num_heads
             
             # Create a dummy tensor with correct shape, device, and dtype
             dummy_x = inputs_embeds.new_empty(batch_size, num_heads, seq_length, head_dim)
             
             position_embeddings = rotary_emb(dummy_x, position_ids)

        return inputs_embeds, extended_attention_mask, position_ids, position_embeddings


    @torch.no_grad()
    def debug_forward_layer(self, layer_idx, hidden_states, attention_mask, position_ids, past_key_values=None, position_embeddings=None):
        """
        Run a single layer forward pass.
        Manually implements the LlamaDecoderLayer forward pass to ensure we capture KV cache correctly,
        bypassing potential return type issues in the layer's forward method.
        """
        layer = self.model.layers[layer_idx]
        past_key_value = past_key_values[layer_idx] if past_key_values is not None else None
        
        # Prepare kwargs for attention
        kwargs = {}
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings

        # 1. Input Norm
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        
        # 2. Self Attention
        # LlamaAttention returns (attn_output, attn_weights, past_key_value) if output_attentions=True
        attn_outputs = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True, # Enable for analysis and consistent return tuple
            use_cache=True,
            **kwargs
        )
        
        # Handle variable return length (SDPA/FlashAttn might not return weights even if requested)
        if len(attn_outputs) == 3:
            attn_output, attn_weights, new_past_key_value = attn_outputs
        elif len(attn_outputs) == 2:
            attn_output, new_past_key_value = attn_outputs
            attn_weights = None
        else:
            raise ValueError(f"Unexpected output length from LlamaAttention: {len(attn_outputs)}")
        
        # Residual Connection
        hidden_states = residual + attn_output
        
        # 3. MLP
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        
        # Residual Connection
        hidden_states = residual + hidden_states
        
        return hidden_states, new_past_key_value, attn_weights




    @torch.no_grad()
    def debug_forward_end(self, hidden_states):
        """
        Finalize the forward pass (Norm + LM Head).
        """
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

