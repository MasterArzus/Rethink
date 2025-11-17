"""Rethink-enabled adapters for Hugging Face Llama models."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

from ..core.base import BaseRethinkAdapter
from ..core.cache import HiddenStateCache
from ..core.controller import RethinkAction, RethinkController
from ..core.detokenizer import DecodeResult, HiddenStateDecoder
from ..core.metrics import ConfidenceResult, ConfidenceScorer
from ..core.options import RethinkOptions


class RethinkLlamaConfig(LlamaConfig):
    """Extends :class:`~LlamaConfig` with rethink-specific defaults."""

    model_type = "rethink-llama"

    def __init__(
        self,
        rethink_layers: Optional[list[int]] = None,
        rethink_options: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.rethink_layers = list(rethink_layers) if rethink_layers is not None else None
        opts = rethink_options or {}
        self.rethink_options = RethinkOptions(
            capture_layers=opts.get("capture_layers", self.rethink_layers),
            capture_last_token_only=opts.get("capture_last_token_only", True),
            store_past_key_values=opts.get("store_past_key_values", False),
            detach_hidden_states=opts.get("detach_hidden_states", True),
            cache_max_steps=opts.get("cache_max_steps", 2048),
            decode_strategy=opts.get("decode_strategy", "argmax"),
            metric_set=opts.get("metric_set", ("cosine",)),
            confidence_threshold=opts.get("confidence_threshold", 0.75),
            temperature=opts.get("temperature", 1.0),
            logits_softmax_temperature=opts.get("logits_softmax_temperature"),
            controller_window=opts.get("controller_window", 4),
            save_metrics=opts.get("save_metrics", True),
        )

    @property
    def total_layers(self) -> int:
        return getattr(self, "num_hidden_layers", 0)

    def target_layers(self) -> list[int]:
        return self.rethink_options.normalized_layers(self.total_layers)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        base_config = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        base_dict = base_config.to_dict()
        rethink_layers = base_dict.pop("rethink_layers", None)
        rethink_options = base_dict.pop("rethink_options", None)
        return cls(rethink_layers=rethink_layers, rethink_options=rethink_options, **base_dict)

    def to_dict(self) -> dict:
        data = super().to_dict()
        opts = self.rethink_options
        if isinstance(opts, RethinkOptions):
            data["rethink_options"] = asdict(opts)
        return data


class RethinkLlamaModel(LlamaModel):
    """Extends :class:`~LlamaModel` to stream hidden states into a cache."""

    def forward(
        self,
        *args: Any,
        output_hidden_states: Optional[bool] = None,
        rethink_cache: Optional[HiddenStateCache] = None,
        rethink_options: Optional[RethinkOptions] = None,
        rethink_step: Optional[int] = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if rethink_cache is not None:
            output_hidden_states = True
        outputs = super().forward(
            *args,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        if rethink_cache is not None:
            self._record_hidden_states(outputs, rethink_cache, rethink_options, rethink_step)
        return outputs

    def _record_hidden_states(
        self,
        outputs: BaseModelOutputWithPast,
        cache: HiddenStateCache,
        options: Optional[RethinkOptions],
        step: Optional[int],
    ) -> None:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None and isinstance(outputs, tuple) and len(outputs) > 2:
            hidden_states = outputs[2]
        if hidden_states is None:
            return
        opts = options or getattr(self.config, "rethink_options", None) or RethinkOptions()
        layer_outputs = hidden_states[1:]  # skip embeddings
        target_layers = opts.normalized_layers(len(layer_outputs))
        if not target_layers:
            target_layers = list(range(len(layer_outputs)))
        step_idx = step if step is not None else cache.next_step()
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None and isinstance(outputs, tuple) and len(outputs) > 1:
            past_key_values = outputs[1]
        for layer_idx in target_layers:
            tensor = layer_outputs[layer_idx]
            if opts.capture_last_token_only:
                tensor = tensor[:, -1:, :]
                token_index = -1
            else:
                token_index = tensor.size(1) - 1
            layer_pkv = None
            if opts.store_past_key_values and past_key_values is not None:
                layer_pkv = past_key_values[layer_idx]
            cache.record(
                layer=layer_idx,
                hidden_state=tensor,
                step=step_idx,
                token_index=token_index,
                past_key_value=layer_pkv,
                detach=opts.detach_hidden_states,
            )


def _coerce_options(value: Optional[object]) -> RethinkOptions:
    if isinstance(value, RethinkOptions):
        return value
    if isinstance(value, dict):
        return RethinkOptions(**value)
    return RethinkOptions()


def _merge_options(base, override: Optional[object]) -> RethinkOptions:
    base_opts = _coerce_options(base)
    if override is None:
        return base_opts
    if isinstance(override, RethinkOptions):
        override_dict = asdict(override)
    elif isinstance(override, dict):
        override_dict = dict(override)
    else:
        return base_opts
    merged: Dict[str, Any] = {**asdict(base_opts), **override_dict}
    return RethinkOptions(**merged)


class RethinkLlamaForCausalLM(LlamaForCausalLM, BaseRethinkAdapter):
    """Adds rethink-aware helpers on top of the standard causal LM."""

    config_class = RethinkLlamaConfig

    def __init__(self, config: LlamaConfig):
        config = self._prepare_rethink_config(config)
        super().__init__(config)
        if not isinstance(self.model, RethinkLlamaModel):
            base_state = self.model.state_dict()
            rethink_model = RethinkLlamaModel(config)
            rethink_model.load_state_dict(base_state)
            self.model = rethink_model
        self.hidden_state_decoder = HiddenStateDecoder(
            lm_head=self.lm_head,
            norm_layer=getattr(self.model, "norm", None),
        )
        self.confidence_scorer = ConfidenceScorer(metrics=config.rethink_options.metric_set)

    @classmethod
    def _prepare_rethink_config(cls, config: LlamaConfig, overrides: Optional[object] = None) -> RethinkLlamaConfig:
        if isinstance(config, RethinkLlamaConfig):
            rethink_config = config
        else:
            rethink_config = RethinkLlamaConfig(**config.to_dict())
        rethink_config.rethink_options = _merge_options(rethink_config.rethink_options, overrides)
        return rethink_config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        rethink_options: Optional[object] = None,
        config: Optional[LlamaConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = RethinkLlamaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs.get("config_kwargs", {}))
        config = cls._prepare_rethink_config(config, overrides=rethink_options)
        kwargs["config"] = config
        kwargs.pop("config_kwargs", None)
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        rethink_cache: Optional[HiddenStateCache] = None,
        rethink_options: Optional[RethinkOptions] = None,
        rethink_step: Optional[int] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rethink_cache=rethink_cache,
            rethink_options=rethink_options,
            rethink_step=rethink_step,
            **kwargs,
        )
        hidden_states = model_outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + model_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )

    def detokenize_from_cache(
        self,
        cache: HiddenStateCache,
        *,
        layer: int,
        step: Optional[int] = None,
        tokenizer=None,
        top_k: int = 1,
        strategy: Optional[str] = None,
    ) -> DecodeResult:
        record = cache.get(layer, step) if step is not None else cache.latest(layer)
        if record is None:
            raise ValueError(f"No cached state for layer={layer} step={step}")
        return self.hidden_state_decoder.decode(
            record.hidden_state,
            top_k=top_k,
            strategy=strategy or self.config.rethink_options.decode_strategy,
            tokenizer=tokenizer,
        )

    def analyze_cache(
        self,
        cache: HiddenStateCache,
        *,
        scorer: Optional[ConfidenceScorer] = None,
        reference_step: Optional[int] = None,
    ) -> list[ConfidenceResult]:
        scorer = scorer or self.confidence_scorer
        return scorer.score_cache(cache, reference_step=reference_step)

    def generate_with_rethink(
        self,
        input_ids: torch.LongTensor,
        *,
        cache: Optional[HiddenStateCache] = None,
        controller: Optional[RethinkController] = None,
        scorer: Optional[ConfidenceScorer] = None,
        tokenizer=None,
        max_rethink_loops: int = 1,
        **generate_kwargs: Any,
    ) -> tuple[Any, HiddenStateCache, list[ConfidenceResult], RethinkAction]:
        cache = cache or HiddenStateCache(max_steps=self.config.rethink_options.cache_max_steps)
        scorer = scorer or self.confidence_scorer
        target_layers = self.config.target_layers()
        max_layer = max(target_layers) if target_layers else None
        controller = controller or RethinkController(
            confidence_threshold=getattr(self.config, "rethink_options", RethinkOptions()).confidence_threshold,
            max_layer=max_layer,
        )
        action = RethinkAction(decision="continue")
        outputs = None
        scores: list[ConfidenceResult] = []
        for _ in range(max(1, max_rethink_loops)):
            outputs = super().generate(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict_in_generate=True,
                rethink_cache=cache,
                **generate_kwargs,
            )
            scores = scorer.score_cache(cache)
            action = controller.decide(scores)
            if action.decision != "rewind":
                break
        if outputs is None:
            raise RuntimeError("Generation failed to run")
        return outputs, cache, scores, action
