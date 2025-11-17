"""Hidden-state caching primitives used by rethink-aware decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Optional, Tuple

import torch


@dataclass
class CachedState:
    """Container for a single layer's hidden state at a given decoding step."""

    layer: int
    step: int
    hidden_state: torch.Tensor
    token_index: int = -1
    past_key_value: Optional[Tuple[torch.Tensor, ...]] = None
    confidence: Optional[float] = None
    metrics: Optional[dict] = field(default_factory=dict)
    metadata: Optional[dict] = field(default_factory=dict)

    def clone(self, detach: bool = True) -> "CachedState":
        hidden = self.hidden_state.detach().clone() if detach else self.hidden_state.clone()
        pkv = None
        if self.past_key_value is not None:
            pkv = tuple(t.detach().clone() for t in self.past_key_value)
        return CachedState(
            layer=self.layer,
            step=self.step,
            hidden_state=hidden,
            token_index=self.token_index,
            past_key_value=pkv,
            confidence=self.confidence,
            metrics=dict(self.metrics or {}),
            metadata=dict(self.metadata or {}),
        )


class HiddenStateCache:
    """Keeps a rolling buffer of hidden states for selected layers."""

    def __init__(self, max_steps: Optional[int] = None, device: Optional[torch.device] = None) -> None:
        self.max_steps = max_steps
        self.device = device
        self._storage: Dict[int, Dict[int, CachedState]] = {}
        self._current_step: int = -1

    @property
    def current_step(self) -> int:
        return self._current_step

    def next_step(self) -> int:
        self._current_step += 1
        return self._current_step

    def record(
        self,
        layer: int,
        hidden_state: torch.Tensor,
        *,
        step: Optional[int] = None,
        token_index: int = -1,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        confidence: Optional[float] = None,
        metrics: Optional[dict] = None,
        metadata: Optional[dict] = None,
        detach: bool = True,
    ) -> CachedState:
        if step is None:
            step = self.next_step()
        tensor = hidden_state.detach() if detach else hidden_state
        if self.device is not None:
            tensor = tensor.to(self.device)
        record = CachedState(
            layer=layer,
            step=step,
            hidden_state=tensor,
            token_index=token_index,
            past_key_value=past_key_value,
            confidence=confidence,
            metrics=metrics,
            metadata=metadata,
        )
        self._storage.setdefault(layer, {})[step] = record
        self._enforce_max_steps(layer)
        return record

    def _enforce_max_steps(self, layer: int) -> None:
        if self.max_steps is None:
            return
        layer_states = self._storage.get(layer)
        if not layer_states:
            return
        while len(layer_states) > self.max_steps:
            oldest = min(layer_states)
            del layer_states[oldest]

    def get(self, layer: int, step: int) -> Optional[CachedState]:
        return self._storage.get(layer, {}).get(step)

    def latest(self, layer: int) -> Optional[CachedState]:
        layer_map = self._storage.get(layer)
        if not layer_map:
            return None
        latest_step = max(layer_map)
        return layer_map[latest_step]

    def iter_layer(self, layer: int) -> Iterator[CachedState]:
        layer_map = self._storage.get(layer, {})
        for _, record in sorted(layer_map.items()):
            yield record

    def layers(self) -> Iterable[int]:
        return sorted(self._storage)

    def steps(self, layer: int) -> Iterable[int]:
        return sorted(self._storage.get(layer, {}))

    def to(self, device: torch.device) -> "HiddenStateCache":
        self.device = device
        for layer_map in self._storage.values():
            for record in layer_map.values():
                record.hidden_state = record.hidden_state.to(device)
                if record.past_key_value is not None:
                    record.past_key_value = tuple(t.to(device) for t in record.past_key_value)
        return self

    def clear(self) -> None:
        self._storage.clear()
        self._current_step = -1

    def __len__(self) -> int:
        return sum(len(layer_map) for layer_map in self._storage.values())

    def summary(self) -> dict:
        return {
            "layers": {layer: len(steps) for layer, steps in self._storage.items()},
            "total": len(self),
            "current_step": self.current_step,
        }
