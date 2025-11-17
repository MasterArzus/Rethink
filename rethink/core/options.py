"""Model-agnostic rethink option dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class RethinkOptions:
    capture_layers: Optional[Sequence[int]] = None
    capture_last_token_only: bool = True
    store_past_key_values: bool = False
    detach_hidden_states: bool = True
    cache_max_steps: Optional[int] = 2048
    decode_strategy: str = "argmax"
    metric_set: Sequence[str] = ("cosine",)
    confidence_threshold: float = 0.75
    temperature: float = 1.0
    logits_softmax_temperature: Optional[float] = None
    controller_window: int = 4
    save_metrics: bool = True

    def normalized_layers(self, total_layers: int) -> List[int]:
        if not self.capture_layers:
            return list(range(total_layers))
        normalized: List[int] = []
        for idx in self.capture_layers:
            value = idx
            if value < 0:
                value = total_layers + value
            if 0 <= value < total_layers:
                normalized.append(value)
        return sorted(set(normalized))
