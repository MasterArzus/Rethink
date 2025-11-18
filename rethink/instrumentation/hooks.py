"""Reusable hooks for collecting hidden states and token statistics."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import torch


@dataclass
class TokenLogProb:
    """Store log-probability metadata for a generated token."""

    token: str
    token_id: int
    prob: float
    log_prob: float
    step: int
    meta: Dict[str, float] = field(default_factory=dict)


class HiddenStateRecorder:
    """Forward hooks that capture hidden states during autoregressive decoding."""

    def __init__(self, layers: Optional[List[int]] = None, device: Optional[str] = None):
        self.layers = layers
        self.device = device
        self.storage: Dict[int, List[torch.Tensor]] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _should_capture(self, layer_idx: int) -> bool:
        return self.layers is None or layer_idx in self.layers

    def _hook(self, layer_idx: int):
        def wrapper(_module, _inputs, outputs):
            if not self._should_capture(layer_idx):
                return
            hidden_state = outputs[0] if isinstance(outputs, tuple) else outputs
            # clone the last token vector for lightweight storage
            token_vec = hidden_state[:, -1, :].detach().to(self.device or "cpu").cpu()
            self.storage.setdefault(layer_idx, []).append(token_vec)

        return wrapper

    @contextmanager
    def attach(self, model) -> Iterator["HiddenStateRecorder"]:
        """Context manager that installs hooks and clears them afterwards."""

        transformer = getattr(getattr(model, "model", model), "layers", None)
        if transformer is None:
            raise RuntimeError("Model does not expose decoder layers via model.layers")

        self.storage.clear()
        for idx, layer in enumerate(transformer):
            handle = layer.register_forward_hook(self._hook(idx))
            self.handles.append(handle)
        try:
            yield self
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
