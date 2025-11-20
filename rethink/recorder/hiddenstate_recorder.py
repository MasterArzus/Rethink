from typing import Dict, List, Optional, Iterator
from contextlib import contextmanager
import torch

class HiddenState:
    '''
    Data structure representing the hidden state of a single layer for a single token.
    '''
    def __init__(self, layer_idx: int, value: torch.Tensor):
        self.layer_idx = layer_idx
        self.value = value  # Expected shape: (hidden_dim,) or (1, hidden_dim)

    def get_value(self) -> torch.Tensor:
        return self.value

    def to(self, device: str):
        self.value = self.value.to(device)
        return self

    def __repr__(self):
        return f"HiddenState(layer={self.layer_idx}, shape={self.value.shape})"


class HiddenStateRecorder:
    '''
    Collector that attaches to the model to capture hidden states during inference.
    '''
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
            # Assuming we want to capture the last token's hidden state
            token_vec = hidden_state[:, -1, :].detach().to(self.device or "cpu").cpu()
            self.storage.setdefault(layer_idx, []).append(token_vec)

        return wrapper

    @contextmanager
    def attach(self, model) -> Iterator["HiddenStateRecorder"]:
        """Context manager that installs hooks and clears them afterwards."""
        
        # Try to find the transformer layers. This might need adjustment for different models.
        # For Llama, it's usually model.model.layers
        transformer = getattr(getattr(model, "model", model), "layers", None)
        if transformer is None:
            # Fallback or raise error
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
