from typing import Dict, List, Optional, Iterator
from contextlib import contextmanager
import torch

class HiddenState:
    '''
    Data structure representing the hidden state of a single layer for a single token.
    '''
    def __init__(self, layer_idx: int, value: torch.Tensor, attention_data: Optional[Dict] = None, mlp_output: Optional[torch.Tensor] = None, residual_norm: Optional[float] = None):
        self.layer_idx = layer_idx
        self.value = value  # Expected shape: (hidden_dim,) or (1, hidden_dim)
        self.attention_data = attention_data # Optional: Attention weights or scores
        self.mlp_output = mlp_output # Optional: Output of the MLP sub-layer
        self.residual_norm = residual_norm # Optional: L2 norm of the residual stream

    def get_value(self) -> torch.Tensor:
        return self.value

    def to(self, device: str):
        self.value = self.value.to(device)
        if self.mlp_output is not None:
            self.mlp_output = self.mlp_output.to(device)
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
        # Storage now holds List[HiddenState] instead of List[torch.Tensor]
        self.storage: Dict[int, List[HiddenState]] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _should_capture(self, layer_idx: int) -> bool:
        return self.layers is None or layer_idx in self.layers

    def _hook(self, layer_idx: int):
        def wrapper(_module, _inputs, outputs):
            if not self._should_capture(layer_idx):
                return
            
            # Parse outputs
            # HuggingFace LlamaDecoderLayer output: (hidden_states, self_attn_weights, present_key_value)
            # If output_attentions=False, it might be just (hidden_states,) or (hidden_states, present_key_value)
            
            hidden_state = None
            attn_weights = None
            
            if isinstance(outputs, tuple):
                hidden_state = outputs[0]
                # Try to find attention weights
                # Usually if output_attentions=True, it's the second element
                if len(outputs) > 1:
                    # Check if the second element looks like attention weights
                    # Shape: (batch, heads, seq_len, seq_len)
                    possible_attn = outputs[1]
                    if isinstance(possible_attn, torch.Tensor) and possible_attn.dim() == 4:
                        attn_weights = possible_attn
            else:
                hidden_state = outputs

            # Clone and move to CPU
            # Assuming we want to capture the last token's hidden state
            token_vec = hidden_state[:, -1, :].detach().to(self.device or "cpu").cpu()
            
            attn_data = None
            if attn_weights is not None:
                # We usually only care about the attention for the last token query
                # attn_weights shape: (batch, heads, query_len, key_len)
                # For generation, query_len is usually 1.
                attn_data = {"attn_weights": attn_weights.detach().to(self.device or "cpu").cpu()}

            # Create HiddenState object
            hs = HiddenState(layer_idx=layer_idx, value=token_vec, attention_data=attn_data)
            
            self.storage.setdefault(layer_idx, []).append(hs)

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
