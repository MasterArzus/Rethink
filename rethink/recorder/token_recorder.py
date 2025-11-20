from typing import Dict, Optional
import torch
from .hiddenstate_recorder import HiddenState

class TokenRecorder:
    '''
    Record a token during inference, including its probability and hidden states.
    '''
    def __init__(self, idx: int, step: int, token: str, prob: float, log_prob: float, hidden_states: Dict[int, HiddenState]):
        self.idx = idx  # token idx
        self.step = step  # step to generate this token
        self.token = token  # detokenized token
        self.prob = prob  # probability to generate this word
        self.log_prob = log_prob # log probability
        self.hidden_states = hidden_states  # Map layer_idx -> HiddenState

    def get_hidden_state(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Retrieve the hidden state tensor for a specific layer."""
        if layer_idx in self.hidden_states:
            return self.hidden_states[layer_idx].get_value()
        return None

    def __repr__(self):
        return f"TokenRecorder(token='{self.token}', step={self.step}, prob={self.prob:.4f})"
