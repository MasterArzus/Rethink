from typing import Dict, Optional, Union
import torch
from .hiddenstate_recorder import HiddenState
from .trajectory import Trajectory

class TokenRecorder:
    '''
    Record a token during inference, including its probability and hidden states.
    '''
    def __init__(self, idx: int, step: int, token: str, prob: float, log_prob: float, hidden_states: Union[Dict[int, HiddenState], Trajectory], input_ids: Optional[torch.Tensor] = None):
        self.idx = idx  # token idx
        self.step = step  # step to generate this token
        self.token = token  # detokenized token
        self.prob = prob  # probability to generate this word
        self.log_prob = log_prob # log probability
        self.input_ids = input_ids # The input context used to generate this token
        
        # Support both legacy Dict and new Trajectory
        if isinstance(hidden_states, Trajectory):
            self.trajectory = hidden_states
        else:
            # Backwards compatibility: wrap dict in Trajectory
            self.trajectory = Trajectory()
            for layer_idx, state in hidden_states.items():
                self.trajectory.add(state)


    @property
    def hidden_states(self) -> Dict[int, HiddenState]:
        """Legacy accessor for compatibility."""
        return self.trajectory.to_dict()

    def get_hidden_state(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Retrieve the hidden state tensor for a specific layer."""
        state = self.trajectory.get_by_layer(layer_idx)
        if state:
            return state.get_value()
        return None


    def __repr__(self):
        return f"TokenRecorder(token='{self.token}', step={self.step}, prob={self.prob:.4f})"
