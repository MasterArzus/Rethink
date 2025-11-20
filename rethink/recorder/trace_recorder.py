from typing import List, Dict, Optional
import torch
from .token_recorder import TokenRecorder

class TraceRecorder:
    '''
    Record a complete inference trace, containing a sequence of TokenRecorders.
    '''
    def __init__(self, question: str, answer: str, tokenlist: List[TokenRecorder], metadata: Optional[Dict] = None):
        self.question = question
        self.answer = answer
        self.tokenlist = tokenlist
        self.metadata = metadata or {}

    def get_token_strings(self) -> List[str]:
        """Get the list of token strings."""
        return [t.token for t in self.tokenlist]

    def get_full_text(self) -> str:
        """Reconstruct the full generated text."""
        return "".join(self.get_token_strings())

    def get_layer_hidden_states(self, layer_idx: int) -> torch.Tensor:
        '''
        Get all hidden states for a specific layer as a stacked tensor.
        Returns:
            torch.Tensor: Shape (seq_len, hidden_dim)
        '''
        states = []
        for t in self.tokenlist:
            s = t.get_hidden_state(layer_idx)
            if s is not None:
                states.append(s)
        
        if not states:
            return torch.empty(0)
            
        # Ensure all states are on the same device (CPU usually) before stacking
        return torch.stack(states)

    def __repr__(self):
        return f"TraceRecorder(question='{self.question[:20]}...', tokens={len(self.tokenlist)})"
