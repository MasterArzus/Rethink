from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import torch
from rethink.recorder.trajectory import Trajectory
from rethink.recorder.token_recorder import TokenRecorder
from rethink.recorder.hiddenstate_recorder import HiddenState

class DebugStrategy(ABC):
    """
    Abstract base class for model-specific debugging logic.
    Manages the state of the generation process (KV cache, current hidden states)
    and provides a unified interface for step-by-step execution.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # History of generated tokens
        self.history: List[TokenRecorder] = []
        
        # The trajectory of the token currently being generated
        self.current_trajectory: Optional[Trajectory] = None
        
        # Current generation context
        self.full_text: str = ""
        self.current_input_ids: Optional[torch.Tensor] = None
        self.status: str = "idle" # idle, running, finished

    @abstractmethod
    def start_generation(self, prompt: str):
        """Initialize the generation process with a prompt."""
        pass

    @abstractmethod
    def step_layer(self) -> Dict[str, Any]:
        """Execute the next layer in the model."""
        pass

    @abstractmethod
    def finish_token(self) -> Dict[str, Any]:
        """Execute all remaining layers for the current token."""
        pass

    @abstractmethod
    def sample_next_token(self) -> Dict[str, Any]:
        """Sample the next token using the final hidden state and prepare for the next step."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot of the current debug state for the UI."""
        current_hidden = None
        if self.current_trajectory:
            last_state = self.current_trajectory.get_last_state()
            if last_state:
                current_hidden = last_state.value

        return {
            "status": self.status,
            "layer_idx": self.current_trajectory.current_layer_count() if self.current_trajectory else 0,
            "total_layers": self._get_total_layers(),
            "full_text": self.full_text,
            "current_token_preview": self._preview_current_token(),
            "hidden_states": current_hidden
        }

    def modify_current_hidden_state(self, new_value: torch.Tensor):
        """
        Modify the hidden state of the most recently executed layer.
        This allows 'intervention' before the next layer consumes this state.
        """
        if not self.current_trajectory:
            return
        
        last_state = self.current_trajectory.get_last_state()
        if last_state:
            # Update the value in the Trajectory
            last_state.value = new_value
            # Subclasses should also update their internal 'current_hidden_states' variable
            self._update_internal_hidden_state(new_value)

    @abstractmethod
    def _update_internal_hidden_state(self, new_value: torch.Tensor):
        """Update the internal tensor used for the next forward step."""
        pass

    @abstractmethod
    def _get_total_layers(self) -> int:
        pass

    def _preview_current_token(self) -> str:
        """Helper to guess what the current token might be (Logit Lens)."""
        return ""
