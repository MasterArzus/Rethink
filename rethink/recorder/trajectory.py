from typing import List, Optional, Dict
import torch
from .hiddenstate_recorder import HiddenState

class Trajectory:
    """
    Intermediate layer: Records the complete inference path from Layer 0 to Layer N for a single token step.
    Acts as a dynamic container for HiddenStates during the generation process.
    """
    def __init__(self):
        self.states: List[HiddenState] = []
        self._layer_map: Dict[int, HiddenState] = {} # Cache for O(1) access
    
    def add(self, state: HiddenState):
        """Append a new layer state to the trajectory."""
        self.states.append(state)
        self._layer_map[state.layer_idx] = state
        
    def get_by_layer(self, layer_idx: int) -> Optional[HiddenState]:
        """Retrieve state by layer index."""
        return self._layer_map.get(layer_idx)
    
    def current_layer_count(self) -> int:
        """Returns how many layers have been recorded so far."""
        return len(self.states)

    def get_last_state(self) -> Optional[HiddenState]:
        """Get the most recently added state (usually the top-most layer computed so far)."""
        return self.states[-1] if self.states else None

    def to_dict(self) -> Dict[int, HiddenState]:
        """Convert to the legacy dictionary format {layer_idx: HiddenState}."""
        return self._layer_map.copy()

    def __repr__(self):
        return f"Trajectory(layers={len(self.states)})"
