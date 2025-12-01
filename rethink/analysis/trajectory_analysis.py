from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from rethink.recorder.trajectory import Trajectory
from rethink.analysis.hiddenstate_analysis import HiddenStateAnalysis

class TrajectoryAnalysis:
    """
    Analyzes the evolution of hidden states across layers for a single token generation step.
    """
    def __init__(self, trajectory: Trajectory, model: Any, tokenizer: Any):
        self.trajectory = trajectory
        self.model = model
        self.tokenizer = tokenizer
        self._layer_analyses: Dict[int, HiddenStateAnalysis] = {}

    def _get_analysis(self, layer_idx: int) -> Optional[HiddenStateAnalysis]:
        """Lazy loader for HiddenStateAnalysis objects."""
        state = self.trajectory.get_by_layer(layer_idx)
        if not state:
            return None
        
        if layer_idx not in self._layer_analyses:
            self._layer_analyses[layer_idx] = HiddenStateAnalysis(state, self.model, self.tokenizer)
        
        return self._layer_analyses[layer_idx]

    def compute_drift(self) -> List[Dict[str, Any]]:
        """
        Compute the cosine distance between consecutive layers.
        High drift indicates significant transformation of the representation.
        """
        drifts = []
        sorted_layers = sorted(self.trajectory.to_dict().keys())
        
        for i in range(len(sorted_layers) - 1):
            layer_from = sorted_layers[i]
            layer_to = sorted_layers[i+1]
            
            state_from = self.trajectory.get_by_layer(layer_from).get_value()
            state_to = self.trajectory.get_by_layer(layer_to).get_value()
            
            # Ensure shapes match and are flattened
            v1 = state_from.view(-1).float()
            v2 = state_to.view(-1).float()
            
            # Cosine Similarity
            sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            dist = 1.0 - sim
            
            drifts.append({
                "layer_from": layer_from,
                "layer_to": layer_to,
                "cosine_distance": dist
            })
        return drifts

    def project_to_vocab(self, layer_idx: int, k: int = 5) -> List[Dict[str, Any]]:
        """
        Project a specific layer's hidden state to the vocabulary (Logit Lens).
        Returns top-k tokens and their probabilities.
        """
        analyzer = self._get_analysis(layer_idx)
        if not analyzer:
            return []
        
        # Use the decode method from HiddenStateAnalysis
        # It returns List[Tuple[str, float]]
        decoded = analyzer.decode(k=k)
        
        return [{"token": t, "prob": p} for t, p in decoded]

    def get_entropy_evolution(self) -> List[Dict[str, Any]]:
        """
        Compute the entropy of the projected distribution at each layer.
        """
        evolution = []
        sorted_layers = sorted(self.trajectory.to_dict().keys())
        
        for layer_idx in sorted_layers:
            analyzer = self._get_analysis(layer_idx)
            if analyzer:
                entropy = analyzer.compute_entropy()
                evolution.append({
                    "layer": layer_idx,
                    "entropy": entropy
                })
        return evolution

    def get_attention_data(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Retrieve attention weights for a specific layer.
        Returns tensor of shape (batch, heads, seq_len, seq_len) or similar.
        """
        state = self.trajectory.get_by_layer(layer_idx)
        if state and state.attention_data and "attn_weights" in state.attention_data:
            return state.attention_data["attn_weights"]
        return None
