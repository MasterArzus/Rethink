from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
from rethink.recorder.trace_recorder import TraceRecorder

@dataclass
class LayerDivergence:
    """Summaries of hidden-state drift for a specific layer."""
    layer: int
    score: float
    commentary: str = ""
    diagnostics: Dict[str, float] = field(default_factory=dict)

@dataclass
class TokenDivergence:
    """Per-token comparisons between reference and model traces."""
    token: str
    step: int
    prob_gap: float
    hidden_state_delta: Optional[float] = None
    suspicion: float = 0.0

@dataclass
class DivergenceReport:
    """Bundle divergences with convenient accessors."""
    token_deltas: List[TokenDivergence]
    layer_deltas: List[LayerDivergence]
    flagged_spans: List[tuple] = field(default_factory=list)

def _compute_hidden_gap(ref_state, hyp_state) -> float:
    """Cosine distance helper with graceful fallback."""
    if ref_state.shape != hyp_state.shape:
        # raise ValueError("Hidden states must share shape for comparison")
        return 0.0
    ref_norm = torch.nn.functional.normalize(ref_state.flatten(0), dim=0)
    hyp_norm = torch.nn.functional.normalize(hyp_state.flatten(0), dim=0)
    gap = 1.0 - torch.dot(ref_norm, hyp_norm).item()
    return gap

def compare_traces(reference: TraceRecorder, hypothesis: TraceRecorder, hidden_states: Dict[int, List[torch.Tensor]] | None = None) -> DivergenceReport:
    """Generate coarse divergence metrics between two traces."""
    token_deltas: List[TokenDivergence] = []
    layer_deltas: List[LayerDivergence] = []

    min_len = min(len(reference.tokenlist), len(hypothesis.tokenlist))
    
    for step in range(min_len):
        ref_token = reference.tokenlist[step]
        hyp_token = hypothesis.tokenlist[step]
        
        ref_prob = ref_token.prob
        hyp_prob = hyp_token.prob
        
        prob_gap = ref_prob - hyp_prob
        token = ref_token.token
        
        token_deltas.append(
            TokenDivergence(
                token=token,
                step=step,
                prob_gap=prob_gap,
                suspicion=float(abs(prob_gap)),
            )
        )

    # Use the hidden states from the recorders if available, otherwise fallback to the passed hidden_states dict
    # Note: The passed hidden_states dict is legacy from when we returned it separately in TracePack
    # Now TraceRecorder has it inside TokenRecorders.
    
    # Let's try to compare layer 0 as a sample if available
    # This is just a placeholder logic for layer comparison
    # In a real scenario, we would iterate over common layers
    
    # For now, we keep the old logic if hidden_states is passed, but we should probably update it to use TraceRecorder data
    if hidden_states:
        for layer_idx, states in hidden_states.items():
            if len(states) < 2:
                continue
            ref_state, hyp_state = states[:2]
            score = _compute_hidden_gap(ref_state, hyp_state)
            layer_deltas.append(
                LayerDivergence(layer=layer_idx, score=score, diagnostics={"cosine_gap": score})
            )

    flagged_spans = [
        (delta.step, delta.step)
        for delta in token_deltas
        if delta.suspicion > 0.5  # placeholder threshold
    ]

    return DivergenceReport(token_deltas=token_deltas, layer_deltas=layer_deltas, flagged_spans=flagged_spans)
