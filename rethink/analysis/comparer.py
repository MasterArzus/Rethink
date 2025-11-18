"""Functions that compare reference traces with model traces."""

from __future__ import annotations

from typing import Dict, List

import torch

from ..data.benchmark import TokenTrace
from .structures import DivergenceReport, LayerDivergence, TokenDivergence


def _compute_hidden_gap(ref_state, hyp_state) -> float:
    """Cosine distance helper with graceful fallback."""

    if ref_state.shape != hyp_state.shape:
        raise ValueError("Hidden states must share shape for comparison")
    ref_norm = torch.nn.functional.normalize(ref_state.flatten(0), dim=0)
    hyp_norm = torch.nn.functional.normalize(hyp_state.flatten(0), dim=0)
    gap = 1.0 - torch.dot(ref_norm, hyp_norm).item()
    return gap


def compare_traces(reference: TokenTrace, hypothesis: TokenTrace, hidden_states: Dict[int, List[torch.Tensor]] | None = None) -> DivergenceReport:
    """Generate coarse divergence metrics between two traces.

    Parameters
    ----------
    reference:
        Trace obtained by forcing the gold answer.
    hypothesis:
        Trace obtained from the model's own reasoning path.
    hidden_states:
        Optional shared-layer hidden state stacks (indexed by layer).
    """

    token_deltas: List[TokenDivergence] = []
    layer_deltas: List[LayerDivergence] = []

    for step, (ref_prob, hyp_prob) in enumerate(zip(reference.log_probs, hypothesis.log_probs)):
        prob_gap = ref_prob - hyp_prob
        token = reference.tokens[step] if step < len(reference.tokens) else "?"
        token_deltas.append(
            TokenDivergence(
                token=token,
                step=step,
                prob_gap=prob_gap,
                suspicion=float(abs(prob_gap)),
            )
        )

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
