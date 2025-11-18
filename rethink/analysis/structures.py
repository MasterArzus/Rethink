"""Shared dataclasses for analysis outputs."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
