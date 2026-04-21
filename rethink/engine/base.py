"""Shared components for instrumented models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class TracePack:
    """Aggregate token-wise statistics and raw hidden states."""

    token_logprobs: List
    hidden_states: Dict[int, List[torch.Tensor]]
    extra: Dict[str, torch.Tensor]
