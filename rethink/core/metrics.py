"""Distance metrics and confidence scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .cache import CachedState, HiddenStateCache


@dataclass
class ConfidenceResult:
    layer: int
    step: int
    value: float
    metrics: Dict[str, float]


class ConfidenceScorer:
    """Computes layer-wise confidence using configurable similarities."""

    def __init__(
        self,
        metrics: Sequence[str] = ("cosine",),
        reduction: str = "softmax",
        temperature: float = 1.0,
    ) -> None:
        self.metric_names = tuple(metrics)
        self.reduction = reduction
        self.temperature = temperature

    def _cosine(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(a, b, dim=-1)

    def _l2(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return -torch.norm(a - b, dim=-1)

    def _kl(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_logit = a
        b_logit = b
        a_log_prob = F.log_softmax(a_logit, dim=-1)
        b_prob = F.softmax(b_logit, dim=-1)
        return -F.kl_div(a_log_prob, b_prob, reduction="batchmean")

    def _apply_metric(self, name: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if name == "cosine":
            return self._cosine(a, b)
        if name == "l2":
            return self._l2(a, b)
        if name == "kl":
            return self._kl(a, b)
        raise ValueError(f"Unsupported metric: {name}")

    def _aggregate(self, values: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(values, dim=0)
        if self.reduction == "mean":
            return stacked.mean(dim=0)
        if self.reduction == "softmax":
            weights = F.softmax(stacked / max(self.temperature, 1e-5), dim=0)
            return (weights * stacked).sum(dim=0)
        if self.reduction == "min":
            return stacked.min(dim=0).values
        if self.reduction == "max":
            return stacked.max(dim=0).values
        raise ValueError(f"Unsupported reduction: {self.reduction}")

    def score_pair(self, anchor: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        tensors: List[torch.Tensor] = []
        for name in self.metric_names:
            result = self._apply_metric(name, anchor, candidate)
            scalar = result.mean().item()
            metrics[name] = float(scalar)
            tensors.append(torch.tensor(scalar))
        metrics["aggregate"] = float(self._aggregate(tensors).item())
        return metrics

    def _select_record(self, records: List[CachedState], reference_step: Optional[int]) -> CachedState:
        if reference_step is None or reference_step < 0:
            return records[-2]
        for record in records:
            if record.step == reference_step:
                return record
        return records[0]

    def score_layer_history(
        self,
        cache: HiddenStateCache,
        layer: int,
        *,
        reference_step: Optional[int] = None,
    ) -> Optional[ConfidenceResult]:
        records = list(cache.iter_layer(layer))
        if len(records) < 2:
            return None
        anchor = self._select_record(records, reference_step)
        candidate = records[-1]
        anchor_tensor = anchor.hidden_state
        candidate_tensor = candidate.hidden_state
        if anchor_tensor.shape != candidate_tensor.shape:
            min_len = min(anchor_tensor.size(-2), candidate_tensor.size(-2))
            anchor_tensor = anchor_tensor[..., -min_len:, :]
            candidate_tensor = candidate_tensor[..., -min_len:, :]
        metrics = self.score_pair(anchor_tensor, candidate_tensor)
        return ConfidenceResult(layer=layer, step=candidate.step, value=metrics["aggregate"], metrics=metrics)

    def score_cache(self, cache: HiddenStateCache, reference_step: Optional[int] = None) -> List[ConfidenceResult]:
        results: List[ConfidenceResult] = []
        for layer in cache.layers():
            result = self.score_layer_history(cache, layer, reference_step=reference_step)
            if result is not None:
                results.append(result)
        return results
