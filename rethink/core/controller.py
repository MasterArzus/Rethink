"""Decision logic for selecting trustworthy hidden states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .metrics import ConfidenceResult


@dataclass
class RethinkAction:
    decision: str
    target_layer: Optional[int] = None
    reference_step: Optional[int] = None
    confidence: Optional[float] = None


class RethinkController:
    """Consumes confidence scores and decides whether to rewind or continue."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.75,
        patience: int = 2,
        min_layer: int = 0,
        max_layer: Optional[int] = None,
    ) -> None:
        self.threshold = confidence_threshold
        self.patience = patience
        self.min_layer = min_layer
        self.max_layer = max_layer
        self._below_threshold = 0

    def decide(self, scores: Iterable[ConfidenceResult]) -> RethinkAction:
        best = self._select_best(scores)
        if best is None:
            self._below_threshold += 1
            return RethinkAction(decision="continue")
        if best.value >= self.threshold:
            self._below_threshold = 0
            return RethinkAction(
                decision="continue",
                target_layer=best.layer,
                reference_step=best.step,
                confidence=best.value,
            )
        self._below_threshold += 1
        if self._below_threshold >= self.patience:
            self._below_threshold = 0
            return RethinkAction(
                decision="rewind",
                target_layer=best.layer,
                reference_step=best.step,
                confidence=best.value,
            )
        return RethinkAction(decision="continue", target_layer=best.layer, confidence=best.value)

    def _select_best(self, scores: Iterable[ConfidenceResult]) -> Optional[ConfidenceResult]:
        best: Optional[ConfidenceResult] = None
        for item in scores:
            if item is None:
                continue
            if self.max_layer is not None and item.layer > self.max_layer:
                continue
            if item.layer < self.min_layer:
                continue
            if best is None or item.value > best.value:
                best = item
        return best
