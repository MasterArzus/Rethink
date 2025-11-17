"""Abstract interfaces implemented by rethink-aware model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from .cache import HiddenStateCache
from .controller import RethinkAction, RethinkController
from .metrics import ConfidenceResult, ConfidenceScorer


class BaseRethinkAdapter(ABC):
    """Defines the hooks that every rethink-capable model must expose."""

    @abstractmethod
    def detokenize_from_cache(
        self,
        cache: HiddenStateCache,
        *,
        layer: int,
        step: Optional[int] = None,
        tokenizer=None,
        top_k: int = 1,
        strategy: Optional[str] = None,
    ):  # pragma: no cover - interface only
        raise NotImplementedError

    @abstractmethod
    def analyze_cache(
        self,
        cache: HiddenStateCache,
        *,
        scorer: Optional[ConfidenceScorer] = None,
        reference_step: Optional[int] = None,
    ) -> Sequence[ConfidenceResult]:  # pragma: no cover - interface only
        raise NotImplementedError

    @abstractmethod
    def generate_with_rethink(
        self,
        input_ids,
        *,
        cache: Optional[HiddenStateCache] = None,
        controller: Optional[RethinkController] = None,
        scorer: Optional[ConfidenceScorer] = None,
        tokenizer=None,
        max_rethink_loops: int = 1,
        **generate_kwargs: Any,
    ):  # pragma: no cover - interface only
        raise NotImplementedError
