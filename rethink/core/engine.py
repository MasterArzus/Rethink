"""High-level orchestration utilities for rethink-enabled generation."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .base import BaseRethinkAdapter
from .cache import HiddenStateCache
from .controller import RethinkAction, RethinkController
from .metrics import ConfidenceResult, ConfidenceScorer


class RethinkEngine:
    """Runs generation using any adapter that implements the rethink interface."""

    def __init__(
        self,
        adapter: BaseRethinkAdapter,
        *,
        cache: Optional[HiddenStateCache] = None,
        controller: Optional[RethinkController] = None,
        scorer: Optional[ConfidenceScorer] = None,
        tokenizer=None,
    ) -> None:
        self.adapter = adapter
        self.cache = cache
        self.controller = controller
        self.scorer = scorer
        self.tokenizer = tokenizer

    def generate(
        self,
        input_ids,
        *,
        max_rethink_loops: int = 1,
        **generate_kwargs: Any,
    ) -> tuple[Any, HiddenStateCache, Sequence[ConfidenceResult], RethinkAction]:
        return self.adapter.generate_with_rethink(
            input_ids,
            cache=self.cache,
            controller=self.controller,
            scorer=self.scorer,
            tokenizer=self.tokenizer,
            max_rethink_loops=max_rethink_loops,
            **generate_kwargs,
        )
