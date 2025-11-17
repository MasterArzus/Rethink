"""Utilities for projecting cached hidden states back to tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class DecodeResult:
    token_ids: List[int]
    scores: List[float]
    text: Optional[str]


class HiddenStateDecoder:
    """Projects hidden states through the LM head for inspection."""

    def __init__(self, lm_head, norm_layer=None, tokenizer=None) -> None:
        self.lm_head = lm_head
        self.norm_layer = norm_layer
        self.tokenizer = tokenizer

    def logits_from_hidden(self, hidden_state: torch.Tensor) -> torch.Tensor:
        tensor = hidden_state
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 3:
            raise ValueError("hidden_state must be [batch, seq, hidden]")
        if self.norm_layer is not None:
            tensor = self.norm_layer(tensor)
        logits = torch.matmul(tensor, self.lm_head.weight.T)
        if hasattr(self.lm_head, "bias") and self.lm_head.bias is not None:
            logits = logits + self.lm_head.bias
        return logits

    def decode(
        self,
        hidden_state: torch.Tensor,
        *,
        top_k: int = 1,
        strategy: str = "argmax",
        temperature: float = 1.0,
        tokenizer=None,
    ) -> DecodeResult:
        logits = self.logits_from_hidden(hidden_state)
        last_logits = logits[:, -1, :]
        top_k = max(1, min(top_k, last_logits.size(-1)))
        if strategy == "sample":
            probs = F.softmax(last_logits / max(temperature, 1e-5), dim=-1)
            top_indices = torch.multinomial(probs, num_samples=top_k)
            scores = probs.gather(-1, top_indices)
        else:
            scores, top_indices = torch.topk(last_logits, k=top_k, dim=-1)
            scores = F.softmax(scores, dim=-1)
        token_ids = top_indices[0].tolist()
        score_list = scores[0].tolist()
        tok = tokenizer or self.tokenizer
        text = tok.decode(token_ids) if tok is not None else None
        return DecodeResult(token_ids=token_ids, scores=score_list, text=text)
