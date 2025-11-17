"""Minimal demo exercising rethink-specific hooks with a toy config."""

from __future__ import annotations

import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rethink import HiddenStateCache, RethinkController, RethinkEngine
from rethink.adapters import RethinkLlamaConfig, RethinkLlamaForCausalLM


def main() -> None:
    config = RethinkLlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rethink_layers=[0, 1],
        rethink_options={
            "capture_last_token_only": True,
            "metric_set": ("cosine", "l2"),
            "confidence_threshold": 0.75,
        },
    )
    model = RethinkLlamaForCausalLM(config)
    model.eval()

    input_ids = torch.randint(low=0, high=config.vocab_size, size=(1, 4))
    cache = HiddenStateCache(max_steps=16)
    controller = RethinkController(confidence_threshold=0.5)

    engine = RethinkEngine(
        model,
        cache=cache,
        controller=controller,
    )
    outputs, cache, scores, action = engine.generate(
        input_ids,
        max_rethink_loops=2,
        max_new_tokens=4,
    )

    print("Generated ids:", outputs.sequences)
    print("Cache summary:", cache.summary())
    print("Confidence scores:", scores)
    print("Controller action:", action)


if __name__ == "__main__":
    main()
