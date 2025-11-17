"""Model-specific adapters for rethink instrumentation."""

from .llama import (
    RethinkLlamaConfig,
    RethinkLlamaModel,
    RethinkLlamaForCausalLM,
)

__all__ = [
    "RethinkLlamaConfig",
    "RethinkLlamaModel",
    "RethinkLlamaForCausalLM",
]
