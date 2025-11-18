"""Global configuration objects shared across the Rethink toolkit."""

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class DatasetSlice:
    """Describe which subset of a dataset to use for quick experiments."""

    name: str
    split: str = "train"
    max_examples: Optional[int] = None
    filter_ids: Optional[Sequence[int]] = None


@dataclass
class InstrumentationConfig:
    """Control hooks for capturing logits, hidden states, and attention."""

    track_hidden_states: bool = True
    track_attentions: bool = False
    track_logits: bool = True
    layers_to_capture: Optional[Sequence[int]] = None
    max_tokens: Optional[int] = None


@dataclass
class RethinkConfig:
    """Aggregate experiment configuration."""

    dataset: DatasetSlice = field(default_factory=DatasetSlice)
    instrumentation: InstrumentationConfig = field(default_factory=InstrumentationConfig)
    output_dir: str = "outputs/rethink"
    device: str = "cuda"
    seed: int = 42
