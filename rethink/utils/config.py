"""Global configuration objects shared across the Rethink toolkit."""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Any, Dict
import yaml
from pathlib import Path


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
class GenerationConfig:
    """Hyperparameters for text generation."""
    
    temperature: float = 0.4
    top_p: float = 0.7
    top_k: Optional[int] = 40
    repetition_penalty: float = 1.15
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_new_tokens: int = 512
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
    do_sample: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "GenerationConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Filter keys that match the dataclass fields
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_name_or_path: str
    device_map: str = "auto"
    torch_dtype: str = "float16"
    attn_implementation: str = "eager"
    
    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class PromptConfig:
    """Configuration for prompts and chat templates."""
    system_prompt: str = "You are a helpful assistant."
    template_type: str = "chat"
    user_role: str = "user"
    system_role: str = "system"
    
    @classmethod
    def from_yaml(cls, path: str) -> "PromptConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}



@dataclass
class RethinkConfig:
    """Aggregate experiment configuration."""

    dataset: DatasetSlice = field(default_factory=lambda: DatasetSlice(name="gsm8k"))
    instrumentation: InstrumentationConfig = field(default_factory=InstrumentationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    output_dir: str = "outputs/rethink"
    device: str = "cuda"
    seed: int = 42


