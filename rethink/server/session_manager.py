import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
from rethink.engine.llama import RethinkLlamaForCausalLM
from rethink.engine.qwen import RethinkQwenForCausalLM
try:
    from rethink.engine.qwen import RethinkQwen3ForCausalLM
except ImportError:
    RethinkQwen3ForCausalLM = None
import streamlit as st


def _resolve_torch_dtype(dtype_name):
    if dtype_name is None:
        return torch.float16
    normalized = str(dtype_name).strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported torch_dtype '{dtype_name}'. Supported values: {', '.join(mapping.keys())}."
        )
    return mapping[normalized]


def _describe_checkpoint_layout(model_name_or_path):
    path = Path(str(model_name_or_path))
    if not path.exists() or not path.is_dir():
        return "unknown (remote_or_missing_local_path)"

    safetensors = list(path.glob("*.safetensors"))
    bins = list(path.glob("pytorch_model*.bin"))
    shard_index = list(path.glob("*.safetensors.index.json"))
    shard_count = len(safetensors) + len(bins)

    if shard_count <= 1 and not shard_index:
        return f"single-file checkpoint (files={shard_count})"
    return f"sharded checkpoint (files={shard_count}, index_files={len(shard_index)})"

@st.cache_resource
def load_model_and_tokenizer(
    model_name_or_path,
    torch_dtype_name="float16",
    device_map="auto",
    attn_implementation="eager",
):
    """
    Load the model and tokenizer. Cached by Streamlit to avoid reloading.
    """
    print(f"Loading model from {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Checkpoint layout: {_describe_checkpoint_layout(model_name_or_path)}")
    
    # Detect model architecture from both explicit architectures and model_type fallback.
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    architectures = config.architectures if config.architectures else []
    model_type = (getattr(config, "model_type", "") or "").lower()
    
    model_class = None
    if "LlamaForCausalLM" in architectures or "llama" in model_type:
        model_class = RethinkLlamaForCausalLM
    elif "Qwen2ForCausalLM" in architectures or "qwen2" in model_type:
        model_class = RethinkQwenForCausalLM
    elif (
        ("Qwen3ForCausalLM" in architectures or "qwen3" in model_type)
        and RethinkQwen3ForCausalLM is not None
    ):
        model_class = RethinkQwen3ForCausalLM
    else:
        raise ValueError(
            "Unsupported architecture for Rethink instrumentation. "
            f"architectures={architectures}, model_type={model_type}. "
            "Supported: LlamaForCausalLM/llama, Qwen2ForCausalLM/qwen2, Qwen3ForCausalLM/qwen3."
        )

    print(f"Selected model class: {model_class.__name__}")

    # Load the model directly using our instrumented class
    # This ensures the config is passed correctly and the model is initialized properly
    # We use attn_implementation="eager" to ensure output_attentions=True works for visualization
    resolved_dtype = _resolve_torch_dtype(torch_dtype_name)
    model = model_class.from_pretrained(
        model_name_or_path,
        torch_dtype=resolved_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=True
    )
    
    print("Model loaded successfully.")
    return model, tokenizer

class SessionManager:
    def __init__(self):
        pass

    @staticmethod
    def get_resources(model_path, model_cfg_data=None):
        model_cfg_data = model_cfg_data or {}
        torch_dtype_name = model_cfg_data.get("torch_dtype", "float16")
        device_map = model_cfg_data.get("device_map", "auto")
        attn_implementation = model_cfg_data.get("attn_implementation", "eager")
        return load_model_and_tokenizer(
            model_path,
            torch_dtype_name=torch_dtype_name,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
