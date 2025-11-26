import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from rethink.engine.llama import RethinkLlamaForCausalLM
from rethink.engine.qwen import RethinkQwenForCausalLM
import streamlit as st

@st.cache_resource
def load_model_and_tokenizer(model_name_or_path):
    """
    Load the model and tokenizer. Cached by Streamlit to avoid reloading.
    """
    print(f"Loading model from {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Detect model architecture
    config = AutoConfig.from_pretrained(model_name_or_path)
    architectures = config.architectures if config.architectures else []
    
    model_class = None
    if "LlamaForCausalLM" in architectures:
        model_class = RethinkLlamaForCausalLM
    elif "Qwen2ForCausalLM" in architectures:
        model_class = RethinkQwenForCausalLM
    else:
        # Fallback or error? For now let's default to AutoModel but that won't have our instrumentation
        # Or maybe we should raise an error if we want to enforce instrumentation
        print(f"Warning: Architecture {architectures} not explicitly supported for instrumentation. Trying Llama fallback.")
        model_class = RethinkLlamaForCausalLM

    print(f"Selected model class: {model_class.__name__}")

    # Load the model directly using our instrumented class
    # This ensures the config is passed correctly and the model is initialized properly
    # We use attn_implementation="eager" to ensure output_attentions=True works for visualization
    model = model_class.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager"
    )
    
    print("Model loaded successfully.")
    return model, tokenizer

class SessionManager:
    def __init__(self):
        pass

    @staticmethod
    def get_resources(model_path):
        return load_model_and_tokenizer(model_path)
