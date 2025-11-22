import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rethink.engine.llama import RethinkLlamaForCausalLM
import streamlit as st

@st.cache_resource
def load_model_and_tokenizer(model_name_or_path):
    """
    Load the model and tokenizer. Cached by Streamlit to avoid reloading.
    """
    print(f"Loading model from {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Load the model directly using our instrumented class
    # This ensures the config is passed correctly and the model is initialized properly
    # We use attn_implementation="eager" to ensure output_attentions=True works for visualization
    model = RethinkLlamaForCausalLM.from_pretrained(
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
