"""Visualization stubs for notebook or dashboard integrations."""

from __future__ import annotations

from typing import Iterable
import torch
# from rethink.analysis.trace_analysis import TraceAnalysisReport

try:
    from bertviz import head_view
    BERTVIZ_AVAILABLE = True
except ImportError:
    BERTVIZ_AVAILABLE = False

def generate_attention_html(model, tokenizer, text):
    if not BERTVIZ_AVAILABLE:
        return "<div style='color:red'>BertViz is not installed. Please run `pip install bertviz`.</div>"
    
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    
    # Limit sequence length to avoid OOM or browser crash
    if inputs.input_ids.shape[1] > 512:
        # Truncate if too long, though this might break if we want to see the end
        # For now, just warn or take the last 512? 
        # Taking last 512 might lose context for attention.
        # Let's just warn.
        pass

    with torch.no_grad():
        outputs = model(inputs.input_ids, output_attentions=True)
        
    attention = outputs.attentions
    
    if attention is None:
        return "<div style='color:red'>Error: Model did not return attention weights. Please ensure the model is loaded with attn_implementation='eager'.</div>"

    # Convert ids to tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # Generate HTML
    try:
        html_obj = head_view(attention, tokens, html_action='return')
        return html_obj.data
    except Exception as e:
        return f"<div style='color:red'>Error generating visualization: {str(e)}</div>"

def render_prob_trajectory(report) -> None:
    """Placeholder for a Plotly/Matplotlib implementation."""

    raise NotImplementedError("Visualization layer to be implemented")

def plot_sos_evolution(sos_scores: list[float], tokens: list[str], save_path: str = "sos_plot.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(12, 6))
    plt.plot(sos_scores, marker='o', linestyle='-')
    plt.title("SOS Score Evolution")
    plt.xlabel("Token Step")
    plt.ylabel("SOS Score")
    
    # If too many tokens, only show some ticks
    if len(tokens) > 50:
        step = len(tokens) // 50
        plt.xticks(range(0, len(tokens), step), [tokens[i] for i in range(0, len(tokens), step)], rotation=90)
    else:
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def render_hidden_state_heatmap(report) -> None:
    """Placeholder function for hidden-state diagnostics."""

    raise NotImplementedError("Visualization layer to be implemented")
