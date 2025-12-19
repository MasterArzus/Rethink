import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from rethink.recorder.token_recorder import TokenRecorder
from rethink.analysis.token_analysis import TokenAnalysis

def compute_js_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-9) -> float:
    """
    Compute Jensen-Shannon Divergence between two probability distributions.
    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)
    
    Args:
        p: Probability tensor (1, vocab_size)
        q: Probability tensor (1, vocab_size)
    """
    # Ensure inputs are probabilities
    p = p + epsilon
    q = q + epsilon
    
    m = 0.5 * (p + q)
    
    # KL Divergence: sum(p * log(p/q))
    kl_p_m = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
    kl_q_m = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
    
    jsd = 0.5 * (kl_p_m + kl_q_m)
    return jsd.item()

def compute_probability_margin(probs: torch.Tensor) -> float:
    """
    Compute the margin between the top-1 and top-2 probabilities.
    Margin = P(w1) - P(w2)
    """
    top_k = torch.topk(probs, 2, dim=-1)
    values = top_k.values[0] # (2,)
    if len(values) < 2:
        return 0.0
    return (values[0] - values[1]).item()

def analyze_layer_divergence(token_rec: TokenRecorder, model: Any, tokenizer: Any) -> Dict[str, Any]:
    """
    Analyze the divergence between early/middle layers and the final layer.
    Returns a profile of JSD scores across layers.
    """
    analyzer = TokenAnalysis(token_rec, model, tokenizer)
    
    # Get all available layers
    layers = sorted(token_rec.hidden_states.keys())
    if not layers:
        return {}
        
    final_layer_idx = layers[-1]
    final_analyzer = analyzer._get_analysis(final_layer_idx)
    if not final_analyzer:
        return {}
        
    final_logits = final_analyzer._compute_logits()
    final_probs = F.softmax(final_logits, dim=-1)
    
    divergence_profile = []
    
    for layer_idx in layers[:-1]: # Skip final layer comparing to itself
        layer_analyzer = analyzer._get_analysis(layer_idx)
        if not layer_analyzer:
            continue
            
        layer_logits = layer_analyzer._compute_logits()
        layer_probs = F.softmax(layer_logits, dim=-1)
        
        jsd = compute_js_divergence(layer_probs, final_probs)
        margin = compute_probability_margin(layer_probs)
        
        divergence_profile.append({
            "layer": layer_idx,
            "js_divergence": jsd,
            "margin": margin
        })
        
    return {
        "final_layer": final_layer_idx,
        "profile": divergence_profile,
        "final_margin": compute_probability_margin(final_probs)
    }
