from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn.functional as F
from rethink.recorder.token_recorder import TokenRecorder
from rethink.analysis.hiddenstate_analysis import HiddenStateAnalysis

'''
from rethink.analysis.token_analysis import TokenAnalysis

# 假设您有一个 token_recorder 对象
analyzer = TokenAnalysis(token_recorder, model, tokenizer)

# 1. 看看这个 Token 是怎么被“想”出来的
evolution = analyzer.analyze_semantic_evolution()
for step in evolution:
    print(f"Layer {step['layer']}: Top1='{step['top_k'][0][0]}' (Prob: {step['target_prob']:.4f})")

# 2. 看看哪一层发生了剧烈变化
drifts = analyzer.analyze_layer_drift()
for d in drifts:
    if d['cosine_distance'] > 0.1: # 设定一个阈值
        print(f"Big jump between {d['layer_from']} -> {d['layer_to']}")
'''

class TokenAnalysis:
    '''
    Analyzes a single Token generation step, focusing on the evolution of hidden states across layers.
    '''
    def __init__(self, token_recorder: TokenRecorder, model: Any, tokenizer: Any):
        self.recorder = token_recorder
        self.model = model
        self.tokenizer = tokenizer
        self._layer_analyses: Dict[int, HiddenStateAnalysis] = {}

    def _get_analysis(self, layer_idx: int) -> Optional[HiddenStateAnalysis]:
        """Lazy loader for HiddenStateAnalysis objects."""
        if layer_idx not in self.recorder.hidden_states:
            return None
        
        if layer_idx not in self._layer_analyses:
            hs = self.recorder.hidden_states[layer_idx]
            self._layer_analyses[layer_idx] = HiddenStateAnalysis(hs, self.model, self.tokenizer)
        
        return self._layer_analyses[layer_idx]

    def analyze_semantic_evolution(self, k: int = 5) -> List[Dict[str, Any]]:
        """
        Trace the 'Logit Lens' evolution: what the model 'thought' the token was at each layer.
        
        This method implements the core idea of tracking semantic shifts. By decoding the 
        hidden state at each layer, we can see when the model 'decided' on the final token,
        or if it considered other alternatives in intermediate layers.

        Returns:
            A list of dicts, each containing:
            - layer: layer index
            - top_k: list of (token, prob) tuples
            - entropy: entropy of the distribution at this layer (uncertainty)
            - target_prob: probability assigned to the actually generated token at this layer
            - target_rank: rank of the actually generated token (0 = top choice)
        """
        evolution = []
        sorted_layers = sorted(self.recorder.hidden_states.keys())
        
        # The token that was actually generated
        target_token_id = self.recorder.idx

        for layer_idx in sorted_layers:
            analyzer = self._get_analysis(layer_idx)
            if not analyzer:
                continue

            # 1. Top-K decoding (Logit Lens)
            top_k = analyzer.decode(k=k)
            
            # 2. Entropy
            entropy = analyzer.compute_entropy()
            
            # 3. Probability of the target token
            # We need to access the logits from the analyzer to get specific token prob
            logits = analyzer._compute_logits() # (1, vocab_size)
            probs = F.softmax(logits, dim=-1)
            target_prob = probs[0, target_token_id].item()
            
            # 4. Rank of the target token
            # How far down the list was the actual choice?
            # argsort is ascending, so we reverse or take from end
            # Note: This can be slow for very large vocabs, but usually acceptable for analysis
            sorted_indices = torch.argsort(probs[0], descending=True)
            # find index of target_token_id
            rank = (sorted_indices == target_token_id).nonzero(as_tuple=True)[0].item()

            evolution.append({
                "layer": layer_idx,
                "top_k": top_k,
                "entropy": entropy,
                "target_prob": target_prob,
                "target_rank": rank
            })
            
        return evolution

    def analyze_layer_drift(self) -> List[Dict[str, Any]]:
        """
        Analyze how much the hidden state changes (cosine distance) between consecutive captured layers.
        
        High drift implies significant information processing or 'realization' (a semantic jump).
        Low drift implies the layer is mostly passing information through (residual stream dominance).
        """
        drift_stats = []
        sorted_layers = sorted(self.recorder.hidden_states.keys())
        
        for i in range(len(sorted_layers) - 1):
            l1, l2 = sorted_layers[i], sorted_layers[i+1]
            
            vec1 = self.recorder.hidden_states[l1].get_value().float()
            vec2 = self.recorder.hidden_states[l2].get_value().float()
            
            # Ensure shapes match and are flat for cosine sim
            if vec1.dim() > 1: vec1 = vec1.flatten()
            if vec2.dim() > 1: vec2 = vec2.flatten()
            
            # Cosine Similarity
            cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
            distance = 1.0 - cos_sim
            
            drift_stats.append({
                "layer_from": l1,
                "layer_to": l2,
                "cosine_distance": distance,
                "similarity": cos_sim
            })
            
        return drift_stats

    def get_trajectory_matrix(self, method: str = "stats") -> torch.Tensor:
        """
        Get a matrix representing the trajectory of the token through layers.
        
        Args:
            method: 'stats' (5 dims) or 'pca' (requires projector in HiddenStateAnalysis)
            
        Returns:
            torch.Tensor: Shape (num_layers, feature_dim)
        """
        features = []
        sorted_layers = sorted(self.recorder.hidden_states.keys())
        
        for layer_idx in sorted_layers:
            analyzer = self._get_analysis(layer_idx)
            if analyzer:
                feat = analyzer.get_low_dim_representation(method=method)
                features.append(feat)
        
        if not features:
            return torch.empty(0)
            
        return torch.stack(features)
