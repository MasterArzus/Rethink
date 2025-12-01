from typing import Tuple, List, Optional, Any, Union
import torch
import torch.nn.functional as F
from rethink.recorder.hiddenstate_recorder import HiddenState

'''
example usage:
# 假设您已经有了一个 hidden_state 对象，以及加载好的 model 和 tokenizer
from rethink.analysis.hiddenstate_analysis import HiddenStateAnalysis

analyzer = HiddenStateAnalysis(hidden_state, model, tokenizer)

# 1. 看看这一层想说什么
top_tokens = analyzer.decode(k=5)
print(f"Layer {hidden_state.layer_idx} thinks next token is: {top_tokens}")

# 2. 获取低维统计特征
stats = analyzer.get_low_dim_representation(method="stats")
print(f"Stats vector: {stats}")

# 3. 计算不确定性
entropy = analyzer.compute_entropy()
print(f"Confusion level (Entropy): {entropy}")

'''

class HiddenStateAnalysis:
    '''Analysis a HiddenState during a inference in one layer in one token'''
    def __init__(self, hidden_state: HiddenState, model: Any, tokenizer: Any):
        """
        Args:
            hidden_state: The HiddenState object to analyze.
            model: The model (e.g., LlamaForCausalLM) used for inference. Needed for unembedding.
            tokenizer: The tokenizer used for decoding.
        """
        self.hidden_state = hidden_state
        # Ensure vector is (1, hidden_dim)
        self.vector = hidden_state.get_value().detach()
        
        # Handle 3D shape (batch, seq, hidden) -> take last token (batch, hidden)
        # This ensures we are analyzing the prediction for the *next* token
        if self.vector.dim() == 3:
            self.vector = self.vector[:, -1, :]
            
        if self.vector.dim() == 1:
            self.vector = self.vector.unsqueeze(0)
        
        self.model = model
        self.tokenizer = tokenizer
        self._logits: Optional[torch.Tensor] = None

    def _get_unembedding_components(self):
        """Helper to retrieve the final norm and lm_head from the model."""
        # Standard HF Llama structure: model.model.norm -> model.lm_head
        # Or generic: model.norm -> model.lm_head
        
        final_norm = getattr(self.model, 'norm', None)
        if final_norm is None and hasattr(self.model, 'model'):
            final_norm = getattr(self.model.model, 'norm', None)
            
        lm_head = getattr(self.model, 'lm_head', None)
        
        return final_norm, lm_head

    def _compute_logits(self) -> torch.Tensor:
        """
        Project the hidden state to the vocabulary space (Logit Lens).
        Applies the model's final normalization (if present) and then the LM head.
        """
        if self._logits is None:
            final_norm, lm_head = self._get_unembedding_components()
            
            if lm_head is None:
                raise ValueError("Model does not have an accessible lm_head for decoding.")

            # Move vector to model device for computation
            x = self.vector.to(self.model.device)
            
            # Apply final normalization (crucial for Llama/Transformer models)
            if final_norm:
                x = final_norm(x)
            
            self._logits = lm_head(x)
            
        return self._logits

    def decode(self, k: int = 5) -> List[Tuple[str, float]]:
        """
        Decode the hidden state to see its semantic representation (Logit Lens).
        
        Args:
            k: Number of top tokens to return.
            
        Returns:
            List of (token_string, probability) tuples.
        """
        logits = self._compute_logits()
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k, dim=-1)
        
        results = []
        # top_probs is (1, k)
        for prob, idx in zip(top_probs[0], top_indices[0]):
            token = self.tokenizer.decode([idx.item()])
            results.append((token, prob.item()))
        return results

    def get_low_dim_representation(self, method: str = "stats", projector: Optional[Any] = None) -> torch.Tensor:
        """
        Convert the high-dimensional hidden state to a low-dimensional representation.
        
        Args:
            method: The method to use.
                - 'stats': Returns basic statistics [mean, std, norm, min, max].
                - 'pca': Projects using a provided PCA object (must have transform method).
                - 'projection': Projects using a provided matrix (torch.Tensor).
            projector: The PCA object or projection matrix required for 'pca' or 'projection'.
            
        Returns:
            A low-dimensional torch.Tensor.
        """
        x = self.vector.float().cpu()
        
        if method == "stats":
            # Simple statistical summary (dim=5)
            return torch.tensor([
                x.mean(),
                x.std(),
                torch.norm(x),
                x.min(),
                x.max()
            ])
        
        elif method == "pca":
            if projector is None or not hasattr(projector, "transform"):
                raise ValueError("Method 'pca' requires a projector object with a 'transform' method (e.g., sklearn PCA).")
            # sklearn expects numpy
            x_np = x.numpy()
            projected = projector.transform(x_np)
            return torch.from_numpy(projected)
            
        elif method == "projection":
            if projector is None or not isinstance(projector, torch.Tensor):
                raise ValueError("Method 'projection' requires a torch.Tensor as projector.")
            # x is (1, D), projector should be (D, d)
            projector = projector.to(x.device)
            return torch.matmul(x, projector)
            
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")

    def compute_entropy(self) -> float:
        """
        Compute the entropy of the probability distribution implied by this hidden state.
        High entropy indicates the layer is 'confused' or 'undecided' about the next token.
        """
        logits = self._compute_logits()
        probs = F.softmax(logits, dim=-1)
        # Entropy = -sum(p * log(p))
        # Add epsilon to avoid log(0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9))
        return entropy.item()

    def compute_norm(self) -> float:
        """Compute the L2 norm of the hidden state vector."""
        return torch.norm(self.vector.float()).item()
