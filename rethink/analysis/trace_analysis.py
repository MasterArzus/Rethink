from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from rethink.recorder.trace_recorder import TraceRecorder
from rethink.recorder.hiddenstate_recorder import HiddenStateRecorder
from rethink.analysis.token_analysis import TokenAnalysis

@dataclass
class Interval:
    start: int
    end: int
    type: str  # "divergence", "high_entropy", "low_prob", "high_drift", "latent_conflict"
    score: float
    description: str

@dataclass
class TraceAnalysisReport:
    full_text: str
    divergence_index: Optional[int]
    critical_intervals: List[Interval]
    token_metrics: List[Dict[str, Any]]  # Per-token metrics (prob, entropy, etc.)

class TraceAnalysis:
    '''
    Analyzes a complete inference trace to locate errors and understand reasoning flow.
    '''
    def __init__(self, trace: TraceRecorder, model: Any, tokenizer: Any, reference_trace: Optional[TraceRecorder] = None):
        self.trace = trace
        self.reference = reference_trace
        self.model = model
        self.tokenizer = tokenizer

    def analyze_trace_evolution(self, k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform a deep scan of the entire trace, analyzing each token's layer-wise evolution.
        
        Returns:
            List of dicts, where each dict contains the analysis for one token step.
        """
        evolution_trace = []
        for token_recorder in self.trace.tokenlist:
            # Initialize TokenAnalysis for this specific token
            token_analyzer = TokenAnalysis(token_recorder, self.model, self.tokenizer)
            
            # Get semantic evolution (Logit Lens)
            semantic_evo = token_analyzer.analyze_semantic_evolution(k=k)
            
            # Get layer drift
            drift_stats = token_analyzer.analyze_layer_drift()
            
            evolution_trace.append({
                "step": token_recorder.step,
                "token": token_recorder.token,
                "prob": token_recorder.prob,
                "semantic_evolution": semantic_evo,
                "layer_drift": drift_stats
            })
        return evolution_trace

    def _find_divergence_point(self) -> int:
        """Find the first index where hypothesis differs from reference."""
        if not self.reference:
            return -1
            
        min_len = min(len(self.trace.tokenlist), len(self.reference.tokenlist))
        for i in range(min_len):
            # Compare token strings (normalized)
            t1 = self.trace.tokenlist[i].token.strip()
            t2 = self.reference.tokenlist[i].token.strip()
            if t1 != t2:
                return i
        
        if len(self.trace.tokenlist) != len(self.reference.tokenlist):
            return min_len
            
        return -1

    def _compute_step_hidden_states(self, step_idx: int) -> Dict[int, Any]:
        """
        Re-compute hidden states for a specific step to analyze latent conflicts.
        """
        device = self.model.device
        # Reconstruct input
        prompt_text = self.trace.question
        prev_tokens = [t.token for t in self.trace.tokenlist[:step_idx]]
        full_text = prompt_text + "".join(prev_tokens)
        
        inputs = self.tokenizer(full_text, return_tensors="pt").to(device)
        
        # We only need the last layer for logit lens
        # Assuming the model has 'config.num_hidden_layers' or similar
        num_layers = getattr(self.model.config, "num_hidden_layers", 32)
        last_layer_idx = num_layers - 1
        
        recorder = HiddenStateRecorder(layers=[last_layer_idx]) 
        
        with recorder.attach(self.model):
            with torch.no_grad():
                # We don't need to generate, just forward
                self.model(**inputs, output_hidden_states=True)
        
        states = {}
        if recorder.storage:
            for layer_idx, state_list in recorder.storage.items():
                if state_list:
                    # state_list[-1] is the HiddenState object for the last token
                    states[layer_idx] = state_list[-1]
                    
        return states

    def _project_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        state = hidden_state.to(self.model.device)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        # Try to find normalization layer
        norm_layer = getattr(self.model.model, "norm", None)
        if norm_layer:
            normalized = norm_layer(state)
        else:
            normalized = state # Fallback
            
        return self.model.lm_head(normalized)

    def _decode_hidden_state(self, hidden_state: torch.Tensor, top_k: int = 10) -> List[Tuple[str, float]]:
        logits = self._project_hidden_state(hidden_state)
        probs = torch.nn.functional.softmax(logits[0, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k)
        return [
            (self.tokenizer.decode([idx.item()]), prob.item())
            for prob, idx in zip(top_probs, top_indices)
        ]

    def analyze_latent_conflict(self, step_idx: int) -> Optional[Interval]:
        token_rec = self.trace.tokenlist[step_idx]
        
        # 1. Compute hidden states
        try:
            states = self._compute_step_hidden_states(step_idx)
        except Exception as e:
            print(f"Warning: Failed to compute hidden states for step {step_idx}: {e}")
            return None

        if not states:
            return None
            
        # 2. Decode last layer
        last_layer = max(states.keys())
        hs = states[last_layer]
        top_k = self._decode_hidden_state(hs.get_value(), top_k=5)
        
        # 3. Check conflict
        chosen_token = token_rec.token.strip()
        top_token = top_k[0][0].strip()
        top_prob = top_k[0][1]
        
        # If the chosen token is NOT the top token
        # And the top token is significantly more probable
        if chosen_token != top_token and top_prob > token_rec.prob * 1.2:
             return Interval(
                start=step_idx,
                end=step_idx + 1,
                type="latent_conflict",
                score=top_prob - token_rec.prob,
                description=f"Latent Conflict: Model latent preferred '{top_token}' ({top_prob:.2f}) over sampled '{chosen_token}' ({token_rec.prob:.2f})."
            )
        return None

    def locate_critical_intervals(self, entropy_threshold: float = 2.0, prob_threshold: float = 0.1) -> List[Interval]:
        """
        Identify suspicious or important intervals in the trace.
        
        Strategies:
        1. Divergence: Where did we deviate from the gold standard?
        2. Uncertainty: Where was the model confused (High Entropy)?
        3. Surprisal: Where did the model pick a low-probability token?
        4. Latent Conflict: Where did the model sample against its own latent preference?
        """
        intervals = []
        
        # 1. Divergence Analysis
        div_idx = self._find_divergence_point()
        if div_idx != -1:
            intervals.append(Interval(
                start=div_idx,
                end=min(div_idx + 5, len(self.trace.tokenlist)), # Mark a small window after divergence
                type="divergence",
                score=1.0,
                description=f"Diverged from reference at token '{self.trace.tokenlist[div_idx].token}'"
            ))

        # 2. Metric-based Analysis (Entropy & Probability)
        
        # Limit expensive checks to avoid performance hit
        conflict_checks = 0
        max_conflict_checks = 5

        for i, token_rec in enumerate(self.trace.tokenlist):
            # Check for low probability (Surprisal)
            if token_rec.prob < prob_threshold:
                intervals.append(Interval(
                    start=i,
                    end=i+1,
                    type="low_prob",
                    score=1.0 - token_rec.prob,
                    description=f"Low probability token '{token_rec.token}' ({token_rec.prob:.4f})"
                ))
                
                # Trigger Latent Conflict Check for low prob tokens
                if conflict_checks < max_conflict_checks:
                    conflict = self.analyze_latent_conflict(i)
                    if conflict:
                        intervals.append(conflict)
                        conflict_checks += 1
            
            # Check for high entropy (requires computation, so we do it on demand)
            # Note: This can be slow for long traces. In production, maybe pre-compute or sample.
            # Here we assume we want deep analysis.
            # token_analyzer = TokenAnalysis(token_rec, self.model, self.tokenizer)
            # # We pick the last layer's entropy as a proxy for final uncertainty
            # # (Or we could average across layers)
            # # For speed, let's skip full entropy computation here and rely on prob, 
            # # unless the user explicitly calls analyze_trace_evolution first.
            # pass 

        return intervals

    def generate_report(self) -> TraceAnalysisReport:
        """Generate a comprehensive report for data mining."""
        intervals = self.locate_critical_intervals()
        div_idx = self._find_divergence_point()
        
        token_metrics = []
        for t in self.trace.tokenlist:
            token_metrics.append({
                "step": t.step,
                "token": t.token,
                "prob": t.prob,
                "log_prob": t.log_prob
            })

        return TraceAnalysisReport(
            full_text=self.trace.get_full_text(),
            divergence_index=div_idx if div_idx != -1 else None,
            critical_intervals=intervals,
            token_metrics=token_metrics
        )

    def compute_sos_scores(self, start_idx: int = 0, end_idx: int = None) -> List[float]:
        """
        Compute SOS (Steering Opportunity Score) for tokens in the trace.

        Args:
            start_idx: Start index for computation (for incremental updates after rethink)
            end_idx: End index (exclusive). None means to end of trace.

        Returns:
            List of float SOS scores, one per token, in [0, 1].
        """
        if end_idx is None:
            end_idx = len(self.trace.tokenlist)

        scores = [0.0] * start_idx  # Pre-fill with zeros for unchanged tokens

        for i in range(start_idx, end_idx):
            token_rec = self.trace.tokenlist[i]
            layers = sorted(token_rec.hidden_states.keys())
            if len(layers) < 2:
                scores.append(0.0)
                continue

            reference_layer = layers[-1]
            mid_layer = layers[len(layers) // 2]

            token_analyzer = TokenAnalysis(token_rec, self.model, self.tokenizer)
            sos = token_analyzer.compute_sos_metric(mid_layer, reference_layer)
            scores.append(max(0.0, sos))
        return scores

    def get_token_alternatives(self, token_index: int, k: int = 5) -> Dict[str, Any]:
        """
        Get the top-k alternative tokens for a specific position.
        Uses the last layer's hidden state to decode logits.
        """
        if token_index < 0 or token_index >= len(self.trace.tokenlist):
            return {}
            
        token_rec = self.trace.tokenlist[token_index]
        token_analyzer = TokenAnalysis(token_rec, self.model, self.tokenizer)
        
        # Use the last available layer
        if not token_rec.hidden_states:
            return {}
            
        last_layer = max(token_rec.hidden_states.keys())
        
        # Get analysis for this layer
        hs_analysis = token_analyzer._get_analysis(last_layer)
        if not hs_analysis:
            return {}
            
        top_k = hs_analysis.decode(k=k)
        entropy = hs_analysis.compute_entropy()
        
        return {
            "top_k": top_k,
            "entropy": entropy,
            "layer": last_layer
        }

# Legacy function wrapper for backward compatibility if needed
def compare_traces(reference: TraceRecorder, hypothesis: TraceRecorder, hidden_states: Dict[int, List[torch.Tensor]] | None = None) -> Any:
    """Legacy wrapper: Use TraceAnalysis class instead."""
    # This is a simplified return to match the old signature's expected behavior roughly,
    # or we can just return the new report.
    # For now, let's just return a dummy object or adapt the new class to return the old DivergenceReport
    # But since we are refactoring, it's better to update the caller (Controller).
    pass
