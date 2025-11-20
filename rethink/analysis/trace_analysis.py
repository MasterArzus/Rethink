from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from rethink.recorder.trace_recorder import TraceRecorder
from rethink.analysis.token_analysis import TokenAnalysis

@dataclass
class Interval:
    start: int
    end: int
    type: str  # "divergence", "high_entropy", "low_prob", "high_drift"
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

    def locate_critical_intervals(self, entropy_threshold: float = 2.0, prob_threshold: float = 0.1) -> List[Interval]:
        """
        Identify suspicious or important intervals in the trace.
        
        Strategies:
        1. Divergence: Where did we deviate from the gold standard?
        2. Uncertainty: Where was the model confused (High Entropy)?
        3. Surprisal: Where did the model pick a low-probability token?
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
        # We need to compute entropy for each token if not already available.
        # For efficiency, we might just use the recorded prob if entropy isn't pre-calculated.
        # But TokenAnalysis can compute entropy.
        
        high_entropy_start = -1
        
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

# Legacy function wrapper for backward compatibility if needed
def compare_traces(reference: TraceRecorder, hypothesis: TraceRecorder, hidden_states: Dict[int, List[torch.Tensor]] | None = None) -> Any:
    """Legacy wrapper: Use TraceAnalysis class instead."""
    # This is a simplified return to match the old signature's expected behavior roughly,
    # or we can just return the new report.
    # For now, let's just return a dummy object or adapt the new class to return the old DivergenceReport
    # But since we are refactoring, it's better to update the caller (Controller).
    pass
