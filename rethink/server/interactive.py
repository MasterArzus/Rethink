import torch
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from rethink.engine.controller import RethinkController
from rethink.recorder.trace_recorder import TraceRecorder
from rethink.recorder.token_recorder import TokenRecorder
from rethink.analysis.trace_analysis import TraceAnalysis
from rethink.utils.config import RethinkConfig, DatasetSlice, InstrumentationConfig

class InteractiveSession:
    def __init__(self, model, tokenizer, cfg: RethinkConfig = None):
        if cfg is None:
            # Default configuration for interactive mode
            cfg = RethinkConfig(
                dataset=DatasetSlice(name="gsm8k", split="test"), # Default to gsm8k for prompt loading
                instrumentation=InstrumentationConfig(
                    track_hidden_states=False,
                    layers_to_capture=[-1], # Hidden states will be recomputed lazily per-click
                    max_tokens=512
                ),
                output_dir="outputs/interactive_sessions"
            )
        
        self.cfg = cfg
        self.controller = RethinkController(model, tokenizer, cfg)
        self.current_trace = None
        self.analysis_results = None
        
        # Ensure output directory exists
        os.makedirs(self.cfg.output_dir, exist_ok=True)

    def save_session(self, filename=None):
        """
        Save the current trace and analysis to a JSON file.
        """
        if not self.current_trace:
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"
            
        filepath = os.path.join(self.cfg.output_dir, filename)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": self.current_trace.question if hasattr(self.current_trace, 'question') else "",
            "generated_text": self.current_trace.get_full_text(),
            "tokens": [
                {
                    "token": t.token,
                    "prob": t.prob,
                    "step": t.step
                } for t in self.current_trace.tokenlist
            ],
            "analysis": [
                {
                    "start": interval.start,
                    "end": interval.end,
                    "type": interval.type,
                    "description": interval.description
                } for interval in (self.analysis_results or [])
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        return filepath

    def run_initial_inference(self, prompt_text, use_template=True, max_new_tokens=128, stream_callback=None):
        """
        Run the initial inference to get the baseline trace.
        """
        # Apply template if requested
        if use_template:
            final_prompt = self.controller._format_prompt(prompt_text)
        else:
            final_prompt = prompt_text

        # Determine stop tokens
        terminators = [self.controller.tokenizer.eos_token_id]
        # Add <|eot_id|> for Llama 3 if present
        if "<|eot_id|>" in self.controller.tokenizer.get_vocab():
            terminators.append(self.controller.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        # Add <|im_end|> for Qwen if present
        if "<|im_end|>" in self.controller.tokenizer.get_vocab():
            terminators.append(self.controller.tokenizer.convert_tokens_to_ids("<|im_end|>"))
            
        # Generate and record
        trace_pack = self.controller.model.generate_autoregressive_trace(
            tokenizer=self.controller.tokenizer,
            prompt=final_prompt,
            generation_kwargs={
                "max_new_tokens": max_new_tokens,
                "eos_token_id": terminators
            },
            stream_callback=stream_callback
        )
        
        # Convert to TraceRecorder
        generated_text = "".join([t.token for t in trace_pack.token_logprobs])
        self.current_trace = self.controller._tracepack_to_trace_recorder(trace_pack, final_prompt, generated_text)
        
        self._analyze_trace()
        return self.current_trace, self.analysis_results

    def rethink_from_step(self, trace_recorder, step_idx, max_new_tokens=128, force_token=None, steering_prompt: Optional[str] = None):
        """
        Truncate the trace at step_idx and regenerate from there.
        If force_token is provided, replace the token at step_idx with it.
        """
        # 1. Construct the new prompt context
        # The original prompt (including system prompts etc)
        base_prompt = trace_recorder.question
        
        # The tokens generated BEFORE step_idx
        prefix_tokens = trace_recorder.tokenlist[:step_idx]
        prefix_text = "".join([t.token for t in prefix_tokens])
        
        # The token at the rethink point
        if force_token is not None:
            current_token_text = force_token
            # Create a dummy recorder for the forced token
            # Try to get ID from tokenizer
            ids = self.controller.tokenizer.encode(force_token, add_special_tokens=False)
            token_id = ids[0] if ids else -1
            
            forced_recorder = TokenRecorder(
                idx=token_id,
                step=len(prefix_tokens),
                token=force_token,
                prob=1.0, # Forced
                log_prob=0.0,
                hidden_states={} # No hidden states
            )
            middle_token_list = [forced_recorder]
        else:
            current_token_text = trace_recorder.tokenlist[step_idx].token
            middle_token_list = [trace_recorder.tokenlist[step_idx]]
        
        guide_text = steering_prompt or ""
        new_full_prompt = base_prompt + prefix_text + current_token_text + guide_text
        
        # 2. Run generation
        # We pass use_template=False because new_full_prompt is already fully formatted
        new_trace, analysis = self.run_initial_inference(new_full_prompt, use_template=False, max_new_tokens=max_new_tokens)
        
        # 3. Merge traces for visualization continuity
        # The new_trace will contain tokens starting from index 0 relative to the new generation.
        # But we want to visualize it as a continuation.
        # Actually, run_initial_inference returns a full trace of the NEW generation.
        # But since we provided the prefix as prompt, the new trace only contains the NEWLY generated tokens
        # (because generate_autoregressive_trace usually returns only new tokens in token_logprobs).
        
        # We need to prepend the kept tokens to make a complete trace for visualization
        full_token_list = prefix_tokens + middle_token_list + new_trace.tokenlist
        
        # Re-index the new tokens
        start_idx = 0
        for i, t in enumerate(full_token_list):
            t.step = i # Update step index
            
        # Create a new combined TraceRecorder
        combined_trace = TraceRecorder(
            question=trace_recorder.question, # Original question
            answer=prefix_text + current_token_text + new_trace.get_full_text(),
            tokenlist=full_token_list,
            metadata=new_trace.metadata
        )
        
        # Re-analyze the combined trace
        analyzer = TraceAnalysis(combined_trace, self.controller.model, self.controller.tokenizer)
        combined_analysis = analyzer.locate_critical_intervals()
        
        return combined_trace, combined_analysis

    @dataclass
    class BranchResult:
        trace: TraceRecorder
        analysis: List[object]
        label: str
        meta: dict

    def branch_from_step(
        self,
        trace_recorder: TraceRecorder,
        step_idx: int,
        k: int = 3,
        strategy: str = "sample",
        max_new_tokens: int = 128,
        steering_prompt: Optional[str] = None,
    ) -> List["InteractiveSession.BranchResult"]:
        """
        Generate K alternative continuations from a given step, optionally with a steering prompt.
        """
        k = max(1, min(k, 8))

        base_prompt = trace_recorder.question
        prefix_tokens = trace_recorder.tokenlist[: step_idx + 1]
        prefix_text = "".join([t.token for t in prefix_tokens])
        guide_text = steering_prompt or ""
        prompt = base_prompt + prefix_text + guide_text

        # Build generation kwargs per strategy
        def _gen_kwargs(seed_variation: int):
            if strategy == "beam":
                return {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "seed": None,
                }
            # default: sample/high-temp
            return {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 1.1,
                "top_p": 0.9,
                "seed": None,
            }

        results: List[InteractiveSession.BranchResult] = []
        for i in range(k):
            gen_kwargs = _gen_kwargs(i)
            trace_pack = self.controller.model.generate_autoregressive_trace(
                tokenizer=self.controller.tokenizer,
                prompt=prompt,
                generation_kwargs=gen_kwargs,
                stream_callback=None,
            )

            generated_text = "".join([t.token for t in trace_pack.token_logprobs])
            new_tokens = trace_pack.token_logprobs

            # Merge prefix tokens with new tokens
            merged_tokens = prefix_tokens + new_tokens
            for idx, tok in enumerate(merged_tokens):
                tok.step = idx

            combined_trace = TraceRecorder(
                question=trace_recorder.question,
                answer=prefix_text + generated_text,
                tokenlist=merged_tokens,
                metadata=trace_pack.extra,
            )

            analyzer = TraceAnalysis(combined_trace, self.controller.model, self.controller.tokenizer)
            combined_analysis = analyzer.locate_critical_intervals()

            results.append(
                InteractiveSession.BranchResult(
                    trace=combined_trace,
                    analysis=combined_analysis,
                    label=f"Branch {i+1}",
                    meta={"strategy": strategy},
                )
            )

        return results

    def run_intervention(self, modified_tokens, start_index):
        """
        Run inference starting from a specific point with modified tokens.
        modified_tokens: list of strings (new tokens to force)
        start_index: index in the original trace where modification starts
        """
        # This is a simplified version of intervention. 
        # In a real scenario, we would need to:
        # 1. Truncate the KV cache or re-compute up to start_index
        # 2. Force the modified_tokens
        # 3. Continue generation
        
        # For this prototype, we will re-generate from scratch with the new prefix
        # constructed from: original_prefix + modified_tokens
        
        if not self.current_trace:
            return None

        # Reconstruct the prefix
        # The trace stores tokens. We need to get the text up to start_index.
        # trace.tokenlist is a list of Token objects.
        
        # Get original tokens up to start_index
        prefix_tokens = [t.token for t in self.current_trace.tokenlist[:start_index]]
        prefix_text = "".join(prefix_tokens)
        
        # Add modified tokens
        # modified_tokens is expected to be a string or list of strings
        if isinstance(modified_tokens, list):
            intervention_text = "".join(modified_tokens)
        else:
            intervention_text = modified_tokens
            
        new_prompt = prefix_text + intervention_text
        
        # Determine stop tokens
        terminators = [self.controller.tokenizer.eos_token_id]
        # Add <|eot_id|> for Llama 3 if present
        if "<|eot_id|>" in self.controller.tokenizer.get_vocab():
            terminators.append(self.controller.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        # Re-run inference
        new_trace_pack = self.controller.model.generate_autoregressive_trace(
            tokenizer=self.controller.tokenizer,
            prompt=new_prompt,
            generation_kwargs={
                "max_new_tokens": 512,
                "eos_token_id": terminators
            }
        )
        
        generated_text = "".join([t.token for t in new_trace_pack.token_logprobs])
        self.current_trace = self.controller._tracepack_to_trace_recorder(new_trace_pack, new_prompt, generated_text)
        
        self._analyze_trace()
        return self.current_trace, self.analysis_results

    def _analyze_trace(self):
        if self.current_trace:
            analyzer = TraceAnalysis(
                trace=self.current_trace,
                model=self.controller.model,
                tokenizer=self.controller.tokenizer,
                reference_trace=None # Or pass if available
            )
            self.analysis_results = analyzer.locate_critical_intervals()


