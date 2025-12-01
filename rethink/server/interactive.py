import torch
import json
import os
from datetime import datetime
from rethink.engine.controller import RethinkController
from rethink.recorder.trace_recorder import TraceRecorder
from rethink.analysis.trace_analysis import TraceAnalysis
from rethink.utils.config import RethinkConfig, DatasetSlice, InstrumentationConfig

class InteractiveSession:
    def __init__(self, model, tokenizer, cfg: RethinkConfig = None):
        if cfg is None:
            # Default configuration for interactive mode
            cfg = RethinkConfig(
                dataset=DatasetSlice(name="gsm8k", split="test"), # Default to gsm8k for prompt loading
                instrumentation=InstrumentationConfig(
                    track_hidden_states=True,
                    layers_to_capture=[-1], # Capture last layer by default for efficiency
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

    def run_initial_inference(self, prompt_text, use_template=True, max_new_tokens=128):
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
            
        # Generate and record
        trace_pack = self.controller.model.generate_autoregressive_trace(
            tokenizer=self.controller.tokenizer,
            prompt=final_prompt,
            generation_kwargs={
                "max_new_tokens": max_new_tokens,
                "eos_token_id": terminators
            }
        )
        
        # Convert to TraceRecorder
        generated_text = "".join([t.token for t in trace_pack.token_logprobs])
        self.current_trace = self.controller._tracepack_to_trace_recorder(trace_pack, final_prompt, generated_text)
        
        self._analyze_trace()
        return self.current_trace, self.analysis_results

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
            analyzer = TraceAnalysis(self.current_trace, self.controller.model, self.controller.tokenizer)
            self.analysis_results = analyzer.locate_critical_intervals()


from rethink.engine.llama import LlamaDebugStrategy

class InteractiveDebugSession:
    def __init__(self, model, tokenizer):
        # In a real scenario, we would detect the model type and instantiate the correct strategy
        # For now, we default to Llama
        self.strategy = LlamaDebugStrategy(model, tokenizer)

    def start(self, prompt):
        self.strategy.start_generation(prompt)
        return self.strategy.get_state()

    def step_layer(self):
        return self.strategy.step_layer()

    def finish_token(self):
        return self.strategy.finish_token()

    def sample_and_next(self):
        return self.strategy.sample_next_token()
    
    @property
    def generated_tokens(self):
        return [r.token for r in self.strategy.history]
    
    @property
    def current_trajectory(self):
        return self.strategy.current_trajectory


