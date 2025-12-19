
import argparse
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, RepetitionPenaltyLogitsProcessor
from rethink.engine.llama import RethinkLlamaForCausalLM, TracePack
from rethink.recorder.token_recorder import TokenRecorder
from rethink.analysis.token_analysis import TokenAnalysis
from rethink.utils.config import RethinkConfig
from transformers import AutoTokenizer, AutoConfig
import yaml
import os

class SimulationLlama(RethinkLlamaForCausalLM):
    @torch.no_grad()
    def generate_with_simulation(
        self,
        tokenizer,
        prompt: str,
        generation_kwargs: Optional[dict] = None,
        sos_threshold: float = 0.5,
        reference_layer_idx: int = 20,
    ) -> TracePack:
        """
        Run generation with SOS-based simulation intervention.
        """
        generation_kwargs = generation_kwargs or {"max_new_tokens": 128}
        device = self.device
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        token_logs: List[TokenRecorder] = []

        # Setup Logits Processors
        logits_processor = LogitsProcessorList()
        if generation_kwargs.get("repetition_penalty", 1.0) != 1.0:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=generation_kwargs["repetition_penalty"]))
        
        logits_warper = LogitsProcessorList()
        if generation_kwargs.get("temperature", 1.0) != 1.0:
            logits_warper.append(TemperatureLogitsWarper(temperature=generation_kwargs["temperature"]))
        if generation_kwargs.get("top_k", 0) > 0:
            logits_warper.append(TopKLogitsWarper(top_k=generation_kwargs["top_k"]))
        if generation_kwargs.get("top_p", 1.0) < 1.0:
            logits_warper.append(TopPLogitsWarper(top_p=generation_kwargs["top_p"]))

        recorder_ctx = self._recorder.attach(self) if self.instrumentation_cfg.track_hidden_states else None
        ctx_manager = recorder_ctx if recorder_ctx is not None else torch.no_grad()

        intervention_count = 0

        with ctx_manager:
            past_key_values = None
            generated_ids = input_ids
            for step in range(generation_kwargs.get("max_new_tokens", 128)):
                outputs = super(RethinkLlamaForCausalLM, self).forward(
                    input_ids=generated_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                    return_dict=True,
                    output_attentions=self.instrumentation_cfg.track_attentions,
                )
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                # Extract hidden states for this step
                current_states = {}
                if self.instrumentation_cfg.track_hidden_states and self._recorder.storage:
                    for l, states in self._recorder.storage.items():
                        if states:
                            current_states[l] = states[-1]
                
                # --- SOS Calculation & Intervention Logic ---
                sos_score = 0.0
                if current_states and reference_layer_idx in current_states:
                    # Create a temporary TokenRecorder to use TokenAnalysis
                    temp_recorder = TokenRecorder(
                        idx=0, step=step, token="", prob=0.0, log_prob=0.0, 
                        hidden_states=current_states
                    )
                    analyzer = TokenAnalysis(temp_recorder, self, tokenizer)
                    
                    # We need to pick a layer to compare against reference. 
                    # Usually the last layer (before final norm/head) or a mid layer.
                    # Let's assume we compare layer 15 vs 20 (reference).
                    # Or iterate through layers to find max SOS?
                    # For simplicity, let's compare layer 15 (mid) with reference.
                    mid_layer = 15 
                    if mid_layer in current_states:
                        sos_score = analyzer.compute_sos_metric(mid_layer, reference_layer_idx)
                
                # Intervention: If SOS is high, we might want to change sampling strategy
                if sos_score > sos_threshold:
                    intervention_count += 1
                    # Strategy: "Reject Top-1" (Simulate user saying 'No, not that')
                    # We mask the highest probability token and force the model to choose the next best
                    top1_idx = torch.argmax(next_token_logits, dim=-1)
                    next_token_logits[0, top1_idx] = -float('inf')
                    
                    # Optional: We could also increase temperature here to encourage exploration
                    # generation_kwargs['temperature'] = 1.0 
                
                # --------------------------------------------

                # Apply processors
                next_token_logits = logits_processor(generated_ids, next_token_logits)
                next_token_scores = logits_warper(generated_ids, next_token_logits)
                
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                
                if generation_kwargs.get("do_sample", True):
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
                
                token_id = next_token.item()
                token_str = tokenizer.decode([token_id])

                token_logs.append(
                    TokenRecorder(
                        idx=token_id,
                        step=step,
                        token=token_str,
                        prob=probs[0, token_id].item(),
                        log_prob=torch.log(probs[0, token_id]).item(),
                        hidden_states=current_states,
                    )
                )
                generated_ids = torch.cat([generated_ids, next_token.to(device)], dim=-1)
                
                eos_token_id = generation_kwargs.get("eos_token_id")
                if eos_token_id is not None:
                    if isinstance(eos_token_id, int) and token_id == eos_token_id:
                        break
                    elif isinstance(eos_token_id, (list, tuple)) and token_id in eos_token_id:
                        break

        print(f"Simulation complete. Total interventions triggered: {intervention_count}")
        print(f"Total tokens generated: {len(token_logs)}")
        
        return TracePack(
            token_logprobs=token_logs,
            hidden_states=dict(self._recorder.storage) if self.instrumentation_cfg.track_hidden_states else {},
            extra={"prompt_ids": input_ids.cpu()},
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Solve: 24 + 5 * 3 = ?")
    parser.add_argument("--sos-threshold", type=float, default=0.3)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    # Load config to get reference_layer_idx if possible, or default
    # For now, hardcode or pass via args
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    
    # Initialize SimulationLlama
    # We need to pass instrumentation config
    from rethink.utils.config import InstrumentationConfig
    instr_cfg = InstrumentationConfig(layers_to_capture=[15, 20, 31]) # Capture mid, ref, and last
    
    model = SimulationLlama.from_pretrained(
        args.model_path, 
        config=config, 
        instrumentation_cfg=instr_cfg,
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    print("Starting generation...")
    trace = model.generate_with_simulation(
        tokenizer, 
        args.prompt, 
        sos_threshold=args.sos_threshold,
        reference_layer_idx=20
    )
    
    print("Generation Result:")
    print("".join([t.token for t in trace.token_logprobs]))

if __name__ == "__main__":
    main()
