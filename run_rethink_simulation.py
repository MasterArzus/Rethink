
import argparse
import torch
import torch.nn.functional as F
import json
import re
from typing import Optional, List, Dict, Any
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, RepetitionPenaltyLogitsProcessor
from rethink.engine.llama import RethinkLlamaForCausalLM, TracePack
from rethink.recorder.token_recorder import TokenRecorder
from rethink.analysis.token_analysis import TokenAnalysis
from rethink.utils.config import RethinkConfig, InstrumentationConfig
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset

def extract_answer(text):
    """Extract the numerical answer from GSM8K text."""
    match = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if match:
        return match.group(1)
    matches = re.findall(r"-?\d+\.?\d*", text)
    if matches:
        return matches[-1]
    return None

class SimulationLlama(RethinkLlamaForCausalLM):
    @torch.no_grad()
    def generate_with_simulation(
        self,
        tokenizer,
        prompt: str,
        generation_kwargs: Optional[dict] = None,
        sos_threshold: float = 0.5,
        reference_layer_idx: int = 20,
        max_interventions: int = 5
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
        total_tokens = 0

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
                intervened = False
                
                if current_states and reference_layer_idx in current_states:
                    # Create a temporary TokenRecorder to use TokenAnalysis
                    temp_recorder = TokenRecorder(
                        idx=0, step=step, token="", prob=0.0, log_prob=0.0, 
                        hidden_states=current_states
                    )
                    analyzer = TokenAnalysis(temp_recorder, self, tokenizer)
                    
                    # Compare mid layer (e.g., 15) with reference (e.g., 20 or last)
                    mid_layer = 15 
                    if mid_layer in current_states:
                        # Use the new compute_sos_metric
                        sos_score = analyzer.compute_sos_metric(mid_layer, reference_layer_idx)
                
                # Intervention: If SOS is high, we might want to change sampling strategy
                if sos_score > sos_threshold and intervention_count < max_interventions:
                    intervention_count += 1
                    intervened = True
                    # Strategy: "Reject Top-1" (Simulate user saying 'No, not that')
                    # We mask the highest probability token and force the model to choose the next best
                    top1_idx = torch.argmax(next_token_logits, dim=-1)
                    next_token_logits[0, top1_idx] = -float('inf')
                    
                    # Optional: Increase temperature to encourage exploration
                    # generation_kwargs['temperature'] = 1.2
                
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
                total_tokens += 1

                token_logs.append(
                    TokenRecorder(
                        idx=token_id,
                        step=step,
                        token=token_str,
                        prob=probs[0, token_id].item(),
                        log_prob=torch.log(probs[0, token_id]).item(),
                        hidden_states=current_states,
                        extra={"sos": sos_score, "intervened": intervened}
                    )
                )
                generated_ids = torch.cat([generated_ids, next_token.to(device)], dim=-1)
                
                eos_token_id = generation_kwargs.get("eos_token_id")
                if eos_token_id is not None:
                    if isinstance(eos_token_id, int) and token_id == eos_token_id:
                        break
                    elif isinstance(eos_token_id, (list, tuple)) and token_id in eos_token_id:
                        break

        return TracePack(
            token_logprobs=token_logs,
            hidden_states=dict(self._recorder.storage) if self.instrumentation_cfg.track_hidden_states else {},
            extra={
                "prompt_ids": input_ids.cpu(),
                "intervention_count": intervention_count,
                "total_tokens": total_tokens
            },
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--sos-threshold", type=float, default=0.3)
    parser.add_argument("--output-file", type=str, default="rethink_simulation_results.json")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    
    # Initialize SimulationLlama
    # We need to pass instrumentation config
    # Capture mid (15), ref (20), and last (31/32 depending on model)
    # Assuming Llama-3-8B has 32 layers.
    instr_cfg = InstrumentationConfig(layers_to_capture=[15, 20, 31]) 
    
    model = SimulationLlama.from_pretrained(
        args.model_path, 
        config=config, 
        instrumentation_cfg=instr_cfg,
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    # Load Dataset
    if args.dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test[:10]")
    else:
        raise ValueError("Dataset not supported")
        
    results = []
    
    for i, example in enumerate(ds):
        question = example['question']
        answer = example['answer']
        print(f"\n--- Example {i} ---")
        print(f"Question: {question}")
        
        trace = model.generate_with_simulation(
            tokenizer, 
            question, 
            sos_threshold=args.sos_threshold,
            reference_layer_idx=20
        )
        
        generated_text = "".join([t.token for t in trace.token_logprobs])
        print(f"Generated: {generated_text[:50]}...")
        
        # Check correctness
        pred = extract_answer(generated_text)
        gt = extract_answer(answer)
        success = (pred == gt) if (pred and gt) else False
        
        print(f"Success: {success}, Interventions: {trace.extra['intervention_count']}")
        
        results.append({
            "question": question,
            "ground_truth": answer,
            "generated_text": generated_text,
            "success": success,
            "interventions": trace.extra['intervention_count'],
            "total_tokens": trace.extra['total_tokens']
        })
        
    # Summary
    csr = sum(1 for r in results if r['success']) / len(results)
    avg_interventions = sum(r['interventions'] for r in results) / len(results)
    
    print("\n=== Summary ===")
    print(f"Rethink CSR: {csr:.2%}")
    print(f"Avg Interventions: {avg_interventions:.2f}")
    
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
