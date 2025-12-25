
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import argparse
import torch
import torch.nn.functional as F
import json
import re
import os
import time
import random
from typing import Optional, List, Dict, Any
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, RepetitionPenaltyLogitsProcessor
from rethink.engine.llama import RethinkLlamaForCausalLM, TracePack
from rethink.recorder.token_recorder import TokenRecorder
from rethink.analysis.token_analysis import TokenAnalysis
from rethink.utils.config import RethinkConfig, InstrumentationConfig
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset, load_from_disk

try:
    import openai
except ImportError:
    openai = None

class LLMUser:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-4"):
        self.client = None
        if openai and api_key:
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def critique_answer(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Ask the LLM Judge to critique the answer.
        Returns: {'correct': bool, 'last_correct_segment': str, 'reason': str}
        """
        if not self.client:
            return {'correct': True, 'last_correct_segment': answer, 'reason': "No API"}

        prompt = f"""
                    You are a strict teacher grading a math problem.
                    Problem: {question}

                    Student Answer:
                    {answer}

                    Is the student's answer logically correct and complete?
                    - If YES, output exactly: "STATUS: CORRECT"
                    - If NO (logic error, calculation error, or stops prematurely), output:
                    "STATUS: INCORRECT"
                    "LAST_CORRECT: <quote the exact text of the answer up to the last correct point>"
                    "REASON: <brief explanation>"

                    Example of premature stop:
                    Answer: "15 + 3 = "
                    LAST_CORRECT: "15 + 3 ="
                    (This implies the next token should be the result, not stop)
                    """
        max_retries = 5
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150
                )
                content = response.choices[0].message.content.strip()
                
                is_correct = "STATUS: CORRECT" in content
                last_correct = answer # Default to full answer
                reason = ""
                
                if not is_correct:
                    # Extract LAST_CORRECT
                    match = re.search(r"LAST_CORRECT:\s*\"?(.*?)\"?\n", content, re.DOTALL)
                    if not match:
                        match = re.search(r"LAST_CORRECT:\s*\"?(.*)", content, re.DOTALL)
                    
                    if match:
                        last_correct = match.group(1).strip()
                    
                    # Extract REASON
                    match_r = re.search(r"REASON:\s*(.*)", content, re.DOTALL)
                    if match_r:
                        reason = match_r.group(1).strip()
                        
                return {
                    'correct': is_correct,
                    'last_correct_segment': last_correct,
                    'reason': reason
                }

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"LLM Judge Rate Limit (429). Retrying in {delay:.2f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"LLM Judge Error: {e}")
                    break
        
        return {'correct': True, 'last_correct_segment': answer, 'reason': "Error"}

    def select_best_token(self, question: str, generated_so_far: str, candidates: List[str], layer_evolution: str = "", internal_explanation: str = "") -> Optional[int]:
        """
        Ask the LLM Judge to pick the best token from candidates.
        Returns the index of the best token (0-based), or None if failed.
        """
        if not self.client:
            return None
            
        candidates_str = "\n".join([f"{i}. {repr(c)}" for i, c in enumerate(candidates)])
        
        evolution_context = ""
        if layer_evolution:
            evolution_context = f"\n\n[Internal Layer Evolution]\n{layer_evolution}\n"
            
        explanation_context = ""
        if internal_explanation:
            explanation_context = f"\n\n[Model's Self-Explanation]\nThe model explains why it is considering these tokens:\n{internal_explanation}\n"

        prompt = f"""You are an expert math tutor helping a student model.
                    The student is solving this problem:
                    {question}

                    The student has written this so far:
                    {generated_so_far}
                    {evolution_context}{explanation_context}
                    The student is unsure about the next step. Here are the candidate next tokens:
                    {candidates_str}

                    Which token is the most logically correct continuation to solve the problem?
                    Return ONLY the number (index) of the best token (e.g., 0, 1, 2...)."""

        max_retries = 5
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=5
                )
                content = response.choices[0].message.content.strip()
                # Extract number
                match = re.search(r"\d+", content)
                if match:
                    idx = int(match.group(0))
                    if 0 <= idx < len(candidates):
                        return idx
                # If successful but no valid index found, maybe break or return None? 
                # For now, let's assume if it returns text without number, it's a failure of the model, not rate limit.
                break 
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"LLM Judge Rate Limit (429). Retrying in {delay:.2f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"LLM Judge Error: {e}")
                    break
            
        return None

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
    def generate_self_explanation(self, tokenizer, context_text: str, top_tokens_str: str, layer_idx: int) -> str:
        """
        Generate a self-explanation for the current state using the model itself.
        """
        system_msg = "You are an expert in interpreting language model internal states. Analyze the provided context and the current layer's token predictions to explain the model's reasoning."
        user_msg = (
            f"Context: {context_text}\n\n"
            f"At the current step, the model's internal state at Layer {layer_idx} is most strongly predicting these tokens: {top_tokens_str}.\n"
            "Explain why the model is focusing on these tokens given the context. Keep the explanation concise."
        )
        
        # Construct prompt
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            probe_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            probe_prompt = f"{system_msg}\n\n{user_msg}\n\nExplanation:"

        # Generate explanation
        # We use a separate generation call. Note: This adds computational cost.
        inputs = tokenizer(probe_prompt, return_tensors="pt").to(self.device)
        
        # Ensure we don't track gradients or hooks for this side-generation
        with torch.no_grad():
            # Temporarily disable our recorder hooks if they are active globally (though they are usually context managed)
            # The outer loop has `with ctx_manager`. 
            # If we call self.generate, it might trigger hooks again if we are not careful.
            # However, RethinkLlamaForCausalLM.forward calls super().forward.
            # The hooks are attached to the model instance.
            # We should probably detach hooks or use a clean context.
            # But `self._recorder` is managed via context manager. 
            # If we are inside `with ctx_manager`, hooks are active.
            # We don't want to record traces for the explanation generation itself.
            
            # Hack: Temporarily detach recorder
            original_storage = self._recorder.storage
            self._recorder.storage = None # Disable recording
            
            try:
                probe_output = super(RethinkLlamaForCausalLM, self).generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            finally:
                self._recorder.storage = original_storage # Restore

        generated_text = tokenizer.decode(probe_output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated_text.strip()

    @torch.no_grad()
    def generate_with_simulation(
        self,
        tokenizer,
        prompt: str,
        generation_kwargs: Optional[dict] = None,
        sos_threshold: float = 0.5,
        mid_layer_idx: int = 15,
        reference_layer_idx: int = 31,
        max_interventions: int = 5,
        llm_user: Optional[LLMUser] = None
    ) -> TracePack:
        """
        Run generation with Post-hoc Critique & Rethink Loop.
        1. Generate full answer.
        2. Judge critiques.
        3. If error, rewind to error step, analyze (SOS/LogitLens), and intervene.
        4. Repeat.
        """
        generation_kwargs = generation_kwargs or {"max_new_tokens": 128}
        device = self.device
        
        # Initial Generation
        current_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        intervention_count = 0
        total_tokens = 0
        evolution_history = []
        
        # Max correction turns
        max_turns = 3
        
        final_generated_ids = current_input_ids
        
        for turn in range(max_turns + 1):
            # 1. Generate (Draft)
            # We use standard generate, but we need to handle the fact that we might be continuing
            # Calculate how many new tokens allowed
            current_len = current_input_ids.shape[1]
            max_new = generation_kwargs.get("max_new_tokens", 128)
            
            if turn > 0:
                print(f"  [Turn {turn}] Resuming generation from: {tokenizer.decode(current_input_ids[0][-10:], skip_special_tokens=True)}...")
            
            # Disable recorder for the draft pass to save memory/time
            # We only need recorder for the analysis step
            original_storage = self._recorder.storage
            self._recorder.storage = None 
            
            try:
                outputs = super(RethinkLlamaForCausalLM, self).generate(
                    input_ids=current_input_ids,
                    max_new_tokens=max_new,
                    do_sample=generation_kwargs.get("do_sample", False),
                    temperature=generation_kwargs.get("temperature", 1.0),
                    pad_token_id=tokenizer.eos_token_id
                )
            finally:
                self._recorder.storage = original_storage
            
            final_generated_ids = outputs
            full_text = tokenizer.decode(final_generated_ids[0][current_input_ids.shape[1]:], skip_special_tokens=True)
            full_response = tokenizer.decode(final_generated_ids[0], skip_special_tokens=True)
            
            # Update total tokens count
            total_tokens += (final_generated_ids.shape[1] - current_input_ids.shape[1])

            # If this is the last turn, stop
            if turn == max_turns:
                break

            # 2. Critique
            if not llm_user or not llm_user.client:
                break # No judge, standard generation only
                
            critique = llm_user.critique_answer(prompt, full_text)
            if critique['correct']:
                print("  [Judge] Answer is correct.")
                break
            
            print(f"  [Judge] Error detected. Last correct segment: \"{critique['last_correct_segment'][-50:]}...\"")
            
            # 3. Locate Intervention Point
            # Heuristic: Tokenize the last_correct_segment and find the index
            # Note: This is approximate. 
            # We assume the prompt is fixed. We look at the generated part.
            last_correct_text = critique['last_correct_segment']
            
            # Re-tokenize to find length. 
            # Warning: Tokenization is not always additive. 
            # Ideally we should match the token sequence.
            # Let's try to match the prefix of the generated ids.
            
            # Full sequence including prompt
            full_ids = final_generated_ids[0]
            prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            
            # We want to find where `last_correct_text` ends in `full_text`
            # Simple approach: Tokenize `last_correct_text` and use its length as offset from prompt
            # This assumes `last_correct_text` starts from the beginning of generation
            
            # Handle case where Judge quotes the whole thing including prompt (unlikely given prompt instructions but possible)
            # The prompt instructions say "quote the exact text of the answer".
            
            correct_ids = tokenizer(last_correct_text, add_special_tokens=False).input_ids
            intervention_idx = prompt_len + len(correct_ids)
            
            # Boundary checks
            if intervention_idx >= len(full_ids):
                # Judge said everything is correct? But status was INCORRECT.
                # Maybe it means "incomplete"?
                # We intervene at the very end (replace EOS or append)
                intervention_idx = len(full_ids) - 1 
            
            if intervention_idx < prompt_len:
                intervention_idx = prompt_len # Should not happen
                
            # 4. Analyze at Intervention Point (The "Rethink" Step)
            # We need to run a forward pass up to `intervention_idx` to get the state 
            # that produced the *next* token (the first wrong one).
            
            # Context is everything up to intervention_idx
            context_ids = full_ids[:intervention_idx].unsqueeze(0)
            
            # Run forward pass with recording enabled
            # We only care about the last step
            recorder_ctx = self._recorder.attach(self)
            with recorder_ctx:
                with torch.no_grad():
                    outputs = super(RethinkLlamaForCausalLM, self).forward(
                        input_ids=context_ids,
                        return_dict=True
                    )
            
            # Extract Analysis Info
            next_token_logits = outputs.logits[:, -1, :]
            current_states = {}
            if self._recorder.storage:
                for l, states in self._recorder.storage.items():
                    if states:
                        current_states[l] = states[-1]
            
            # Create Analyzer
            temp_recorder = TokenRecorder(0, 0, "", 0.0, 0.0, current_states)
            analyzer = TokenAnalysis(temp_recorder, self, tokenizer)
            
            # Get Candidates
            probs = torch.softmax(next_token_logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 5, dim=-1)
            candidates = [tokenizer.decode([idx.item()]) for idx in topk_indices[0]]
            
            # Get Evolution & Explanation
            evolution_data = analyzer.analyze_semantic_evolution(k=3)
            evolution_str_lines = []
            for step_data in evolution_data:
                l = step_data['layer']
                if l % 4 == 0 or l > 20:
                    top_k_str = ", ".join([f"'{t[0]}'({t[1]:.2f})" for t in step_data['top_k']])
                    evolution_str_lines.append(f"Layer {l}: {top_k_str}")
            evolution_str = "\n".join(evolution_str_lines)
            
            # Self-Explanation
            mid_layer_tokens = analyzer._get_analysis(mid_layer_idx).decode(k=5)
            mid_tokens_str = ", ".join([f"'{t[0]}'({t[1]:.2f})" for t in mid_layer_tokens])
            internal_explanation = self.generate_self_explanation(
                tokenizer, 
                tokenizer.decode(context_ids[0], skip_special_tokens=True), 
                mid_tokens_str, 
                mid_layer_idx
            )
            
            # Save history
            evolution_history.append({
                "turn": turn,
                "step": intervention_idx,
                "evolution": evolution_data,
                "explanation": internal_explanation,
                "critique": critique
            })
            
            # 5. Select New Token
            generated_so_far = tokenizer.decode(context_ids[0], skip_special_tokens=True)
            # Remove prompt from generated_so_far for cleaner display to Judge
            generated_only = generated_so_far[len(prompt):]
            
            best_idx = llm_user.select_best_token(
                prompt, 
                generated_only, 
                candidates, 
                layer_evolution=evolution_str,
                internal_explanation=internal_explanation
            )
            
            if best_idx is not None:
                chosen_token_id = topk_indices[0][best_idx]
                print(f"  [Rethink] Intervening at step {intervention_idx}. Replacing with '{candidates[best_idx]}'")
                
                # 6. Rewind & Force
                # New input is context + chosen token
                current_input_ids = torch.cat([
                    context_ids, 
                    chosen_token_id.unsqueeze(0).unsqueeze(0)
                ], dim=-1)
                
                intervention_count += 1
            else:
                print("  [Rethink] Judge could not select a better token. Stopping intervention.")
                break

        # Final TracePack (Mocking the structure for compatibility)
        # We don't have per-token logs for the whole sequence in this mode, 
        # but we have the final text and intervention history.
        return TracePack(
            token_logprobs=[], # Empty for now as we didn't track every token in draft mode
            hidden_states={},
            extra={
                "prompt_ids": current_input_ids.cpu(),
                "intervention_count": intervention_count,
                "total_tokens": total_tokens,
                "evolution_history": evolution_history
            },
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--sos-threshold", type=float, default=0.3)
    parser.add_argument("--output-file", type=str, default="rethink_simulation_results.json")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API Key for LLM Judge")
    parser.add_argument("--api-base", type=str, default=None, help="OpenAI API Base URL")
    parser.add_argument("--judge-model", type=str, default="gpt-4", help="Model name for LLM Judge")
    parser.add_argument("--mid-layer", type=int, default=15, help="Layer index for internal thought (hypothesis)")
    parser.add_argument("--ref-layer", type=int, default=31, help="Layer index for final output (reference)")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples to run")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    
    # Initialize LLM User
    llm_user = LLMUser(api_key=args.api_key, base_url=args.api_base, model=args.judge_model)
    if llm_user.client:
        print(f"Using LLM Judge: {args.judge_model}")
    else:
        print("No API Key provided. Using heuristic intervention (Temperature Scaling).")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    
    # Initialize SimulationLlama
    # We need to pass instrumentation config
    # Capture mid, ref, and potentially others if needed.
    # Ensure we capture the layers we want to analyze.
    layers_to_capture = sorted(list(set([args.mid_layer, args.ref_layer])))
    instr_cfg = InstrumentationConfig(layers_to_capture=layers_to_capture) 
    
    model = SimulationLlama.from_pretrained(
        args.model_path, 
        config=config, 
        instrumentation_cfg=instr_cfg,
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    # Load Dataset
    dataset_path = os.path.join("dataset", args.dataset)
    if os.path.exists(dataset_path):
        print(f"Loading local dataset from {dataset_path}...")
        ds = load_from_disk(dataset_path)
        if "test" in ds:
            ds = ds["test"]
        elif "validation" in ds:
            ds = ds["validation"]
        # Slice for quick testing
        ds = ds.select(range(min(args.num_examples, len(ds))))
    elif args.dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=f"test[:{args.num_examples}]")
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
            mid_layer_idx=args.mid_layer,
            reference_layer_idx=args.ref_layer,
            llm_user=llm_user
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
            "total_tokens": trace.extra['total_tokens'],
            "evolution_history": trace.extra.get("evolution_history", [])
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
