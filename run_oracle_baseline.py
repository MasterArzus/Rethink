
import argparse
import torch
import json
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig
from rethink.engine.llama import RethinkLlamaForCausalLM
from rethink.utils.config import RethinkConfig, InstrumentationConfig

def extract_answer(text):
    """Extract the numerical answer from GSM8K text."""
    # Look for #### Number
    match = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if match:
        return match.group(1)
    # Fallback: look for last number
    matches = re.findall(r"-?\d+\.?\d*", text)
    if matches:
        return matches[-1]
    return None

class OracleJudge:
    def __init__(self, ground_truth):
        self.ground_truth = extract_answer(ground_truth)
    
    def check_correctness(self, generated_text):
        pred = extract_answer(generated_text)
        if pred and self.ground_truth:
            return float(pred) == float(self.ground_truth)
        return False

    def check_trace(self, trace_text):
        """
        Returns the index of the first error token/step, or -1 if correct.
        For Oracle Baseline, we simulate 'Prompting with ground-truth error location'.
        Since we don't have a real step-by-step verifier, we'll simulate it:
        If the final answer is wrong, we assume the error happened at 50% of the generation
        (or random, or we just say 'You are wrong' which is Reflexion).
        
        The experiment description says: 'Oracle Prompting: Prompting with ground-truth error location ("Error at step t, rewrite")'.
        To simulate this without a real verifier, we can:
        1. Check final answer.
        2. If wrong, return a synthetic 'error index' (e.g., half of the new tokens).
        """
        if self.check_correctness(trace_text):
            return -1
        else:
            # Simulate finding an error halfway through the generated text
            # This is a heuristic for the simulation
            return len(trace_text) // 2

def run_standard_generation(model, tokenizer, question):
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        do_sample=True,
        temperature=0.7
    )
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text, outputs[0].shape[0] - inputs.input_ids.shape[1]

def run_oracle_loop(model, tokenizer, question, ground_truth, max_retries=3):
    current_prompt = question
    total_generated_tokens = 0
    
    judge = OracleJudge(ground_truth)
    
    for attempt in range(max_retries + 1):
        # Generate
        inputs = tokenizer(current_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=True,
            temperature=0.7
        )
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        total_generated_tokens += len(new_tokens)
        
        full_text = current_prompt + generated_text # Note: current_prompt grows
        
        print(f"Attempt {attempt}: {generated_text[:50]}...")
        
        # Judge
        if judge.check_correctness(generated_text):
            print("Judge: Correct.")
            return {
                "final_text": generated_text,
                "success": True,
                "total_tokens": total_generated_tokens,
                "attempts": attempt + 1
            }
        
        # If not correct and we have retries left
        if attempt < max_retries:
            print(f"Judge: Error detected. Retrying...")
            # Oracle Prompting Strategy: "Error at step t, rewrite"
            # We simulate this by keeping the first half of the wrong generation (simulating we found the error there)
            # and asking to fix.
            # Or simpler for 'Reflexion': "You are wrong, fix it."
            # The paper distinguishes Reflexion vs Oracle Prompting.
            # Oracle Prompting: "Prompting with ground-truth error location".
            
            # Let's simulate "Error at step t".
            # We keep 50% of the generated text, and append feedback.
            keep_len = len(generated_text) // 2
            kept_text = generated_text[:keep_len]
            feedback = "\n[System: Error detected in the reasoning above. Please rewrite from this point.]\n"
            current_prompt = current_prompt + kept_text + feedback
            
    return {
        "final_text": generated_text,
        "success": False,
        "total_tokens": total_generated_tokens,
        "attempts": max_retries + 1
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--output-file", type=str, default="oracle_baseline_results.json")
    args = parser.parse_args()
    
    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = RethinkLlamaForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load Dataset
    if args.dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test[:10]") # Increased to 10 for better stats
    else:
        raise ValueError("Dataset not supported")
        
    results = []
    
    for i, example in enumerate(ds):
        question = example['question']
        answer = example['answer']
        print(f"\n--- Example {i} ---")
        print(f"Question: {question}")
        
        # 1. Standard Generation Baseline
        print("Running Standard Generation...")
        std_text, std_tokens = run_standard_generation(model, tokenizer, question)
        std_judge = OracleJudge(answer)
        std_success = std_judge.check_correctness(std_text)
        print(f"Standard: Success={std_success}, Tokens={std_tokens}")
        
        # 2. Oracle Prompting Loop
        print("Running Oracle Prompting...")
        oracle_res = run_oracle_loop(model, tokenizer, question, answer)
        print(f"Oracle: Success={oracle_res['success']}, Tokens={oracle_res['total_tokens']}")
        
        results.append({
            "question": question,
            "ground_truth": answer,
            "standard": {
                "text": std_text,
                "success": std_success,
                "tokens": std_tokens
            },
            "oracle": oracle_res
        })

    # Calculate Metrics
    total_std_tokens = sum(r['standard']['tokens'] for r in results)
    total_oracle_tokens = sum(r['oracle']['total_tokens'] for r in results)
    std_csr = sum(1 for r in results if r['standard']['success']) / len(results)
    oracle_csr = sum(1 for r in results if r['oracle']['success']) / len(results)
    
    # Token Saving Rate (TSR) - usually Rethink vs Prompting. 
    # Here we just save the data.
    
    print("\n=== Summary ===")
    print(f"Standard CSR: {std_csr:.2%}")
    print(f"Oracle CSR: {oracle_csr:.2%}")
    print(f"Total Standard Tokens: {total_std_tokens}")
    print(f"Total Oracle Tokens: {total_oracle_tokens}")
    
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
