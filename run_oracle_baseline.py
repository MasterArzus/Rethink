
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig
from rethink.engine.llama import RethinkLlamaForCausalLM
from rethink.utils.config import RethinkConfig, InstrumentationConfig

class OracleJudge:
    def __init__(self, ground_truth):
        self.ground_truth = ground_truth
    
    def check_trace(self, trace_text):
        """
        Returns the index of the first error token/step, or -1 if correct.
        This is a placeholder. In a real scenario, this would use a stronger model 
        or symbolic solver to verify the reasoning chain.
        """
        # Placeholder: Just check if the final answer matches
        # If we wanted step-by-step, we'd need to parse the trace.
        return -1

def run_oracle_loop(model, tokenizer, question, ground_truth, max_retries=3):
    current_prompt = question
    
    for attempt in range(max_retries + 1):
        # Generate
        inputs = tokenizer(current_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=True,
            temperature=0.7
        )
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        full_text = current_prompt + generated_text
        
        print(f"Attempt {attempt}: {generated_text[:50]}...")
        
        # Judge
        judge = OracleJudge(ground_truth)
        error_idx = judge.check_trace(generated_text)
        
        if error_idx == -1:
            print("Judge: Correct (or no error detected).")
            return generated_text
        else:
            print(f"Judge: Error detected at index {error_idx}.")
            # Truncate and prompt
            # For simplicity, let's say we truncate at the error and add feedback
            # This requires mapping char index to token index or just string manipulation
            truncated_text = generated_text[:error_idx]
            feedback = "\n[System: You made a mistake in the previous step. Please correct it.]\n"
            current_prompt = current_prompt + truncated_text + feedback
            
    return generated_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
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
        ds = load_dataset("gsm8k", "main", split="test[:5]")
    else:
        raise ValueError("Dataset not supported")
        
    for example in ds:
        question = example['question']
        answer = example['answer']
        print(f"\nQuestion: {question}")
        
        final_output = run_oracle_loop(model, tokenizer, question, answer)
        print("Final Output:", final_output)

if __name__ == "__main__":
    main()
