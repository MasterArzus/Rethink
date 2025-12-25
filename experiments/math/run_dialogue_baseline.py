import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import argparse
import torch
import json
import re
import os
import time
import random
from typing import Optional, List, Dict, Any
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoConfig
from rethink.engine.llama import RethinkLlamaForCausalLM

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

    def critique_and_refine(self, question: str, current_answer: str, history: List[Dict[str, str]]) -> Optional[str]:
        """
        Simulate a Black-box User who only sees the text output.
        Returns a natural language prompt to correct the model, or None if satisfied.
        """
        if not self.client:
            return None
            
        # Construct conversation history for the Judge
        messages = [
            {"role": "system", "content": "You are an expert math tutor. You are interacting with a student (an AI model). Your goal is to guide the student to the correct answer using natural language hints. Do not give the answer directly, but point out logical errors. If the student's answer is correct, output 'CORRECT'."}
        ]
        
        # Add history
        for turn in history:
            messages.append({"role": "user", "content": turn['user']}) # This is actually the 'student' output in this context? No, wait.
            # Let's reframe: The Judge is the User. The Model is the Assistant.
            # But here we are simulating the User.
            # So the 'messages' sent to the Judge should represent the context the User sees.
            pass

        # Simpler Prompt for the Judge acting as User
        prompt = f"""Problem: {question}

Student's Current Answer:
{current_answer}

Is the student's answer correct?
- If YES, reply exactly "CORRECT".
- If NO, provide a short, helpful hint to guide the student to fix the error. Do not give the final number directly.
"""
        
        max_retries = 5
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=100
                )
                content = response.choices[0].message.content.strip()
                if "CORRECT" in content:
                    return None
                return content
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

def run_dialogue_loop(model, tokenizer, question, ground_truth, llm_user, max_turns=3):
    current_prompt = question
    history = []
    
    for turn in range(max_turns):
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
        
        print(f"Turn {turn} Model Output: {generated_text[:50]}...")
        
        # Check correctness (Oracle check for experiment stats, not visible to Judge)
        # Actually, for the baseline to be fair, the Judge needs to know if it's right or wrong to give feedback?
        # In real life, the user knows the answer (e.g. teacher) or verifies it.
        # We assume the Judge is an expert.
        
        # Ask Judge (Simulated User)
        feedback = llm_user.critique_and_refine(question, generated_text, history)
        
        if feedback is None: # Judge says CORRECT
            return {
                "final_text": generated_text,
                "success": True, # We trust the Judge? Or should we verify with GT?
                # Let's verify with GT for the final stat
                "turns": turn + 1
            }
            
        print(f"Turn {turn} User Feedback: {feedback}")
        
        # Append to prompt
        current_prompt += f"\n{generated_text}\nUser: {feedback}\nAssistant:"
        history.append({"model": generated_text, "user": feedback})
        
    return {
        "final_text": generated_text,
        "success": False,
        "turns": max_turns
    }

def extract_answer(text):
    match = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if match: return match.group(1)
    matches = re.findall(r"-?\d+\.?\d*", text)
    if matches: return matches[-1]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--output-file", type=str, default="dialogue_baseline_results.json")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = RethinkLlamaForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16)
    
    llm_user = LLMUser(api_key=args.api_key, base_url=args.api_base, model=args.judge_model)
    
    # Load Dataset
    dataset_path = os.path.join("dataset", args.dataset)
    if os.path.exists(dataset_path):
        ds = load_from_disk(dataset_path)
        if "test" in ds: ds = ds["test"]
        ds = ds.select(range(min(10, len(ds))))
    else:
        ds = load_dataset("gsm8k", "main", split="test[:10]")

    results = []
    for i, example in enumerate(ds):
        print(f"\n--- Example {i} ---")
        res = run_dialogue_loop(model, tokenizer, example['question'], example['answer'], llm_user)
        
        # Final Ground Truth Check
        pred = extract_answer(res['final_text'])
        gt = extract_answer(example['answer'])
        success = (pred == gt) if (pred and gt) else False
        res['success'] = success # Override with real truth
        
        results.append(res)
        print(f"Success: {success}")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
