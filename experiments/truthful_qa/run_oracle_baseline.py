
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import argparse
import torch
import json
import re
import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoConfig
from rethink.engine.llama import RethinkLlamaForCausalLM
from rethink.utils.config import RethinkConfig, InstrumentationConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="truthful_qa")
    parser.add_argument("--output-file", type=str, default="oracle_baseline_results.json")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples to run")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = RethinkLlamaForCausalLM.from_pretrained(
        args.model_path, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    # Load Dataset
    dataset_path = os.path.join("dataset", args.dataset)
    if os.path.exists(dataset_path):
        print(f"Loading local dataset from {dataset_path}...")
        ds = load_from_disk(dataset_path)
        if "validation" in ds:
            ds = ds["validation"]
        elif "test" in ds:
            ds = ds["test"]
        ds = ds.select(range(min(args.num_examples, len(ds))))
    elif args.dataset == "truthful_qa":
        ds = load_dataset("truthful_qa", "generation", split=f"validation[:{args.num_examples}]")
    else:
        raise ValueError("Dataset not supported")
        
    results = []
    
    for i, example in enumerate(ds):
        question = example['question']
        answer = example.get('best_answer', example.get('answer', ''))
        
        # Standard Generation
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(question):].strip()
        
        # Oracle Prompting (Not fully applicable to TruthfulQA in the same way as Math, 
        # but we can simulate "Try again, be truthful" if we knew it was wrong.
        # For now, we just run standard generation as the baseline comparison point.)
        
        results.append({
            "question": question,
            "ground_truth": answer,
            "generated_text": generated_text,
            "method": "standard_generation"
        })
        
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_file}")
