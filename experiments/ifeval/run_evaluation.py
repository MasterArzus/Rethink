import os
import sys
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add dataset/ifeval to path to import checkers
sys.path.append("/root/Rethink/dataset/ifeval")
try:
    from checkers import get_checker
except ImportError:
    print("Error: Could not import checkers.py. Make sure /root/Rethink/dataset/ifeval/checkers.py exists.")
    sys.exit(1)

MODELS = {
    "deepseek_r1": "/root/autodl-fs/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llama3_8b": "/root/autodl-fs/LLM-Research/Meta-Llama-3.1-8B-Instruct",
    "qwen3_8b": "/root/autodl-fs/Qwen/Qwen3-8B"
}

DATASET_PATH = "/root/Rethink/dataset/ifeval/taskset_60_hard.json"

def load_dataset(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data['tasks']

def run_evaluation():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        return

    all_tasks = load_dataset(DATASET_PATH)
    
    # Split tasks
    taboo_tasks = [t for t in all_tasks if t['type'] == 'taboo']
    json_tasks = [t for t in all_tasks if t['type'] == 'json']
    
    task_groups = [
        ("taboo_hard", taboo_tasks),
        ("json_hard", json_tasks)
    ]
    
    for model_name, model_path in MODELS.items():
        print(f"Loading {model_name} from {model_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="auto", 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
            
        for group_name, tasks in task_groups:
            print(f"Processing {group_name} with {len(tasks)} tasks...")
            results = []
            
            log_file = open(f"{group_name}_{model_name}.log", "w", encoding="utf-8")
            
            for i, task in enumerate(tasks):
                print(f"[{group_name}] Processing task {i+1}/{len(tasks)} (ID: {task.get('id', 'unknown')})")
                prompt = task['prompt']
                log_file.write(f"=========\nQuestion: {prompt}\n")
                constraints = task['constraints']
                task_type = task['type']
                
                try:
                    checker = get_checker(task_type)
                except ValueError:
                    print(f"Skipping task {task['id']}: Unknown task type {task_type}")
                    continue
                
                messages = [{"role": "user", "content": prompt}]
                
                init_tokens = 0
                total_tokens = 0
                turns = 0
                
                max_turns = 5
                success = False
                
                for turn in range(max_turns):
                    turns += 1
                    print(f"    Turn {turn + 1}/{max_turns}")
                    
                    # Generate
                    try:
                        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
                    except Exception as e:
                        print(f"Error applying chat template for task {i}: {e}")
                        break

                    input_len = inputs.shape[1]
                    
                    with torch.no_grad():
                        outputs = model.generate(inputs, max_new_tokens=512, do_sample=False)
                    
                    generated_ids = outputs[0][input_len:]
                    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    gen_tokens = len(generated_ids)
                    total_tokens += gen_tokens
                    
                    if turn == 0:
                        init_tokens = gen_tokens
                        log_file.write(f"Init_answer: {response}\nInit_tokens: {gen_tokens}\n")
                    else:
                        log_file.write(f"\nRound {turn}\nPrompt: {messages[-1]['content']}\nAnswer: {response}\n")
                    
                    messages.append({"role": "assistant", "content": response})
                    
                    # Clean response for checking (remove chain of thought)
                    response_to_check = response
                    if "</think>" in response:
                        response_to_check = response.split("</think>")[-1].strip()
                    
                    # Check
                    passed, error_msg = checker.check(response_to_check, constraints)
                    log_file.write(f"If_violated: {not passed}\n")
                    
                    if passed:
                        success = True
                        break
                    
                    # Prepare correction
                    if turn < max_turns - 1:
                        if task_type == "taboo" and error_msg and "Found forbidden words:" in error_msg:
                            # Extract words
                            words = error_msg.replace("Found forbidden words:", "").strip()
                            correction = f'Don\'t use the following words in subsequent sentence "{words}"'
                        else:
                            # Fallback for JSON or other errors
                            correction = f"Your answer does not satisfy the constraints: {error_msg}. Please correct it."
                        
                        messages.append({"role": "user", "content": correction})
                
                results.append({
                    "question_index": i,
                    "init_tokens": init_tokens,
                    "correct_tokens": total_tokens,
                    "total_turn": turns
                })
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(tasks)} tasks for {group_name}")
            
            log_file.close()
            
            # Save CSV
            output_file = f"{group_name}_{model_name}.csv"
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            print(f"Saved results to {output_file}")
        
        # Unload model to free memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_evaluation()
