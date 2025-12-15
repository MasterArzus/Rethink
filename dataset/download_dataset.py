import os

# Use HF Mirror for better connectivity in some regions
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

def download_gsm8k(output_dir="dataset/gsm8k"):
    """Download GSM8K dataset and save to disk."""
    print("Downloading GSM8K dataset...")
    try:
        # Load from huggingface
        dataset = load_dataset("gsm8k", "main")
        
        # Save to disk
        save_path = os.path.join(os.getcwd(), output_dir)
        dataset.save_to_disk(save_path)
        print(f"GSM8K dataset saved to {save_path}")
    except Exception as e:
        print(f"Error downloading GSM8K: {e}")

def download_math500(output_dir="dataset/math500"):
    print("Downloading MATH-500 dataset...")
    try:
        dataset = load_dataset("HuggingFaceH4/MATH-500")
        # Map columns to question/answer
        # MATH-500 usually has 'problem' and 'solution'
        def map_columns(example):
            return {
                "question": example.get("problem", ""),
                "answer": example.get("solution", "")
            }
        
        dataset = dataset.map(map_columns)
        
        save_path = os.path.join(os.getcwd(), output_dir)
        dataset.save_to_disk(save_path)
        print(f"MATH-500 dataset saved to {save_path}")
    except Exception as e:
        print(f"Error downloading MATH-500: {e}")

def download_humaneval(output_dir="dataset/humaneval"):
    print("Downloading HumanEval dataset...")
    try:
        dataset = load_dataset("openai_humaneval")
        # HumanEval has 'prompt' and 'canonical_solution'
        def map_columns(example):
            return {
                "question": example.get("prompt", ""),
                "answer": example.get("canonical_solution", "")
            }
        
        dataset = dataset.map(map_columns)
        
        save_path = os.path.join(os.getcwd(), output_dir)
        dataset.save_to_disk(save_path)
        print(f"HumanEval dataset saved to {save_path}")
    except Exception as e:
        print(f"Error downloading HumanEval: {e}")

def download_gpqa(output_dir="dataset/gpqa"):
    print("Downloading GPQA dataset...")
    try:
        # Using gpqa_main subset
        dataset = load_dataset("idavidrein/gpqa", "gpqa_main")
        # GPQA has 'Question' and 'Correct Answer'
        def map_columns(example):
            return {
                "question": example.get("Question", ""),
                "answer": example.get("Correct Answer", "")
            }
        
        dataset = dataset.map(map_columns)
        
        save_path = os.path.join(os.getcwd(), output_dir)
        dataset.save_to_disk(save_path)
        print(f"GPQA dataset saved to {save_path}")
    except Exception as e:
        print(f"Error downloading GPQA: {e}")

def download_hh_rlhf(output_dir="dataset/hh_rlhf"):
    print("Downloading Anthropic HH-RLHF dataset...")
    try:
        dataset = load_dataset("Anthropic/hh-rlhf")
        
        def map_columns(example):
            # HH-RLHF 'chosen' field format: "\n\nHuman: ...\n\nAssistant: ..."
            # We extract the prompt (up to the last Assistant:) as 'question'
            # and the chosen response as 'answer'
            chosen = example.get("chosen", "")
            split_marker = "\n\nAssistant:"
            
            if split_marker in chosen:
                parts = chosen.rpartition(split_marker)
                # parts[0] is context, parts[1] is marker, parts[2] is response
                prompt = parts[0] + parts[1]
                response = parts[2].strip()
            else:
                prompt = chosen
                response = ""
                
            return {
                "question": prompt,
                "answer": response
            }
        
        dataset = dataset.map(map_columns)
        
        save_path = os.path.join(os.getcwd(), output_dir)
        dataset.save_to_disk(save_path)
        print(f"HH-RLHF dataset saved to {save_path}")
    except Exception as e:
        print(f"Error downloading HH-RLHF: {e}")

def download_truthful_qa(output_dir="dataset/truthful_qa"):
    print("Downloading TruthfulQA dataset...")
    try:
        # Use 'generation' task for Q&A format
        dataset = load_dataset("truthful_qa", "generation")
        
        def map_columns(example):
            return {
                "question": example.get("question", ""),
                "answer": example.get("best_answer", "")
            }
        
        dataset = dataset.map(map_columns)
        
        save_path = os.path.join(os.getcwd(), output_dir)
        dataset.save_to_disk(save_path)
        print(f"TruthfulQA dataset saved to {save_path}")
    except Exception as e:
        print(f"Error downloading TruthfulQA: {e}")

def download_stsb(output_dir="dataset/stsb"):
    print("Downloading STS-B (Semantic Textual Similarity Benchmark)...")
    try:
        # STS-B is part of the GLUE benchmark
        dataset = load_dataset("glue", "stsb")
        
        def map_columns(example):
            # STS-B has 'sentence1', 'sentence2', and 'label' (score 0-5)
            # We format this as a prompt for the LLM to evaluate similarity
            s1 = example.get("sentence1", "")
            s2 = example.get("sentence2", "")
            score = example.get("label", 0.0)
            
            prompt = (
                f"Assess the semantic similarity between the following two sentences on a scale from 0.0 to 5.0.\n\n"
                f"Sentence 1: {s1}\n"
                f"Sentence 2: {s2}\n\n"
                f"Similarity Score:"
            )
            
            return {
                "question": prompt,
                "answer": str(score)
            }
        
        dataset = dataset.map(map_columns)
        
        save_path = os.path.join(os.getcwd(), output_dir)
        dataset.save_to_disk(save_path)
        print(f"STS-B dataset saved to {save_path}")
    except Exception as e:
        print(f"Error downloading STS-B: {e}")

if __name__ == "__main__":
    # Create dataset directory if it doesn't exist
    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    download_gsm8k()
    download_math500()
    download_humaneval()
    download_gpqa()
    download_hh_rlhf()
    download_truthful_qa()
    download_stsb()
