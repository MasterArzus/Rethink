import os
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

if __name__ == "__main__":
    # Create dataset directory if it doesn't exist
    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    download_gsm8k()
    download_math500()
    download_humaneval()
    download_gpqa()
