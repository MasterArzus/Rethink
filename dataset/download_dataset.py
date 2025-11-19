import os
from datasets import load_dataset

def download_gsm8k(output_dir="dataset/gsm8k"):
    """Download GSM8K dataset and save to disk."""
    print("Downloading GSM8K dataset...")
    # Load from huggingface
    dataset = load_dataset("gsm8k", "main")
    
    # Save to disk
    save_path = os.path.join(os.getcwd(), output_dir)
    dataset.save_to_disk(save_path)
    print(f"GSM8K dataset saved to {save_path}")

if __name__ == "__main__":
    download_gsm8k()
