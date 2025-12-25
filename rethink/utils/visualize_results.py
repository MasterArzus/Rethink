import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import argparse
import os
import pandas as pd

def plot_logit_lens(input_file, output_dir):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Find an example with interventions
    target_example = None
    for ex in data:
        if ex.get('interventions', 0) > 0 and 'evolution_history' in ex:
            target_example = ex
            break
            
    if not target_example:
        print("No examples with interventions found to visualize.")
        return

    print(f"Visualizing example: {target_example['question'][:50]}...")
    
    # Visualize the first intervention
    history = target_example['evolution_history'][0]
    evolution = history['evolution'] # List of {layer, top_k, target_prob...}
    
    layers = [e['layer'] for e in evolution]
    target_probs = [e['target_prob'] for e in evolution]
    entropies = [e['entropy'] for e in evolution]
    
    # Plot 1: Probability of the chosen token across layers
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=layers, y=target_probs, marker='o', label='Target Token Prob')
    plt.xlabel('Layer Index')
    plt.ylabel('Probability')
    plt.title(f'Logit Lens: Target Token Probability Evolution (Step {history["step"]})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'logit_lens_prob.png'))
    plt.close()
    
    # Plot 2: Entropy Evolution
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=layers, y=entropies, marker='s', color='orange', label='Entropy')
    plt.xlabel('Layer Index')
    plt.ylabel('Entropy')
    plt.title(f'Internal Uncertainty Evolution (Step {history["step"]})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'logit_lens_entropy.png'))
    plt.close()
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="plots")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    plot_logit_lens(args.input_file, args.output_dir)
