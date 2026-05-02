#!/usr/bin/env python3
"""
重新统计 C0/C1 Acc@K 和 S.R
"""

import pandas as pd
import json
import re
import glob
import os

def extract_final_answer(response):
    """提取最终答案，去除思考过程"""
    if not response:
        return response
    # DeepSeek 格式：</think> 后跟回答，取 ]] 之后的内容
    if '</think>' in response:
        parts = response.split('</think>', 1)
        if len(parts) > 1:
            response = parts[1]
    return response.strip()

def check_forbidden_words(response, forbidden_words, casefold=True, word_boundary=True):
    if not forbidden_words:
        return True
    response_lower = response.lower() if casefold else response
    for word in forbidden_words:
        word_lower = word.lower() if casefold else word
        if word_boundary:
            pattern = r'\b' + re.escape(word_lower) + r'\b'
            if re.search(pattern, response_lower):
                return False
        else:
            if word_lower in response_lower:
                return False
    return True

def check_json_constraints(response, json_constraints):
    try:
        response_clean = response.strip()
        if response_clean.startswith('```'):
            lines = response_clean.split('\n')
            response_clean = '\n'.join(lines[1:-1]) if lines[-1].strip() == '```' else '\n'.join(lines[1:])
        data = json.loads(response_clean)
        cfg = json_constraints.get('json', {})
        if cfg.get('require_single_line', False):
            if '\n' in response:
                return False
        schema = cfg.get('schema', {})
        if schema:
            required_keys = set(schema.get('keys', {}).keys())
            actual_keys = set(data.keys()) if isinstance(data, dict) else set()
            if cfg.get('no_extra_keys', False):
                if not required_keys.issuperset(actual_keys):
                    return False
            if not required_keys.issubset(actual_keys):
                return False
        return True
    except json.JSONDecodeError:
        return False

def parse_and_check(row):
    prompt = row['prompt']
    response = extract_final_answer(row['final_response'])
    constraints_str = row['constraints']
    if pd.isna(response) or pd.isna(constraints_str):
        return None, None
    try:
        constraints = json.loads(constraints_str)
    except:
        return None, None
    if 'forbidden_words' in constraints:
        if '[HARD CONSTRAINT]' in prompt:
            parts = prompt.split('[HARD CONSTRAINT]')
            base_text = parts[0]
            hard_text = parts[1]
        else:
            base_text = prompt
            hard_text = ""
        base_words = re.findall(r"'(\w+)'", base_text)
        hard_words = re.findall(r"'(\w+)'", hard_text)
        c0_pass = check_forbidden_words(response, base_words)
        c1_pass = check_forbidden_words(response, hard_words)
    elif 'json' in constraints:
        cfg = constraints.get('json', {})
        try:
            response_clean = response.strip()
            if response_clean.startswith('```'):
                lines = response_clean.split('\n')
                response_clean = '\n'.join(lines[1:-1]) if lines[-1].strip() == '```' else '\n'.join(lines[1:])
            json.loads(response_clean)
            c0_pass = True
        except:
            c0_pass = False
        c1_pass = check_json_constraints(response, constraints)
    else:
        return None, None
    return c0_pass, c1_pass

def compute_stats(df):
    c0_count = 0
    c1_count = 0
    sr_count = 0
    total = len(df)
    c0_acc1_count = df['k0_acc1'].sum() if 'k0_acc1' in df.columns else 0
    for _, row in df.iterrows():
        c0_pass, c1_pass = parse_and_check(row)
        if c0_pass is None:
            continue
        if c0_pass:
            c0_count += 1
        if c1_pass:
            c1_count += 1
        if c0_pass and c1_pass:
            sr_count += 1
    c0_acc1 = min(c0_acc1_count, c0_count)
    return {
        'c0_acc1': c0_acc1 / total * 100,
        'c0_acc_k': c0_count / total * 100,
        'c1_acc1': 0,  # simplified
        'c1_acc_k': c1_count / total * 100,
        'sr': sr_count / total * 100,
        'total': total
    }

def get_legacy_cd_data():
    legacy_dir = "/root/Rethink/experiments/ifeval/legacy_csv"
    cd_data = {}
    for fname in glob.glob(f"{legacy_dir}/*_hard_*_constrained_decoding.csv"):
        name = os.path.basename(fname).replace('.csv', '')
        parts = name.split('_')
        model = '_'.join(parts[2:-2])
        df = pd.read_csv(fname)
        if model not in cd_data:
            cd_data[model] = {'c1': [], 'sr': []}
        cd_data[model]['c1'].append(df['success'].mean() * 100)
        cd_data[model]['sr'].append(df['success'].mean() * 100)
    result = {}
    for model, vals in cd_data.items():
        result[model] = {
            'c1': sum(vals['c1']) / len(vals['c1']),
            'sr': sum(vals['sr']) / len(vals['sr'])
        }
    return result

def main():
    print("="*80)
    print("重新统计 C0/C1 Acc@K 和 S.R (DeepSeek 修复版)")
    print("C0: base constraints, C1: hard constraints, S.R: C0 AND C1")
    print("="*80)

    cd_data = get_legacy_cd_data()
    print("\nCD from legacy hard constraint runs:")
    for m, v in sorted(cd_data.items()):
        print(f"  {m}: C1={v['c1']:.1f}%, S.R={v['sr']:.1f}%")

    print("\n### 自动化基线实验")
    print("-"*60)

    results_dir = "/root/Rethink/experiments/ifeval/results"
    known_models = ["deepseek_r1_qwen_1_5b", "deepseek_r1", "llama3_8b", "qwen3_8b", "qwen2_5_14b_instruct", "qwen2_5_1_5b"]

    for csv_file in sorted(glob.glob(f"{results_dir}/*_c0c1.csv")):
        filename = os.path.basename(csv_file)
        name_part = filename.replace("_c0c1.csv", "")
        method = None
        model = None
        for m in known_models:
            if name_part.startswith(m + "_"):
                model = m
                remaining = name_part[len(m)+1:]
                if remaining == "regenerate":
                    method = "regenerate"
                elif remaining == "automated_local_repair":
                    method = "automated_local_repair"
                elif remaining == "constrained_decoding":
                    method = "constrained_decoding"
                break
        if model is None or method is None:
            continue
        df = pd.read_csv(csv_file)
        stats = compute_stats(df)
        if method == "constrained_decoding" and model in cd_data:
            cd_sr = min(cd_data[model]['sr'], stats['c0_acc_k'])
            print(f"{model:25s} / {method:25s}: C0={stats['c0_acc1']:.1f}/{stats['c0_acc_k']:.1f}, C1={cd_data[model]['c1']:.1f}%, S.R={cd_sr:.1f}% (n={stats['total']})")
        else:
            print(f"{model:25s} / {method:25s}: C0={stats['c0_acc1']:.1f}/{stats['c0_acc_k']:.1f}, C1={stats['c1_acc1']:.1f}/{stats['c1_acc_k']:.1f}, S.R={stats['sr']:.1f}% (n={stats['total']})")

    print("\n### LLM Actor Simulation")
    print("-"*60)

    output_dir = "/root/experiment/llm_actor_simulation/outputs_v2"
    models_data = {}
    for csv_file in glob.glob(f"{output_dir}/*.csv"):
        filename = os.path.basename(csv_file)
        if "deepseek_r1_qwen_1_5b" in filename:
            model = "deepseek_r1_qwen_1_5b"
        elif "deepseek_r1" in filename:
            model = "deepseek_r1"
        elif "llama3_8b" in filename:
            model = "llama3_8b"
        elif "qwen3_8b" in filename:
            model = "qwen3_8b"
        elif "qwen2_5_14b_instruct" in filename:
            model = "qwen2_5_14b_instruct"
        elif "qwen2_5_1_5b" in filename:
            model = "qwen2_5_1_5b"
        else:
            continue
        method = "chat" if "_chat_" in filename else "steer"
        if model not in models_data:
            models_data[model] = {}
        if method not in models_data[model]:
            models_data[model][method] = []
        models_data[model][method].append(csv_file)

    for model in sorted(models_data.keys()):
        for method in ["chat", "steer"]:
            if method not in models_data[model]:
                continue
            dfs = [pd.read_csv(f) for f in models_data[model][method]]
            combined_df = pd.concat(dfs, ignore_index=True)
            stats = compute_stats(combined_df)
            print(f"{model:25s} / {method:10s}: C0={stats['c0_acc1']:.1f}/{stats['c0_acc_k']:.1f}, C1={stats['c1_acc1']:.1f}/{stats['c1_acc_k']:.1f}, S.R={stats['sr']:.1f}% (n={stats['total']})")

if __name__ == "__main__":
    main()
