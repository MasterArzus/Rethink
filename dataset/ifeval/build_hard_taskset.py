import json
import random
import copy
import os

# 高频停用词池 (Taboo Hard Mode)
STOP_WORDS = ["the", "and", "is", "of", "to", "in", "a", "it", "that", "for"]

def harden_taskset(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    original_tasks = data['tasks']
    
    # 分离 Taboo 和 JSON 任务
    taboo_tasks = [t for t in original_tasks if t['type'] == 'taboo']
    json_tasks = [t for t in original_tasks if t['type'] == 'json']
    
    # 随机打乱以保证多样性
    random.seed(42)
    random.shuffle(taboo_tasks)
    random.shuffle(json_tasks)
    
    # 选取前 30 个进行强化
    selected_taboo = taboo_tasks[:30]
    selected_json = json_tasks[:30]
    
    hard_tasks = []

    # --- 强化 Taboo 任务 ---
    for task in selected_taboo:
        new_task = copy.deepcopy(task)
        new_task['id'] = new_task['id'] + "_hard"
        
        # 策略：随机选 1-2 个高频停用词
        num_bans = random.choice([1, 2])
        extra_bans = random.sample(STOP_WORDS, num_bans)
        
        # 1. 更新 Prompt
        extra_bans_str = ", ".join([f"'{w}'" for w in extra_bans])
        new_task['prompt'] += f"\n\n[HARD CONSTRAINT] You must also AVOID using the following common words: {extra_bans_str}."
        
        # 2. 更新 Constraints
        current_bans = new_task['constraints'].get('forbidden_words', [])
        new_task['constraints']['forbidden_words'] = current_bans + extra_bans
        
        new_task['difficulty'] = 'hard'
        hard_tasks.append(new_task)

    # --- 强化 JSON 任务 ---
    for task in selected_json:
        new_task = copy.deepcopy(task)
        new_task['id'] = new_task['id'] + "_hard"
        
        # 策略：增加 "No Newlines" (压缩 JSON) 限制
        # 这迫使模型不能输出换行符，增加了可读性难度，也容易让模型在长文本生成时出错
        new_task['prompt'] += "\n\n[HARD CONSTRAINT] The JSON must be compacted into a SINGLE line. Do not use any newlines."
        
        # 更新 Constraints (虽然 JsonChecker 目前可能不检查换行，但我们可以加上标记)
        if 'json' not in new_task['constraints']:
             new_task['constraints']['json'] = {}
        
        # 我们可以在 Checker 中增加对 no_newlines 的支持，或者仅作为 Prompt 难度
        # 这里我们在 constraints 中加一个标记，Checker 可以选择性实现
        new_task['constraints']['json']['require_single_line'] = True
        
        new_task['difficulty'] = 'hard'
        hard_tasks.append(new_task)

    # 构建新的数据集对象
    new_data = {
        "meta": {
            "name": "rethink_ifeval_hard_60",
            "version": "0.1",
            "seed": 42,
            "counts": {
                "taboo_hard": len(selected_taboo),
                "json_hard": len(selected_json),
                "total": len(hard_tasks)
            },
            "sources": data['meta']['sources']
        },
        "tasks": hard_tasks
    }
    
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"Generated {len(hard_tasks)} hard tasks (30 Taboo + 30 JSON) to {output_path}")

if __name__ == "__main__":
    harden_taskset(
        "/root/Rethink/dataset/ifeval/taskset_120.json", 
        "/root/Rethink/dataset/ifeval/taskset_60_hard.json"
    )
