import torch
import os
import json
from tqdm import tqdm
import numpy as np
import re 


def calc_target_probs(prompt: str, target_text: str, model, tokenizer, eps: float = 1e-12, device: str = "cuda") -> list:
    """
    计算模型在给定 prompt + target_text 下，
    target 每个 token 的条件概率 P(a_i | prompt, a_<i)
    避免浮点下溢导致的 prob=0
    """
    # Tokenize
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    target_ids = tokenizer(target_text, return_tensors="pt").input_ids.to(device)[0]

    generated_ids = prompt_ids.clone()
    token_info = []
    token_probs = []

    for idx, token_id in enumerate(target_ids):
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits[:, -1, :]
        # 数值稳定 softmax
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.nn.functional.softmax(logits, dim=-1)

        prob = probs[0, token_id].item()
        prob = max(prob, eps)  # 避免出现 0

        token_str = tokenizer.decode([token_id])

        # 暂时只要概率数值
        token_probs.append(prob)  #[0.232, 0.221, ]
        
        # token_info.append({
        #     "step": idx + 1,
        #     "token": token_str,
        #     "token_id": int(token_id),
        #     "prob": prob
        # })

        # 固定当前 token，继续预测下一个
        next_token = token_id.unsqueeze(0).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    decoded_target = tokenizer.decode(target_ids, skip_special_tokens=True)

    # # 保存到 JSON 文件
    # data = {
    #     "prompt": prompt,
    #     "target": decoded_target,
    #     "token_probs": token_probs
    # }
    # with open(save_path, "w", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=2)
    # print(f"save to {save_path}")
    # print(f"token_probs: {token_probs}")
    return token_probs


def cal_Entropy(token_probs: list) -> float:
    """
    计算sample的熵值
    :param file_path: 文件路径
    :return: 熵值
    """
    # 提取 token_probs 列表

    # 构造 DataFrame，只保留 step 和 prob 两列
    Entropy = -np.log2(token_probs).sum() / len(token_probs)
    return Entropy


