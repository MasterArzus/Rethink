from typing import Tuple, Optional, Any, List
import torch

class TokenRecorder:
    '''Record a token during a inference'''
    def __init__(self, idx: int, step: int, token: str, prob: float, statelist: Tuple[Any, ...]):
        self.idx = idx # token idx
        self.step = step # step to generate this token
        self.token = token # detokenized token
        self.prob = prob # probability to generate this word
        self.state = statelist # tuple to store all hidden states to generate this word
