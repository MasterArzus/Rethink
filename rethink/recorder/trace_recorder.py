from typing import List
from .token_recorder import TokenRecorder

class TraceRecorder:
    '''Record a inference trace during a time'''
    def __init__(self, question: str, answer: str, tokenlist: List[TokenRecorder]):
        self.question = question
        self.answer = answer
        self.tokenlist = tokenlist
