from abc import ABC, abstractmethod

class Engine(ABC):
    @abstractmethod
    def generate(self, prompt: str):
        pass
