from abc import ABC, abstractmethod


class BaseGenerativeModel(ABC):
    def __init__(self, api_key: str, model_name: str):
        if not api_key or not model_name:
            raise ValueError("A api_key must be provided.")
        self.api_key = api_key
        self.model_name = model_name

    @abstractmethod
    def get_response(self, prompt: str) -> str:
        pass
