from abc import ABC, abstractmethod
import torch

class BaseEmbeddingModel(ABC):
    def __init__(self, model_name: str):
        if not model_name:
            raise ValueError("A model_name must be provided.")
        self.model_name = model_name

    @abstractmethod
    def encode_texts(self, texts: list, batch_size: int) -> torch.Tensor:
        """
        Encode a list of texts into embeddings.

        Args:
            texts (list): A list of text strings to encode.
            batch_size (int): The number of texts to process in each batch.

        Returns:
            torch.Tensor: A tensor containing the embeddings for the input texts.
        """
        pass