from abc import ABC, abstractmethod
import torch
import numpy as np


class BaseDimensionalityReductionModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def reduce_dimensionality(self, X: torch.Tensor) -> np.ndarray:
        raise NotImplementedError
