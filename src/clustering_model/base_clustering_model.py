from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
import numpy as np


class BaseClusteringModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def perform_clustering(self, X: np.ndarray, umap_embeddings: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        raise NotImplementedError