import torch
from umap import UMAP
import numpy as np

from src.dimensionality_reduction_model.base_dimensionality_reduction_model import BaseDimensionalityReductionModel


class UmapModel(BaseDimensionalityReductionModel):
    def __init__(self, n_components: int = 256, n_neighbors: int = 15):
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.model = UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=0.0,
            metric="cosine"
        )

    def reduce_dimensionality(self, X: torch.Tensor) -> np.ndarray:
        return self.model.fit_transform(X)
