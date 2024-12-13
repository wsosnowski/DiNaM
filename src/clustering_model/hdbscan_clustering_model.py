import collections
from typing import Tuple

import pandas as pd
import numpy as np
from hdbscan import HDBSCAN

from src.clustering_model.base_clustering_model import BaseClusteringModel


class HDBSCANClusteringModel(BaseClusteringModel):
    def __init__(self, min_cluster_size: int = 20, min_samples: int = 20):
        super().__init__()
        self.hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean', prediction_data=True,
                                     cluster_selection_method="eom")

    def _update_topic_size(self, documents: pd.DataFrame):
        """Calculate the topic sizes.

        Arguments:
            documents: Updated dataframe with documents and their corresponding IDs and newly added Topics
        """
        self.topic_sizes_ = collections.Counter(documents.Topic.values.tolist())
        self.topics_ = documents.Topic.astype(int).tolist()

    def perform_clustering(self, umap_embeddings: np.ndarray, documents: pd.DataFrame) -> Tuple[
        pd.DataFrame, np.ndarray]:
        self.hdbscan_model.fit(umap_embeddings)
        labels = self.hdbscan_model.labels_
        documents["Topic"] = labels
        self._update_topic_size(documents)

        probabilities = None
        if hasattr(self.hdbscan_model, "probabilities_"):
            probabilities = self.hdbscan_model.probabilities_

        return documents.Topic.tolist(), probabilities
