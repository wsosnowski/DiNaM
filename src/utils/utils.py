import os

import numpy as np
import torch
from numpy.f2py.auxfuncs import throw_error
from sklearn.metrics import silhouette_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def convert_to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
    return None


def compute_semantic_similarity(embeddings1, embeddings2):
    """
    Compute the semantic similarity matrix between two sets of embeddings.

    Args:
        embeddings1 (torch.Tensor): First set of embeddings (n x d tensor).
        embeddings2 (torch.Tensor): Second set of embeddings (m x d tensor).

    Returns:
        np.array: Similarity matrix.
    """
    # Ensure embeddings are 2D arrays
    if len(embeddings1.shape) == 1:
        embeddings1 = embeddings1.unsqueeze(0)
    if len(embeddings2.shape) == 1:
        embeddings2 = embeddings2.unsqueeze(0)

    # Convert to numpy for cosine similarity computation
    embeddings1_np = embeddings1.cpu().numpy()
    embeddings2_np = embeddings2.cpu().numpy()

    similarity_matrix = cosine_similarity(embeddings1_np, embeddings2_np)
    return similarity_matrix


def weighted_chamfer_distance(embeddings_g, embeddings_p):
    """
    Calculate the Chamfer distance between two sets of embeddings.

    Args:
        embeddings_g: Ground truth embeddings (n x d tensor, where n is the number of ground truth embeddings).
        embeddings_p: Predicted embeddings (m x d tensor, where m is the number of predicted embeddings).

    Returns:
        The Chamfer distance between the two sets.
    """
    similarity_gt_to_pred = compute_semantic_similarity(embeddings_g, embeddings_p)
    similarity_pred_to_gt = compute_semantic_similarity(embeddings_p, embeddings_g)

    # Calculate scores
    score_gt_to_pred = np.mean(np.max(similarity_gt_to_pred, axis=1))
    score_pred_to_gt = np.mean(np.max(similarity_pred_to_gt, axis=1))

    # Compute final score
    weight_gt = len(similarity_gt_to_pred)
    weight_pred = len(similarity_pred_to_gt)
    total_weight = weight_gt + weight_pred

    # Compute the weighted average
    wsd_score = (weight_gt * score_gt_to_pred + weight_pred * score_pred_to_gt) / total_weight

    return wsd_score


def save_embeddings(embeddings, embedding_path):
    """Saves the embeddings to a file"""
    torch.save(embeddings, embedding_path)
    print(f"Embeddings saved to {embedding_path}")


def load_embeddings(embedding_path):
    """Loads embeddings from a file"""
    if os.path.exists(embedding_path):
        embeddings = torch.load(embedding_path)
        print(f"Embeddings loaded from {embedding_path}")
        return embeddings
    else:
        print(f"No embeddings file found at {embedding_path}")
        return None


def calculate_silhouette_score(reduced_embeddings, labels):
    """Calculate the Silhouette score for the given data and cluster labels."""
    # Check for valid number of clusters
    if len(set(labels)) <= 1:  # Silhouette score is undefined for 1 or fewer clusters
        return -1  # Return -1 for invalid cases
    valid_indices = [i for i, label in enumerate(labels) if label != -1]
    filtered_embeddings = reduced_embeddings[valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]

    return silhouette_score(filtered_embeddings, filtered_labels)


def calculate_false_class_metrics(ground_truth, predictions):
    """
    Calculate evaluation metrics for the false class.
    """
    try:
        tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    except ValueError as e:
        raise ValueError("Check your input data: Ensure ground truth and predictions are aligned.") from e

    precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, f1
