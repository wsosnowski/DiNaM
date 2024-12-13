import argparse
import os
from src.embedding_model.salesforce_embedding_model import SalesForceEmbeddingModel
from src.utils.utils import weighted_chamfer_distance


def evaluate_narratives(predicted_narratives_path, ground_truth_narratives_path, embedding_model_name):
    # Check if the predicted narratives file exists
    if not os.path.exists(predicted_narratives_path):
        raise FileNotFoundError(
            f"Predicted narratives file not found: {predicted_narratives_path}\n"
            "Ensure the following scripts have been executed in order:\n"
            "1. identify_false_information\n"
            "2. cluster_false_information\n"
            "3. derive_disinformtion_narratives\n"
            "Refer to the README for detailed instructions."
        )

    # Load ground truth narratives
    with open(ground_truth_narratives_path, "r") as f:
        ground_truth_narratives = f.readlines()

    # Load predicted narratives
    with open(predicted_narratives_path, "r") as f:
        predicted_narratives = f.readlines()

    # Encode texts
    embedding_model = SalesForceEmbeddingModel(model_name=embedding_model_name)
    ground_truth_embeddings = embedding_model.encode_texts(ground_truth_narratives)
    predicted_embeddings = embedding_model.encode_texts(predicted_narratives)

    # Compute weighted Chamfer distance
    wsd_score = weighted_chamfer_distance(ground_truth_embeddings, predicted_embeddings)

    print(f"Weighted Chamfer Distance: {wsd_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predicted narratives against ground truth using Weighted Chamfer Distance.")
    parser.add_argument("--predicted_narratives_path", type=str, default='./data/processed/predicted_narratives.txt',
                        help="Path to the predicted narratives text file.")
    parser.add_argument("--ground_truth_narratives_path", type=str, default='./data/gt/ground_truth_narratives.txt',
                        help="Path to the ground truth narratives text file.")
    parser.add_argument("--embedding_model_name", type=str, default="Salesforce/SFR-Embedding-2_R",
                        help="Name of the embedding model to use.")

    args = parser.parse_args()

    evaluate_narratives(
        predicted_narratives_path=args.predicted_narratives_path,
        ground_truth_narratives_path=args.ground_truth_narratives_path,
        embedding_model_name=args.embedding_model_name
    )
