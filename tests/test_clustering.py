import os
import argparse
import pandas as pd

from src.clustering_model.hdbscan_clustering_model import HDBSCANClusteringModel
from src.dimensionality_reduction_model.umap_model import UmapModel
from src.embedding_model.salesforce_embedding_model import SalesForceEmbeddingModel
from src.utils.utils import save_embeddings, load_embeddings, calculate_silhouette_score

def load_and_preprocess_data(refined_narratives_path):
    """Load and preprocess data based on refined narratives path."""
    if not os.path.exists(refined_narratives_path):
        raise FileNotFoundError(
            "Required dataset is missing. "
            "Please run the identify_false_information .\n"
            f"Missing files:\n"
            f"- {refined_narratives_path}"
        )

    final_claims_set = pd.read_csv(refined_narratives_path).explode("ExtractedClaims").reset_index(drop=True)
    return final_claims_set[final_claims_set["ExtractedClaims"].notna()]

def load_or_generate_embeddings(false_claims, embedding_paths, embedding_model_name):
    """Load embeddings from file or generate them if not available."""
    embedding_model = SalesForceEmbeddingModel(model_name=embedding_model_name)
    embeddings = load_embeddings(embedding_paths)

    if embeddings is None:
        embeddings = embedding_model.encode_texts(false_claims)
        save_embeddings(embeddings, embedding_paths)

    return embeddings

def prepare_documents(false_claims_df):
    """Prepare documents DataFrame for clustering."""
    documents = []
    for row in false_claims_df.itertuples():
        if isinstance(row.ExtractedClaims, str):
            documents.append({"Document": row.ExtractedClaims, "ID": row.Index, "Timestamp": row.date})

    return pd.DataFrame(documents)

def perform_clustering(param_grid, embeddings, documents, output_file):
    """Perform clustering with the given parameter grid and embeddings."""
    with open(output_file, "a") as f:
        for n_components in param_grid["n_components"]:
            for n_neighbors in param_grid["n_neighbors"]:
                dimensionality_reduction_model = UmapModel(n_components=n_components, n_neighbors=n_neighbors)
                reduced_embeddings = dimensionality_reduction_model.reduce_dimensionality(embeddings.cpu())

                for min_cluster_size in param_grid["min_cluster_size"]:
                    for min_samples in param_grid["min_samples"]:
                        clustering_model = HDBSCANClusteringModel(min_cluster_size=min_cluster_size,
                                                                  min_samples=min_samples)
                        topics, prob = clustering_model.perform_clustering(reduced_embeddings, documents)
                        sil_score = calculate_silhouette_score(reduced_embeddings, topics)
                        result = (f"min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
                                  f"silhouette_score={sil_score}\n")
                        print(result.strip())
                        f.write(result)

def main():
    parser = argparse.ArgumentParser(description="Run clustering pipeline.")

    parser.add_argument("--refined_narratives_path", type=str, default="./data/processed/refined_claims.csv",
                        help="Path to the refined narratives CSV file.")
    parser.add_argument("--embedding_paths", type=str, default="./data/embeddings/embeddings.pt",
                        help="Path to save or load embeddings.")
    parser.add_argument("--embedding_model_name", type=str, default="Salesforce/SFR-Embedding-2_R",
                        help="Name of the embedding model to use.")
    parser.add_argument("--clustering_output_file", type=str, default="./data/results/clustering_results.txt",
                        help="Path to save clustering results.")
    parser.add_argument("--date_range_start", type=str, default="2021-07-01",
                        help="Start date for filtering data.")
    parser.add_argument("--date_range_end", type=str, default="2023-02-01",
                        help="End date for filtering data.")


    args = parser.parse_args()

    # Load and preprocess data
    false_claims_df = load_and_preprocess_data(args.refined_narratives_path)
    false_claims_df = false_claims_df[
        (false_claims_df["date"] >= args.date_range_start) & (false_claims_df["date"] <= args.date_range_end)]

    false_claims = false_claims_df['ExtractedClaims'].tolist()

    # Load or generate embeddings
    embeddings = load_or_generate_embeddings(false_claims, args.embedding_paths, args.embedding_model_name)

    # Prepare documents
    documents = prepare_documents(false_claims_df)

    # Perform clustering
    param_grid = {
        "min_cluster_size": [10, 15, 20, 25, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        "min_samples": [10,15, 20, 25, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        "n_components": [256],
        "n_neighbors": [15],
    }

    perform_clustering(param_grid, embeddings, documents, args.clustering_output_file)

if __name__ == "__main__":
    main()
