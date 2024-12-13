import argparse

from src.clustering.cluster_claims import handle_clustering

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform clustering on refined claims.")
    parser.add_argument("--refined_claims", type=str, default='./data/processed/refined_claims.csv',
                        help="Path to the refined claims.")
    parser.add_argument("--clustered_claims_path", type=str, default='./data/processed/clustered_claims.csv',
                        help="Path to save the clustered claims.")
    parser.add_argument("--start_date", type=str, default="2021-07-01",
                        help="Start date for filtering claims.")
    parser.add_argument("--end_date", type=str, default="2023-02-01",
                        help="End date for filtering claims.")
    parser.add_argument("--umap_params", type=dict, default={"n_components": 256, "n_neighbors": 15},
                        help="Parameters for UMAP dimensionality reduction.")
    parser.add_argument("--hdbscan_params", type=dict, default={"min_cluster_size": 10, "min_samples": 15},
                        help="Parameters for HDBSCAN clustering.")
    parser.add_argument("--model_name", type=str, default="Salesforce/SFR-Embedding-2_R",
                        help="Name of the model to use for embedding generation.")

    args = parser.parse_args()

    handle_clustering(
        refined_claims_path=args.refined_claims,
        clustered_claims_path=args.clustered_claims_path,
        start_date=args.start_date,
        end_date=args.end_date,
        umap_params=args.umap_params,
        hdbscan_params=args.hdbscan_params,
        model_name=args.model_name
    )
