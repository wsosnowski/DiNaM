import pandas as pd
from src.clustering_model.hdbscan_clustering_model import HDBSCANClusteringModel
from src.dimensionality_reduction_model.umap_model import UmapModel
from src.embedding_model.salesforce_embedding_model import SalesForceEmbeddingModel


def handle_clustering(refined_claims_path, clustered_claims_path, start_date, end_date, umap_params, hdbscan_params, model_name):
    # Load and preprocess data
    claims_refined = pd.read_csv(refined_claims_path)

    # Filter by date
    df = claims_refined[
        (claims_refined["date"] >= start_date) & (claims_refined["date"] <= end_date)]

    # Prepare documents
    documents = pd.DataFrame({"Document": df["ExtractedClaims"].tolist(), "ID": range(len(df)), "Topic": None})

    # Generate embeddings
    embedding_model = SalesForceEmbeddingModel(model_name=model_name)
    embeddings = embedding_model.encode_texts(documents.Document.tolist(), batch_size=64)

    # Dimensionality reduction
    dimensionality_reduction_model = UmapModel(**umap_params)
    reduced_embeddings = dimensionality_reduction_model.reduce_dimensionality(embeddings.cpu())

    # Clustering
    clustering_model = HDBSCANClusteringModel(**hdbscan_params)
    topics, prob = clustering_model.perform_clustering(reduced_embeddings, documents)

    # Derive narratives
    df = pd.DataFrame({'topic': topics, 'document': documents.Document})

    df.to_csv(clustered_claims_path, index=False)
