import numpy as np
import pandas as pd
import argparse
from src.embedding_model.salesforce_embedding_model import SalesForceEmbeddingModel
from src.extraction.extract_false_info import handle_extracting
from src.extraction.refine_info import handle_refining
from src.extraction.verify_claims import handle_verifying
from src.generative_model.openai_generative_model import OpenAIGenerativeModel
from src.utils.utils import weighted_chamfer_distance


def load_model_and_tokenizer(model_name):
    """Load the embedding model."""
    return SalesForceEmbeddingModel(model_name=model_name)


def encode_texts_list_of_lists(embedding_model, text_lists, batch_size=1):
    """Encode a list of texts into embeddings."""
    all_embeddings = []
    for texts in text_lists:
        if isinstance(texts, str):
            texts = [texts]
        try:
            embeddings = embedding_model.encode_texts(texts, batch_size=batch_size)
        except Exception as e:
            print(f"Encoding error: {e}")
            continue
        all_embeddings.append(embeddings)
    return all_embeddings


def load_data(input_path):
    """Load and preprocess data."""
    claims_refined = pd.read_csv(input_path)
    grouped = claims_refined.groupby('false_claim')['ExtractedClaims'].apply(list)
    texts_orig_full = grouped.tolist()
    claims = grouped.index.tolist()
    return texts_orig_full, claims


def calculate_chamfer_distances(embedding_model, texts_orig_full, claims):
    """Compute Chamfer distances."""
    predicted_embeddings_list_of_lists = encode_texts_list_of_lists(embedding_model, texts_orig_full)
    narrative_embeddings = embedding_model.encode_texts(claims, batch_size=64)

    chamfer_distances = [
        weighted_chamfer_distance(narr_emb, predicted_embeddings).item()
        for predicted_embeddings, narr_emb in zip(predicted_embeddings_list_of_lists, narrative_embeddings)
    ]
    return chamfer_distances


def main():
    parser = argparse.ArgumentParser(
        description="Process articles through various stages of filtering, extracting, verifying, and refining.")

    parser.add_argument("--ground_truth_fact_checking_articles_path", type=str,
                        default="./data/gt/ground_truth_extraction.csv",
                        help="Path to the raw fact-checking articles.")
    parser.add_argument("--filtered_articles_path", type=str, default="./data/processed/filtered_articles.csv",
                        help="Path to save the filtered articles.")
    parser.add_argument("--extracted_claims_path", type=str, default="./data/processed_test/extracted_claims.csv",
                        help="Path to save the extracted claims.")
    parser.add_argument("--verified_claims_path", type=str, default="./data/processed_test/verified_claims.csv",
                        help="Path to save the verified claims.")
    parser.add_argument("--refined_claims_path", type=str, default="./data/processed_test/refined_claims.csv",
                        help="Path to save the refined claims.")
    parser.add_argument("--api_key", type=str,
                        required=True,
                        help="API key for OpenAI generative model.")
    parser.add_argument("--embedding_model_name", type=str, default="Salesforce/SFR-Embedding-2_R",
                        help="Name of the embedding model to use.")

    args = parser.parse_args()

    # Initialize the generative model
    llm = OpenAIGenerativeModel(args.api_key)

    # Run extraction, verification, and refinement stages
    handle_extracting(args.ground_truth_fact_checking_articles_path, args.extracted_claims_path, llm)
    handle_verifying(args.extracted_claims_path, args.verified_claims_path, llm)
    handle_refining(args.verified_claims_path, args.refined_claims_path, llm)

    # Load the embedding model and data
    embedding_model = load_model_and_tokenizer(args.embedding_model_name)
    texts_orig_full, claims = load_data(args.refined_claims_path)

    # Calculate Chamfer distances
    chamfer_distances = calculate_chamfer_distances(embedding_model, texts_orig_full, claims)

    # Output results
    print("Mean Chamfer Distance:", np.mean(chamfer_distances))


if __name__ == "__main__":
    main()
