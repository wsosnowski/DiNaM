import argparse
import os

import pandas as pd
from src.filtering.filter_articles import process_filtering
from src.generative_model.openai_generative_model import OpenAIGenerativeModel
from src.utils.utils import convert_to_bool, calculate_false_class_metrics

def evaluate_false_claims(api_key, claims_filtering_path):
    # Initialize the OpenAI Generative Model
    llm = OpenAIGenerativeModel(api_key=api_key)

    # Load and preprocess dataset
    if not claims_filtering_path or not os.path.exists(claims_filtering_path):
        raise FileNotFoundError(
            f"Claims classification file not found: {claims_filtering_path}\n"
            "Ensure the ground truth data is available and correctly specified."
        )

    df = pd.read_csv(claims_filtering_path)

    # Filter and process the dataset
    df = process_filtering(df, llm)
    df['DebunkingVerdict'] = df['DebunkingVerdict'].apply(convert_to_bool)
    df = df[df['DebunkingVerdict'].notnull()]

    # Compute confusion matrix and evaluation metrics
    precision, recall, f1 = calculate_false_class_metrics(
        df['OnlyDisinfoClaims'].tolist(),
        df['DebunkingVerdict'].tolist()
    )

    # Display results
    print("Evaluation Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate false claims classification using OpenAI's API.")
    parser.add_argument("--api_key", type=str, help="API key for the OpenAI generative model.", required=True)
    parser.add_argument("--claims_filtering_path", type=str, default='./data/gt/ground_truth_filtering.csv',
                        help="Path to the claims classification ground truth CSV file.")

    args = parser.parse_args()

    evaluate_false_claims(api_key=args.api_key, claims_filtering_path=args.claims_filtering_path)
