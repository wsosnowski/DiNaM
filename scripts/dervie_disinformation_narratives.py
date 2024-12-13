import argparse
from src.narratives.derive_narratives import handle_deriving_narratives

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Derive disinformation narratives from clustered claims.")
    parser.add_argument("--clustered_claims_path", type=str, default='./data/processed/clustered_claims.csv',
                        help="Path to the clustered claims CSV file.")
    parser.add_argument("--derived_narratives_path", type=str, default='./data/processed/predicted_narratives_ds.csv',
                        help="Path to save the derived narratives CSV file.")
    parser.add_argument("--predicted_narratives_path", type=str, default='./data/processed/predicted_narratives.txt',
                        help="Path to save the predicted narratives text file.")
    parser.add_argument("--api_key", type=str,
                        help="API key for the OpenAI generative model.", required=True)

    args = parser.parse_args()

    handle_deriving_narratives(
        clustered_claims_path=args.clustered_claims_path,
        derived_narratives_path=args.derived_narratives_path,
        predicted_narratives_path=args.predicted_narratives_path,
        api_key=args.api_key
    )
