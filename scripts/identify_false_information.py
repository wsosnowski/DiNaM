import argparse
from src.extraction.extract_false_info import handle_extracting
from src.extraction.refine_info import handle_refining
from src.extraction.verify_claims import handle_verifying
from src.filtering.filter_articles import handle_filtering
from src.generative_model.openai_generative_model import OpenAIGenerativeModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process articles through various stages of filtering, extracting, verifying, and refining.")

    parser.add_argument("--fact_checking_articles_path", type=str, default="./data/raw/fact_checking_articles.csv",
                        help="Path to the raw fact-checking articles.")
    parser.add_argument("--filtered_articles_path", type=str, default="./data/processed/filtered_articles.csv",
                        help="Path to save the filtered articles.")
    parser.add_argument("--extracted_claims_path", type=str, default="./data/processed/extracted_claims.csv",
                        help="Path to save the extracted claims.")
    parser.add_argument("--verified_claims_path", type=str, default="./data/processed/verified_claims.csv",
                        help="Path to save the verified claims.")
    parser.add_argument("--refined_claims_path", type=str, default="./data/processed/refined_claims.csv",
                        help="Path to save the refined claims.")
    parser.add_argument("--api_key", type=str, help="API key for OpenAI generative model.",
                        required=True)

    args = parser.parse_args()

    llm = OpenAIGenerativeModel(args.api_key)

    handle_filtering(args.fact_checking_articles_path, args.filtered_articles_path, llm)
    handle_extracting(args.filtered_articles_path, args.extracted_claims_path, llm)
    handle_verifying(args.extracted_claims_path, args.verified_claims_path, llm)
    handle_refining(args.verified_claims_path, args.refined_claims_path, llm)
