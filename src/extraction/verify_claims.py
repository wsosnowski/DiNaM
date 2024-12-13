import ast
import json

from src.generative_model.base_generative_model import BaseGenerativeModel
import concurrent.futures
from tqdm import tqdm
import pandas as pd

def verify_narrative(client: BaseGenerativeModel, text: str, debunking_claim: str) -> dict:
    prompt = f"""
    Your task is to analyze the debunking article and determine if the claim is supported by the article or not.
    
    ### Debunking Article 
    '{text}'
    
    ### Claim
    '{debunking_claim}'
    
    
    ### GUIDELINES
    - If the article confirms that the claim is true, return True.
    - If the article debunks the claim as false, return False.


    Respond in the following JSON format:

    {{    
        "response": ""
    }}
    """

    client_response = client.get_response(prompt)

    try:
        extracted_narratives = json.loads(client_response)
    except json.JSONDecodeError:
        extracted_narratives = {"response": None}
    return extracted_narratives


def process_verifying(df: pd.DataFrame, llm: BaseGenerativeModel) -> pd.DataFrame:
    """
    Verifies extracted claims from the input DataFrame using a generative language model.

    Args:
        df (pd.DataFrame): The input DataFrame containing at least two columns:
            - "Translation": The text to be analyzed.
            - "ExtractedClaims": A list of claims extracted from the "Translation" column.
        llm (BaseGenerativeModel): A generative language model used for verifying claims.

    Returns:
        pd.DataFrame: The updated DataFrame with a new column "VerifiedVerdict" containing
        the verification results for each extracted claim. The DataFrame is exploded such
        that each row corresponds to a single claim.
    """
    df["ExtractedClaims"] = df["ExtractedClaims"].apply(ast.literal_eval)
    df = df.explode("ExtractedClaims").reset_index(drop=True)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = {executor.submit(verify_narrative, llm, row["Translation"], row["ExtractedClaims"]): index for
                   index, row in df.iterrows()}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Verifying claims"):
            try:
                index = futures[future]
                result = future.result()
                results[index] = result["response"] if result["response"] is not None else None
            except Exception as exc:
                print(f"Error processing row: {exc}")
                results[index] = None
    df["VerifiedVerdict"] = df.index.map(results)
    return df


def handle_verifying(input_path, output_path, llm):
    df = pd.read_csv(input_path)
    df = process_verifying(df, llm)
    df.to_csv(output_path, index=False)
