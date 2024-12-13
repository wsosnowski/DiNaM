import json

from src.generative_model.base_generative_model import BaseGenerativeModel
import concurrent.futures
from tqdm import tqdm
import pandas as pd


def refine_narrative(client: BaseGenerativeModel, text: str, debunking_claim: str) -> dict:
    prompt = f"""
    
    Your task is to rewrite a claim based on the misleading perspective that the article debunks, making it sound credible and true.

    ### Debunking Article    
    '{text}'    


    ### Debunking Claim
    '{debunking_claim}'
    
    ### GUIDELINES
    - Rewrite the claim to reflect the misleading perspective that the article debunks. 
    - The new claim must be very simple, very concise, straightforward, and written in active voice, without any additional context, commentary, or interpretation. 

    Return in JSON format:
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


def process_refining(df: pd.DataFrame, llm: BaseGenerativeModel) -> pd.DataFrame:
    """
    Refines extracted claims from the input DataFrame using a generative language model.

    Args:
        df (pd.DataFrame): The input DataFrame containing at least two columns:
            - "Translation": The text to be analyzed.
            - "ExtractedClaims": Claims extracted from the "Translation" column.
        llm (BaseGenerativeModel): A generative language model used for refining claims.

    Returns:
        pd.DataFrame: The updated DataFrame with a new column "RefinedClaims" containing
        the refined claims for each row.
    """
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = {executor.submit(refine_narrative, llm, row["Translation"], row["ExtractedClaims"]): index for
                   index, row in df.iterrows()}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Refining claims"):
            try:
                index = futures[future]
                result = future.result()
                results[index] = result["response"] if result["response"] is not None else None
            except Exception as exc:
                print(f"Error processing row: {exc}")
                results[index] = None
    df["RefinedClaims"] = df.index.map(results)
    return df


def handle_refining(input_path, output_path, llm):
    df_verified = pd.read_csv(input_path)

    df_correct = df_verified[~df_verified["VerifiedVerdict"].isin(["True", True])]

    df_incorrect = df_verified[df_verified["VerifiedVerdict"].isin(["True", True])]
    df_refined = process_refining(df_incorrect, llm)
    df_refined["ExtractedClaims"] = df_refined["RefinedClaims"]

    final_claims_set = pd.concat([df_refined, df_correct], ignore_index=True)
    final_claims_set = final_claims_set[final_claims_set["ExtractedClaims"].notna()]

    final_claims_set.to_csv(output_path, index=False)
