import json

import pandas as pd

from src.generative_model.base_generative_model import BaseGenerativeModel
import concurrent.futures
from tqdm import tqdm

def extract_narrative(client: BaseGenerativeModel, text: str) -> dict:
    prompt = f"""
    You are tasked with analyzing a debunking article and extracting the claims that the article debunks.

    ### GUIDELINES
    - The extracted claims should fully represent the misleading or debunked information that the article aims to refute.
    - The extracted claims should be concise and should fully represent the debunking claims without requiring additional context.   
    - Return one claim if there is one claim debunked in the article.

    ### Debunking Article
    '{text}'

    Output JSON Format:
    {{
        "response": []  #Debunking Claims: This are the claims that the article debunks.
    }}
    """

    client_response = client.get_response(prompt)

    try:
        extracted_narratives = json.loads(client_response)
    except json.JSONDecodeError:
        extracted_narratives = {"response": None}
    return extracted_narratives


def process_extracting(df: pd.DataFrame, llm: BaseGenerativeModel) -> pd.DataFrame:
    """
    Process and extract narratives from a DataFrame using a generative language model.

    Args:
        df (pd.DataFrame): The input DataFrame containing a column named "Translation".
        llm (BaseGenerativeModel): An instance of a generative language model used for extracting narratives.

    Returns:
        pd.DataFrame: The updated DataFrame with a new column "ExtractedClaims" containing the extraction results.
    """
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = {executor.submit(extract_narrative, llm, row["Translation"]): index for index, row in df.iterrows()}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting claims"):
            try:
                index = futures[future]
                result = future.result()
                results[index] = result["response"] if result["response"] is not None else None
            except Exception as exc:
                print(f"Error processing row: {exc}")
                results[index] = None
    df["ExtractedClaims"] = df.index.map(results)
    return df

def handle_extracting(input_path, output_path, llm):
    df = pd.read_csv(input_path)
    if "DebunkingVerdict" in df.columns:
        df = df[df["DebunkingVerdict"].isin([False, "False"])]
    df = process_extracting(df, llm)
    df.to_csv(output_path, index=False)