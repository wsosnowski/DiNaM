import json

import pandas as pd

from src.generative_model.base_generative_model import BaseGenerativeModel
import concurrent.futures
from tqdm import tqdm


def filter_articles(client: BaseGenerativeModel, text: str) -> dict:
    prompt = f"""

    ### Article Text
    '{text}'

    Analyze a fact-checking article and determine if the debunking claims are false.

    ### GUIDELINES
    - If the article debunks false information, return "False". 
    - If the article confirms true information, return "True".

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


def process_filtering(df: pd.DataFrame, llm: BaseGenerativeModel) -> pd.DataFrame:
    """
    Process and filter a DataFrame using a generative language model.

    Args:
        df (pd.DataFrame): The input DataFrame containing a column named "Translation".
        llm (BaseGenerativeModel): An instance of a generative language model used for filtering articles.

    Returns:
        pd.DataFrame: The updated DataFrame with a new column "DebunkingVerdict" containing the filtering results.
    """
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = {executor.submit(filter_articles, llm, row["Translation"]): index for index, row in df.iterrows()}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Filtering texts"):
            try:
                index = futures[future]
                result = future.result()
                results[index] = result["response"] if result["response"] is not None else None
            except Exception as exc:
                print(f"Error processing row: {exc}")
                results[index] = None
    df["DebunkingVerdict"] = df.index.map(results)
    return df


def handle_filtering(input_path, output_path, llm):
    df = pd.read_csv(input_path)
    df = process_filtering(df, llm)
    df.to_csv(output_path, index=False)