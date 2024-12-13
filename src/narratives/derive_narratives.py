import json

import pandas as pd
from pandas import DataFrame

from src.generative_model.base_generative_model import BaseGenerativeModel
import concurrent.futures
from tqdm import tqdm

from src.generative_model.openai_generative_model import OpenAIGenerativeModel


def derive_narratives(client: BaseGenerativeModel, df: DataFrame, topic_id: str) -> tuple[dict, str]:
    texts = df[df["topic"] == topic_id]["document"].tolist()
    prompt = f"""
    Analyze a list of false information and provide a simple, short narrative underlying false intention of all the sentences.
    
    ### FALSE INFORMATION:
    '{', '.join(texts)[:128000]}'

    ### GUIDELINES
    - Provide one narrative that best fits all of those false information. 
    - It must be straightforward, standalone and enough descriptive, so it is clear without additional context.
    - It must be simple and concise, not longer than 15 words.
    - It must be clear enough, to easily be understood by a person who is not familiar with the topic.
    - It must reflect the false perspective those information underlie.
    - It must not reveal it is false narrative.
    
    Return JSON object in following format:

    {{
        "response": <narrative>
    }}
    """

    client_response = client.get_response(prompt)

    try:
        extracted_narratives = json.loads(client_response)
    except json.JSONDecodeError:
        extracted_narratives = {"response": None}
    return extracted_narratives, topic_id


def process_deriving_narratives(df: pd.DataFrame, llm: BaseGenerativeModel) -> dict:
    """
    Process and filter a DataFrame using a generative language model.
    """
    topic_list = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(derive_narratives, llm, df, topic_id) for topic_id in df.topic.unique()]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            narrative, topic_id = future.result()
            topic_list[topic_id] = narrative['response']
    topic_list_dict = {int(el): topic_list[el] for el in topic_list}

    return topic_list_dict


def handle_deriving_narratives(clustered_claims_path, derived_narratives_path, predicted_narratives_path, api_key):
    # Initialize generative model
    llm = OpenAIGenerativeModel(api_key)

    # Load clustered claims
    df = pd.read_csv(clustered_claims_path)

    # Derive narratives
    topic_list_dict = process_deriving_narratives(df, llm)

    # Map derived topics to descriptions
    df['topic_description'] = df['topic'].map(topic_list_dict)
    df.to_csv(derived_narratives_path, index=False)

    # Save predicted narratives
    with open(predicted_narratives_path, "w") as f:
        f.write("\n".join([el[1] for el in topic_list_dict.items() if el[0] != -1]))
