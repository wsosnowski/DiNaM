from src.generative_model.base_generative_model import BaseGenerativeModel
from openai import OpenAI


class DeepInfraGenerativeModel(BaseGenerativeModel):
    def __init__(self,
                 api_key="",
                 model_name=""):
        super().__init__(api_key, model_name)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )

    def get_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={'type': 'json_object'}
        )
        return response.choices[0].message.content.strip()
