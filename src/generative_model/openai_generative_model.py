from src.generative_model.base_generative_model import BaseGenerativeModel
from openai import OpenAI


class OpenAIGenerativeModel(BaseGenerativeModel):
    def __init__(self,
                 api_key="",
                 model_name="gpt-4o-mini"):
        super().__init__(api_key, model_name)
        self.client = OpenAI(
            api_key=self.api_key
        )

    def get_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content.strip()
