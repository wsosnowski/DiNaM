from src.embedding_model.base_embedding_model import BaseEmbeddingModel
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from torch import Tensor


class SalesForceEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="Salesforce/SFR-Embedding-2_R"):
        super(SalesForceEmbeddingModel, self).__init__(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto", load_in_8bit=True, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def last_token_pool(self, last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode_texts(self, texts, batch_size=64):
        all_embeddings = []

        self.model.eval()  # Ensure the model is in evaluation mode

        # Set up progress bar
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Texts", unit="batch"):
                batch_texts = texts[i:i + batch_size]

                # Tokenize and get embeddings for the current batch
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,  # Consistent truncation
                    return_tensors='pt'
                ).to(self.model.device)

                outputs = self.model(**inputs)
                embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])

                all_embeddings.append(embeddings)

        # Concatenate all the embeddings
        return torch.cat(all_embeddings, dim=0)

