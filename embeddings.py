import numpy as np
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType


def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def _emb(text: str):
    name = OpenAIEmbeddingModelType.TEXT_EMBED_3_LARGE
    # name = OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL
    model = OpenAIEmbedding(model_name=name)
    return model._get_text_embedding(text)


similarity = cosine_similarity(_emb("Bábovka"), _emb("Buchty s mákem"))
print(f"Cosine Similarity: {similarity}")

similarity = cosine_similarity(_emb("Bábovka"), _emb("Řízek"))
print(f"Cosine Similarity: {similarity}")

similarity = cosine_similarity(_emb("Bábovka"), _emb("Uhlí"))
print(f"Cosine Similarity: {similarity}")
