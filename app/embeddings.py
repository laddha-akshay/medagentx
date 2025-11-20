from sentence_transformers import SentenceTransformer
MODEL_NAME = 'all-MiniLM-L6-v2'
class Embedder:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)
    def embed_texts(self, texts):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
