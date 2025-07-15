from sentence_transformers import SentenceTransformer
import numpy as np

def get_text_embeddings(texts, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Generates text embeddings using a SentenceTransformer model."""
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, convert_to_tensor=False)

def get_label_embeddings(labels, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Generates label embeddings."""
    model = SentenceTransformer(model_name)
    label_embs = model.encode(labels, show_progress_bar=False, convert_to_tensor=False)
    return {label: vec for label, vec in zip(labels, label_embs)}
