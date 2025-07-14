import os
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_and_encode_texts(train_file, test_file):
    train_df = pd.read_json(train_file, lines=True)
    test_df = pd.read_json(test_file, lines=True)

    text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    train_text_emb = text_model.encode(train_df['text'].tolist(), show_progress_bar=True, convert_to_tensor=False)
    test_text_emb = text_model.encode(test_df['text'].tolist(), show_progress_bar=True, convert_to_tensor=False)

    return train_text_emb, test_text_emb

def encode_labels(train_df):
    unique_labels = train_df['label'].unique().tolist()
    label_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    label_embs = label_model.encode(unique_labels, show_progress_bar=False, convert_to_tensor=False)
    label_to_vec = {label: vec for label, vec in zip(unique_labels, label_embs)}
    
    return label_to_vec

def save_embeddings(train_df, test_df, train_text_emb, test_text_emb, label_to_vec):
    train_df['text_repr'] = list(train_text_emb)
    test_df['text_repr'] = list(test_text_emb)
    train_df['label_vec'] = train_df['label'].map(label_to_vec)
    test_df['label_vec'] = test_df['label'].map(label_to_vec)

    return train_df, test_df