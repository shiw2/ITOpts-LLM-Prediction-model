import pandas as pd
import numpy as np
import os
from ts2vec import TS2Vec
from src.embedding import get_text_embeddings, get_label_embeddings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def train_ts2vec_and_embed():
    """Trains the TS2Vec model and saves the embedded data."""
    train_df = pd.read_json('data/sampledatasets/messages-train.jsonl', lines=True)
    test_df = pd.read_json('data/sampledatasets/messages-test.jsonl', lines=True)
    train_df = train_df.sort_values('timestamp')
    test_df = test_df.sort_values('timestamp')

    # Text Embedding
    train_text_emb = get_text_embeddings(train_df['text'].tolist())
    test_text_emb = get_text_embeddings(test_df['text'].tolist())

    # Label Embedding
    unique_labels = train_df['label'].unique().tolist()
    label_to_vec = get_label_embeddings(unique_labels)
    
    # Timestamp Normalization
    for df in [train_df, test_df]:
        df['timestamp_float'] = df['timestamp'].astype(np.int64) // 10**9
    ts_min = min(train_df['timestamp_float'].min(), test_df['timestamp_float'].min())
    ts_max = max(train_df['timestamp_float'].max(), test_df['timestamp_float'].max())
    for df in [train_df, test_df]:
        df['timestamp_norm'] = (df['timestamp_float'] - ts_min) / (ts_max - ts_min)

    # Prepare data for TS2Vec
    train_data = train_text_emb[np.newaxis, :]
    test_data = test_text_emb[np.newaxis, :]
    
    # Train TS2Vec model
    model_ts2vec = TS2Vec(input_dims=train_data.shape[2], device=0, output_dims=320)
    model_ts2vec.fit(train_data, verbose=True)

    # Encode data
    train_repr = model_ts2vec.encode(train_data)
    test_repr = model_ts2vec.encode(test_data)

    train_df['text_repr'] = list(train_repr[0])
    test_df['text_repr'] = list(test_repr[0])
    train_df['label_vec'] = train_df['label'].map(label_to_vec)
    test_df['label_vec'] = test_df['label'].map(label_to_vec)

    # Visualize embeddings
    pca = PCA(n_components=2)
    train_vec_2d = pca.fit_transform(np.stack(train_df['text_repr'].values))
    plt.figure(figsize=(8, 6))
    for label in train_df['label'].unique():
        idx = train_df['label'] == label
        plt.scatter(train_vec_2d[idx, 0], train_vec_2d[idx, 1], label=label, alpha=0.6)
    plt.title('TS2Vec Representation (PCA 2D)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()

    # Save processed data
    os.makedirs('data/embeddata', exist_ok=True)
    train_df.to_json('data/embeddata/messages-train-ts2vec.jsonl', orient='records', lines=True, force_ascii=False)
    test_df.to_json('data/embeddata/messages-test-ts2vec.jsonl', orient='records', lines=True, force_ascii=False)
    print("TS2Vec training and embedding complete.")

if __name__ == '__main__':
    train_ts2vec_and_embed()
