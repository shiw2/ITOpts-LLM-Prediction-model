import sys
import os
import pandas as pd
import numpy as np
from ts2vec import TS2Vec

# Function to load data
def load_data(train_path, test_path):
    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    return train_df, test_df

# Function to prepare data for TS2Vec
def prepare_data(train_df, test_df):
    # Normalize timestamps
    for df in [train_df, test_df]:
        df['timestamp_float'] = df['timestamp'].astype(np.int64) // 10**9
    ts_min = min(train_df['timestamp_float'].min(), test_df['timestamp_float'].min())
    ts_max = max(train_df['timestamp_float'].max(), test_df['timestamp_float'].max())
    for df in [train_df, test_df]:
        df['timestamp_norm'] = (df['timestamp_float'] - ts_min) / (ts_max - ts_min)

    # Prepare data for TS2Vec
    train_data = train_df['text_repr'].values[np.newaxis, :]
    test_data = test_df['text_repr'].values[np.newaxis, :]
    
    return train_data, test_data

# Function to train TS2Vec model
def train_ts2vec(train_data):
    model_ts2vec = TS2Vec(input_dims=train_data.shape[2], device=0, output_dims=320)
    model_ts2vec.fit(train_data, verbose=True)
    return model_ts2vec

# Function to encode data using the trained TS2Vec model
def encode_data(model_ts2vec, train_data, test_data):
    train_repr = model_ts2vec.encode(train_data)
    test_repr = model_ts2vec.encode(test_data)
    return train_repr, test_repr

# Main function to execute the training process
def main():
    train_df, test_df = load_data('data/sampledatasets/messages-train.jsonl', 'data/sampledatasets/messages-test.jsonl')
    train_data, test_data = prepare_data(train_df, test_df)
    model_ts2vec = train_ts2vec(train_data)
    train_repr, test_repr = encode_data(model_ts2vec, train_data, test_data)

    # Save representations back to DataFrame
    train_df['text_repr'] = list(train_repr[0])
    test_df['text_repr'] = list(test_repr[0])
    
    # Save processed data
    os.makedirs('data/embeddata', exist_ok=True)
    train_df.to_json('data/embeddata/messages-train-ts2vec.jsonl', orient='records', lines=True, force_ascii=False)
    test_df.to_json('data/embeddata/messages-test-ts2vec.jsonl', orient='records', lines=True, force_ascii=False)

if __name__ == "__main__":
    main()