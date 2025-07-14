import os
import pandas as pd
import numpy as np
from datetime import timedelta
import re

def load_data(train_file: str, test_file: str) -> (pd.DataFrame, pd.DataFrame):
    train_df = pd.read_json(train_file, lines=True)
    test_df = pd.read_json(test_file, lines=True)
    return train_df, test_df

def preprocess_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp_float'] = df['timestamp'].astype(np.int64) // 10**9
    return df

def normalize_timestamps(train_df: pd.DataFrame, test_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    ts_min = min(train_df['timestamp_float'].min(), test_df['timestamp_float'].min())
    ts_max = max(train_df['timestamp_float'].max(), test_df['timestamp_float'].max())
    
    for df in [train_df, test_df]:
        df['timestamp_norm'] = (df['timestamp_float'] - ts_min) / (ts_max - ts_min)
    
    return train_df, test_df

def label_encoding(df: pd.DataFrame) -> pd.DataFrame:
    def label_fn(txt):
        if re.search(r'(?i)(error)', txt):
            return 'abnormal'
        else:
            return 'normal'
    
    df['label'] = df['text'].apply(label_fn)
    return df

def preprocess_data(train_file: str, test_file: str) -> (pd.DataFrame, pd.DataFrame):
    train_df, test_df = load_data(train_file, test_file)
    train_df = preprocess_timestamps(train_df)
    test_df = preprocess_timestamps(test_df)
    train_df, test_df = normalize_timestamps(train_df, test_df)
    train_df = label_encoding(train_df)
    test_df = label_encoding(test_df)
    
    return train_df, test_df