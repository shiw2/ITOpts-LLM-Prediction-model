import os
import pandas as pd
from datetime import timedelta
import re

# label the logs(Design your own)
def label_fn(txt):
    if re.search(r'(?i)(error)', txt):
        return 'abnormal'
    else:
        return 'normal'

def preprocess_data(file_path='data/sourcedata/messages-20250602'):
    records = []
    with open(file_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            ts, host, text = line.strip().split(' ', 2)
            records.append({'timestamp': ts, 'host': host, 'text': text})
    
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['label'] = df['text'].apply(label_fn)

    # Split data
    start_time = df['timestamp'].min()
    trainingset_end_time = start_time + timedelta(days=4)
    training_set = df[(df['timestamp'] >= start_time) & (df['timestamp'] < trainingset_end_time)].reset_index(drop=True)
    testing_set = df[df['timestamp'] >= trainingset_end_time].reset_index(drop=True)

    # Preprocess training data
    train_df = training_set.copy()
    train_df = train_df.sort_values(['label', 'text', 'timestamp'])
    limit_map = {'abnormal': timedelta(seconds=10), 'normal': timedelta(seconds=10)}
    train_df['prev_time'] = train_df.groupby(['label', 'text'])['timestamp'].shift()
    train_df['time_diff'] = train_df['timestamp'] - train_df['prev_time']
    train_df['limit'] = train_df['label'].map(limit_map)
    mask = (train_df['prev_time'].isna()) | (train_df['time_diff'] >= train_df['limit'])
    train_df = train_df[mask].drop(['prev_time', 'time_diff', 'limit'], axis=1)
    train_df = train_df.sort_values(by='timestamp').drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)

    # Downsample training data
    n_per_class = 500
    train_abnormal_sample = train_df[train_df['label'] == 'abnormal'].sample(n=min(n_per_class, len(train_df[train_df['label'] == 'abnormal'])), random_state=42)
    train_normal_sample = train_df[train_df['label'] == 'normal'].sample(n=n_per_class, random_state=42)
    train_df_sampled = pd.concat([train_normal_sample, train_abnormal_sample]).sort_values('timestamp').reset_index(drop=True)

    # Downsample testing data
    test_sample_normal = testing_set[testing_set['label'] == 'normal'].sample(n=400, random_state=42)
    test_sample_abnormal = testing_set[testing_set['label'] == 'abnormal'].sample(n=100, random_state=42)
    test_df_sampled = pd.concat([test_sample_normal, test_sample_abnormal]).sort_values('timestamp').reset_index(drop=True)
    
    # Save processed data
    os.makedirs('data/sampledatasets', exist_ok=True)
    train_df_sampled.to_json('data/sampledatasets/messages-train.jsonl', orient='records', lines=True, force_ascii=False)
    test_df_sampled.to_json('data/sampledatasets/messages-test.jsonl', orient='records', lines=True, force_ascii=False)
    
    print("Data preprocessing complete.")
    print(f"Training samples: {len(train_df_sampled)}")
    print(f"Testing samples: {len(test_df_sampled)}")

if __name__ == '__main__':
    # Assuming latest file is the one to be processed
    folder = "data/sourcedata"
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("messages-")]
    latest_file = max(files, key=os.path.getmtime)
    preprocess_data(latest_file)
