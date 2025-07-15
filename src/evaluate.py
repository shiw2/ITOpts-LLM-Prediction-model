import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from src.gpt2_predictor import GPT2VecPredictor

class VLTDataset(Dataset):
    def __init__(self, X, Y): self.X, self.Y = X, Y
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i], self.Y[i]

def evaluate_models():
    """Trains and evaluates the GPT-2 predictor and baseline models."""
    train_df = pd.read_json('data/embeddata/messages-train-ts2vec.jsonl', lines=True)
    test_df  = pd.read_json('data/embeddata/messages-test-ts2vec.jsonl',  lines=True)

    def vec2tensor(df):
        tr = np.stack(df['text_repr'].values)
        tm = df['timestamp_norm'].values[:,None]
        return torch.tensor(np.concatenate([tr, tm], axis=1), dtype=torch.float32)

    X_train = vec2tensor(train_df)
    X_test  = vec2tensor(test_df)
    Y_train = torch.tensor(np.stack(train_df['label_vec'].values), dtype=torch.float32)
    Y_test  = torch.tensor(np.stack(test_df['label_vec'].values),  dtype=torch.float32)

    labels = list(train_df['label'].unique())
    label_to_id = {l:i for i,l in enumerate(labels)}
    id_to_label = {i:l for l,i in label_to_id.items()}
    Txt2Vec = { l: np.stack(train_df.loc[train_df['label']==l,'label_vec'].values)[0] for l in labels }

    bs = 32
    train_loader = DataLoader(VLTDataset(X_train, Y_train), batch_size=bs, shuffle=True)
    test_loader  = DataLoader(VLTDataset(X_test,  Y_test),  batch_size=bs)

    device = 0
    model = GPT2VecPredictor(X_train.shape[1], hidden_dim=128, out_dim=Y_train.shape[1]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=5e-4)
    cos   = nn.CosineSimilarity(dim=1)
    
    epochs, thresh = 10, 0.7
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = torch.mean(1 - cos(model(xb), yb))
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    y_test_ids = test_df['label'].map(label_to_id).tolist()
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            pv = model(xb.to(device)).cpu().numpy()
            for v in pv:
                sims = [np.dot(v, Txt2Vec[l])/(np.linalg.norm(v)*np.linalg.norm(Txt2Vec[l])) for l in labels]
                best, idx = max((s,i) for i,s in enumerate(sims))
                all_preds.append(idx if best >= thresh else -1)

    # Evaluation results
    results = {}
    y_te = y_test_ids
    results['ITOpts-LLM-Prediction model'] = {
        'Accuracy': accuracy_score(y_te, all_preds),
        'Macro_Recall': recall_score(y_te, all_preds, average='macro'),
        'Macro_Precision': precision_score(y_te, all_preds, average='macro'),
        'Macro_F1': f1_score(y_te, all_preds, average='macro')
    }
    
    print(classification_report(y_te, all_preds, target_names=labels))
    
    # Baseline models
    X_tr, y_tr = X_train.numpy(), train_df['label'].map(label_to_id).values
    X_te = X_test.numpy()
    for name, Clf in [('SVM', SVC()), ('RF', RandomForestClassifier()), ('DT', DecisionTreeClassifier())]:
        clf = Clf.fit(X_tr, y_tr)
        y_p = clf.predict(X_te)
        results[name] = {
            'Accuracy': accuracy_score(y_te, y_p),
            'Macro_Recall': recall_score(y_te, y_p, average='macro'),
            'Macro_Precision': precision_score(y_te, y_p, average='macro'),
            'Macro_F1': f1_score(y_te, y_p, average='macro')
        }
    
    # Display results
    results_df = pd.DataFrame(results).T
    print("\nModel Performance Comparison:")
    print(results_df)

if __name__ == '__main__':
    evaluate_models()
