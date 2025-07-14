import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model

class GPT2VecPredictor(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim):
        super().__init__()
        cfg = GPT2Config(n_embd=hidden_dim, n_layer=4, n_head=4)
        self.proj = nn.Linear(inp_dim, hidden_dim)
        self.enc = GPT2Model(cfg)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.proj(x).unsqueeze(1)  # [B,1,H]
        z = self.enc(inputs_embeds=h).last_hidden_state  # [B,1,H]
        return self.out(z[:, 0])  # [B, out_dim]

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def predict(model, input_data):
    with torch.no_grad():
        predictions = model(input_data)
    return predictions.argmax(dim=1)  # Return the index of the highest score as the predicted class