import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

class GPT2VecPredictor(nn.Module):
    """GPT-2 based predictor for log anomaly detection."""
    def __init__(self, inp_dim, hidden_dim, out_dim):
        super().__init__()
        cfg = GPT2Config(n_embd=hidden_dim, n_layer=4, n_head=4)
        self.proj = nn.Linear(inp_dim, hidden_dim)
        self.enc  = GPT2Model(cfg)
        self.out  = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.proj(x).unsqueeze(1)
        z = self.enc(inputs_embeds=h).last_hidden_state
        return self.out(z[:,0])
