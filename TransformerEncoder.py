import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerEncoder(nn.Module):
    """Per-modality transformer encoder for the hybrid DCIS/IBC classifier.
    No disease-stage embedding — classifies from omics features only."""

    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2,
                 dropout=0.1, max_seq_len=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(1, hidden_dim)
        self.pos_enc    = self._sinusoidal_encoding(max_seq_len, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def _sinusoidal_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def _pathway_attention_mask(self, x, threshold_pct=75):
        x_norm = F.normalize(x, p=2, dim=-1)
        sim    = torch.mm(x_norm, x_norm.t())
        thresh = torch.quantile(sim.flatten(), threshold_pct / 100.0)
        attend = sim >= thresh
        attend.fill_diagonal_(True)
        return ~attend

    def forward(self, x):
        seq_len = x.shape[0]
        x = x.unsqueeze(-1)
        x = self.input_proj(x)

        if seq_len <= self.pos_enc.shape[1]:
            pe = self.pos_enc[:, :seq_len, :]
        else:
            pe = F.interpolate(
                self.pos_enc.permute(0, 2, 1), size=seq_len, mode='linear', align_corners=False
            ).permute(0, 2, 1)
        x = x + pe

        try:
            attn_mask = self._pathway_attention_mask(x.squeeze(0))
        except Exception:
            attn_mask = None

        x = x if x.dim() == 3 else x.unsqueeze(0)
        out = self.transformer(x, mask=attn_mask.to(x.device) if attn_mask is not None else None)
        out = out.squeeze(0)
        return self.norm(out)
