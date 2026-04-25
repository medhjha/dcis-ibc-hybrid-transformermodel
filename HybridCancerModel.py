import torch
import torch.nn as nn
import torch.nn.functional as F
from TransformerEncoder import TransformerEncoder


class HybridDCISModel(nn.Module):
    """Hybrid GNN-Transformer model for DCIS vs IBC classification.
    
    Four per-modality transformer encoders process gene expression,
    methylation proxy, pathway scores, and per-sample immune markers
    independently. Mean-pooled outputs are concatenated into a 512-d
    fused embedding passed to the classification head.
    
    No disease-stage label is used as input at any stage.
    """

    def __init__(self, gene_dim, meth_dim, pathway_dim, immune_dim,
                 hidden_dim=128, num_classes=2, dropout=0.3, proj_dim=64):
        super().__init__()

        self.gene_enc = TransformerEncoder(gene_dim,    hidden_dim, num_heads=4, num_layers=2, dropout=dropout)
        self.meth_enc = TransformerEncoder(meth_dim,    hidden_dim, num_heads=4, num_layers=2, dropout=dropout)
        self.path_enc = TransformerEncoder(pathway_dim, hidden_dim, num_heads=4, num_layers=2, dropout=dropout)
        self.imm_enc  = TransformerEncoder(immune_dim,  hidden_dim, num_heads=4, num_layers=2, dropout=dropout)

        fusion_dim = hidden_dim * 4  # 512-d after concatenating 4 modalities

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.proj_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, data):
        gene_emb = self.gene_enc(data['gene']).mean(dim=0)
        meth_emb = self.meth_enc(data['methylation']).mean(dim=0)
        path_emb = self.path_enc(data['pathway']).mean(dim=0)
        imm_emb  = self.imm_enc(data['immune']).mean(dim=0)

        h_fused = torch.cat([gene_emb, meth_emb, path_emb, imm_emb], dim=-1)
        logits  = self.classifier(h_fused)
        z       = F.normalize(self.proj_head(h_fused), dim=-1)

        return logits, z, h_fused


class ContrastiveLoss(nn.Module):
    """Combined classification + supervised contrastive loss."""

    def __init__(self, lambda_cls=0.6, temperature=0.07, num_classes=2):
        super().__init__()
        self.lambda_cls  = lambda_cls
        self.temperature = temperature
        self.ce          = nn.CrossEntropyLoss()

    def forward(self, logits, z, labels, prog_labels=None):
        cls_loss   = self.ce(logits, labels)
        target     = prog_labels if prog_labels is not None else labels
        contr_loss = self._nt_xent(z, target)
        total      = self.lambda_cls * cls_loss + (1 - self.lambda_cls) * contr_loss
        return total, cls_loss, contr_loss

    def _nt_xent(self, z, labels):
        n = z.shape[0]
        if n < 2:
            return torch.tensor(0.0, requires_grad=True, device=z.device)

        sim = torch.mm(z, z.t()) / self.temperature
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()).float()
        pos_mask.fill_diagonal_(0)
        neg_mask = 1.0 - pos_mask
        neg_mask.fill_diagonal_(0)

        denom    = (torch.exp(sim) * neg_mask).sum(dim=1, keepdim=True) + 1e-8
        log_prob = sim - torch.log(denom)
        n_pos    = pos_mask.sum(dim=1).clamp(min=1)
        loss     = -(log_prob * pos_mask).sum(dim=1) / n_pos

        return loss.mean()
