import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

os.makedirs('results', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(1, hidden_dim)
        self.drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.proj(x)
        x = self.drop(x)
        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)
        return self.norm(x).mean(dim=0)


class HybridTransformer(nn.Module):
    def __init__(self, gene_dim, meth_dim, path_dim, imm_dim,
                 hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.gene_enc = ModalityEncoder(gene_dim, hidden_dim, 4, 2, dropout)
        self.meth_enc = ModalityEncoder(meth_dim, hidden_dim, 4, 2, dropout)
        self.path_enc = ModalityEncoder(path_dim, hidden_dim, 4, 2, dropout)
        self.imm_enc  = ModalityEncoder(imm_dim,  hidden_dim, 4, 2, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, gene, meth, path, imm):
        h = torch.cat([
            self.gene_enc(gene), self.meth_enc(meth),
            self.path_enc(path), self.imm_enc(imm)
        ], dim=-1)
        return self.classifier(h), h


def load_data():
    gene  = pd.read_csv('data/gene_expression_75.csv',   index_col=0)
    meth  = pd.read_csv('data/methylation_proxy.csv',    index_col=0)
    path  = pd.read_csv('data/pathway_faime_scores.csv', index_col=0)
    imm   = pd.read_csv('data/immune_fractions_fixed.csv', index_col=0)
    labs  = pd.read_csv('data/sample_labels.csv').set_index('sample_id')

    common = sorted(set(gene.index) & set(meth.index) & set(path.index) & set(imm.index) & set(labs.index))
    gene, meth, path, imm, labs = [df.loc[common] for df in [gene, meth, path, imm, labs]]

    print(f"Loaded {len(common)} samples | Labels: {dict(labs['label'].value_counts())}")
    print(f"Gene:{gene.shape} Meth:{meth.shape} Path:{path.shape} Imm:{imm.shape}")

    mask = labs['label'].isin([1, 2])
    gene_b, meth_b, path_b, imm_b, labs_b = [
        df.loc[mask] for df in [gene, meth, path, imm, labs]
    ]
    labs_b = labs_b.copy()
    labs_b['label'] = labs_b['label'].map({1: 0, 2: 1})

    print(f"Training: DCIS={int((labs_b['label']==0).sum())} IBC={int((labs_b['label']==1).sum())}")
    return gene_b, meth_b, path_b, imm_b, labs_b


def scale_fold(gene_b, meth_b, path_b, imm_b, tr_idx, vl_idx):
    results = []
    for df in [gene_b, meth_b, path_b, imm_b]:
        sc = StandardScaler()
        tr = sc.fit_transform(df.iloc[tr_idx].values.astype(np.float32))
        vl = sc.transform(df.iloc[vl_idx].values.astype(np.float32))
        results.extend([tr, vl])
    return results  # g_tr, g_vl, m_tr, m_vl, p_tr, p_vl, i_tr, i_vl


def train_epoch(model, optimizer, g, m, p, i, y, device, batch_size=32):
    model.train()
    criterion = nn.CrossEntropyLoss()
    idx = np.random.permutation(len(y))
    total_loss = 0
    for start in range(0, len(y), batch_size):
        batch = idx[start:start + batch_size]
        g_b = torch.tensor(g[batch]).to(device)
        m_b = torch.tensor(m[batch]).to(device)
        p_b = torch.tensor(p[batch]).to(device)
        i_b = torch.tensor(i[batch]).to(device)
        y_b = torch.tensor(y[batch]).to(device)
        optimizer.zero_grad()
        logits = torch.cat([model(g_b[s], m_b[s], p_b[s], i_b[s])[0].unsqueeze(0)
                            for s in range(len(batch))])
        loss = criterion(logits, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss


@torch.no_grad()
def evaluate(model, g, m, p, i, y, device):
    model.eval()
    probs, preds = [], []
    for s in range(len(y)):
        logits, _ = model(
            torch.tensor(g[s]).to(device), torch.tensor(m[s]).to(device),
            torch.tensor(p[s]).to(device), torch.tensor(i[s]).to(device)
        )
        prob = torch.softmax(logits, dim=-1)[1].item()
        probs.append(prob)
        preds.append(logits.argmax().item())
    auc = roc_auc_score(y, probs) if len(set(y)) > 1 else 0.5
    acc = accuracy_score(y, preds)
    f1  = f1_score(y, preds, average='macro', zero_division=0)
    return acc, f1, auc, np.array(preds), np.array(probs)


def run_crossval(gene_b, meth_b, path_b, imm_b, labs_b, device):
    print("\n=== 5-fold Stratified Cross-Validation ===")
    y_all = labs_b['label'].values.astype(np.int64)
    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_probs = np.zeros(len(y_all))
    all_preds = np.zeros(len(y_all), dtype=int)
    all_embs  = {}
    fold_results = []
    sids = labs_b.index.tolist()

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all)):
        print(f"\n--- Fold {fold+1}/5 ---")
        scaled = scale_fold(gene_b, meth_b, path_b, imm_b, tr_idx, vl_idx)
        g_tr, g_vl = scaled[0], scaled[1]
        m_tr, m_vl = scaled[2], scaled[3]
        p_tr, p_vl = scaled[4], scaled[5]
        i_tr, i_vl = scaled[6], scaled[7]
        y_tr = y_all[tr_idx]
        y_vl = y_all[vl_idx]

        model = HybridTransformer(g_tr.shape[1], m_tr.shape[1], p_tr.shape[1], i_tr.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        best_auc, patience, best_state = 0, 0, None
        for epoch in range(50):
            train_epoch(model, optimizer, g_tr, m_tr, p_tr, i_tr, y_tr, device)
            scheduler.step()
            _, _, auc, _, _ = evaluate(model, g_vl, m_vl, p_vl, i_vl, y_vl, device)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1} | Val AUC {auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    print(f"  Early stop at epoch {epoch+1}")
                    break

        model.load_state_dict(best_state)
        acc, f1, auc, preds, probs = evaluate(model, g_vl, m_vl, p_vl, i_vl, y_vl, device)
        print(f"  Fold {fold+1} — Acc {acc:.4f} F1 {f1:.4f} AUC {auc:.4f}")
        fold_results.append({'fold': fold+1, 'accuracy': acc, 'f1': f1, 'auc': auc})

        all_probs[vl_idx] = probs
        all_preds[vl_idx] = preds

        model.eval()
        with torch.no_grad():
            for li, gi in enumerate(vl_idx):
                _, h = model(
                    torch.tensor(g_vl[li]).to(device), torch.tensor(m_vl[li]).to(device),
                    torch.tensor(p_vl[li]).to(device), torch.tensor(i_vl[li]).to(device)
                )
                all_embs[sids[gi]] = h.cpu().numpy()

        torch.save(model.state_dict(), f'checkpoints/fold{fold+1}_best.pt')

    print("\n=== Cross-Validation Summary ===")
    accs = [r['accuracy'] for r in fold_results]
    f1s  = [r['f1']       for r in fold_results]
    aucs = [r['auc']      for r in fold_results]
    print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"F1 Macro : {np.mean(f1s):.4f}  ± {np.std(f1s):.4f}")
    print(f"AUC      : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Overall held-out AUC: {roc_auc_score(y_all, all_probs):.4f}")

    pd.DataFrame(fold_results).to_csv('results/crossval_results.csv', index=False)
    return all_probs, all_preds, y_all, all_embs


def score_dcis(gene_b, meth_b, path_b, imm_b, labs_b, fold_results, device):
    print("\n=== Invasive-like DCIS Scoring ===")
    best_fold = int(pd.DataFrame(fold_results).sort_values('auc').iloc[-1]['fold'])
    print(f"Using fold {best_fold} model")

    sc_g = StandardScaler().fit(gene_b.values.astype(np.float32))
    sc_m = StandardScaler().fit(meth_b.values.astype(np.float32))
    sc_p = StandardScaler().fit(path_b.values.astype(np.float32))
    sc_i = StandardScaler().fit(imm_b.values.astype(np.float32))

    g_dim = gene_b.shape[1]; m_dim = meth_b.shape[1]
    p_dim = path_b.shape[1]; i_dim = imm_b.shape[1]
    model = HybridTransformer(g_dim, m_dim, p_dim, i_dim, dropout=0.0).to(device)
    model.load_state_dict(torch.load(f'checkpoints/fold{best_fold}_best.pt', map_location=device))
    model.eval()

    dcis_ids = labs_b[labs_b['label'] == 0].index.tolist()
    scores = {}
    with torch.no_grad():
        for sid in dcis_ids:
            g = torch.tensor(sc_g.transform(gene_b.loc[[sid]].values.astype(np.float32))[0]).to(device)
            m = torch.tensor(sc_m.transform(meth_b.loc[[sid]].values.astype(np.float32))[0]).to(device)
            p = torch.tensor(sc_p.transform(path_b.loc[[sid]].values.astype(np.float32))[0]).to(device)
            i = torch.tensor(sc_i.transform(imm_b.loc[[sid]].values.astype(np.float32))[0]).to(device)
            logits, _ = model(g, m, p, i)
            scores[sid] = logits[1].item()

    vals = np.array(list(scores.values()))
    vmin, vmax = vals.min(), vals.max()
    for sid in scores:
        scores[sid] = (scores[sid] - vmin) / (vmax - vmin + 1e-8)

    threshold = float(np.percentile(list(scores.values()), 66.7))
    print(f"Threshold (66.7th percentile): {threshold:.4f}")

    rows = [{'sample_id': sid, 'invasive_score': scores[sid],
             'invasive_like': scores[sid] >= threshold} for sid in scores]
    df = pd.DataFrame(rows).sort_values('invasive_score', ascending=False)
    df.to_csv('results/invasive_like_dcis.csv', index=False)

    n_inv = df['invasive_like'].sum()
    print(f"Invasive-like DCIS: {n_inv}/{len(dcis_ids)} ({100*n_inv/len(dcis_ids):.1f}%)")
    return df


def plot_umap(all_embs, labs_b, device):
    if not HAS_UMAP or not HAS_PLOT:
        print("Skipping UMAP — install umap-learn and matplotlib")
        return

    print("\n=== UMAP ===")
    inv_df = pd.read_csv('results/invasive_like_dcis.csv', index_col='sample_id')
    sids = list(all_embs.keys())
    emb_matrix = np.array([all_embs[s] for s in sids])
    labels = labs_b.loc[sids, 'label'].values

    reducer  = umap.UMAP(n_neighbors=12, min_dist=0.05, random_state=42, metric='cosine', spread=1.2)
    embedding = reducer.fit_transform(emb_matrix)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    C = {'dcis': '#4878D0', 'ibc': '#EE854A', 'invasive': '#D65F5F', 'dcislike': '#6ACC65'}

    ax = axes[0]
    for cls, label, col, mk in [(0,'DCIS','#4878D0','o'), (1,'IBC','#EE854A','s')]:
        mask = labels == cls
        ax.scatter(embedding[mask, 0], embedding[mask, 1], c=col, label=label,
                   marker=mk, s=65, alpha=0.8, edgecolors='white', linewidths=0.6)
    ax.set_title('(A) DCIS vs IBC — held-out embeddings', fontweight='bold')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.legend(); ax.grid(True, alpha=0.15)

    ax = axes[1]
    ax.scatter(embedding[labels==1, 0], embedding[labels==1, 1], c='#EE854A',
               label='IBC', marker='s', s=55, alpha=0.5, edgecolors='white', linewidths=0.5)
    inv_idx  = [i for i, s in enumerate(sids) if labels[i]==0 and s in inv_df.index and inv_df.loc[s,'invasive_like']]
    dcis_idx = [i for i, s in enumerate(sids) if labels[i]==0 and (s not in inv_df.index or not inv_df.loc[s,'invasive_like'])]

    if dcis_idx:
        ax.scatter(embedding[dcis_idx, 0], embedding[dcis_idx, 1], c='#6ACC65',
                   label=f'DCIS-like (n={len(dcis_idx)})', marker='o', s=65, alpha=0.8,
                   edgecolors='white', linewidths=0.6)
    if inv_idx:
        ax.scatter(embedding[inv_idx, 0], embedding[inv_idx, 1], c='#D65F5F',
                   label=f'Invasive-like (n={len(inv_idx)})', marker='^', s=100, alpha=0.95,
                   edgecolors='black', linewidths=0.8, zorder=5)

    ax.set_title(f'(B) Invasive-like DCIS at IBC boundary', fontweight='bold')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig('results/figures/Fig5A_UMAP_heldout.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/Fig5A_UMAP_heldout.png")


def run_shap(gene_b, meth_b, path_b, imm_b, labs_b, device):
    if not HAS_SHAP or not HAS_PLOT:
        print("Skipping SHAP — install shap and matplotlib")
        return

    print("\n=== SHAP Analysis ===")
    sc_g = StandardScaler().fit(gene_b.values.astype(np.float32))
    sc_m = StandardScaler().fit(meth_b.values.astype(np.float32))
    sc_p = StandardScaler().fit(path_b.values.astype(np.float32))
    sc_i = StandardScaler().fit(imm_b.values.astype(np.float32))

    g = sc_g.transform(gene_b.values.astype(np.float32))
    m_z = torch.zeros(meth_b.shape[1]).to(device)
    p_z = torch.zeros(path_b.shape[1]).to(device)
    i_z = torch.zeros(imm_b.shape[1]).to(device)
    gene_cols = gene_b.columns.tolist()

    model = HybridTransformer(
        gene_b.shape[1], meth_b.shape[1], path_b.shape[1], imm_b.shape[1], dropout=0.0
    ).to(device)

    best_fold = int(pd.read_csv('results/crossval_results.csv').sort_values('auc').iloc[-1]['fold'])
    model.load_state_dict(torch.load(f'checkpoints/fold{best_fold}_best.pt', map_location=device))
    model.eval()

    def predict_logodds(gene_np):
        results = []
        with torch.no_grad():
            for row in gene_np:
                logits, _ = model(torch.tensor(row.astype(np.float32)).to(device), m_z, p_z, i_z)
                results.append(logits[1].item())
        return np.array(results)

    print(f"  Computing SHAP on log-odds scale (n={len(g)})...")
    background  = shap.kmeans(g, 20)
    explainer   = shap.KernelExplainer(predict_logodds, background)
    shap_values = explainer.shap_values(g, nsamples=100, silent=True)

    mean_shap = np.abs(shap_values).mean(axis=0)
    top20_idx = np.argsort(mean_shap)[::-1][:20]
    top20_genes = [gene_cols[j] for j in top20_idx]
    top20_shap  = mean_shap[top20_idx]
    top20_sv    = shap_values[:, top20_idx]
    top20_expr  = g[:, top20_idx]

    pd.DataFrame({'gene': top20_genes, 'mean_abs_shap': top20_shap}).to_csv(
        'results/shap_top20_logodds.csv', index=False)

    print("\n  Top 10 SHAP genes (log-odds scale):")
    for rank, (gene, val) in enumerate(zip(top20_genes[:10], top20_shap[:10]), 1):
        print(f"    {rank:2d}. {gene:14s} |SHAP|={val:.6f}")

    os.makedirs('results/figures', exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.RdBu_r
    for yi in range(len(top20_genes)):
        sv   = top20_sv[:, yi]
        expr = top20_expr[:, yi]
        e_norm = (expr - expr.min()) / (expr.max() - expr.min() + 1e-8)
        np.random.seed(yi)
        jitter = np.random.uniform(-0.35, 0.35, len(sv))
        ax.scatter(sv, yi + jitter, c=e_norm, cmap=cmap, vmin=0, vmax=1,
                   alpha=0.65, s=28, linewidths=0)

    ax.axvline(0, color='#333333', lw=1.0, linestyle='--', alpha=0.6)
    ax.set_yticks(range(len(top20_genes)))
    ax.set_yticklabels(top20_genes, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('SHAP value (log-odds scale)', fontsize=11)
    ax.set_title(f'SHAP Feature Importance — Top 20 Gene Drivers\n(n={len(g)} samples)', fontsize=11)
    ax.grid(True, axis='x', alpha=0.25)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.35, pad=0.03, aspect=15)
    cbar.set_ticks([0.05, 0.95])
    cbar.set_ticklabels(['Low\nexpression', 'High\nexpression'], fontsize=8)

    plt.tight_layout()
    plt.savefig('results/figures/Fig5B_SHAP_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/Fig5B_SHAP_beeswarm.png")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    gene_b, meth_b, path_b, imm_b, labs_b = load_data()

    all_probs, all_preds, y_all, all_embs = run_crossval(
        gene_b, meth_b, path_b, imm_b, labs_b, device
    )

    fold_results = pd.read_csv('results/crossval_results.csv').to_dict('records')
    score_df = score_dcis(gene_b, meth_b, path_b, imm_b, labs_b, fold_results, device)

    os.makedirs('results/figures', exist_ok=True)
    plot_umap(all_embs, labs_b, device)
    run_shap(gene_b, meth_b, path_b, imm_b, labs_b, device)

    print("\n=== Done ===")
    print("Results saved to results/")
