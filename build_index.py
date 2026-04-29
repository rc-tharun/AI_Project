"""Build the kNN reference set from cached CLIP features and tune the safety-gate thresholds.

Outputs:
- demo_thresholds.json  (gate config: k, thresholds, val metrics)
- id_bank.npy           (in-distribution training features used by the kNN search)
"""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def knn_mean_l2(query: torch.Tensor, bank: torch.Tensor, k: int, batch: int = 256) -> np.ndarray:
    """Mean L2 distance from each query vector to its k nearest bank vectors. CPU torch."""
    bank_sq = (bank * bank).sum(dim=1)  # (N,)
    out = np.empty(query.shape[0], dtype=np.float32)
    for start in range(0, query.shape[0], batch):
        q = query[start:start + batch]
        q_sq = (q * q).sum(dim=1, keepdim=True)            # (B,1)
        # squared L2: |q|^2 + |b|^2 - 2 q.b^T
        d2 = q_sq + bank_sq.unsqueeze(0) - 2.0 * q @ bank.T
        d2.clamp_(min=0.0)
        topk_sq, _ = torch.topk(d2, k=k, dim=1, largest=False)
        topk = topk_sq.sqrt()
        out[start:start + batch] = topk.mean(dim=1).numpy()
    return out


train_feats = torch.load('Vit+knn/train_clip_features.pt', weights_only=False).float().cpu()
train_ood   = torch.load('Vit+knn/train_clip_ood_labels.pt', weights_only=False).cpu().numpy()
val_feats   = torch.load('Vit+knn/val_clip_features.pt',   weights_only=False).float().cpu()
val_ood     = torch.load('Vit+knn/val_clip_ood_labels.pt', weights_only=False).cpu().numpy()

print(f'train: {tuple(train_feats.shape)}  val: {tuple(val_feats.shape)}')
print(f'train OOD share: {train_ood.mean():.3f}  val OOD share: {val_ood.mean():.3f}')

id_train = train_feats[train_ood == 0]  # the ID-only feature bank used for kNN
print(f'ID train bank: {tuple(id_train.shape)}')

K = 10
val_scores = knn_mean_l2(val_feats, id_train, k=K)

auroc = float(roc_auc_score(val_ood, val_scores))
aupr  = float(average_precision_score(val_ood, val_scores))
print(f'AUROC: {auroc:.4f}   AUPR: {aupr:.4f}')


def tune_trust(scores, labels, budget=0.05):
    thrs = np.quantile(scores, np.linspace(0.01, 0.99, 400))
    best = None
    for t in thrs:
        trust = scores < t
        fsr = float(trust[labels == 1].mean()) if (labels == 1).any() else 0.0
        cov = float(trust.mean())
        if fsr <= budget and (best is None or cov > best['cov']):
            best = {'thr': float(t), 'fsr': fsr, 'cov': cov}
    return best


trust_gate = tune_trust(val_scores, val_ood, 0.05)
print(f'TRUST threshold (false-safe<=5%): {trust_gate}')

id_val = val_scores[val_ood == 0]
abstain_thr = float(np.quantile(id_val, 0.95))
print(f'ABSTAIN threshold (95th pct of ID val score): {abstain_thr:.4f}')

config = {
    'k': K,
    'feature_dim': int(id_train.shape[1]),
    'id_bank_size': int(id_train.shape[0]),
    'trust_threshold': trust_gate['thr'],
    'abstain_threshold': abstain_thr,
    'val_auroc': auroc,
    'val_aupr':  aupr,
    'trust_coverage': trust_gate['cov'],
    'trust_false_safe_rate': trust_gate['fsr'],
    'val_score_min': float(val_scores.min()),
    'val_score_max': float(val_scores.max()),
    'val_score_id_median': float(np.median(id_val)),
    'val_score_ood_median': float(np.median(val_scores[val_ood == 1])),
}
with open('demo_thresholds.json', 'w') as f:
    json.dump(config, f, indent=2)

np.save('id_bank.npy', id_train.numpy())
print('saved demo_thresholds.json + id_bank.npy')
