#!/usr/bin/env python3
"""
ODD + OOD Detection for Safe Autonomy
Train on TAMU HPRC (Grace cluster) with GPU
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_EPOCHS = 20
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
NUM_ENSEMBLE = 3

# Paths — adjust DATA_ROOT to where you put BDD100K on Grace
DATA_ROOT = Path(os.environ.get('SCRATCH', '.')) / 'data'
BDD_RAW = DATA_ROOT / 'raw' / 'bdd100k'
OUTPUT_DIR = Path(os.environ.get('SCRATCH', '.')) / 'safety_gate_output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')


# ─── 1. Dataset Setup ────────────────────────────────────────

WEATHER_BUCKET = {
    'clear': 'in_odd',
    'overcast': 'in_odd',
    'partly cloudy': 'in_odd',
    'rainy': 'rain',
    'foggy': 'fog_snow',
    'snowy': 'fog_snow',
}


def collect_image_paths(bdd_raw, weather_bucket_map):
    buckets = {'in_odd': [], 'rain': [], 'fog_snow': []}
    img_ext = {'.jpg', '.jpeg', '.png'}
    for split in ['train', 'val']:
        split_dir = bdd_raw / split
        if not split_dir.exists():
            continue
        for weather_name, bucket in weather_bucket_map.items():
            folder = split_dir / weather_name
            if not folder.exists():
                continue
            for img in folder.iterdir():
                if img.suffix.lower() in img_ext:
                    buckets[bucket].append((str(img), split))
    return buckets


print(f'Looking for BDD100K data at {BDD_RAW}')
assert BDD_RAW.exists(), f'BDD100K not found at {BDD_RAW}. Download it first (see README).'

buckets = collect_image_paths(BDD_RAW, WEATHER_BUCKET)
print('=== Dataset Bucket Summary ===')
for name, paths in buckets.items():
    splits = Counter(s for _, s in paths)
    print(f'  {name:>10}: {len(paths):>6} images  (train={splits["train"]}, val={splits["val"]})')
print(f'  {"TOTAL":>10}: {sum(len(v) for v in buckets.values()):>6} images')


# ─── 2. Dataset & DataLoaders ────────────────────────────────

class ODD_OOD_Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_splits(buckets, max_in_odd=20000, max_ood_per_type=5000):
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    val_ood_types = []

    in_odd_train = [p for p, s in buckets['in_odd'] if s == 'train']
    in_odd_val = [p for p, s in buckets['in_odd'] if s == 'val']
    random.shuffle(in_odd_train)
    random.shuffle(in_odd_val)
    in_odd_train = in_odd_train[:max_in_odd]
    in_odd_val = in_odd_val[:max_in_odd // 4]

    train_paths.extend(in_odd_train)
    train_labels.extend([0] * len(in_odd_train))
    val_paths.extend(in_odd_val)
    val_labels.extend([0] * len(in_odd_val))
    val_ood_types.extend(['in_odd'] * len(in_odd_val))

    for ood_type in ['rain', 'fog_snow']:
        ood_train = [p for p, s in buckets[ood_type] if s == 'train']
        ood_val = [p for p, s in buckets[ood_type] if s == 'val']
        random.shuffle(ood_train)
        random.shuffle(ood_val)
        ood_train = ood_train[:max_ood_per_type]
        ood_val = ood_val[:max_ood_per_type // 4]

        train_paths.extend(ood_train)
        train_labels.extend([1] * len(ood_train))
        val_paths.extend(ood_val)
        val_labels.extend([1] * len(ood_val))
        val_ood_types.extend([ood_type] * len(ood_val))

    return (train_paths, train_labels), (val_paths, val_labels), val_ood_types


(train_paths, train_labels), (val_paths, val_labels), val_ood_types = build_splits(buckets)

train_ds = ODD_OOD_Dataset(train_paths, train_labels, train_transform)
val_ds = ODD_OOD_Dataset(val_paths, val_labels, val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

print(f'Train: {len(train_ds)} images ({Counter(train_labels)})')
print(f'Val:   {len(val_ds)} images ({Counter(val_labels)})')
print(f'Batches per epoch: {len(train_loader)}')


# ─── 3. Model ────────────────────────────────────────────────

class SafetyGateModel(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        self.frozen_layers = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2,
        )
        self.finetune_layers = nn.Sequential(
            backbone.layer3, backbone.layer4,
        )
        self.avgpool = backbone.avgpool

        for param in self.frozen_layers.parameters():
            param.requires_grad = False
        for param in self.finetune_layers.parameters():
            param.requires_grad = True

        feat_dim = 2048
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def features(self, x):
        with torch.no_grad():
            x = self.frozen_layers(x)
        x = self.finetune_layers(x)
        x = self.avgpool(x)
        return x.flatten(1)

    def forward(self, x):
        feats = self.features(x)
        return self.head(feats)

    def get_features(self, x):
        return self.features(x).detach()


model = SafetyGateModel(num_classes=2).to(DEVICE)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params:     {total:>10,}')
print(f'Trainable params: {trainable:>10,} ({100*trainable/total:.1f}%)')


# ─── 4. Training ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc='Train', leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc='Val', leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW([
    {'params': model.finetune_layers.parameters(), 'lr': LR_BACKBONE},
    {'params': model.head.parameters(), 'lr': LR_HEAD},
], weight_decay=1e-4)


def lr_lambda(epoch):
    warmup_epochs = 3
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (NUM_EPOCHS - warmup_epochs)
    return 0.5 * (1 + np.cos(np.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0

print('\n' + '=' * 60)
print('  TRAINING MAIN MODEL')
print('=' * 60)

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    tag = ''
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), OUTPUT_DIR / 'best_safety_gate.pth')
        tag = ' *best*'

    print(f'Epoch {epoch:2d}/{NUM_EPOCHS} | '
          f'Train Loss={train_loss:.4f} Acc={train_acc:.4f} | '
          f'Val Loss={val_loss:.4f} Acc={val_acc:.4f}{tag}')

model.load_state_dict(torch.load(OUTPUT_DIR / 'best_safety_gate.pth', weights_only=True))
print(f'\nBest validation accuracy: {best_val_acc:.4f}')

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['train_loss'], label='Train')
ax1.plot(history['val_loss'], label='Val')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Loss Curves'); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(history['train_acc'], label='Train')
ax2.plot(history['val_acc'], label='Val')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Curves'); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
plt.close()


# ─── 5. Ensemble Training ────────────────────────────────────

print('\n' + '=' * 60)
print('  TRAINING ENSEMBLE MODELS')
print('=' * 60)

ensemble_models = []
for i in range(NUM_ENSEMBLE):
    print(f'\n--- Ensemble Model {i+1}/{NUM_ENSEMBLE} ---')
    torch.manual_seed(SEED + i * 100)

    m = SafetyGateModel(num_classes=2).to(DEVICE)
    opt = torch.optim.AdamW([
        {'params': m.finetune_layers.parameters(), 'lr': LR_BACKBONE},
        {'params': m.head.parameters(), 'lr': LR_HEAD},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_acc = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        tl, ta = train_one_epoch(m, train_loader, opt, criterion)
        vl, va = evaluate(m, val_loader, criterion)
        sched.step()
        if va > best_acc:
            best_acc = va
            torch.save(m.state_dict(), OUTPUT_DIR / f'ensemble_{i}.pth')
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            print(f'  Epoch {epoch:2d} | Val Acc={va:.4f}')

    m.load_state_dict(torch.load(OUTPUT_DIR / f'ensemble_{i}.pth', weights_only=True))
    m.eval()
    ensemble_models.append(m)
    print(f'  Best Val Acc: {best_acc:.4f}')

print(f'\nEnsemble of {NUM_ENSEMBLE} models ready.')


# ─── 6. OOD Scoring ──────────────────────────────────────────

def score_msp(logits):
    probs = F.softmax(logits, dim=1)
    return probs.max(dim=1).values

def score_energy(logits, temperature=1.0):
    return temperature * torch.logsumexp(logits / temperature, dim=1)

def score_mc_dropout(model, imgs, T=30):
    model.train()
    all_probs = []
    with torch.no_grad():
        feats = model.features(imgs)
        for _ in range(T):
            logits = model.head(feats)
            all_probs.append(F.softmax(logits, dim=1))
    all_probs = torch.stack(all_probs)
    mean_probs = all_probs.mean(dim=0)
    entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)
    expected_entropy = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=2).mean(dim=0)
    mutual_info = entropy - expected_entropy
    model.eval()
    return mean_probs, entropy, mutual_info

def score_ensemble(models_list, imgs):
    all_probs = []
    with torch.no_grad():
        for m in models_list:
            m.eval()
            logits = m(imgs)
            all_probs.append(F.softmax(logits, dim=1))
    all_probs = torch.stack(all_probs)
    mean_probs = all_probs.mean(dim=0)
    entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)
    return mean_probs, entropy


@torch.no_grad()
def collect_scores(model, ensemble_models, loader):
    all_labels, all_msp, all_energy = [], [], []
    all_mc_entropy, all_mc_mi, all_ens_entropy = [], [], []
    all_logits = []

    model.eval()
    for imgs, labels in tqdm(loader, desc='Scoring'):
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        all_msp.append(score_msp(logits).cpu())
        all_energy.append(score_energy(logits).cpu())
        all_logits.append(logits.cpu())
        all_labels.append(labels)
        _, mc_ent, mc_mi = score_mc_dropout(model, imgs, T=30)
        all_mc_entropy.append(mc_ent.cpu())
        all_mc_mi.append(mc_mi.cpu())
        _, ens_ent = score_ensemble(ensemble_models, imgs)
        all_ens_entropy.append(ens_ent.cpu())

    return {
        'labels': torch.cat(all_labels).numpy(),
        'logits': torch.cat(all_logits).numpy(),
        'msp': torch.cat(all_msp).numpy(),
        'energy': torch.cat(all_energy).numpy(),
        'mc_entropy': torch.cat(all_mc_entropy).numpy(),
        'mc_mutual_info': torch.cat(all_mc_mi).numpy(),
        'ensemble_entropy': torch.cat(all_ens_entropy).numpy(),
    }


print('\n' + '=' * 60)
print('  COLLECTING OOD SCORES')
print('=' * 60)

scores = collect_scores(model, ensemble_models, val_loader)
print(f'Collected scores for {len(scores["labels"])} samples')
print(f'In-ODD: {(scores["labels"]==0).sum()}, OOD: {(scores["labels"]==1).sum()}')


# ─── 7. Temperature Scaling ──────────────────────────────────

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def fit_temperature(logits_np, labels_np, lr=0.01, max_iter=200):
    ts = TemperatureScaling().to(DEVICE)
    logits_t = torch.tensor(logits_np, dtype=torch.float32).to(DEVICE)
    labels_t = torch.tensor(labels_np, dtype=torch.long).to(DEVICE)
    optimizer = torch.optim.LBFGS([ts.temperature], lr=lr, max_iter=max_iter)
    crit = nn.CrossEntropyLoss()
    def closure():
        optimizer.zero_grad()
        loss = crit(ts(logits_t), labels_t)
        loss.backward()
        return loss
    optimizer.step(closure)
    return ts.temperature.item()


optimal_temp = fit_temperature(scores['logits'], scores['labels'])
print(f'Optimal temperature T* = {optimal_temp:.4f}')

scores['logits_calibrated'] = scores['logits'] / optimal_temp
scores['probs_calibrated'] = torch.softmax(
    torch.tensor(scores['logits_calibrated']), dim=1).numpy()
scores['probs_uncalibrated'] = torch.softmax(
    torch.tensor(scores['logits']), dim=1).numpy()


# ─── 8. Evaluation ───────────────────────────────────────────

def compute_ece(probs, labels, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_count = mask.sum()
            ece += (bin_count / len(labels)) * abs(bin_acc - bin_conf)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_counts.append(bin_count)
        else:
            bin_accs.append(0); bin_confs.append(0); bin_counts.append(0)
    return ece, bin_accs, bin_confs, bin_counts


def compute_brier_score(probs, labels):
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1
    return ((probs - one_hot) ** 2).sum(axis=1).mean()


def compute_fpr_at_tpr(labels, ood_scores, target_tpr=0.95):
    fpr, tpr, _ = roc_curve(labels, ood_scores)
    idx = np.argmin(np.abs(tpr - target_tpr))
    return fpr[idx]


def false_safe_rate(labels, ood_scores, threshold):
    ood_mask = labels == 1
    if ood_mask.sum() == 0:
        return 0.0
    return (ood_scores[ood_mask] < threshold).mean()


labels = scores['labels']

ood_scores = {
    'MSP (baseline)':     -scores['msp'],
    'Energy Score':       -scores['energy'],
    'MC Dropout Entropy': scores['mc_entropy'],
    'MC Mutual Info':     scores['mc_mutual_info'],
    'Ensemble Entropy':   scores['ensemble_entropy'],
}

print('\n' + '=' * 60)
print('  OOD DETECTION RESULTS')
print('=' * 60)
print(f'{"Method":<25} {"AUROC":>7} {"AUPR":>7} {"FPR@95":>8}')
print('-' * 50)

results = {}
for name, s in ood_scores.items():
    auroc = roc_auc_score(labels, s)
    aupr = average_precision_score(labels, s)
    fpr95 = compute_fpr_at_tpr(labels, s, target_tpr=0.95)
    results[name] = {'auroc': auroc, 'aupr': aupr, 'fpr95': fpr95}
    print(f'{name:<25} {auroc:>7.4f} {aupr:>7.4f} {fpr95:>8.4f}')

# Calibration
ece_before, _, _, _ = compute_ece(scores['probs_uncalibrated'], labels)
ece_after, _, _, _ = compute_ece(scores['probs_calibrated'], labels)
brier_before = compute_brier_score(scores['probs_uncalibrated'], labels)
brier_after = compute_brier_score(scores['probs_calibrated'], labels)

print(f'\n=== Calibration Metrics ===')
print(f'{"":20} {"Before T-scaling":>18} {"After T-scaling":>18}')
print(f'{"ECE":20} {ece_before:>18.4f} {ece_after:>18.4f}')
print(f'{"Brier Score":20} {brier_before:>18.4f} {brier_after:>18.4f}')
print(f'Temperature T* = {optimal_temp:.4f}')


# ─── 9. Safety Gate Analysis ─────────────────────────────────

def safety_gate_analysis(labels, ood_score, method_name, n_thresholds=200):
    thresholds = np.linspace(ood_score.min(), ood_score.max(), n_thresholds)
    fsr_list, abstain_list = [], []
    for tau in thresholds:
        accepted = ood_score <= tau
        abstain_rate = 1 - accepted.mean()
        ood_mask = labels == 1
        fsr = (accepted & ood_mask).sum() / ood_mask.sum() if ood_mask.sum() > 0 else 0
        fsr_list.append(fsr)
        abstain_list.append(abstain_rate)
    return thresholds, np.array(fsr_list), np.array(abstain_list)


best_method = 'Energy Score'
best_score = ood_scores[best_method]

thresholds, fsr_arr, abstain_arr = safety_gate_analysis(labels, best_score, best_method)

target_fsr = 0.05
valid = fsr_arr <= target_fsr
if valid.any():
    best_idx = np.where(valid)[0][np.argmin(abstain_arr[valid])]
    tau_star = thresholds[best_idx]
    print(f'\nOptimal threshold tau* = {tau_star:.4f}')
    print(f'  FSR = {fsr_arr[best_idx]:.4f} (target <= {target_fsr})')
    print(f'  Abstain rate = {abstain_arr[best_idx]:.4f}')
else:
    tau_star = np.median(thresholds)
    best_idx = len(thresholds) // 2
    print(f'Could not achieve FSR <= {target_fsr}, using median threshold')

gate_decisions = np.where(best_score > tau_star, 'ABSTAIN', 'PROCEED')
gate_decisions = np.where(
    (best_score > tau_star * 0.8) & (best_score <= tau_star),
    'SLOW_DOWN', gate_decisions)

print('\n=== Safety Gate Decisions ===')
for decision in ['PROCEED', 'SLOW_DOWN', 'ABSTAIN']:
    mask = gate_decisions == decision
    n = mask.sum()
    n_ood = (labels[mask] == 1).sum() if n > 0 else 0
    n_id = (labels[mask] == 0).sum() if n > 0 else 0
    print(f'  {decision:>10}: {n:>5} frames  (In-ODD={n_id}, OOD={n_ood})')

# Per-type analysis
ood_type_arr = np.array(val_ood_types)
print(f'\n{"OOD Type":<15} {"Count":>6} {"AUROC":>7} {"Avg Score":>10} {"FSR@tau*":>9}')
print('-' * 55)
for ood_type in ['rain', 'fog_snow']:
    type_mask = (ood_type_arr == 'in_odd') | (ood_type_arr == ood_type)
    type_labels = (ood_type_arr[type_mask] != 'in_odd').astype(int)
    type_scores = best_score[type_mask]
    n_ood = (ood_type_arr == ood_type).sum()
    auroc = roc_auc_score(type_labels, type_scores)
    avg_score = type_scores[type_labels == 1].mean()
    ood_accepted = (type_scores[type_labels == 1] <= tau_star).mean()
    print(f'{ood_type:<15} {n_ood:>6} {auroc:>7.4f} {avg_score:>10.4f} {ood_accepted:>9.4f}')


# ─── 10. Visualizations ──────────────────────────────────────

# ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
for name, s in ood_scores.items():
    fpr, tpr, _ = roc_curve(labels, s)
    auroc = roc_auc_score(labels, s)
    ax.plot(fpr, tpr, label=f'{name} (AUROC={auroc:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — OOD Detection Methods'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# Score Distributions
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(best_score[labels == 0], bins=60, alpha=0.6, label='In-ODD', color='#2ecc71', density=True)
ax.hist(best_score[labels == 1], bins=60, alpha=0.6, label='OOD', color='#e74c3c', density=True)
ax.axvline(tau_star, color='black', linestyle='--', linewidth=2, label=f'tau*={tau_star:.2f}')
ax.set_xlabel('Energy Score (negated)'); ax.set_ylabel('Density')
ax.set_title('Score Distribution — In-ODD vs OOD'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'score_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Reliability Diagram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for ax, probs, title, ece_val in [
    (ax1, scores['probs_uncalibrated'], 'Before Temperature Scaling', ece_before),
    (ax2, scores['probs_calibrated'], f'After Temperature Scaling (T*={optimal_temp:.2f})', ece_after),
]:
    _, bin_accs, bin_confs, bin_counts = compute_ece(probs, labels, n_bins=10)
    bin_centers = np.linspace(0.05, 0.95, 10)
    ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color='#3498db', label='Accuracy')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
    ax.set_title(f'{title}\nECE = {ece_val:.4f}')
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'reliability_diagram.png', dpi=150, bbox_inches='tight')
plt.close()

# FSR vs Abstain Rate
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(abstain_arr, fsr_arr, linewidth=2, color='#e74c3c')
ax.axhline(target_fsr, color='gray', linestyle='--', alpha=0.5, label=f'Target FSR={target_fsr}')
if valid.any():
    ax.scatter([abstain_arr[best_idx]], [fsr_arr[best_idx]],
               s=100, c='black', zorder=5, label=f'Operating point (FSR={fsr_arr[best_idx]:.3f})')
ax.set_xlabel('Abstain Rate'); ax.set_ylabel('False-Safe Rate (FSR)')
ax.set_title('Safety-Coverage Tradeoff (Energy Score)'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fsr_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()

# Risk-Coverage Curve
def risk_coverage_curve(labels, ood_score):
    order = np.argsort(ood_score)
    sorted_labels = labels[order]
    coverages, risks = [], []
    n = len(labels)
    for k in range(1, n + 1, max(1, n // 200)):
        coverage = k / n
        risk = (sorted_labels[:k] == 1).mean()
        coverages.append(coverage)
        risks.append(risk)
    return np.array(coverages), np.array(risks)

fig, ax = plt.subplots(figsize=(8, 5))
for name, s in [('MSP (baseline)', ood_scores['MSP (baseline)']),
                ('Energy Score', ood_scores['Energy Score']),
                ('Ensemble Entropy', ood_scores['Ensemble Entropy'])]:
    cov, risk = risk_coverage_curve(labels, s)
    ax.plot(cov, risk, linewidth=2, label=name)
ax.set_xlabel('Coverage'); ax.set_ylabel('Risk (OOD error rate)')
ax.set_title('Risk-Coverage Curve'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'risk_coverage.png', dpi=150, bbox_inches='tight')
plt.close()

# Confusion Matrix
preds = (best_score > tau_star).astype(int)
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=['In-ODD', 'OOD'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title(f'Confusion Matrix — Energy Score (tau*={tau_star:.2f})')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
acc = accuracy_score(labels, preds)

# Method Comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
methods = list(results.keys())
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
for ax, metric, title in zip(axes, ['auroc', 'aupr', 'fpr95'],
                              ['AUROC (higher=better)', 'AUPR (higher=better)', 'FPR@95%TPR (lower=better)']):
    vals = [results[m][metric] for m in methods]
    short_names = ['MSP', 'Energy', 'MC Ent.', 'MC MI', 'Ens. Ent.']
    bars = ax.bar(short_names, vals, color=colors)
    ax.set_title(title); ax.set_ylim(0, 1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
plt.suptitle('OOD Detection Method Comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'method_comparison.png', dpi=150, bbox_inches='tight')
plt.close()


# ─── 11. Final Summary ───────────────────────────────────────

print('\n' + '=' * 65)
print('  ODD + OOD SAFETY GATE — FINAL RESULTS')
print('=' * 65)
print(f'\n  Dataset: BDD100K (weather-classified)')
print(f'  In-ODD: Day + Clear/Overcast ({(labels==0).sum()} val samples)')
print(f'  OOD:    Rain + Fog/Snow      ({(labels==1).sum()} val samples)')
print(f'\n  Backbone: ResNet-50 (layer3+layer4 fine-tuned)')
print(f'  Head:     MLP (2048 -> 512 -> 128 -> 2) with BatchNorm')
print(f'  Best Val Accuracy: {best_val_acc:.4f}')
print(f'\n  --- OOD Detection ---')
print(f'  {"Method":<25} {"AUROC":>7} {"AUPR":>7} {"FPR@95":>8}')
print(f'  {"-"*50}')
for name, r in results.items():
    print(f'  {name:<25} {r["auroc"]:>7.4f} {r["aupr"]:>7.4f} {r["fpr95"]:>8.4f}')
print(f'\n  --- Calibration ---')
print(f'  Temperature T* = {optimal_temp:.4f}')
print(f'  ECE:   {ece_before:.4f} -> {ece_after:.4f}')
print(f'  Brier: {brier_before:.4f} -> {brier_after:.4f}')
print(f'\n  --- Safety Gate (Energy Score) ---')
print(f'  Threshold tau* = {tau_star:.4f}')
if valid.any():
    print(f'  FSR = {fsr_arr[best_idx]:.4f}')
    print(f'  Abstain Rate = {abstain_arr[best_idx]:.4f}')
print(f'  Gate Accuracy = {acc:.4f}')
print(f'\n  Output saved to: {OUTPUT_DIR}')
print('=' * 65)
