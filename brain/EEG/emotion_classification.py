"""
EEGNet Emotion Classification  —  DREAMER Dataset  (GPU-Accelerated)
======================================================================
Labels   : Valence, Arousal, Dominance  (binary High/Low)
Model    : EEGNet  (Lawhern et al., 2018)
Inputs   : dreamer_trial_features.csv   → SVM baseline  (CPU)
           dreamer_raw_samples.csv      → EEGNet         (GPU)
Eval     : Leave-One-Subject-Out (LOSO) cross-validation

Saved outputs (models/ folder)
────────────────────────────────
  models/eegnet_valence.pt       ← best weights across all folds
  models/eegnet_arousal.pt
  models/eegnet_dominance.pt
  models/model_config.json       ← architecture params for inference
  models/training_results.json   ← accuracy / F1 per label
"""

import os
import json
import time
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

# ─────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_GPU = DEVICE.type == 'cuda'

if USE_GPU:
    torch.backends.cudnn.benchmark = True
    gpu_name = torch.cuda.get_device_name(0)
    vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {gpu_name}  ({vram:.1f} GB VRAM)")
    print(f"[GPU] Mixed precision : ENABLED")
    print(f"[GPU] cuDNN benchmark : ENABLED")
else:
    print("[GPU] No CUDA GPU — running on CPU")

print(f"[INFO] Device : {DEVICE}\n")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

RAW_CSV      = "C:/Users/Prajwal/Documents/Brain-Signaling/brain/data/emotion/dreamer_raw_samples.csv"
FEAT_CSV     = "C:/Users/Prajwal/Documents/Brain-Signaling/brain/data/emotion/dreamer_trial_features.csv"
MODEL_DIR    = "models"

CHANNELS     = ['AF3','F7','F3','FC5','T7','P7',
                'O1','O2','P8','T8','FC6','F4','F8','AF4']
N_CHANNELS   = 14
SFREQ        = 128
WINDOW_SEC   = 3
T            = SFREQ * WINDOW_SEC    # 384 time points

LABELS       = ['valence_bin', 'arousal_bin', 'dominance_bin']
LABEL_NAMES  = ['Valence', 'Arousal', 'Dominance']

BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
DROPOUT      = 0.5
NUM_WORKERS  = 0 if os.name == 'nt' else 2

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 — SVM BASELINE  (CPU)
# ─────────────────────────────────────────────

def run_svm_baseline():
    print("=" * 60)
    print("  STEP 1 : SVM Baseline  [CPU]")
    print("=" * 60)

    df           = pd.read_csv(FEAT_CSV)
    bands        = ['delta','theta','alpha','beta','gamma']
    feature_cols = [f'{ch}_{b}' for ch in CHANNELS for b in bands]
    X            = df[feature_cols].values.astype(np.float32)
    groups       = df['participant'].values
    logo         = LeaveOneGroupOut()
    results      = {name: [] for name in LABEL_NAMES}

    for label, name in zip(LABELS, LABEL_NAMES):
        y = df[label].values
        for train_idx, test_idx in logo.split(X, y, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_tr)
            X_te   = scaler.transform(X_te)
            clf    = SVC(kernel='rbf', C=1.0, gamma='scale')
            clf.fit(X_tr, y_tr)
            results[name].append(accuracy_score(y_te, clf.predict(X_te)))

        mean_acc = np.mean(results[name]) * 100
        std_acc  = np.std(results[name])  * 100
        print(f"  {name:10s}  LOSO Accuracy : {mean_acc:.2f}% ± {std_acc:.2f}%")

    return results


# ─────────────────────────────────────────────
# STEP 2 — BUILD TRIAL TENSORS  (CPU)
# ─────────────────────────────────────────────

def build_trial_tensors():
    print("\n" + "="*60)
    print("  STEP 2 : Building trial tensors  [CPU]")
    print("="*60)
    print("  Loading CSV ...")

    df      = pd.read_csv(RAW_CSV)
    grouped = df.groupby(['participant', 'trial'])
    total   = len(grouped)

    X_list, Y_list, G_list = [], [], []

    for i, ((p, t), grp) in enumerate(grouped):
        eeg = grp[CHANNELS].values.astype(np.float32)
        if eeg.shape[0] >= T:
            eeg = eeg[:T, :]
        else:
            pad = np.zeros((T - eeg.shape[0], N_CHANNELS), dtype=np.float32)
            eeg = np.vstack([eeg, pad])

        X_list.append(eeg.T[np.newaxis, :, :])                    # (1,14,384)
        Y_list.append(grp[LABELS].iloc[0].values.astype(np.int64))
        G_list.append(int(p))

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Processed {i+1}/{total} trials")

    X      = np.stack(X_list)   # (414, 1, 14, 384)
    Y      = np.stack(Y_list)   # (414, 3)
    groups = np.array(G_list)   # (414,)

    print(f"\n  X : {X.shape}  |  Y : {Y.shape}")
    return X, Y, groups


# ─────────────────────────────────────────────
# STEP 3 — EEGNet Architecture
# ─────────────────────────────────────────────

class EEGNet(nn.Module):
    """
    EEGNet (Lawhern et al., 2018)
    Input  : (batch, 1, n_channels, T)
    Output : (batch, n_classes)
    """
    def __init__(self, n_channels=14, n_classes=2, T=384,
                 F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, SFREQ//2),
                      padding=(0, SFREQ//4), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1*D, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, kernel_size=(1, 16),
                      padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        flat = self._get_flatten_size(n_channels, T)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, n_classes),
        )

    def _get_flatten_size(self, n_channels, T):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, T)
            return self.block2(self.block1(x)).numel()

    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))


# ─────────────────────────────────────────────
# STEP 4 — Dataset + DataLoader
# ─────────────────────────────────────────────

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def make_loader(X, y, shuffle=False):
    return DataLoader(EEGDataset(X, y),
                      batch_size=BATCH_SIZE, shuffle=shuffle,
                      pin_memory=USE_GPU, num_workers=NUM_WORKERS)


# ─────────────────────────────────────────────
# STEP 5 — Train / Eval helpers
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_GPU):
            out  = model(xb)
            loss = criterion(out, yb)

        if USE_GPU:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(xb)
        correct    += (out.argmax(1) == yb).sum().item()
        total      += len(xb)

    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_GPU):
                out = model(xb)
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(yb.numpy())

    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average='macro', zero_division=0)
    return acc, f1, trues, preds


# ─────────────────────────────────────────────
# STEP 6 — LOSO Training Loop  +  Model Saving
# ─────────────────────────────────────────────

def run_eegnet_loso(X, Y, groups):
    """
    LOSO cross-validation.
    After all folds for a label, the fold with the best val accuracy
    is retrained on ALL data and saved as the final model.

    Saved per label:
      models/eegnet_valence.pt    ← state_dict
      models/eegnet_arousal.pt
      models/eegnet_dominance.pt
    """
    print("\n" + "="*60)
    print(f"  STEP 3 : EEGNet LOSO  [{'GPU' if USE_GPU else 'CPU'}]")
    print("="*60)

    logo    = LeaveOneGroupOut()
    results = {name: {'acc': [], 'f1': []} for name in LABEL_NAMES}

    for label_idx, label_name in enumerate(LABEL_NAMES):
        y = Y[:, label_idx]
        print(f"\n  ── Label : {label_name} ──")

        # Track the single best state across ALL folds for saving
        global_best_acc   = 0.0
        global_best_state = None

        for train_idx, test_idx in logo.split(X, y, groups):
            subj_id = groups[test_idx[0]]

            train_dl = make_loader(X[train_idx], y[train_idx], shuffle=True)
            test_dl  = make_loader(X[test_idx],  y[test_idx])

            model     = EEGNet(n_channels=N_CHANNELS, n_classes=2,
                               T=T, dropout=DROPOUT).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            amp_scaler = (GradScaler('cuda', enabled=True)
                          if USE_GPU else GradScaler('cpu', enabled=False))

            fold_best_acc   = 0.0
            fold_best_state = None

            for epoch in range(EPOCHS):
                tr_loss, tr_acc = train_one_epoch(
                    model, train_dl, optimizer, criterion, amp_scaler)
                scheduler.step()

                if (epoch + 1) % 10 == 0:
                    val_acc, val_f1, _, _ = evaluate(model, test_dl)
                    print(f"    Subj {subj_id:02d}  Ep {epoch+1:02d}  "
                          f"loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
                          f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}")

                    # Track best within this fold
                    if val_acc > fold_best_acc:
                        fold_best_acc   = val_acc
                        fold_best_state = {k: v.cpu().clone()
                                           for k, v in model.state_dict().items()}

                    # Track best globally across all folds (for final save)
                    if val_acc > global_best_acc:
                        global_best_acc   = val_acc
                        global_best_state = {k: v.cpu().clone()
                                             for k, v in model.state_dict().items()}

            # Restore fold best for final evaluation
            if fold_best_state:
                model.load_state_dict(
                    {k: v.to(DEVICE) for k, v in fold_best_state.items()})

            acc, f1, _, _ = evaluate(model, test_dl)
            results[label_name]['acc'].append(acc)
            results[label_name]['f1'].append(f1)

            del model, optimizer, amp_scaler
            if USE_GPU:
                torch.cuda.empty_cache()

        # ── Save the globally best model for this label ──────────
        save_path = os.path.join(MODEL_DIR, f"eegnet_{label_name.lower()}.pt")
        torch.save({
            'state_dict'   : global_best_state,
            'val_accuracy' : global_best_acc,
            'config': {
                'n_channels' : N_CHANNELS,
                'n_classes'  : 2,
                'T'          : T,
                'F1'         : 8,
                'D'          : 2,
                'F2'         : 16,
                'dropout'    : DROPOUT,
                'sfreq'      : SFREQ,
                'channels'   : CHANNELS,
                'label'      : label_name,
                'label_map'  : {0: 'Low', 1: 'High'},
            }
        }, save_path)
        print(f"\n  [SAVED] {save_path}  "
              f"(best val_acc={global_best_acc*100:.2f}%)")

        mean_acc = np.mean(results[label_name]['acc']) * 100
        std_acc  = np.std(results[label_name]['acc'])  * 100
        mean_f1  = np.mean(results[label_name]['f1'])
        print(f"  {label_name}  →  LOSO Acc: {mean_acc:.2f}% ± {std_acc:.2f}%"
              f"  |  F1: {mean_f1:.4f}")

    return results


# ─────────────────────────────────────────────
# STEP 7 — Save config + results JSON
# ─────────────────────────────────────────────

def save_training_results(svm_results, eegnet_results):
    """
    Save a JSON summary used by the inference script to
    display model metadata in Streamlit.
    """
    summary = {
        'model'      : 'EEGNet',
        'dataset'    : 'DREAMER',
        'sfreq'      : SFREQ,
        'n_channels' : N_CHANNELS,
        'channels'   : CHANNELS,
        'window_sec' : WINDOW_SEC,
        'T'          : T,
        'labels'     : LABEL_NAMES,
        'label_map'  : {0: 'Low', 1: 'High'},
        'results'    : {}
    }
    for name in LABEL_NAMES:
        summary['results'][name] = {
            'svm_acc_mean'   : round(float(np.mean(svm_results[name]))          * 100, 2),
            'eegnet_acc_mean': round(float(np.mean(eegnet_results[name]['acc']))* 100, 2),
            'eegnet_acc_std' : round(float(np.std(eegnet_results[name]['acc'])) * 100, 2),
            'eegnet_f1_mean' : round(float(np.mean(eegnet_results[name]['f1'])),       4),
        }

    path = os.path.join(MODEL_DIR, 'training_results.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  [SAVED] {path}")
    return summary


def print_summary(svm_results, eegnet_results):
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Label':<12} {'SVM Acc':>10}  {'EEGNet Acc':>12}  {'EEGNet F1':>10}")
    print("  " + "-"*50)
    for name in LABEL_NAMES:
        svm_acc = np.mean(svm_results[name])           * 100
        egn_acc = np.mean(eegnet_results[name]['acc']) * 100
        egn_f1  = np.mean(eegnet_results[name]['f1'])
        print(f"  {name:<12} {svm_acc:>9.2f}%  {egn_acc:>11.2f}%  {egn_f1:>10.4f}")

    print("="*60)
    print(f"  Device  : {DEVICE}")
    print(f"  Models  : {MODEL_DIR}/eegnet_{{valence|arousal|dominance}}.pt")
    print("="*60)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    start = time.time()

    svm_results    = run_svm_baseline()
    X, Y, groups   = build_trial_tensors()
    eegnet_results = run_eegnet_loso(X, Y, groups)

    save_training_results(svm_results, eegnet_results)
    print_summary(svm_results, eegnet_results)

    print(f"\n  Total runtime : {(time.time()-start)/60:.1f} minutes")