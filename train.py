"""
train.py — Full training pipeline (v2).

Improvements over v1:
  • Focal Loss + label smoothing     → handles class imbalance, reduces overconfidence
  • SpecAugment (on-the-fly)         → spectrogram regularisation during training
  • Mixup                            → smoother decision boundaries
  • OneCycleLR scheduler             → ~2× faster convergence than fixed LR
  • 70 / 15 / 15 train/val/test      → honest evaluation on unseen data
  • Early stopping + best checkpoint → saves the epoch that actually generaliSes best
  • Gradient clipping                → stable training, prevents exploding gradients
  • GPU / MPS / CPU auto-select      → runs on Apple Silicon, CUDA, or CPU

Usage:
    python train.py
"""
from __future__ import annotations
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from augmentation import mixup_batch, spec_augment
from config import (
    BATCH_SIZE, COMMAND_CLASSES, EARLY_STOP_PATIENCE,
    EPOCHS, FOCAL_GAMMA, LABEL_SMOOTHING, LEARNING_RATE,
    MIXUP_ALPHA, MODEL_DIR, MODEL_PATH, N_MELS, NUM_CLASSES,
    PROCESSED_DIR, WEIGHT_DECAY,
)
from model import SpeechCommandCNN

MODEL_DIR.mkdir(exist_ok=True)


# ── Loss function ──────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss with label smoothing (Lin et al., 2017 + Müller et al., 2019).

    Focal term  : (1 - p_t)^gamma  downweights easy/correctly-classified examples
                  so the gradient is dominated by hard/misclassified ones.
    Label smooth: avoids over-confident predictions by distributing 'label_smoothing'
                  probability mass evenly across all wrong classes.
    """
    def __init__(self, gamma: float = FOCAL_GAMMA, label_smoothing: float = LABEL_SMOOTHING):
        super().__init__()
        self.gamma = gamma
        self.ls    = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = logits.size(1)
        # Build smoothed target distribution
        smooth = torch.full_like(logits, self.ls / max(n - 1, 1))
        smooth.scatter_(1, targets.view(-1, 1), 1.0 - self.ls)
        log_p = F.log_softmax(logits, dim=1)
        p     = log_p.exp()
        # Focal weight based on confidence for the true class
        p_t       = (smooth * p).sum(dim=1, keepdim=True)
        focal_w   = (1.0 - p_t) ** self.gamma
        return -(focal_w * smooth * log_p).sum(dim=1).mean()


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy for soft (Mixup-blended) targets."""
    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


# ── Dataset ────────────────────────────────────────────────────────────────────

class SpeechDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.augment:
            x = spec_augment(x)   # SpecAugment on each spectrogram
        return x, self.y[idx]


# ── Early stopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = EARLY_STOP_PATIENCE, min_delta: float = 1e-3):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> tuple:
    print("Loading pre-processed tensors …")
    X = np.load(PROCESSED_DIR / "X_features.npy")
    y = np.load(PROCESSED_DIR / "y_labels.npy")
    print(f"  Loaded  X: {X.shape}  y: {y.shape}")

    # 70 / 15 / 15 stratified split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )
    print(f"  Train: {len(y_tr):,}  Val: {len(y_val):,}  Test: {len(y_te):,}\n")

    train_ds = SpeechDataset(X_tr,  y_tr,  augment=True)
    val_ds   = SpeechDataset(X_val, y_val, augment=False)
    test_ds  = SpeechDataset(X_te,  y_te,  augment=False)

    kw = dict(num_workers=2, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, **kw),
        y_te,
    )


# ── Training utilities ─────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    use_mixup: bool = True,
) -> tuple[float, float]:

    model.train()
    total_loss = correct = total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Mixup ~50% of batches during training for label regularisation
        if use_mixup and random.random() < 0.5:
            mixed, soft_labels = mixup_batch(inputs, labels, NUM_CLASSES, MIXUP_ALPHA)
            outputs = model(mixed)
            loss    = soft_cross_entropy(outputs, soft_labels)
        else:
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int]]:

    model.eval()
    total_loss = correct = total = 0
    all_preds: list[int] = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())

    return total_loss / len(loader), 100.0 * correct / total, all_preds


# ── Main ───────────────────────────────────────────────────────────────────────

def train() -> None:
    # Device selection (CUDA → Apple MPS → CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    train_dl, val_dl, test_dl, y_test_true = load_data()

    model     = SpeechCommandCNN(num_classes=NUM_CLASSES, n_mels=N_MELS).to(device)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # OneCycleLR: ramps LR up then down over the whole run — faster than fixed LR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_dl),
    )
    stopper       = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    best_val_loss = float("inf")

    print(f"Training for up to {EPOCHS} epochs (early stop patience={EARLY_STOP_PATIENCE})…\n")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(
            model, train_dl, criterion, optimizer, scheduler, device
        )
        va_loss, va_acc, _ = evaluate(model, val_dl, criterion, device)
        elapsed = time.perf_counter() - t0

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss {tr_loss:.4f} acc {tr_acc:.1f}% | "
            f"Val loss {va_loss:.4f} acc {va_acc:.1f}% | "
            f"{elapsed:.1f}s"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"           ↳ ✅ New best saved → {MODEL_PATH}")

        if stopper.step(va_loss):
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    # ── Final test-set evaluation ──────────────────────────────────────────────
    print(f"\nLoading best checkpoint from {MODEL_PATH} …")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    _, test_acc, all_preds = evaluate(model, test_dl, criterion, device)

    f1 = f1_score(y_test_true, all_preds, average="weighted")
    bar = "=" * 58
    print(f"\n{bar}")
    print(f"  TEST ACCURACY  : {test_acc:.2f}%")
    print(f"  WEIGHTED F1    : {f1:.4f}")
    print(f"{bar}\n")
    print(classification_report(y_test_true, all_preds, target_names=COMMAND_CLASSES))


if __name__ == "__main__":
    train()
