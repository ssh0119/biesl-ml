"""Training primitives shared by CNN cross-validation and production fit."""
from __future__ import annotations

import copy
import random
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    SEED,
    WEIGHT_DECAY,
)
from .model import Phase2CNN1D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SignalDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, *, augment: bool):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        x = self.X[index].copy() if self.augment else self.X[index]
        if self.augment:
            shift = int(np.random.randint(-125, 126))  # up to 0.5 seconds at 250 Hz
            x = np.roll(x, shift, axis=-1)
            scale = (1.0 + np.random.uniform(-0.05, 0.05, (x.shape[0], 1))).astype(np.float32)
            noise = np.random.normal(0.0, 0.03, x.shape).astype(np.float32)
            x = x * scale + noise
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[index])


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = Counter(y.tolist())
    weights = np.asarray([
        len(y) / (n_classes * max(counts.get(label, 0), 1))
        for label in range(n_classes)
    ], dtype=np.float32)
    return torch.from_numpy(weights).to(DEVICE)


def _loader(X: np.ndarray, y: np.ndarray, *, training: bool, batch_size: int) -> DataLoader:
    return DataLoader(
        SignalDataset(X, y, augment=training),
        batch_size=batch_size,
        shuffle=training,
        num_workers=0,
        pin_memory=DEVICE == "cuda",
    )


@torch.no_grad()
def predict_probabilities(model: Phase2CNN1D, X: np.ndarray, *, batch_size: int = BATCH_SIZE) -> np.ndarray:
    model.eval()
    loader = _loader(X, np.zeros(len(X), dtype=np.int64), training=False, batch_size=batch_size)
    output: list[np.ndarray] = []
    for xb, _ in loader:
        logits = model(xb.to(DEVICE, non_blocking=True))
        output.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(output)


def train_with_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_classes: int,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
) -> tuple[Phase2CNN1D, dict]:
    set_seed()
    model = Phase2CNN1D(in_channels=X_train.shape[1], n_classes=n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_train, n_classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    train_loader = _loader(X_train, y_train, training=True, batch_size=batch_size)

    best_score = -1.0
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
            total_n += len(xb)
        scheduler.step()

        val_prob = predict_probabilities(model, X_val, batch_size=batch_size)
        val_pred = val_prob.argmax(axis=1)
        val_f1 = float(f1_score(y_val, val_pred, labels=range(n_classes), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": total_loss / total_n, "val_macro_f1": val_f1})
        print(f"    epoch={epoch:02d} loss={total_loss / total_n:.4f} val_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_score + 1e-4:
            best_score = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "stopped_at_epoch": len(history),
        "history": history,
    }


def train_fixed_epochs(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    epochs: int,
    batch_size: int = BATCH_SIZE,
    scheduler_epochs: int | None = None,
) -> Phase2CNN1D:
    set_seed()
    model = Phase2CNN1D(in_channels=X.shape[1], n_classes=n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y, n_classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_epochs or epochs)
    loader = _loader(X, y, training=True, batch_size=batch_size)
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"    refit epoch={epoch:02d}/{epochs}", flush=True)
    return model
