"""
training/train.py
Full training script for the CNN+LSTM action recognizer (Suraksha AI).

Usage:
    python training/train.py --epochs 50 --batch 16 --gpu 0 --resume
    python training/train.py --epochs 30 --batch 8 --device cpu
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.action_recognizer import ActionRecognizerModel
from training.dataset_loader import get_dataloaders
from utils.logger import get_logger

logger = get_logger("training.train")


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train Suraksha AI Action Recognizer")
    p.add_argument("--epochs",       type=int,   default=50,           help="Number of training epochs")
    p.add_argument("--batch",        type=int,   default=16,           help="Batch size")
    p.add_argument("--lr",           type=float, default=5e-4,         help="Learning rate")
    p.add_argument("--device",       type=str,   default="auto",       help="'cuda', 'cpu', or 'auto'")
    p.add_argument("--gpu",          type=int,   default=0,            help="GPU index to use")
    p.add_argument("--workers",      type=int,   default=4,            help="DataLoader workers")
    p.add_argument("--data",         type=str,   default="datasets",   help="Path to dataset root")
    p.add_argument("--save-dir",     type=str,   default="saved_models", help="Checkpoint save directory")
    p.add_argument("--resume",       action="store_true",              help="Resume from best checkpoint")
    p.add_argument("--patience",     type=int,   default=10,           help="Early stopping patience")
    p.add_argument("--seq-len",      type=int,   default=16,           help="Sequence length (frames)")
    p.add_argument("--num-classes",  type=int,   default=8,            help="Number of action classes")
    return p.parse_args()


# ── Training Logic ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> dict:
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch_idx, (seqs, labels) in enumerate(loader):
        seqs   = seqs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast(enabled=(device.type == "cuda")):
            logits = model(seqs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

        if batch_idx % 10 == 0:
            logger.info(f"  Batch {batch_idx}/{len(loader)} | loss={loss.item():.4f}")

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "loss":      round(total_loss / len(loader), 4),
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1":        round(float(f1),   4),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for seqs, labels in loader:
        seqs   = seqs.to(device)
        labels = labels.to(device)

        logits = model(seqs)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "loss":      round(total_loss / len(loader), 4),
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1":        round(float(f1),   4),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Training on device: {device}")

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        dataset_root=args.data,
        sequence_length=args.seq_len,
        batch_size=args.batch,
        num_workers=args.workers,
    )

    # ── Model ─────────────────────────────────────────────────
    model = ActionRecognizerModel(num_classes=args.num_classes).to(device)
    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt = os.path.join(args.save_dir, "action_recognizer_best.pt")
    last_ckpt = os.path.join(args.save_dir, "action_recognizer_last.pt")

    start_epoch    = 0
    best_val_loss  = float("inf")
    patience_count = 0
    history        = []

    if args.resume and os.path.exists(best_ckpt):
        logger.info(f"Resuming from {best_ckpt}")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # ── Optimizer & Scheduler ─────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    logger.info(f"Starting training | epochs={args.epochs} | batch={args.batch} | lr={args.lr}")

    # ── Training Loop ─────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        logger.info(f"\n{'='*55}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics   = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["loss"])

        elapsed = time.time() - t0
        logger.info(
            f"  TRAIN → loss={train_metrics['loss']:.4f}  acc={train_metrics['accuracy']:.4f}  "
            f"f1={train_metrics['f1']:.4f}"
        )
        logger.info(
            f"  VAL   → loss={val_metrics['loss']:.4f}    acc={val_metrics['accuracy']:.4f}  "
            f"f1={val_metrics['f1']:.4f}  [{elapsed:.1f}s]"
        )

        history.append({"epoch": epoch+1, "train": train_metrics, "val": val_metrics})

        # ── Save best checkpoint ──────────────────────────────
        if val_metrics["loss"] < best_val_loss:
            best_val_loss  = val_metrics["loss"]
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"  ✓ Best model saved → {best_ckpt}")
        else:
            patience_count += 1
            logger.info(f"  No improvement. Patience: {patience_count}/{args.patience}")

        # Save last checkpoint every epoch
        torch.save(model.state_dict(), last_ckpt)

        # ── Early stopping ────────────────────────────────────
        if patience_count >= args.patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}.")
            break

    # ── Save training history ─────────────────────────────────
    history_path = os.path.join(args.save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training complete. History → {history_path}")
    logger.info(f"Best val loss: {best_val_loss:.4f} | Best model → {best_ckpt}")


if __name__ == "__main__":
    main()
