"""
training/evaluate.py
Evaluation script for Suraksha AI action recognizer.
Generates accuracy, precision, recall, F1, and confusion matrix on the test set.

Usage:
    python training/evaluate.py --weights saved_models/action_recognizer_best.pt
    python training/evaluate.py --weights saved_models/action_recognizer_best.pt --data datasets
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.action_recognizer import ActionRecognizerModel, ACTION_LABELS
from training.dataset_loader import get_dataloaders
from utils.logger import get_logger

logger = get_logger("training.evaluate")

BINARY_LABELS = {0: "Normal", 1: "Harassment"}  # For binary evaluation


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Suraksha AI Action Recognizer")
    p.add_argument("--weights",     type=str, required=True,        help="Path to .pt weights file")
    p.add_argument("--data",        type=str, default="datasets",   help="Dataset root directory")
    p.add_argument("--device",      type=str, default="auto",       help="'cuda' | 'cpu' | 'auto'")
    p.add_argument("--batch",       type=int, default=16,           help="Batch size")
    p.add_argument("--workers",     type=int, default=4,            help="DataLoader workers")
    p.add_argument("--num-classes", type=int, default=8,            help="Number of action classes")
    p.add_argument("--output-dir",  type=str, default="outputs/evaluation", help="Where to save results")
    p.add_argument("--seq-len",     type=int, default=16,           help="Sequence length")
    return p.parse_args()


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str) -> None:
    """Save a colour-coded confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)

    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {output_path}")


@torch.no_grad()
def run_evaluation(model, loader, device) -> tuple:
    """Run model on full test loader, return (all_preds, all_labels)."""
    model.eval()
    all_preds, all_labels = [], []

    for seqs, labels in loader:
        seqs = seqs.to(device)
        logits = model(seqs)
        preds  = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    return all_preds, all_labels


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Evaluating on: {device}")

    # ── Load model ────────────────────────────────────────────
    model = ActionRecognizerModel(num_classes=args.num_classes).to(device)
    if not os.path.exists(args.weights):
        logger.error(f"Weights file not found: {args.weights}")
        sys.exit(1)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    logger.info(f"Weights loaded: {args.weights}")

    # ── Load test data ────────────────────────────────────────
    _, _, test_loader = get_dataloaders(
        dataset_root=args.data,
        sequence_length=args.seq_len,
        batch_size=args.batch,
        num_workers=args.workers,
    )

    logger.info(f"Running evaluation on {len(test_loader.dataset)} test samples ...")
    preds, labels = run_evaluation(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────
    class_names = [ACTION_LABELS.get(i, str(i)) for i in range(args.num_classes)]

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    cm  = confusion_matrix(labels, preds, labels=list(range(args.num_classes)))
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)

    logger.info(f"\n{'='*55}")
    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    logger.info(f"\n{report}")

    # ── Save confusion matrix plot ─────────────────────────────
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)

    # ── Save metrics JSON ─────────────────────────────────────
    results = {
        "weights":    args.weights,
        "accuracy":   round(float(acc),  4),
        "precision":  round(float(prec), 4),
        "recall":     round(float(rec),  4),
        "f1_score":   round(float(f1),   4),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "classification_report": report,
    }
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation results saved → {results_path}")

    # ── Save CSV report ───────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "eval_results.csv")
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        for k in ["accuracy", "precision", "recall", "f1_score"]:
            f.write(f"{k},{results[k]}\n")
    logger.info(f"CSV saved → {csv_path}")


if __name__ == "__main__":
    main()
