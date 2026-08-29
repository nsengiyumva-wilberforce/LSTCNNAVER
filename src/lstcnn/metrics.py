"""Shared classification metrics for PyTorch and TFLite evaluation."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(preds: list[int], labels: list[int], class_names: list[str]) -> dict:
    used = sorted(set(labels) | set(preds))
    names = [class_names[i] for i in used if i < len(class_names)]
    report = classification_report(labels, preds, labels=used, target_names=names, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)) if labels else 0.0,
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)) if labels else 0.0,
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(class_names)))).tolist(),
        "report": report,
        "preds": preds,
        "labels": labels,
    }
