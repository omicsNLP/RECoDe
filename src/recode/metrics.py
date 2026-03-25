"""Evaluation metrics for RECoDe relation extraction."""
import os

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .logic.predict import labels as CLASS_LABELS

TRUE_CLASS = [
    "association",
    "increaseAssociation",
    "decreaseAssociation",
    "positiveCorrelation",
    "negativeCorrelation",
    "consists",
    "causalEffect",
    "substitution",
]

NO_CLASS = [
    "NoAssociation",
    "Unrelated",
]


def map_to_binary(label):
    """Map a relation label to binary: association / NoAssociation."""
    if label in TRUE_CLASS:
        return "association"
    elif label in NO_CLASS:
        return "NoAssociation"
    return None


def compute_metrics(y_true, y_pred, bin=False):
    """Compute accuracy + precision/recall/f1 for multiclass or binary."""
    acc = accuracy_score(y_true, y_pred)

    metrics = {}
    for avg in ["micro", "macro", "weighted"]:
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[avg] = {"precision": p, "recall": r, "f1": f1}

    if bin:
        p, r, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            pos_label="association",
            average="binary",
            zero_division=0,
        )
        metrics["binary_pos"] = {"precision": p, "recall": r, "f1": f1}

    report = classification_report(y_true, y_pred, zero_division=0)

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return acc, metrics, report, labels, cm


def evaluate_re(gold_df, pred_df):
    """Evaluate relation extraction: multiclass + binary.

    Args:
        gold_df: pd.Series of gold labels
        pred_df: pd.Series of predicted labels
    """
    y_true = gold_df
    y_pred = pred_df

    invalid_labels = set(y_pred) - set(CLASS_LABELS)
    if invalid_labels:
        raise ValueError(f"Invalid predicted labels found: {invalid_labels}")

    y_true_bin = y_true.map(map_to_binary)
    y_pred_bin = y_pred.map(map_to_binary)

    acc, metrics, report, labels, cm = compute_metrics(y_true, y_pred)
    acc_bin, metrics_bin, report_bin, labels_bin, cm_bin = compute_metrics(
        y_true_bin, y_pred_bin, bin=True
    )

    return {
        "multiclass": {
            "accuracy": acc,
            "metrics": metrics,
            "classification_report": report,
            "labels": labels,
            "confusion_matrix": cm,
        },
        "binary": {
            "accuracy": acc_bin,
            "metrics": metrics_bin,
            "classification_report": report_bin,
            "labels": labels_bin,
            "confusion_matrix": cm_bin,
        },
    }


def print_results(results, name=""):
    """Pretty-print evaluation results."""
    if name:
        print(f"\n\n===== RESULTS for {name} =====")

    print("===== MULTICLASS =====")
    print("Accuracy:", results["multiclass"]["accuracy"])
    print("\nMicro:", results["multiclass"]["metrics"]["micro"])
    print("\nMacro:", results["multiclass"]["metrics"]["macro"])
    print("\nWeighted:", results["multiclass"]["metrics"]["weighted"])
    print("\nClassification report:\n", results["multiclass"]["classification_report"])
    print("\nConfusion matrix:\n", results["multiclass"]["confusion_matrix"])

    print("\n\n===== BINARY =====")
    print("Accuracy:", results["binary"]["accuracy"])
    print("\nBinary (Positive):", results["binary"]["metrics"]["binary_pos"])
    print("\nMicro:", results["binary"]["metrics"]["micro"])
    print("\nMacro:", results["binary"]["metrics"]["macro"])
    print("\nWeighted:", results["binary"]["metrics"]["weighted"])
    print("\nClassification report:\n", results["binary"]["classification_report"])
    print("\nConfusion matrix:\n", results["binary"]["confusion_matrix"])


def save_results(current_dataset, output_dir, split, model_name):
    """Save prediction results as TSV files."""
    os.makedirs(output_dir, exist_ok=True)
    model_tag = model_name.replace("/", "_") if model_name else "unknown"

    # Extract pmc_id from original_txt_info if available
    pmc_ids = None
    if "original_txt_info" in current_dataset.columns:
        pmc_ids = current_dataset["original_txt_info"].apply(
            lambda x: x.get("pmc_id", "") if isinstance(x, dict) else ""
        ).values

    # Eval TSV: pmc_id, rel_id, type (matches results/real/ format)
    eval_df = pd.DataFrame()
    if pmc_ids is not None:
        eval_df["pmc_id"] = pmc_ids
    eval_df["rel_id"] = range(len(current_dataset))
    eval_df["type"] = current_dataset["recode_result"].values
    eval_path = os.path.join(output_dir, f"{split}_{model_tag}_result.tsv")
    eval_df.to_csv(eval_path, sep="\t", index=False)
    print(f"\nSaved: {eval_path}")

    # Full TSV: pmc_id, rel_id, type_gold, type_pred (for reference)
    full_df = pd.DataFrame()
    if pmc_ids is not None:
        full_df["pmc_id"] = pmc_ids
    full_df["rel_id"] = range(len(current_dataset))
    if "type" in current_dataset.columns:
        full_df["type_gold"] = current_dataset["type"].values
    full_df["type_pred"] = current_dataset["recode_result"].values
    full_path = os.path.join(output_dir, f"{split}_{model_tag}_full.tsv")
    full_df.to_csv(full_path, sep="\t", index=False)
    print(f"Saved full: {full_path}")

    return eval_path, full_path
