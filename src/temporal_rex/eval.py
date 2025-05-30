"""
Evaluation and metrics utilities for temporal relation extraction.

This module provides functions for computing evaluation metrics
and performance assessment of temporal relation models.
"""

import numpy as np
import torch
from datasets import load_metric


def calc_f1(predicted_labels, all_labels, label_type):
    """
    Calculate F1 score and other metrics for temporal relation classification.
    
    Args:
        predicted_labels: Predicted class labels
        all_labels: True class labels
        label_type: LabelType enum for label mapping
        
    Returns:
        tuple: (accuracy, precision, recall, f1_score)
    """
    confusion = np.zeros((len(label_type), len(label_type)))
    for i in range(len(predicted_labels)):
        confusion[all_labels[i]][predicted_labels[i]] += 1

    acc = 1.0 * np.sum([confusion[i][i] for i in range(4)]) / np.sum(confusion)
    true_positive = 0
    for i in range(len(label_type) - 1):
        true_positive += confusion[i][i]
    prec = true_positive / (np.sum(confusion) - np.sum(confusion, axis=0)[-1])
    rec = true_positive / (np.sum(confusion) - np.sum(confusion[-1][:]))
    f1 = 2 * prec * rec / (rec + prec)

    return (
        acc,
        prec,
        rec,
        f1,
    )


def compute_metrics(p):
    """
    Compute metrics for sequence evaluation (used with Hugging Face Trainer).
    
    Args:
        p: Predictions tuple containing (predictions, labels)
        
    Returns:
        dict: Dictionary containing precision, recall, f1, and accuracy scores
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special token) predictions and labels
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def evaluate_model(model, test_dataloader, device="cpu", label_type=None):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained temporal relation model
        test_dataloader: DataLoader for test data
        device: Device to run evaluation on
        label_type: LabelType enum for label mapping
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for batch in test_dataloader:
            batch = [x.to(device) for x in batch]
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "event_ix": batch[2],
                "labels": batch[3],
            }
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]
            all_logits.append(logits)
            all_labels.append(inputs["labels"])

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    predicted_logits, predicted_labels = torch.max(all_logits, dim=1)
    
    if label_type:
        acc, prec, rec, f1 = calc_f1(predicted_labels, all_labels, label_type)
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
    else:
        # Simple accuracy calculation
        accuracy = (predicted_labels == all_labels).float().mean().item()
        return {"accuracy": accuracy}


def print_evaluation_results(results, dataset_name="Test"):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary containing evaluation metrics
        dataset_name: Name of the dataset being evaluated
    """
    print(f"\n{dataset_name} Results:")
    print("-" * 40)
    for metric_name, value in results.items():
        print(f"{metric_name.capitalize()}: {value:.4f}")
    print("-" * 40)


def confusion_matrix_analysis(predicted_labels, true_labels, label_names):
    """
    Create and analyze confusion matrix for temporal relations.
    
    Args:
        predicted_labels: Predicted class labels
        true_labels: True class labels  
        label_names: List of label names
        
    Returns:
        np.ndarray: Confusion matrix
    """
    num_classes = len(label_names)
    confusion = np.zeros((num_classes, num_classes))
    
    for pred, true in zip(predicted_labels, true_labels):
        confusion[true][pred] += 1
    
    print("\nConfusion Matrix:")
    print("-" * 60)
    
    # Header
    header = "True\\Pred"
    for name in label_names:
        header += f"\t{name[:8]}"
    print(header)
    
    # Matrix rows
    for i, true_name in enumerate(label_names):
        row = f"{true_name[:8]}"
        for j in range(num_classes):
            row += f"\t{int(confusion[i][j])}"
        print(row)
    
    return confusion


def per_class_metrics(predicted_labels, true_labels, label_names):
    """
    Calculate precision, recall, and F1 for each class.
    
    Args:
        predicted_labels: Predicted class labels
        true_labels: True class labels
        label_names: List of label names
        
    Returns:
        dict: Per-class metrics
    """
    num_classes = len(label_names)
    confusion = np.zeros((num_classes, num_classes))
    
    for pred, true in zip(predicted_labels, true_labels):
        confusion[true][pred] += 1
    
    metrics = {}
    
    for i, label_name in enumerate(label_names):
        tp = confusion[i][i]
        fp = np.sum(confusion[:, i]) - tp
        fn = np.sum(confusion[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(tp + fn)
        }
    
    return metrics


def macro_average_metrics(per_class_results):
    """
    Calculate macro-averaged metrics across all classes.
    
    Args:
        per_class_results: Dictionary of per-class metrics
        
    Returns:
        dict: Macro-averaged metrics
    """
    precisions = [metrics["precision"] for metrics in per_class_results.values()]
    recalls = [metrics["recall"] for metrics in per_class_results.values()]
    f1s = [metrics["f1"] for metrics in per_class_results.values()]
    
    return {
        "macro_precision": np.mean(precisions),
        "macro_recall": np.mean(recalls),
        "macro_f1": np.mean(f1s)
    }


def weighted_average_metrics(per_class_results):
    """
    Calculate weighted-averaged metrics across all classes.
    
    Args:
        per_class_results: Dictionary of per-class metrics
        
    Returns:
        dict: Weighted-averaged metrics
    """
    total_support = sum(metrics["support"] for metrics in per_class_results.values())
    
    weighted_precision = sum(
        metrics["precision"] * metrics["support"] 
        for metrics in per_class_results.values()
    ) / total_support
    
    weighted_recall = sum(
        metrics["recall"] * metrics["support"]
        for metrics in per_class_results.values()
    ) / total_support
    
    weighted_f1 = sum(
        metrics["f1"] * metrics["support"]
        for metrics in per_class_results.values()
    ) / total_support
    
    return {
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1
    }


def comprehensive_evaluation(predicted_labels, true_labels, label_names):
    """
    Perform comprehensive evaluation with all metrics.
    
    Args:
        predicted_labels: Predicted class labels
        true_labels: True class labels
        label_names: List of label names
        
    Returns:
        dict: Comprehensive evaluation results
    """
    # Overall accuracy
    accuracy = np.mean(np.array(predicted_labels) == np.array(true_labels))
    
    # Per-class metrics
    per_class = per_class_metrics(predicted_labels, true_labels, label_names)
    
    # Macro and weighted averages
    macro_avg = macro_average_metrics(per_class)
    weighted_avg = weighted_average_metrics(per_class)
    
    # Confusion matrix
    confusion = confusion_matrix_analysis(predicted_labels, true_labels, label_names)
    
    return {
        "accuracy": accuracy,
        "per_class_metrics": per_class,
        "macro_average": macro_avg,
        "weighted_average": weighted_avg,
        "confusion_matrix": confusion
    }


# Global variables for metrics computation
metric = None
label_list = None


def initialize_metrics(label_names):
    """
    Initialize global metrics variables.
    
    Args:
        label_names: List of label names for evaluation
    """
    global metric, label_list
    metric = load_metric("seqeval")
    label_list = label_names
