import json
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from collections import defaultdict


def load_and_process_jsonl(file_path):
    """Load JSONL file and extract predictions and ground truth."""
    predictions = []
    ground_truth = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                predictions.append(record['text'])
                ground_truth.append(record['gt'])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing line: {e}")
                continue

    return predictions, ground_truth


def calculate_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall, and F1 score."""
    metrics = {}

    # Calculate accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Calculate macro-averaged metrics
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')

    # Calculate micro-averaged metrics
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro')
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro')
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')

    return metrics


def analyze_per_class(y_true, y_pred):
    """Analyze performance for each class."""
    classes = sorted(list(set(y_true + y_pred)))
    per_class_metrics = defaultdict(dict)

    for cls in classes:
        # Create binary arrays for the current class
        true_binary = [1 if y == cls else 0 for y in y_true]
        pred_binary = [1 if y == cls else 0 for y in y_pred]

        # Calculate metrics for the current class
        per_class_metrics[cls]['precision'] = precision_score(true_binary, pred_binary)
        per_class_metrics[cls]['recall'] = recall_score(true_binary, pred_binary)
        per_class_metrics[cls]['f1'] = f1_score(true_binary, pred_binary)

    return per_class_metrics


def create_confusion_matrix(y_true, y_pred):
    """Create a confusion matrix as a dictionary."""
    matrix = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix


def main():
    # Load and process the data
    y_pred, y_true = load_and_process_jsonl('hop-5/AgentClinic_answer.jsonl')

    # Calculate overall metrics
    metrics = calculate_metrics(y_true, y_pred)

    # Calculate per-class metrics
    per_class_metrics = analyze_per_class(y_true, y_pred)

    # Create confusion matrix
    confusion_matrix = create_confusion_matrix(y_true, y_pred)

    # Print results
    print("\nOverall Metrics:")
    print("-" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nMacro-averaged metrics:")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall: {metrics['recall_macro']:.4f}")
    print(f"F1 Score: {metrics['f1_macro']:.4f}")

    print("\nMicro-averaged metrics:")
    print(f"Precision: {metrics['precision_micro']:.4f}")
    print(f"Recall: {metrics['recall_micro']:.4f}")
    print(f"F1 Score: {metrics['f1_micro']:.4f}")

    print("\nPer-class Metrics:")
    print("-" * 50)
    for cls, cls_metrics in per_class_metrics.items():
        print(f"\nClass {cls}:")
        print(f"Precision: {cls_metrics['precision']:.4f}")
        print(f"Recall: {cls_metrics['recall']:.4f}")
        print(f"F1: {cls_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()