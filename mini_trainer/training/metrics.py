import numpy as np
import torch

from mini_trainer import get_logger


def named_confusion_matrix(
    results: dict[str, list[str]],
    cls2idx: dict[str, int],
    keys: tuple[str, str] = ("preds", "labels"),
    verbose: bool = False,
):
    """Compute dense label-prediction-count dictionary confusion matrix from results.

    Also computes per-class, micro and macro accuracy.
    """
    # Build confusion matrix and compute accuracies
    classes = [k for k, v in sorted(cls2idx.items(), key=lambda x: x[1])]

    # Initialize confusion matrix and counters
    conf_mat = {lab: {pred: 0 for pred in classes} for lab in classes}
    total_correct = 0
    total_samples = 0
    per_class_total = {cls: 0 for cls in classes}
    per_class_correct = {cls: 0 for cls in classes}

    # Populate confusion matrix and count correct predictions
    for prediction, label in zip(results[keys[0]], results[keys[1]]):
        if label not in classes:
            continue
        conf_mat[label][prediction] += 1
        total_samples += 1
        per_class_total[label] += 1
        if label.lower().strip() == prediction.lower().strip():
            total_correct += 1
            per_class_correct[label] += 1

    # Print the confusion matrix (numbers only, aligned)
    if verbose:
        max_cf_n = max(val for d in conf_mat.values() for val in d.values())
        width = len(str(max_cf_n))
        cf_lines = []
        for lab in classes:
            row_str = "|".join(
                "{:>{width}d}".format(conf_mat[lab][pred], width=width) if conf_mat[lab][pred] != 0 else " " * width for pred in classes
            )
            cf_lines.append(row_str)
        get_logger().info("\n" + "\n".join(cf_lines))

    # Compute and print per-class accuracies
    macro_acc = 0.0
    num_class_with_any = 0
    acc_lines = ["\nPer-class Accuracies:"]
    for cls in classes:
        if per_class_total[cls] > 0:
            acc = per_class_correct[cls] / per_class_total[cls]
            num_class_with_any += 1
        else:
            acc = 0.0
        macro_acc += acc
        if verbose:
            acc_lines.append(f"{cls:_<{max(map(len, classes))}}{acc:_>9.1%} ({per_class_correct[cls]}/{per_class_total[cls]})")
    macro_acc /= num_class_with_any or 1

    # Micro accuracy: overall correct predictions / total predictions
    micro_acc = total_correct / total_samples if total_samples > 0 else 0.0

    if verbose:
        acc_lines.append(f"\nMicro Accuracy: {micro_acc:.2%} ({total_correct}/{total_samples})")
        acc_lines.append(f"Macro Accuracy: {macro_acc:.2%}")
        get_logger().info("\n" + "\n".join(acc_lines))

    return {"hits": per_class_correct, "totals": per_class_total, "micro": micro_acc, "macro": macro_acc, "conf_mat": conf_mat}


def raw_confusion_matrix(
    labels: list[int] | torch.Tensor | np.ndarray, predictions: list[int] | torch.Tensor | np.ndarray, n_classes: int | None = None
):
    """Compute dense confusion matrix array from sparse label-prediction pairs."""
    labels = np.asarray(labels).ravel().astype(np.int64)
    predictions = np.asarray(predictions).ravel().astype(np.int64)
    if n_classes is None:
        n_classes = int(max(max(labels), max(predictions))) + 1

    indices = n_classes * labels + predictions
    cm = np.bincount(indices, minlength=n_classes * n_classes)
    cm = cm.reshape((n_classes, n_classes)).astype(np.float64)

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm / row_sums
        cm_norm[~np.isfinite(cm_norm)] = 0.0  # set inf and NaN to zero

    return cm_norm
