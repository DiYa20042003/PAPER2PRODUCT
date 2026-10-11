"""Metric calculations used by the MNIST training loop."""

from collections.abc import Sequence
from math import isfinite


def calculate_confusion_matrix(
    actual: Sequence[int],
    predicted: Sequence[int],
    num_classes: int = 10,
) -> list[list[int]]:
    """Count predictions with actual classes on rows and predicted classes on columns."""
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero")
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same number of items")

    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for actual_label, predicted_label in zip(actual, predicted):
        if not 0 <= actual_label < num_classes:
            raise ValueError("actual labels must be within the configured class range")
        if not 0 <= predicted_label < num_classes:
            raise ValueError("predicted labels must be within the configured class range")
        matrix[actual_label][predicted_label] += 1
    return matrix


def calculate_metrics(loss_sum: float, correct: int, total: int) -> dict[str, float]:
    """Convert accumulated per-example loss and correct counts to metrics."""
    if total <= 0:
        raise ValueError("total must be greater than zero")

    loss = loss_sum / total
    accuracy = correct / total
    if not isfinite(loss) or not isfinite(accuracy):
        raise ValueError("metrics must be finite")

    return {"loss": loss, "accuracy": accuracy}
