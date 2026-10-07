"""Metric calculations used by the MNIST training loop."""

from math import isfinite


def calculate_metrics(loss_sum: float, correct: int, total: int) -> dict[str, float]:
    """Convert accumulated per-example loss and correct counts to metrics."""
    if total <= 0:
        raise ValueError("total must be greater than zero")

    loss = loss_sum / total
    accuracy = correct / total
    if not isfinite(loss) or not isfinite(accuracy):
        raise ValueError("metrics must be finite")

    return {"loss": loss, "accuracy": accuracy}
