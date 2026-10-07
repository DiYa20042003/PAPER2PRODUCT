import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from experiments.lenet_mnist.metrics import calculate_metrics
from experiments.lenet_mnist.model import LeNet5


BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def test_lenet_forward_output_shape() -> None:
    model = LeNet5()
    output = model(torch.randn(4, 1, 28, 28))
    assert output.shape == (4, 10)


def test_metric_calculation() -> None:
    metrics = calculate_metrics(loss_sum=5.0, correct=3, total=4)
    assert metrics == {"loss": 1.25, "accuracy": 0.75}


def test_metric_calculation_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="total must be greater than zero"):
        calculate_metrics(loss_sum=0.0, correct=0, total=0)


def test_smoke_run_writes_real_finite_metrics() -> None:
    completed = subprocess.run(
        [
            PYTHON,
            "-m",
            "experiments.lenet_mnist.run",
            "--epochs",
            "1",
            "--batch-size",
            "64",
            "--lr",
            "0.001",
            "--device",
            "cpu",
            "--max-train-samples",
            "256",
            "--max-test-samples",
            "128",
        ],
        cwd=BACKEND_DIRECTORY,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "epoch=1/1" in completed.stdout
    run_directory = Path(completed.stdout.split("Run artifacts: ")[-1].strip())
    assert run_directory.is_dir()
    assert {path.name for path in run_directory.iterdir()} == {
        "config.json",
        "log.txt",
        "metrics.json",
        "model.pt",
    }

    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    assert config["max_train_samples"] == 256
    assert config["optimizer"] == "Adam"
    assert config["model_spec"]["input_shape"] == [1, 28, 28]
    assert metrics["status"] == "completed"
    assert metrics["training_samples"] == 256
    assert metrics["test_samples"] == 128
    assert metrics["execution_time_seconds"] > 0
    assert metrics["torch_version"] == torch.__version__
    assert metrics["device"] == "cpu"
    assert len(metrics["epochs"]) == 1
    epoch = metrics["epochs"][0]
    assert epoch["epoch"] == 1
    assert all(
        math.isfinite(epoch[field])
        for field in ("training_loss", "test_loss", "test_accuracy")
    )
    assert epoch["training_loss"] > 0
    assert epoch["test_loss"] > 0
    assert 0 <= epoch["test_accuracy"] <= 1
    assert (run_directory / "model.pt").stat().st_size > 0
