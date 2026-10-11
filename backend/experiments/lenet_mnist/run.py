"""Train and evaluate LeNet-5 on the torchvision MNIST dataset."""

import argparse
import json
import math
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from experiments.lenet_mnist.metrics import calculate_confusion_matrix, calculate_metrics
from experiments.lenet_mnist.model import MODEL_SPEC, LeNet5


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data"
RUNS_DIRECTORY = Path(__file__).resolve().parent / "runs"


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def log_line(log_file: Any, message: str) -> None:
    line = f"{timestamp()} {message}"
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be a finite number greater than zero")
    for option in ("max_train_samples", "max_test_samples"):
        limit = getattr(args, option)
        if limit is not None and limit <= 0:
            parser.error(f"--{option.replace('_', '-')} must be greater than zero")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def load_datasets(
    max_train_samples: int | None,
    max_test_samples: int | None,
) -> tuple[Any, Any]:
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root=str(DATA_DIRECTORY), train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(DATA_DIRECTORY), train=False, download=True, transform=transform
    )
    if max_train_samples is not None:
        train_dataset = Subset(train_dataset, range(min(max_train_samples, len(train_dataset))))
    if max_test_samples is not None:
        test_dataset = Subset(test_dataset, range(min(max_test_samples, len(test_dataset))))
    return train_dataset, test_dataset


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size
    return calculate_metrics(loss_sum, correct, total)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    actual_labels: list[int] = []
    predicted_labels: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        actual_labels.extend(labels.detach().cpu().tolist())
        predicted_labels.extend(predictions.detach().cpu().tolist())
        total += batch_size
    return {
        **calculate_metrics(loss_sum, correct, total),
        "confusion_matrix": calculate_confusion_matrix(
            actual_labels, predicted_labels
        ),
    }


def run_experiment(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    run_directory = RUNS_DIRECTORY / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    log_path = run_directory / "log.txt"
    config_path = run_directory / "config.json"
    metrics_path = run_directory / "metrics.json"
    model_path = run_directory / "model.pt"
    device_name = "unresolved"
    config = {
        "run_id": run_id,
        "timestamp": timestamp(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "requested_device": args.device,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "seed": args.seed,
        "dataset": "torchvision.datasets.MNIST",
        "preprocessing": "torchvision.transforms.ToTensor(); no additional augmentation",
        "model_spec": MODEL_SPEC,
        "optimizer": "Adam",
    }
    write_json(config_path, config)
    metrics: dict[str, Any] = {
        "status": "running",
        "run_id": run_id,
        "timestamp": config["timestamp"],
        "device": device_name,
        "torch_version": torch.__version__,
        "execution_time_seconds": None,
        "training_samples": None,
        "test_samples": None,
        "epochs": [],
    }

    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            torch.manual_seed(args.seed)
            device = choose_device(args.device)
            device_name = str(device)
            metrics["device"] = device_name
            model = LeNet5().to(device)
            torch.save(model.state_dict(), model_path)

            log_line(log_file, f"run_id={run_id} status=running device={device_name}")
            train_dataset, test_dataset = load_datasets(
                args.max_train_samples, args.max_test_samples
            )
            metrics["training_samples"] = len(train_dataset)
            metrics["test_samples"] = len(test_dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True
            )
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for epoch in range(1, args.epochs + 1):
                training = train_epoch(model, train_loader, criterion, optimizer, device)
                testing = evaluate(model, test_loader, criterion, device)
                epoch_metrics = {
                    "epoch": epoch,
                    "train_loss": training["loss"],
                    "training_loss": training["loss"],
                    "test_loss": testing["loss"],
                    "test_accuracy": testing["accuracy"],
                }
                if not all(
                    math.isfinite(epoch_metrics[key])
                    for key in (
                        "train_loss",
                        "training_loss",
                        "test_loss",
                        "test_accuracy",
                    )
                ):
                    raise ValueError("epoch metrics contain non-finite values")
                metrics["epochs"].append(epoch_metrics)
                metrics["final_test_accuracy"] = testing["accuracy"]
                metrics["final_test_loss"] = testing["loss"]
                metrics["confusion_matrix"] = testing["confusion_matrix"]
                log_line(
                    log_file,
                    f"epoch={epoch}/{args.epochs} "
                    f"training_loss={training['loss']:.6f} "
                    f"test_loss={testing['loss']:.6f} "
                    f"test_accuracy={testing['accuracy']:.6f}",
                )
                write_json(metrics_path, metrics)

            torch.save(model.state_dict(), model_path)
            metrics["status"] = "completed"
            log_line(log_file, "status=completed")
        except Exception as error:
            metrics["status"] = "failed"
            metrics["error"] = f"{type(error).__name__}: {error}"
            log_line(log_file, f"status=failed error={metrics['error']}")
            write_json(metrics_path, metrics)
            raise
        finally:
            metrics["device"] = device_name
            metrics["execution_time_seconds"] = time.perf_counter() - started
            write_json(metrics_path, metrics)
    return run_directory


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_directory = run_experiment(args)
    except Exception as error:
        print(f"Experiment failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    print(f"Run artifacts: {run_directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
