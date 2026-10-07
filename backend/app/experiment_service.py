"""Run the vetted M1 experiment and persist its lifecycle and output."""

import json
import math
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import sessionmaker

from app.models import Artifact, Experiment, ExperimentLog, Metric


BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]
RUNS_DIRECTORY = BACKEND_DIRECTORY / "experiments" / "lenet_mnist" / "runs"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _save_log(session_factory: sessionmaker, experiment_id: str, stream: str, line: str) -> None:
    with session_factory() as session:
        session.add(
            ExperimentLog(
                experiment_id=experiment_id,
                stream=stream,
                message=line,
                created_at=_now(),
            )
        )
        session.commit()


def _run_directory_from_output(stdout_lines: list[str]) -> Path | None:
    for line in reversed(stdout_lines):
        if line.startswith("Run artifacts: "):
            return Path(line.removeprefix("Run artifacts: ").strip())
    for line in stdout_lines:
        match = re.search(r"\brun_id=([^\s]+)", line)
        if match:
            return RUNS_DIRECTORY / match.group(1)
    return None


def _store_artifacts_and_metrics(
    session_factory: sessionmaker,
    experiment_id: str,
    run_directory: Path | None,
) -> tuple[str | None, str | None]:
    if run_directory is None:
        return None, None

    paths = {
        "config": run_directory / "config.json",
        "metrics": run_directory / "metrics.json",
        "log": run_directory / "log.txt",
        "model": run_directory / "model.pt",
    }
    with session_factory() as session:
        experiment = session.get(Experiment, experiment_id)
        if experiment is None:
            raise RuntimeError(f"Experiment {experiment_id} disappeared during execution")
        experiment.artifact_directory = str(run_directory)
        experiment.config_path = str(paths["config"])
        experiment.metrics_path = str(paths["metrics"])
        for kind, path in paths.items():
            if path.is_file():
                session.add(Artifact(experiment_id=experiment_id, kind=kind, path=str(path)))

        metrics_path = paths["metrics"]
        if metrics_path.is_file():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            for epoch_data in payload.get("epochs", []):
                epoch = epoch_data.get("epoch")
                for name in ("training_loss", "test_loss", "test_accuracy"):
                    value = epoch_data.get(name)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        session.add(
                            Metric(
                                experiment_id=experiment_id,
                                name=name,
                                value=float(value),
                                epoch=epoch,
                            )
                        )
            for name in ("execution_time_seconds",):
                value = payload.get(name)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    session.add(
                        Metric(experiment_id=experiment_id, name=name, value=float(value))
                    )
        session.commit()
    return str(paths["config"]), str(paths["metrics"])


def _build_command(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    max_train_samples: int | None,
    max_test_samples: int | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.lenet_mnist.run",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(learning_rate),
        "--device",
        device,
    ]
    if max_train_samples is not None:
        command.extend(("--max-train-samples", str(max_train_samples)))
    if max_test_samples is not None:
        command.extend(("--max-test-samples", str(max_test_samples)))
    return command


def queue_m1_experiment(
    session_factory: sessionmaker,
    *,
    paper_id: int | None = None,
    plan_id: int | None = None,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = "auto",
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    timeout_seconds: float = 3600,
) -> str:
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be a finite number greater than zero")
    command = _build_command(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )
    experiment_id = str(uuid.uuid4())
    with session_factory() as session:
        session.add(
            Experiment(
                id=experiment_id,
                paper_id=paper_id,
                plan_id=plan_id,
                status="QUEUED",
                command_json=json.dumps(command),
                timeout_seconds=timeout_seconds,
            )
        )
        session.commit()
    return experiment_id


def run_m1_experiment(
    session_factory: sessionmaker,
    *,
    experiment_id: str | None = None,
    paper_id: int | None = None,
    plan_id: int | None = None,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = "auto",
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    timeout_seconds: float = 3600,
) -> str:
    """Run M1 as a subprocess while recording line-by-line logs and status."""
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be a finite number greater than zero")

    command = _build_command(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )
    if experiment_id is None:
        experiment_id = queue_m1_experiment(
            session_factory,
            paper_id=paper_id,
            plan_id=plan_id,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
            timeout_seconds=timeout_seconds,
        )
    with session_factory() as session:
        experiment = session.get(Experiment, experiment_id)
        if experiment is None:
            raise LookupError(f"Queued experiment {experiment_id} was not found")
        if experiment.status != "QUEUED":
            raise ValueError(f"Experiment {experiment_id} is not QUEUED")
        experiment.command_json = json.dumps(command)
        experiment.timeout_seconds = timeout_seconds
        experiment.status = "RUNNING"
        experiment.started_at = _now()
        session.commit()

    stdout_lines: list[str] = []
    readers_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
    process: subprocess.Popen[str] | None = None
    timed_out = False
    error: str | None = None
    exit_code: int | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=BACKEND_DIRECTORY,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def read_stream(stream_name: str, stream: Any) -> None:
            try:
                for line in stream:
                    readers_queue.put((stream_name, line.rstrip("\r\n")))
            finally:
                readers_queue.put((stream_name, None))

        assert process.stdout is not None
        assert process.stderr is not None
        readers = [
            threading.Thread(
                target=read_stream,
                args=("stdout", process.stdout),
                daemon=True,
            ),
            threading.Thread(
                target=read_stream,
                args=("stderr", process.stderr),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()

        deadline = time.monotonic() + timeout_seconds
        completed_streams = 0
        while completed_streams < len(readers) or process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0 and process.poll() is None:
                timed_out = True
                process.kill()
                deadline = float("inf")
            try:
                stream_name, line = readers_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                completed_streams += 1
                continue
            if stream_name == "stdout":
                stdout_lines.append(line)
            _save_log(session_factory, experiment_id, stream_name, line)
        exit_code = process.wait()
        for reader in readers:
            reader.join()
        if timed_out:
            error = f"Experiment exceeded timeout of {timeout_seconds} seconds"
        elif exit_code != 0:
            error = f"M1 subprocess exited with status {exit_code}"
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        _save_log(session_factory, experiment_id, "stderr", error)

    run_directory = _run_directory_from_output(stdout_lines)
    config_path: str | None = None
    metrics_path: str | None = None
    try:
        config_path, metrics_path = _store_artifacts_and_metrics(
            session_factory, experiment_id, run_directory
        )
    except Exception as exception:
        if error is None:
            error = f"Artifact persistence failed: {type(exception).__name__}: {exception}"

    with session_factory() as session:
        experiment = session.get(Experiment, experiment_id)
        assert experiment is not None
        experiment.config_path = config_path
        experiment.metrics_path = metrics_path
        experiment.exit_code = exit_code
        experiment.finished_at = _now()
        if error is None and run_directory is not None and metrics_path is not None:
            metrics_payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
            if metrics_payload.get("status") == "completed":
                experiment.status = "COMPLETED"
            else:
                error = metrics_payload.get("error", "M1 did not report a completed run")
        if experiment.status != "COMPLETED":
            experiment.status = "FAILED"
            experiment.error = error or "M1 run artifacts were not produced"
        session.commit()
    return experiment_id
