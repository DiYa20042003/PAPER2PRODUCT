"""Evidence-backed comparison of the paper setup and a real M1 run."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.models import Experiment, Paper, PaperSection, ReproductionPlan as ReproductionPlanRow
from experiments.lenet_mnist.model import MODEL_SPEC


BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]
RUNS_DIRECTORY = BACKEND_DIRECTORY / "experiments" / "lenet_mnist" / "runs"
REQUIREMENTS_PATH = BACKEND_DIRECTORY / "requirements.txt"
EVIDENCE_TERMS = {
    "dataset_counts": "60 000 training examples, 10 000 test examples",
    "mnist_definition": "resulting database was called",
    "preprocessing": "images were centered in a 28",
    "architecture": "Layer C1 is a convolutional layer",
    "training_duration": "Twenty iterations through the entire training data",
    "learning_rate_schedule": "global learning rate",
    "baseline_performance": "test error rate stabilizes after around ten passes",
    "augmentation_performance": "When distorted data were used for training",
}


class SourceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: int
    section_heading: str
    quote: str


class SetupDifference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str
    original_setup: str
    observed_setup: str
    significance: str
    original_evidence: list[SourceEvidence]
    observed_run_path: str


class ReproductionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: int | None = None
    paper_id: int
    experiment_id: str | None = None
    status: Literal["READY", "BLOCKED"]
    objective: str
    original_setup: dict[str, Any]
    reproduction_setup: dict[str, Any]
    dependencies: list[str]
    compute: dict[str, Any]
    missing_information: list[str]
    blockers: list[str]
    differences: list[SetupDifference]
    observed_run_path: str | None
    created_at: str


def _flat_text(text: str) -> str:
    return " ".join(text.split())


def _find_evidence(
    sections: list[PaperSection],
    term: str,
    *,
    exact_phrase: bool = False,
    exact_sentence: bool = False,
) -> SourceEvidence | None:
    for section in sections:
        flattened = _flat_text(section.content)
        match = re.search(re.escape(term), flattened, re.IGNORECASE)
        if match is None:
            continue
        offset = match.start()
        term_length = match.end() - match.start()
        if exact_phrase:
            quote = flattened[match.start() : match.end()]
        elif exact_sentence:
            preceding = list(
                re.finditer(
                    r"(?<=[.!?])\s+(?=[A-Z0-9])",
                    flattened[: match.start()],
                )
            )
            start = preceding[-1].end() if preceding else 0
            following = re.search(
                r"(?<=[.!?])\s+(?=[A-Z0-9])",
                flattened[match.end() :],
            )
            end = match.end() + following.start() if following else len(flattened)
            quote = flattened[start:end].strip()
        else:
            start = max(0, offset - 140)
            end = min(len(flattened), offset + term_length + 280)
            if start:
                next_space = flattened.find(" ", start)
                start = next_space + 1 if next_space >= 0 else start
            if end < len(flattened):
                previous_space = flattened.rfind(" ", 0, end)
                end = previous_space if previous_space > offset else end
            quote = flattened[start:end].strip()
        return SourceEvidence(
            section_id=section.id,
            section_heading=section.heading,
            quote=quote,
        )
    return None


def _find_completed_run(
    runs_directory: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    candidates: list[tuple[int, str, Path, dict[str, Any], dict[str, Any]]] = []
    if not runs_directory.is_dir():
        return None
    for run_directory in runs_directory.iterdir():
        config_path = run_directory / "config.json"
        metrics_path = run_directory / "metrics.json"
        if not config_path.is_file() or not metrics_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            metrics.get("status") != "completed"
            or not metrics.get("training_samples")
            or not metrics.get("test_samples")
            or not metrics.get("epochs")
        ):
            continue
        coverage = metrics["training_samples"] * metrics["test_samples"]
        candidates.append(
            (coverage, str(metrics.get("timestamp", "")), run_directory, config, metrics)
        )
    if not candidates:
        return None
    _, _, path, config, metrics = max(candidates, key=lambda item: (item[0], item[1]))
    return path, config, metrics


def _read_requirements() -> list[str]:
    dependencies: list[str] = []
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("--"):
            dependencies.append(line)
    return dependencies


def _make_differences(
    evidence: dict[str, SourceEvidence],
    run_path: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> list[SetupDifference]:
    model_spec = config.get("model_spec", MODEL_SPEC)
    preprocessing = config.get(
        "preprocessing",
        "torchvision.transforms.ToTensor(); no additional augmentation",
    )
    dataset_name = config.get("dataset", "torchvision.datasets.MNIST")
    optimizer = config.get("optimizer", "Adam")
    final_epoch = metrics["epochs"][-1]
    original_text = {
        "dataset": evidence["dataset_counts"].quote,
        "mnist": evidence["mnist_definition"].quote,
        "preprocessing": evidence["preprocessing"].quote,
        "architecture": evidence["architecture"].quote,
        "training": evidence["training_duration"].quote,
        "learning_rate": evidence["learning_rate_schedule"].quote,
        "baseline": evidence["baseline_performance"].quote,
        "augmentation": evidence["augmentation_performance"].quote,
    }
    differences = [
        SetupDifference(
            dimension="input preprocessing and dimensions",
            original_setup=original_text["preprocessing"],
            observed_setup=(
                f"{dataset_name}; input shape {model_spec['input_shape']}; "
                f"preprocessing: {preprocessing}"
            ),
            significance=(
                "The paper describes size-normalized, center-of-mass-aligned images "
                "and a 28x28 field that could be extended to 32x32; this run feeds "
                "28x28 tensors directly after ToTensor."
            ),
            original_evidence=[evidence["preprocessing"]],
            observed_run_path=str(run_path),
        ),
        SetupDifference(
            dimension="network architecture",
            original_setup=original_text["architecture"],
            observed_setup=json.dumps(model_spec, ensure_ascii=False),
            significance=(
                "The implementation is a LeNet-5-style variant with ReLU, fixed "
                "average pooling, and a 10-logit classifier rather than a claim of "
                "an exact reproduction of the paper's layer definitions."
            ),
            original_evidence=[evidence["architecture"]],
            observed_run_path=str(run_path),
        ),
        SetupDifference(
            dimension="training schedule and optimizer",
            original_setup=(
                f"{original_text['training']} {original_text['learning_rate']}"
            ),
            observed_setup=(
                f"{config['epochs']} epochs, batch size {config['batch_size']}, "
                f"{optimizer}, learning rate {config['learning_rate']}"
            ),
            significance=(
                "The run uses a different optimizer, duration, batch setting, and "
                "learning-rate schedule from the cited experimental description."
            ),
            original_evidence=[
                evidence["training_duration"],
                evidence["learning_rate_schedule"],
            ],
            observed_run_path=str(run_path),
        ),
        SetupDifference(
            dimension="data augmentation",
            original_setup=original_text["augmentation"],
            observed_setup=preprocessing,
            significance=(
                "The cited 0.8% test-error result used distorted training data; "
                "this M1 configuration records no additional augmentation."
            ),
            original_evidence=[evidence["augmentation_performance"]],
            observed_run_path=str(run_path),
        ),
        SetupDifference(
            dimension="reported performance and evaluation",
            original_setup=original_text["baseline"],
            observed_setup=(
                f"Observed test accuracy {final_epoch['test_accuracy']} after "
                f"{config['epochs']} epochs on {metrics['test_samples']} test examples."
            ),
            significance=(
                "These are different reported quantities and training setups; "
                "the scores are not evidence of an exact reproduction."
            ),
            original_evidence=[evidence["baseline_performance"]],
            observed_run_path=str(run_path),
        ),
    ]
    return differences


def create_reproduction_plan(
    session_factory: sessionmaker,
    paper_id: int,
    *,
    experiment_id: str | None = None,
    runs_directory: str | Path = RUNS_DIRECTORY,
) -> dict[str, Any]:
    with session_factory() as session:
        paper = session.get(Paper, paper_id)
        if paper is None:
            raise LookupError(f"Paper {paper_id} was not found")
        sections = list(
            session.scalars(
                select(PaperSection)
                .where(PaperSection.paper_id == paper_id)
                .order_by(PaperSection.section_order)
            )
        )
        if not sections:
            raise ValueError(f"Paper {paper_id} has no parsed sections")

        evidence = {
            key: _find_evidence(
                sections,
                term,
                exact_phrase=key == "dataset_counts",
                exact_sentence=key
                in {"baseline_performance", "augmentation_performance"},
            )
            for key, term in EVIDENCE_TERMS.items()
        }
        blockers = [
            f"Required source evidence not found: {key}"
            for key, item in evidence.items()
            if item is None
        ]
        evidence = {key: item for key, item in evidence.items() if item is not None}

        run_data: tuple[Path, dict[str, Any], dict[str, Any]] | None = None
        experiment = None
        if experiment_id is not None:
            experiment = session.get(Experiment, experiment_id)
            if experiment is None:
                raise LookupError(f"Experiment {experiment_id} was not found")
            if experiment.status != "COMPLETED" or not experiment.metrics_path:
                blockers.append("The selected experiment has no completed metrics.")
            else:
                metrics_path = Path(experiment.metrics_path)
                config_path = Path(experiment.config_path or "")
                if not metrics_path.is_file() or not config_path.is_file():
                    blockers.append("The selected experiment's config or metrics file is missing.")
                else:
                    run_data = (
                        metrics_path.parent,
                        json.loads(config_path.read_text(encoding="utf-8")),
                        json.loads(metrics_path.read_text(encoding="utf-8")),
                    )
        else:
            run_data = _find_completed_run(Path(runs_directory))
            if run_data is None:
                blockers.append("No completed M1 run with stored metrics was found.")

        if run_data is None:
            plan = ReproductionPlan(
                paper_id=paper_id,
                experiment_id=experiment_id,
                status="BLOCKED",
                objective=f"Plan a reproduction of {paper.title}",
                original_setup={},
                reproduction_setup={},
                dependencies=_read_requirements(),
                compute={"original": "Not identified", "reproduction": "Not identified"},
                missing_information=["Not identified until paper evidence and run metrics are available."],
                blockers=blockers,
                differences=[],
                observed_run_path=None,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        else:
            run_path, config, metrics = run_data
            required = set(EVIDENCE_TERMS)
            if required - set(evidence):
                blockers.extend(
                    f"Missing source evidence: {name}" for name in sorted(required - set(evidence))
                )
            model_spec = config.get("model_spec", MODEL_SPEC)
            preprocessing = config.get(
                "preprocessing",
                "torchvision.transforms.ToTensor(); no additional augmentation",
            )
            dataset_name = config.get("dataset", "torchvision.datasets.MNIST")
            final_epoch = metrics["epochs"][-1]
            differences = (
                _make_differences(evidence, run_path, config, metrics)
                if required <= set(evidence)
                else []
            )
            original_setup = {
                "dataset_counts": evidence.get("dataset_counts").model_dump()
                if evidence.get("dataset_counts")
                else "Not identified",
                "mnist_definition": evidence.get("mnist_definition").model_dump()
                if evidence.get("mnist_definition")
                else "Not identified",
                "preprocessing": evidence.get("preprocessing").model_dump()
                if evidence.get("preprocessing")
                else "Not identified",
                "architecture": evidence.get("architecture").model_dump()
                if evidence.get("architecture")
                else "Not identified",
                "training_duration": evidence.get("training_duration").model_dump()
                if evidence.get("training_duration")
                else "Not identified",
                "learning_rate_schedule": evidence.get("learning_rate_schedule").model_dump()
                if evidence.get("learning_rate_schedule")
                else "Not identified",
                "baseline_performance": evidence.get("baseline_performance").model_dump()
                if evidence.get("baseline_performance")
                else "Not identified",
                "augmentation_performance": evidence.get("augmentation_performance").model_dump()
                if evidence.get("augmentation_performance")
                else "Not identified",
            }
            reproduction_setup = {
                "run_id": metrics.get("run_id"),
                "dataset": dataset_name,
                "training_samples": metrics.get("training_samples"),
                "test_samples": metrics.get("test_samples"),
                "input_shape": model_spec.get("input_shape", "Not identified"),
                "preprocessing": preprocessing,
                "model": model_spec,
                "epochs": config.get("epochs"),
                "batch_size": config.get("batch_size"),
                "optimizer": config.get("optimizer", "Adam"),
                "learning_rate": config.get("learning_rate"),
                "device": metrics.get("device", "Not identified"),
                "torch_version": metrics.get("torch_version", "Not identified"),
                "final_epoch_metrics": final_epoch,
                "execution_time_seconds": metrics.get("execution_time_seconds"),
            }
            plan = ReproductionPlan(
                paper_id=paper_id,
                experiment_id=experiment_id,
                status="BLOCKED" if blockers else "READY",
                objective=f"Reproduce the MNIST experiment described in {paper.title}",
                original_setup=original_setup,
                reproduction_setup=reproduction_setup,
                dependencies=_read_requirements(),
                compute={
                    "original": "Not identified in the ingested paper text.",
                    "reproduction": metrics.get("device", "Not identified"),
                },
                missing_information=[
                    "Original software and library versions: Not identified in the paper.",
                    "Original hardware and accelerator: Not identified in the paper.",
                    "Original mini-batch size: Not identified in the paper.",
                    "The paper's source-code revision: Not identified in the paper.",
                ],
                blockers=blockers,
                differences=differences,
                observed_run_path=str(run_path),
                created_at=datetime.now(timezone.utc).isoformat(),
            )

        session.add(
            ReproductionPlanRow(
                paper_id=paper_id,
                status=plan.status,
                content_json=plan.model_dump_json(),
                differences_json=json.dumps(
                    [difference.model_dump() for difference in plan.differences],
                    ensure_ascii=False,
                ),
            )
        )
        session.commit()
        return {
            **plan.model_dump(),
            "plan_id": session.scalar(
                select(ReproductionPlanRow.id)
                .where(ReproductionPlanRow.paper_id == paper_id)
                .order_by(ReproductionPlanRow.id.desc())
                .limit(1)
            ),
        }
