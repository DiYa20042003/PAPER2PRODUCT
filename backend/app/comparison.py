"""Compare observed experiment results against evidence from the paper."""

import json
import math
import re
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.models import (
    Experiment,
    Paper,
    PaperSection,
    ReproductionPlan as ReproductionPlanRow,
    ReproducibilityScore,
)


class ComparisonThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reproduced_max_error_rate_difference: float = Field(default=0.005, ge=0, le=1)
    partially_reproduced_max_error_rate_difference: float = Field(default=0.02, gt=0, le=1)
    metric_similarity_weight: float = Field(default=0.60, ge=0, le=1)
    setup_similarity_weight: float = Field(default=0.25, ge=0, le=1)
    evidence_quality_weight: float = Field(default=0.15, ge=0, le=1)

    @model_validator(mode="after")
    def validate_thresholds_and_weights(self):
        if (
            self.reproduced_max_error_rate_difference
            > self.partially_reproduced_max_error_rate_difference
        ):
            raise ValueError(
                "reproduced threshold cannot exceed partially reproduced threshold"
            )
        total_weight = (
            self.metric_similarity_weight
            + self.setup_similarity_weight
            + self.evidence_quality_weight
        )
        if not math.isclose(total_weight, 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("score weights must sum to 1")
        return self


class ComparisonReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comparison_id: int | None = None
    plan_id: int
    paper_id: int
    experiment_id: str | None
    status: Literal[
        "REPRODUCED", "PARTIALLY_REPRODUCED", "NOT_REPRODUCED", "BLOCKED"
    ]
    score: float | None
    score_components: dict[str, float | None]
    thresholds: ComparisonThresholds
    metrics: dict[str, Any]
    evidence: dict[str, Any]
    verification: dict[str, list[str]]
    rationale: str
    blockers: list[str]
    created_at: str


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _reference_error_rate(quote: str) -> float:
    match = re.search(
        r"test error rate stabilizes.*?\bat\s+(\d+(?:\.\d+)?)\s*%",
        quote,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError("Paper's reference test error rate was not identified.")
    rate = float(match.group(1)) / 100
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Paper's reference error rate is outside the valid range.")
    return rate


def _blocked_report(
    plan_id: int,
    paper_id: int,
    experiment_id: str | None,
    thresholds: ComparisonThresholds,
    blockers: list[str],
) -> dict[str, Any]:
    return ComparisonReport(
        plan_id=plan_id,
        paper_id=paper_id,
        experiment_id=experiment_id,
        status="BLOCKED",
        score=None,
        score_components={
            "metric_similarity": None,
            "setup_similarity": None,
            "evidence_quality": None,
        },
        thresholds=thresholds,
        metrics={},
        evidence={},
        verification={
            "OBSERVED EVIDENCE": [
                "The persisted reproduction plan or source/run artifacts are incomplete."
            ],
            "POSSIBLE EXPLANATION": [],
        },
        rationale="No comparison score was computed because required evidence is unavailable.",
        blockers=blockers,
        created_at=datetime.now(timezone.utc).isoformat(),
    ).model_dump()


def _persist_report(
    session_factory: sessionmaker,
    report: dict[str, Any],
    plan_id: int,
    experiment_id: str | None,
) -> dict[str, Any]:
    with session_factory() as session:
        score_row = ReproducibilityScore(
            plan_id=plan_id,
            experiment_id=experiment_id,
            score=report["score"],
            status=report["status"],
            explanation_json=json.dumps(report, ensure_ascii=False),
        )
        session.add(score_row)
        session.commit()
        report["comparison_id"] = score_row.id
    return report


def compare_reproduction(
    session_factory: sessionmaker,
    plan_id: int,
    *,
    thresholds: ComparisonThresholds | None = None,
) -> dict[str, Any]:
    policy = thresholds or ComparisonThresholds()
    with session_factory() as session:
        plan_row = session.get(ReproductionPlanRow, plan_id)
        if plan_row is None:
            raise LookupError(f"Reproduction plan {plan_id} was not found")
        plan = json.loads(plan_row.content_json)
        paper = session.get(Paper, plan_row.paper_id)
        sections = list(
            session.scalars(
                select(PaperSection)
                .where(PaperSection.paper_id == plan_row.paper_id)
                .order_by(PaperSection.section_order)
            )
        )
        experiment_id = plan.get("experiment_id")
        experiment = session.get(Experiment, experiment_id) if experiment_id else None

    blockers: list[str] = []
    if plan_row.status != "READY":
        blockers.append(f"Reproduction plan status is {plan_row.status}, not READY.")
    if paper is None:
        blockers.append("Paper record is missing.")
    if not sections:
        blockers.append("Parsed paper sections are missing.")
    if experiment_id and (experiment is None or experiment.status != "COMPLETED"):
        blockers.append("The selected experiment is not marked COMPLETED.")
    run_path = plan.get("observed_run_path")
    metrics_path = Path(experiment.metrics_path) if experiment and experiment.metrics_path else (
        Path(run_path) / "metrics.json" if run_path else None
    )
    if metrics_path is None or not metrics_path.is_file():
        blockers.append("A persisted experiment metrics.json file is missing.")

    original_performance = plan.get("original_setup", {}).get("baseline_performance")
    if not isinstance(original_performance, dict):
        blockers.append("Evidence for the paper's baseline test error is missing.")
    else:
        section_by_id = {section.id: section for section in sections}
        source_section = section_by_id.get(original_performance.get("section_id"))
        if (
            source_section is None
            or _normalize(original_performance.get("quote", ""))
            not in _normalize(source_section.content)
        ):
            blockers.append("The paper's baseline evidence quote does not match its source section.")

    if blockers:
        blocked = _blocked_report(plan_id, plan_row.paper_id, experiment_id, policy, blockers)
        return _persist_report(session_factory, blocked, plan_id, experiment_id)

    assert paper is not None and metrics_path is not None and original_performance is not None
    try:
        run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        blockers.append(f"Experiment metrics could not be read: {error}")
        blocked = _blocked_report(plan_id, plan_row.paper_id, experiment_id, policy, blockers)
        return _persist_report(session_factory, blocked, plan_id, experiment_id)

    if run_metrics.get("status") != "completed":
        blockers.append("Experiment metrics do not report a completed run.")
    epochs = run_metrics.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        blockers.append("Completed experiment has no recorded epoch metrics.")
    if blockers:
        blocked = _blocked_report(plan_id, plan_row.paper_id, experiment_id, policy, blockers)
        return _persist_report(session_factory, blocked, plan_id, experiment_id)

    if not isinstance(epochs[-1], dict):
        blocked = _blocked_report(
            plan_id,
            plan_row.paper_id,
            experiment_id,
            policy,
            ["Final epoch metrics are malformed."],
        )
        return _persist_report(session_factory, blocked, plan_id, experiment_id)
    observed_accuracy = epochs[-1].get("test_accuracy")
    if (
        not isinstance(observed_accuracy, (int, float))
        or not math.isfinite(observed_accuracy)
        or not 0 <= observed_accuracy <= 1
    ):
        blocked = _blocked_report(
            plan_id,
            plan_row.paper_id,
            experiment_id,
            policy,
            ["Observed test accuracy is not a finite fraction between 0 and 1."],
        )
        return _persist_report(session_factory, blocked, plan_id, experiment_id)
    try:
        reference_error = _reference_error_rate(original_performance["quote"])
    except (KeyError, ValueError) as error:
        blocked = _blocked_report(
            plan_id,
            plan_row.paper_id,
            experiment_id,
            policy,
            [str(error)],
        )
        return _persist_report(session_factory, blocked, plan_id, experiment_id)

    reference_error = round(reference_error, 8)
    reference_accuracy = round(1 - reference_error, 8)
    observed_error = round(1 - float(observed_accuracy), 8)
    absolute_error_difference = round(abs(observed_error - reference_error), 8)
    signed_accuracy_difference = float(observed_accuracy) - reference_accuracy
    accuracy_relative_difference = (
        abs(signed_accuracy_difference) / reference_accuracy
        if reference_accuracy
        else None
    )
    error_relative_difference = (
        absolute_error_difference / reference_error if reference_error else None
    )
    differences = plan.get("differences", [])
    setup_similarity = 1.0 if not differences else 0.0
    metric_similarity = max(
        0.0,
        1
        - absolute_error_difference
        / policy.partially_reproduced_max_error_rate_difference,
    )

    all_references = [
        reference
        for difference in differences
        for reference in difference.get("original_evidence", [])
    ]
    if original_performance not in all_references:
        all_references.append(original_performance)
    valid_references = 0
    section_by_id = {section.id: section for section in sections}
    for reference in all_references:
        source_section = section_by_id.get(reference.get("section_id"))
        quote = reference.get("quote", "")
        if source_section and quote and _normalize(quote) in _normalize(source_section.content):
            valid_references += 1
    quote_coverage = valid_references / len(all_references) if all_references else 0.0
    understanding = json.loads(paper.understanding_json) if paper.understanding_json else {}
    extraction_confidence = understanding.get("confidence_score")
    if not isinstance(extraction_confidence, (int, float)) or not math.isfinite(
        extraction_confidence
    ):
        extraction_confidence = 0.0
    evidence_quality = 0.5 * quote_coverage + 0.5 * extraction_confidence
    metric_similarity_for_score = round(metric_similarity, 8)
    setup_similarity_for_score = round(setup_similarity, 8)
    evidence_quality_for_score = round(evidence_quality, 8)
    score = float(
        Decimal("100")
        * (
            Decimal(str(policy.metric_similarity_weight))
            * Decimal(str(metric_similarity_for_score))
            + Decimal(str(policy.setup_similarity_weight))
            * Decimal(str(setup_similarity_for_score))
            + Decimal(str(policy.evidence_quality_weight))
            * Decimal(str(evidence_quality_for_score))
        )
    )

    if (
        setup_similarity == 1
        and absolute_error_difference
        <= policy.reproduced_max_error_rate_difference
    ):
        status: Literal[
            "REPRODUCED", "PARTIALLY_REPRODUCED", "NOT_REPRODUCED", "BLOCKED"
        ] = "REPRODUCED"
        rationale = "Recorded setup dimensions match and the error-rate difference is within the reproduced threshold."
    elif absolute_error_difference <= policy.partially_reproduced_max_error_rate_difference:
        status = "PARTIALLY_REPRODUCED"
        rationale = "A completed result is within the partial error-rate threshold, but setup differences remain."
    else:
        status = "NOT_REPRODUCED"
        rationale = "The absolute error-rate difference exceeds the partial reproduction threshold."

    reference_section = section_by_id[original_performance["section_id"]]
    evidence_summary = {
        "paper_baseline": {
            "error_rate_fraction": reference_error,
            "error_rate_percent": reference_error * 100,
            "accuracy_fraction_derived": reference_accuracy,
            "accuracy_percent_derived": round(reference_accuracy * 100, 6),
            "conversion": "accuracy = 1 - (reported error rate percent / 100)",
            "section_id": reference_section.id,
            "section_heading": reference_section.heading,
            "quote": original_performance["quote"],
        },
        "observed_run": {
            "run_id": run_metrics.get("run_id"),
            "metrics_path": str(metrics_path),
            "test_accuracy_fraction": float(observed_accuracy),
            "test_accuracy_percent": float(observed_accuracy) * 100,
            "test_error_rate_fraction": observed_error,
            "test_error_rate_percent": round(observed_error * 100, 6),
            "test_samples": run_metrics.get("test_samples"),
            "device": run_metrics.get("device", "Not identified"),
        },
        "source_quote_coverage": {
            "valid": valid_references,
            "total": len(all_references),
            "fraction": quote_coverage,
        },
        "extraction_confidence": {
            "value": extraction_confidence,
            "method": understanding.get("extraction_method", "Not identified"),
        },
    }
    metrics_report = {
        "accuracy_signed_difference": round(signed_accuracy_difference, 8),
        "accuracy_absolute_difference": round(abs(signed_accuracy_difference), 8),
        "accuracy_relative_difference": round(accuracy_relative_difference, 8)
        if accuracy_relative_difference is not None
        else None,
        "error_rate_signed_difference": round(observed_error - reference_error, 8),
        "error_rate_absolute_difference": round(absolute_error_difference, 8),
        "error_rate_relative_difference": round(error_relative_difference, 8)
        if error_relative_difference is not None
        else None,
        "error_rate_absolute_difference_percentage_points": round(
            absolute_error_difference * 100, 6
        ),
    }
    possible_explanations = [
        (
            f"The observed {difference['dimension']} mismatch is a plausible contributor "
            "to the measured gap; this comparison does not establish causality."
        )
        for difference in differences
    ]
    report = ComparisonReport(
        plan_id=plan_id,
        paper_id=plan_row.paper_id,
        experiment_id=experiment_id,
        status=status,
        score=float(Decimal(str(score)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
        score_components={
            "metric_similarity": round(metric_similarity, 8),
            "setup_similarity": setup_similarity,
            "evidence_quality": round(evidence_quality, 8),
        },
        thresholds=policy,
        metrics=metrics_report,
        evidence=evidence_summary,
        verification={
            "OBSERVED EVIDENCE": [
                (
                    f"Paper reports {reference_error * 100}% test error: "
                    f"{original_performance['quote']}"
                ),
                (
                    f"Run {run_metrics.get('run_id')} recorded test accuracy "
                    f"{float(observed_accuracy) * 100}% across "
                    f"{run_metrics.get('test_samples')} examples."
                ),
                (
                    f"Absolute error-rate gap is "
                    f"{round(absolute_error_difference * 100, 4)} percentage points."
                ),
                (
                    f"{valid_references} of {len(all_references)} source evidence "
                    "quotes match their stored paper sections."
                ),
            ],
            "POSSIBLE EXPLANATION": possible_explanations,
        },
        rationale=rationale,
        blockers=[],
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    report_payload = report.model_dump()
    return _persist_report(session_factory, report_payload, plan_id, experiment_id)
