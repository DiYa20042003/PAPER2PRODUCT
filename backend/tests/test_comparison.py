import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.comparison import (
    ComparisonThresholds,
    _reference_error_rate,
    compare_reproduction,
)
from app.models import Base, Experiment, Paper, PaperSection, ReproductionPlan, ReproducibilityScore


def make_session_factory() -> sessionmaker:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_reference_error_rate_is_converted_to_accuracy_from_paper_evidence() -> None:
    quote = (
        "The test error rate stabilizes after around ten passes through the "
        "training set at 0.95%."
    )
    error_rate = _reference_error_rate(quote)
    accuracy = 1 - error_rate

    assert error_rate == 0.0095
    assert accuracy == 0.9905


def test_comparison_thresholds_are_validated_and_configurable() -> None:
    thresholds = ComparisonThresholds(
        reproduced_max_error_rate_difference=0.01,
        partially_reproduced_max_error_rate_difference=0.03,
        metric_similarity_weight=0.5,
        setup_similarity_weight=0.3,
        evidence_quality_weight=0.2,
    )
    assert thresholds.reproduced_max_error_rate_difference == 0.01

    with pytest.raises(ValueError, match="cannot exceed"):
        ComparisonThresholds(
            reproduced_max_error_rate_difference=0.03,
            partially_reproduced_max_error_rate_difference=0.02,
        )
    with pytest.raises(ValueError, match="sum to 1"):
        ComparisonThresholds(
            metric_similarity_weight=0.5,
            setup_similarity_weight=0.4,
            evidence_quality_weight=0.2,
        )


def test_blocked_comparison_persists_status_without_fabricated_score() -> None:
    session_factory = make_session_factory()
    with session_factory() as session:
        paper = Paper(title="Paper with unavailable run evidence")
        session.add(paper)
        session.flush()
        plan = ReproductionPlan(
            paper_id=paper.id,
            status="BLOCKED",
            content_json=json.dumps({"paper_id": paper.id}),
        )
        session.add(plan)
        session.commit()
        plan_id = plan.id

    report = compare_reproduction(session_factory, plan_id)

    assert report["status"] == "BLOCKED"
    assert report["score"] is None
    assert report["metrics"] == {}
    assert report["blockers"]
    with session_factory() as session:
        score = session.scalar(
            select(ReproducibilityScore).where(ReproducibilityScore.plan_id == plan_id)
        )
        assert score is not None
        assert score.status == "BLOCKED"
        assert score.score is None
        assert json.loads(score.explanation_json)["comparison_id"] == score.id


def test_comparison_uses_latest_experiment_linked_to_plan(tmp_path) -> None:
    session_factory = make_session_factory()
    quote = (
        "The test error rate stabilizes after around ten passes through the "
        "training set at 0.95%."
    )
    stale_run = tmp_path / "stale"
    stale_run.mkdir()
    stale_metrics = stale_run / "metrics.json"
    stale_metrics.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "stale-run",
                "epochs": [{"test_accuracy": 0.99}],
            }
        ),
        encoding="utf-8",
    )
    linked_run = tmp_path / "linked"
    linked_run.mkdir()
    linked_metrics = linked_run / "metrics.json"
    linked_metrics.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "linked-run",
                "test_samples": 100,
                "device": "cpu",
                "epochs": [{"test_accuracy": 0.90}],
            }
        ),
        encoding="utf-8",
    )

    with session_factory() as session:
        paper = Paper(
            title="Evidence-based comparison",
            understanding_json=json.dumps(
                {"confidence_score": 0.35, "extraction_method": "rule_based"}
            ),
        )
        session.add(paper)
        session.flush()
        section = PaperSection(
            paper_id=paper.id,
            heading="Results",
            content=quote,
            section_order=0,
        )
        session.add(section)
        session.flush()
        plan = ReproductionPlan(
            paper_id=paper.id,
            status="READY",
            content_json=json.dumps(
                {
                    "original_setup": {
                        "baseline_performance": {
                            "section_id": section.id,
                            "quote": quote,
                        }
                    },
                    "observed_run_path": str(stale_run),
                    "differences": [],
                }
            ),
        )
        session.add(plan)
        session.flush()
        session.add(
            Experiment(
                id="older-experiment",
                paper_id=paper.id,
                plan_id=plan.id,
                status="COMPLETED",
                started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                metrics_path=str(stale_metrics),
            )
        )
        session.add(
            Experiment(
                id="linked-experiment",
                paper_id=paper.id,
                plan_id=plan.id,
                status="COMPLETED",
                started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                metrics_path=str(linked_metrics),
            )
        )
        session.commit()
        plan_id = plan.id

    report = compare_reproduction(session_factory, plan_id)

    assert report["experiment_id"] == "linked-experiment"
    assert report["evidence"]["observed_run"]["run_id"] == "linked-run"
    assert report["evidence"]["observed_run"]["test_accuracy_fraction"] == 0.90
