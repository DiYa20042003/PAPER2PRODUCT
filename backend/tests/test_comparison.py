import json

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.comparison import (
    ComparisonThresholds,
    _reference_error_rate,
    compare_reproduction,
)
from app.models import Base, Paper, ReproductionPlan, ReproducibilityScore


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
