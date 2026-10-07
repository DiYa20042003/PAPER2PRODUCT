import json

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.comparison import compare_reproduction
from app.models import (
    Base,
    Experiment,
    Paper,
    PaperSection,
    ReproductionPlan,
    ReproducibilityScore,
)


REPORT_SENTENCE = (
    "Mid-sem reproduction configuration differs from the original experimental setup."
)


@pytest.mark.parametrize(
    ("has_differences", "sentence_expected"),
    [(True, True), (False, False)],
)
def test_comparison_report_mentions_setup_difference_only_when_present(
    tmp_path, has_differences, sentence_expected
) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    quote = (
        "The test error rate stabilizes after around ten passes through the "
        "training set at 0.95%."
    )
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "report-test-run",
                "test_samples": 100,
                "device": "cpu",
                "epochs": [{"test_accuracy": 0.90}],
            }
        ),
        encoding="utf-8",
    )

    with session_factory() as session:
        paper = Paper(
            title="Comparison report test",
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
                    "experiment_id": "report-test-experiment",
                    "original_setup": {
                        "baseline_performance": {
                            "section_id": section.id,
                            "quote": quote,
                        }
                    },
                    "observed_run_path": str(tmp_path),
                    "differences": (
                        [
                            {
                                "dimension": "training schedule",
                                "original_evidence": [
                                    {"section_id": section.id, "quote": quote}
                                ],
                            }
                        ]
                        if has_differences
                        else []
                    ),
                }
            ),
        )
        session.add(plan)
        session.flush()
        session.add(
            Experiment(
                id="report-test-experiment",
                paper_id=paper.id,
                plan_id=plan.id,
                status="COMPLETED",
                metrics_path=str(metrics_path),
            )
        )
        session.commit()
        plan_id = plan.id

    report = compare_reproduction(session_factory, plan_id)
    assert (REPORT_SENTENCE in report["rationale"]) is sentence_expected
    with session_factory() as session:
        persisted = session.scalar(
            select(ReproducibilityScore).where(
                ReproducibilityScore.plan_id == plan_id
            )
        )
        assert persisted is not None
        saved_report = json.loads(persisted.explanation_json)
        assert (REPORT_SENTENCE in saved_report["rationale"]) is sentence_expected

    engine.dispose()
