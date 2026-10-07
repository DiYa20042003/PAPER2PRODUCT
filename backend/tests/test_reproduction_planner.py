from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.models import Base, Paper, PaperSection, ReproductionPlan as ReproductionPlanRow
from app.reproduction_planner import (
    ReproductionPlan,
    _find_evidence,
    create_reproduction_plan,
)


def make_session_factory() -> sessionmaker:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_evidence_reference_is_an_exact_paper_excerpt() -> None:
    section = PaperSection(
        id=17,
        paper_id=1,
        heading="Results",
        content="Counts: 60 000 training examples, 10 000 test examples.",
    )

    evidence = _find_evidence(
        [section],
        "60 000 training examples, 10 000 test examples",
        exact_phrase=True,
    )

    assert evidence is not None
    assert evidence.section_id == 17
    assert evidence.quote == "60 000 training examples, 10 000 test examples"
    assert evidence.quote in section.content


def test_plan_is_blocked_without_paper_evidence_or_completed_run(tmp_path) -> None:
    session_factory = make_session_factory()
    with session_factory() as session:
        paper = Paper(title="Paper without extracted source evidence", source="test")
        paper.sections = [
            PaperSection(heading="Unparsed", content="No supported setup details.")
        ]
        session.add(paper)
        session.commit()
        paper_id = paper.id

    plan = create_reproduction_plan(
        session_factory,
        paper_id,
        runs_directory=tmp_path,
    )

    assert isinstance(ReproductionPlan.model_validate(plan), ReproductionPlan)
    assert plan["status"] == "BLOCKED"
    assert plan["observed_run_path"] is None
    assert plan["reproduction_setup"] == {}
    assert plan["compute"] == {
        "original": "Not identified",
        "reproduction": "Not identified",
    }
    assert any("No completed M1 run" in blocker for blocker in plan["blockers"])
    with session_factory() as session:
        stored_plan = session.scalar(
            select(ReproductionPlanRow).where(ReproductionPlanRow.id == plan["plan_id"])
        )
        assert stored_plan is not None
        assert stored_plan.status == "BLOCKED"
