"""FastAPI service for paper ingestion and reproduction experiments."""

import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.error import URLError

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.api_schemas import (
    ComparisonRequest,
    ExperimentListResponse,
    ExperimentRequest,
    ExperimentResponse,
    IngestRequest,
    LogsResponse,
    PaperListResponse,
    PaperResponse,
    PipelineStatusResponse,
    PlanRequest,
)
from app.comparison import ComparisonThresholds, compare_reproduction
from app.database import SessionLocal, engine, initialize_database
from app.experiment_service import queue_m1_experiment, run_m1_experiment
from app.models import (
    Artifact,
    Claim,
    Experiment,
    ExperimentLog,
    Paper,
    PaperSection,
    ReproductionPlan,
    ReproducibilityScore,
)
from app.paper_analysis import analyze_paper
from app.paper_ingestion import (
    MAX_PDF_BYTES,
    PAPERS_DIRECTORY,
    ingest_pdf,
    ingest_source,
)
from app.reproduction_planner import create_reproduction_plan


UPLOAD_DIRECTORY = PAPERS_DIRECTORY / "uploads"


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database(engine)
    yield


app = FastAPI(
    title="PAPER2PRODUCT API",
    version="0.1.0",
    description="Evidence-backed paper ingestion, analysis, and reproduction runs.",
    lifespan=lifespan,
)
origins = [
    origin.strip()
    for origin in os.environ.get("PAPER2PRODUCT_CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def get_session_factory() -> sessionmaker[Session]:
    return SessionLocal


def _paper_response(session: Session, paper: Paper) -> dict[str, Any]:
    sections = list(
        session.scalars(
            select(PaperSection)
            .where(PaperSection.paper_id == paper.id)
            .order_by(PaperSection.section_order)
        )
    )
    return {
        "id": paper.id,
        "title": paper.title,
        "source": paper.source,
        "source_ref": paper.source_ref,
        "abstract": paper.abstract,
        "status": paper.status,
        "section_count": len(sections),
    }


def _experiment_response(experiment: Experiment) -> dict[str, Any]:
    return {
        "id": experiment.id,
        "paper_id": experiment.paper_id,
        "plan_id": experiment.plan_id,
        "status": experiment.status,
        "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
        "finished_at": experiment.finished_at.isoformat() if experiment.finished_at else None,
        "exit_code": experiment.exit_code,
        "error": experiment.error,
        "config_path": experiment.config_path,
        "metrics_path": experiment.metrics_path,
        "artifact_directory": experiment.artifact_directory,
    }


def _required_paper(session_factory: sessionmaker, paper_id: int) -> None:
    with session_factory() as session:
        if session.get(Paper, paper_id) is None:
            raise HTTPException(status_code=404, detail=f"Paper {paper_id} was not found")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/papers", response_model=PaperListResponse, tags=["papers"])
def list_papers(
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, list[dict[str, Any]]]:
    with session_factory() as session:
        papers = list(session.scalars(select(Paper).order_by(Paper.id.desc())))
        return {"papers": [_paper_response(session, paper) for paper in papers]}


@app.post("/papers/ingest", status_code=201, tags=["ingest"])
def ingest_paper_source(
    body: IngestRequest,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    try:
        paper_id = ingest_source(
            session_factory, body.source_ref, source_type=body.source_type
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except URLError as error:
        raise HTTPException(status_code=502, detail=f"Paper source request failed: {error.reason}") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    with session_factory() as session:
        paper = session.get(Paper, paper_id)
        assert paper is not None
        return _paper_response(session, paper)


@app.post("/papers/upload", status_code=201, tags=["ingest"])
async def upload_paper(
    file: UploadFile = File(...),
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    payload = await file.read(MAX_PDF_BYTES + 1)
    if len(payload) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 50 MiB upload limit")
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pdf_path = UPLOAD_DIRECTORY / f"{uuid.uuid4().hex}.pdf"
    pdf_path.write_bytes(payload)
    try:
        paper_id = ingest_pdf(
            session_factory,
            pdf_path,
            source="upload",
            source_ref=file.filename or pdf_path.name,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    with session_factory() as session:
        paper = session.get(Paper, paper_id)
        assert paper is not None
        return _paper_response(session, paper)


@app.get("/papers/{paper_id}", tags=["papers"])
def get_paper(
    paper_id: int,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    with session_factory() as session:
        paper = session.get(Paper, paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail=f"Paper {paper_id} was not found")
        result = _paper_response(session, paper)
        result["authors"] = json.loads(paper.authors_json)
        result["metadata"] = json.loads(paper.metadata_json)
        result["sections"] = [
            {
                "id": section.id,
                "heading": section.heading,
                "content": section.content,
                "page_start": section.page_start,
                "page_end": section.page_end,
            }
            for section in session.scalars(
                select(PaperSection)
                .where(PaperSection.paper_id == paper_id)
                .order_by(PaperSection.section_order)
            )
        ]
        return result


@app.post("/papers/{paper_id}/analyze", tags=["analysis"])
def analyze(
    paper_id: int,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    try:
        return analyze_paper(session_factory, paper_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.get("/papers/{paper_id}/understanding", tags=["analysis"])
def get_understanding(
    paper_id: int,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    with session_factory() as session:
        paper = session.get(Paper, paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail=f"Paper {paper_id} was not found")
        if not paper.understanding_json:
            raise HTTPException(status_code=409, detail="Paper has not been analyzed")
        return json.loads(paper.understanding_json)


@app.get("/papers/{paper_id}/claims", tags=["analysis"])
def get_claims(
    paper_id: int,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    with session_factory() as session:
        if session.get(Paper, paper_id) is None:
            raise HTTPException(status_code=404, detail=f"Paper {paper_id} was not found")
        claims = list(
            session.scalars(
                select(Claim).where(Claim.paper_id == paper_id).order_by(Claim.id)
            )
        )
        return {
            "paper_id": paper_id,
            "claims": [
                {
                    "id": claim.id,
                    "claim_type": claim.claim_type,
                    "statement": claim.statement,
                    "section_id": claim.section_id,
                    "evidence": claim.evidence,
                    "confidence": claim.confidence,
                    "supported": claim.supported,
                    "extraction_method": claim.extraction_method,
                }
                for claim in claims
            ],
        }


@app.post("/papers/{paper_id}/plan", tags=["planning"])
def create_plan(
    paper_id: int,
    body: PlanRequest,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    try:
        return create_reproduction_plan(
            session_factory, paper_id, experiment_id=body.experiment_id
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/papers/{paper_id}/plan", tags=["planning"])
def get_latest_plan(
    paper_id: int,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    with session_factory() as session:
        if session.get(Paper, paper_id) is None:
            raise HTTPException(status_code=404, detail=f"Paper {paper_id} was not found")
        plan = session.scalar(
            select(ReproductionPlan)
            .where(ReproductionPlan.paper_id == paper_id)
            .order_by(ReproductionPlan.id.desc())
            .limit(1)
        )
        if plan is None:
            raise HTTPException(status_code=404, detail="No reproduction plan exists")
        return {"plan_id": plan.id, **json.loads(plan.content_json)}


@app.post("/experiments", status_code=202, response_model=ExperimentResponse, tags=["experiments"])
def start_experiment(
    body: ExperimentRequest,
    background_tasks: BackgroundTasks,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    paper_id = body.paper_id
    if body.plan_id is not None:
        with session_factory() as session:
            plan = session.get(ReproductionPlan, body.plan_id)
            if plan is None:
                raise HTTPException(status_code=404, detail=f"Plan {body.plan_id} was not found")
            if paper_id is not None and plan.paper_id != paper_id:
                raise HTTPException(
                    status_code=422,
                    detail="plan_id does not belong to the requested paper_id",
                )
            paper_id = plan.paper_id
    if paper_id is not None:
        _required_paper(session_factory, paper_id)
    experiment_id = queue_m1_experiment(
        session_factory,
        paper_id=paper_id,
        plan_id=body.plan_id,
        epochs=body.epochs,
        batch_size=body.batch_size,
        learning_rate=body.learning_rate,
        device=body.device,
        max_train_samples=body.max_train_samples,
        max_test_samples=body.max_test_samples,
        timeout_seconds=body.timeout_seconds,
    )
    background_tasks.add_task(
        run_m1_experiment,
        session_factory,
        experiment_id=experiment_id,
        paper_id=paper_id,
        plan_id=body.plan_id,
        epochs=body.epochs,
        batch_size=body.batch_size,
        learning_rate=body.learning_rate,
        device=body.device,
        max_train_samples=body.max_train_samples,
        max_test_samples=body.max_test_samples,
        timeout_seconds=body.timeout_seconds,
    )
    with session_factory() as session:
        experiment = session.get(Experiment, experiment_id)
        assert experiment is not None
        return _experiment_response(experiment)


@app.get("/experiments", response_model=ExperimentListResponse, tags=["experiments"])
def list_experiments(
    paper_id: int | None = None,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, list[dict[str, Any]]]:
    statement = select(Experiment).order_by(Experiment.started_at.desc(), Experiment.id.desc())
    if paper_id is not None:
        statement = statement.where(Experiment.paper_id == paper_id)
    with session_factory() as session:
        experiments = list(session.scalars(statement))
        return {"experiments": [_experiment_response(item) for item in experiments]}


@app.get("/experiments/{experiment_id}", response_model=ExperimentResponse, tags=["experiments"])
def get_experiment(
    experiment_id: str,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    with session_factory() as session:
        experiment = session.get(Experiment, experiment_id)
        if experiment is None:
            raise HTTPException(status_code=404, detail="Experiment was not found")
        return _experiment_response(experiment)


@app.get("/experiments/{experiment_id}/logs", response_model=LogsResponse, tags=["experiments"])
def get_experiment_logs(
    experiment_id: str,
    after_id: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    with session_factory() as session:
        if session.get(Experiment, experiment_id) is None:
            raise HTTPException(status_code=404, detail="Experiment was not found")
        logs = list(
            session.scalars(
                select(ExperimentLog)
                .where(
                    ExperimentLog.experiment_id == experiment_id,
                    ExperimentLog.id > after_id,
                )
                .order_by(ExperimentLog.id)
                .limit(limit)
            )
        )
        return {
            "experiment_id": experiment_id,
            "logs": [
                {
                    "id": item.id,
                    "stream": item.stream,
                    "message": item.message,
                    "created_at": item.created_at.isoformat(),
                }
                for item in logs
            ],
            "next_after_id": logs[-1].id if len(logs) == limit else None,
        }


@app.post("/plans/{plan_id}/compare", tags=["reports"])
def compare_plan(
    plan_id: int,
    body: ComparisonRequest,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    try:
        thresholds = ComparisonThresholds(**body.model_dump())
        return compare_reproduction(session_factory, plan_id, thresholds=thresholds)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/experiments/{experiment_id}/report", tags=["reports"])
def experiment_report(
    experiment_id: str,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    with session_factory() as session:
        experiment = session.get(Experiment, experiment_id)
        if experiment is None:
            raise HTTPException(status_code=404, detail="Experiment was not found")
        score = session.scalar(
            select(ReproducibilityScore)
            .where(ReproducibilityScore.experiment_id == experiment_id)
            .order_by(ReproducibilityScore.id.desc())
            .limit(1)
        )
        return {
            "experiment": _experiment_response(experiment),
            "comparison": json.loads(score.explanation_json) if score else None,
        }


@app.get("/papers/{paper_id}/report", tags=["reports"])
def paper_report(
    paper_id: int,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    with session_factory() as session:
        paper = session.get(Paper, paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail=f"Paper {paper_id} was not found")
        plan = session.scalar(
            select(ReproductionPlan)
            .where(ReproductionPlan.paper_id == paper_id)
            .order_by(ReproductionPlan.id.desc())
            .limit(1)
        )
        score = (
            session.scalar(
                select(ReproducibilityScore)
                .where(ReproducibilityScore.plan_id == plan.id)
                .order_by(ReproducibilityScore.id.desc())
                .limit(1)
            )
            if plan
            else None
        )
        return {
            "paper": _paper_response(session, paper),
            "understanding": json.loads(paper.understanding_json)
            if paper.understanding_json
            else None,
            "plan": {"plan_id": plan.id, **json.loads(plan.content_json)} if plan else None,
            "comparison": json.loads(score.explanation_json) if score else None,
        }


@app.get("/pipeline/status", response_model=PipelineStatusResponse, tags=["pipeline"])
def pipeline_status(
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    with session_factory() as session:
        paper_count = session.scalar(select(func.count()).select_from(Paper)) or 0
        status_rows = session.execute(
            select(Experiment.status, func.count())
            .group_by(Experiment.status)
        ).all()
        latest_paper = session.scalar(select(Paper).order_by(Paper.id.desc()).limit(1))
        latest_experiment = session.scalar(
            select(Experiment)
            .order_by(Experiment.started_at.desc(), Experiment.id.desc())
            .limit(1)
        )
        latest_plan = session.scalar(
            select(ReproductionPlan)
            .order_by(ReproductionPlan.id.desc())
            .limit(1)
        )
        latest_score = session.scalar(
            select(ReproducibilityScore)
            .order_by(ReproducibilityScore.id.desc())
            .limit(1)
        )
        pipeline_state = (
            latest_score.status
            if latest_score
            else latest_experiment.status
            if latest_experiment
            else latest_paper.status
            if latest_paper
            else "EMPTY"
        )
        return {
            "papers": paper_count,
            "experiments_by_status": {status: count for status, count in status_rows},
            "latest_paper_id": latest_paper.id if latest_paper else None,
            "latest_experiment_id": latest_experiment.id if latest_experiment else None,
            "latest_plan_id": latest_plan.id if latest_plan else None,
            "latest_comparison_id": latest_score.id if latest_score else None,
            "latest_pipeline_status": pipeline_state,
        }
