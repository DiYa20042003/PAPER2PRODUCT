import fitz
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import main
from app.database import initialize_database
from app.models import Base


def make_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Gradient-based learning applied to document recognition\n"
        "Yann Lecun, Léon Bottou, Yoshua Bengio, Patrick Haffner\n"
        "ABSTRACT\n"
        "The resulting database was called the modified NIST, or MNIST, dataset.\n"
        "The test error rate stabilizes after around ten passes through the training set at 0.95%.\n"
        "I.\n"
        "INTRODUCTION\n"
        "Layer C1 is a convolutional layer with six feature maps.",
    )
    document.set_metadata(
        {
            "title": "Gradient-based learning applied to document recognition",
            "author": "Yann Lecun, Léon Bottou, Yoshua Bengio, Patrick Haffner",
        }
    )
    payload = document.tobytes()
    document.close()
    return payload


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_database(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    main.app.dependency_overrides[main.get_session_factory] = lambda: session_factory
    monkeypatch.setattr(main, "UPLOAD_DIRECTORY", tmp_path / "uploads")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("PAPER2PRODUCT_LLM_PROVIDER", "gemini")
    with TestClient(main.app) as client:
        yield client
    main.app.dependency_overrides.clear()
    engine.dispose()


def test_pdf_upload_analysis_claims_plan_and_report(api_client) -> None:
    response = api_client.post(
        "/papers/upload",
        files={"file": ("lecun98.pdf", make_pdf(), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    paper_id = response.json()["id"]
    assert response.json()["title"] == (
        "Gradient-based learning applied to document recognition"
    )
    assert response.json()["section_count"] >= 3

    papers = api_client.get("/papers")
    paper_detail = api_client.get(f"/papers/{paper_id}")
    assert papers.status_code == 200
    assert len(papers.json()["papers"]) == 1
    assert paper_detail.status_code == 200
    assert paper_detail.json()["sections"]

    understanding = api_client.post(f"/papers/{paper_id}/analyze")
    assert understanding.status_code == 200, understanding.text
    assert understanding.json()["extraction_method"] == "rule_based"
    assert understanding.json()["confidence_level"] == "lower"
    assert any("MNIST" in fact["evidence_quote"] for fact in understanding.json()["facts"])

    claims = api_client.get(f"/papers/{paper_id}/claims")
    assert claims.status_code == 200
    assert claims.json()["claims"]
    assert all(claim["supported"] for claim in claims.json()["claims"])

    plan = api_client.post(f"/papers/{paper_id}/plan", json={})
    assert plan.status_code == 200, plan.text
    assert plan.json()["status"] == "BLOCKED"
    plan_id = plan.json()["plan_id"]
    persisted_plan = api_client.get(f"/papers/{paper_id}/plan")
    assert persisted_plan.status_code == 200
    assert persisted_plan.json()["plan_id"] == plan_id
    comparison = api_client.post(f"/plans/{plan_id}/compare", json={})
    assert comparison.status_code == 200
    assert comparison.json()["status"] == "BLOCKED"
    assert comparison.json()["score"] is None

    report = api_client.get(f"/papers/{paper_id}/report")
    assert report.status_code == 200
    assert report.json()["plan"]["plan_id"] == plan_id
    assert report.json()["comparison"]["status"] == "BLOCKED"
    pipeline = api_client.get("/pipeline/status")
    assert pipeline.status_code == 200
    assert pipeline.json()["papers"] == 1


def test_local_ingest_and_request_validation(api_client, tmp_path) -> None:
    pdf_path = tmp_path / "local-paper.pdf"
    pdf_path.write_bytes(make_pdf())
    ingested = api_client.post(
        "/papers/ingest",
        json={"source_type": "local", "source_ref": str(pdf_path)},
    )
    assert ingested.status_code == 201, ingested.text
    invalid_url = api_client.post(
        "/papers/ingest",
        json={"source_type": "url", "source_ref": "http://example.invalid/paper.pdf"},
    )
    assert invalid_url.status_code == 422
    invalid_body = api_client.post(
        "/papers/ingest",
        json={"source_type": "unknown", "source_ref": "paper"},
    )
    assert invalid_body.status_code == 422
    invalid_upload = api_client.post(
        "/papers/upload",
        files={"file": ("not-a-paper.pdf", b"not a PDF", "application/pdf")},
    )
    assert invalid_upload.status_code == 422
    missing_paper = api_client.get("/papers/99999")
    assert missing_paper.status_code == 404


def test_experiment_is_queued_executed_and_logs_are_queryable(api_client) -> None:
    started = api_client.post(
        "/experiments",
        json={
            "epochs": 1,
            "batch_size": 64,
            "device": "cpu",
            "timeout_seconds": 0.001,
        },
    )
    assert started.status_code == 202, started.text
    experiment_id = started.json()["id"]
    assert started.json()["status"] == "QUEUED"

    status = api_client.get(f"/experiments/{experiment_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "FAILED"
    assert "timeout" in status.json()["error"].lower()
    logs = api_client.get(f"/experiments/{experiment_id}/logs")
    assert logs.status_code == 200
    assert logs.json()["experiment_id"] == experiment_id
    assert isinstance(logs.json()["logs"], list)
    assert api_client.get(f"/experiments/{experiment_id}/report").status_code == 200


def test_openapi_and_cors_contract(api_client) -> None:
    openapi = api_client.get("/openapi.json")
    assert openapi.status_code == 200
    paths = openapi.json()["paths"]
    assert {
        "/papers",
        "/papers/ingest",
        "/papers/upload",
        "/papers/{paper_id}/analyze",
        "/papers/{paper_id}/claims",
        "/papers/{paper_id}/plan",
        "/experiments",
        "/experiments/{experiment_id}/logs",
        "/papers/{paper_id}/report",
        "/pipeline/status",
    } <= set(paths)
    cors = api_client.options(
        "/papers",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert cors.status_code == 200
    assert cors.headers["access-control-allow-origin"] == "http://localhost:3000"
