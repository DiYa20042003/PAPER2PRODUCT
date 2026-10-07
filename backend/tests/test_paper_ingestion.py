import json
from pathlib import Path

import fitz
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.models import Base, Paper, PaperSection
from app.paper_ingestion import ingest_pdf, ingest_source


def make_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "A Reproduction Paper\n"
        "A. Researcher\n"
        "ABSTRACT\n"
        "We evaluate a documented method.\n"
        "I.\n"
        "INTRODUCTION\n"
        "The paper describes a measurable result.\n"
        "REFERENCES\n"
        "A. Author, A publication, 2020.",
    )
    document.set_metadata({"title": "A Reproduction Paper", "author": "A. Researcher"})
    document.save(path)
    document.close()


def make_session_factory() -> sessionmaker:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_pdf_ingestion_persists_metadata_and_page_sections(tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    make_pdf(pdf_path)
    session_factory = make_session_factory()

    paper_id = ingest_pdf(session_factory, pdf_path, source="upload")

    with session_factory() as session:
        paper = session.get(Paper, paper_id)
        sections = list(
            session.scalars(
                select(PaperSection)
                .where(PaperSection.paper_id == paper_id)
                .order_by(PaperSection.section_order)
            )
        )
        assert paper is not None
        assert paper.title == "A Reproduction Paper"
        assert paper.source == "upload"
        assert paper.abstract is not None
        assert "documented method" in paper.abstract
        assert json.loads(paper.authors_json) == ["A. Researcher"]
        metadata = json.loads(paper.metadata_json)
        assert metadata["page_count"] == 1
        assert metadata["text_char_count"] > 0
        assert len(metadata["sha256"]) == 64
        assert [section.heading for section in sections] == [
            "Front matter",
            "ABSTRACT",
            "I. INTRODUCTION",
            "REFERENCES",
        ]
        assert all(section.page_start == 1 for section in sections)


def test_ingest_source_supports_local_paths(tmp_path) -> None:
    pdf_path = tmp_path / "local.pdf"
    make_pdf(pdf_path)
    session_factory = make_session_factory()

    paper_id = ingest_source(session_factory, str(pdf_path), source_type="local")

    with session_factory() as session:
        paper = session.get(Paper, paper_id)
        assert paper is not None
        assert paper.source == "local"
        assert paper.source_ref == str(pdf_path)


@pytest.mark.parametrize(
    ("source_ref", "source_type", "message"),
    [
        ("../../paper.pdf", "arxiv", "Invalid arXiv identifier"),
        ("http://example.com/paper.pdf", "url", "HTTPS URL"),
        ("paper.pdf", "unknown", "source_type"),
    ],
)
def test_ingestion_rejects_invalid_sources(source_ref, source_type, message) -> None:
    with pytest.raises(ValueError, match=message):
        ingest_source(make_session_factory(), source_ref, source_type=source_type)


def test_non_pdf_input_is_rejected(tmp_path) -> None:
    invalid = tmp_path / "not-a-pdf.pdf"
    invalid.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid PDF"):
        ingest_pdf(make_session_factory(), invalid)
