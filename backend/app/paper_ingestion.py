"""Resolve, validate, parse, and persist research-paper PDFs."""

import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import fitz
from sqlalchemy.orm import sessionmaker

from app.models import Paper, PaperSection


BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]
PAPERS_DIRECTORY = BACKEND_DIRECTORY / "data" / "papers"
MAX_PDF_BYTES = 50 * 1024 * 1024
ROMAN_HEADING = re.compile(r"^[IVXLCDM]+\.$")
APPENDIX_HEADING = re.compile(r"^APPENDIX\s+[A-Z0-9]+$", re.IGNORECASE)
AUTHOR_SEPARATOR = re.compile(r",\s*(?:and\s+)?")
SINGLE_WORD_HEADINGS = {
    "ABSTRACT",
    "BACKGROUND",
    "CONCLUSION",
    "CONCLUSIONS",
    "DISCUSSION",
    "EXPERIMENTS",
    "INTRODUCTION",
    "METHOD",
    "METHODS",
    "NOMENCLATURE",
    "RESULTS",
}


def _is_heading(line: str) -> bool:
    line = " ".join(line.split())
    if not 4 <= len(line) <= 120 or line != line.upper():
        return False
    if len(line.split()) > 16 or "," in line or ";" in line:
        return False
    if not any(character.isalpha() for character in line):
        return False
    if len(line.split()) == 1 and line not in SINGLE_WORD_HEADINGS:
        return False
    letters = [character for character in line if character.isalpha()]
    return bool(letters) and sum(character.isupper() for character in letters) / len(letters) > 0.9


def _extract_sections(pages: list[str]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_heading = "Front matter"
    current_start = 1
    current_lines: list[str] = []

    def flush(end_page: int) -> None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(
                {
                    "heading": current_heading,
                    "content": content,
                    "page_start": current_start,
                    "page_end": end_page,
                }
            )

    for page_number, page_text in enumerate(pages, start=1):
        lines = page_text.splitlines()
        index = 0
        while index < len(lines):
            line = " ".join(lines[index].split())
            heading: str | None = None
            consumed = 1
            if ROMAN_HEADING.fullmatch(line):
                continuation: list[str] = []
                for candidate in lines[index + 1 : index + 4]:
                    candidate = " ".join(candidate.split())
                    if not candidate or not _is_heading(candidate):
                        break
                    continuation.append(candidate)
                if continuation:
                    heading = f"{line} {' '.join(continuation)}"
                    consumed += len(continuation)
            elif APPENDIX_HEADING.fullmatch(line) or (
                line in {"NOMENCLATURE", "ACKNOWLEDGMENT", "ACKNOWLEDGMENTS", "REFERENCES"}
            ):
                heading = line
            elif _is_heading(line):
                heading = line

            if heading:
                flush(page_number)
                current_heading = heading
                current_start = page_number
                current_lines = []
                index += consumed
            else:
                current_lines.append(lines[index])
                index += 1
    flush(len(pages))
    if not sections:
        text = "\n".join(pages).strip()
        if text:
            sections.append(
                {
                    "heading": "Full text",
                    "content": text,
                    "page_start": 1,
                    "page_end": len(pages),
                }
            )
    return sections


def _read_pdf(pdf_path: Path) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    if pdf_path.stat().st_size > MAX_PDF_BYTES:
        raise ValueError(f"PDF exceeds the {MAX_PDF_BYTES}-byte size limit")
    with pdf_path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError("The selected file is not a valid PDF")

    with fitz.open(pdf_path) as document:
        if document.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported")
        pages = [page.get_text("text") for page in document]
        if not pages or not any(text.strip() for text in pages):
            raise ValueError("PDF contains no extractable text")
        metadata = dict(document.metadata)
        metadata["page_count"] = len(document)

    sections = _extract_sections(pages)
    metadata["text_char_count"] = sum(len(text) for text in pages)
    return metadata, pages, sections


def _authors_from_metadata(author: str | None) -> list[str]:
    if not author:
        return []
    return [name.strip() for name in AUTHOR_SEPARATOR.split(author) if name.strip()]


def ingest_pdf(
    session_factory: sessionmaker,
    pdf_path: str | Path,
    *,
    source: str = "local",
    source_ref: str | None = None,
) -> int:
    path = Path(pdf_path).expanduser().resolve()
    metadata, pages, sections = _read_pdf(path)
    title = str(metadata.get("title") or "").strip()
    if not title:
        title = next(
            (line.strip() for line in pages[0].splitlines() if len(line.strip()) > 8),
            "Untitled paper",
        )
    abstract_section = next(
        (section["content"] for section in sections if "ABSTRACT" in section["heading"].upper()),
        None,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stored_metadata = {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    stored_metadata.update({"pdf_path": str(path), "sha256": digest})

    with session_factory() as session:
        paper = Paper(
            title=title,
            source=source,
            source_ref=source_ref or str(path),
            abstract=abstract_section,
            authors_json=json.dumps(_authors_from_metadata(metadata.get("author"))),
            metadata_json=json.dumps(stored_metadata, ensure_ascii=False),
            status="INGESTED",
        )
        paper.sections = [
            PaperSection(
                heading=section["heading"],
                content=section["content"],
                page_start=section["page_start"],
                page_end=section["page_end"],
                section_order=order,
            )
            for order, section in enumerate(sections)
        ]
        session.add(paper)
        session.commit()
        return paper.id


def _download_pdf(url: str, destination: Path) -> Path:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("PDF downloads must use an HTTPS URL")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PAPER2PRODUCT/1.0 (research paper ingestion)"},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as out:
            content_type = response.headers.get_content_type()
            prefix = response.read(5)
            if prefix != b"%PDF-":
                raise ValueError(f"URL did not return a PDF (content type: {content_type})")
            out.write(prefix)
            total = len(prefix)
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_PDF_BYTES:
                    raise ValueError(f"PDF exceeds the {MAX_PDF_BYTES}-byte size limit")
                out.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def ingest_source(
    session_factory: sessionmaker,
    source_ref: str,
    *,
    source_type: str,
) -> int:
    """Ingest a local PDF, HTTPS URL, or arXiv identifier."""
    if source_type == "local":
        return ingest_pdf(session_factory, source_ref, source="local", source_ref=source_ref)
    if source_type == "arxiv":
        arxiv_id = source_ref.removeprefix("arXiv:").strip()
        if not re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?", arxiv_id):
            raise ValueError("Invalid arXiv identifier")
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        basename = re.sub(r"[^A-Za-z0-9.-]+", "_", arxiv_id)
        destination = PAPERS_DIRECTORY / f"arxiv-{basename}.pdf"
        path = _download_pdf(url, destination)
        return ingest_pdf(session_factory, path, source="arxiv", source_ref=arxiv_id)
    if source_type == "url":
        digest = hashlib.sha256(source_ref.encode("utf-8")).hexdigest()[:16]
        destination = PAPERS_DIRECTORY / f"url-{digest}.pdf"
        path = _download_pdf(source_ref, destination)
        return ingest_pdf(session_factory, path, source="url", source_ref=source_ref)
    raise ValueError("source_type must be one of: local, url, arxiv")
