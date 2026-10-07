"""Evidence-validated understanding and claim extraction."""

import json
import re
import unicodedata
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import delete, select
from sqlalchemy.orm import sessionmaker

from app.llm_provider import configured_provider
from app.models import Claim, Paper, PaperSection


FactKind = Literal["dataset", "metric", "method", "result", "limitation", "table", "other"]


class ExtractedFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: FactKind
    statement: str = Field(min_length=4)
    source_section_id: int = Field(gt=0)
    evidence_quote: str = Field(min_length=8)
    confidence: float = Field(ge=0, le=1)


class ProviderFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[ExtractedFact]


class PaperUnderstanding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: int
    title: str
    authors: list[str]
    abstract: str
    extraction_method: str
    model: str | None
    confidence_level: Literal["lower", "model-generated"]
    confidence_score: float = Field(ge=0, le=1)
    facts: list[ExtractedFact]
    unsupported_facts: list[dict[str, Any]]
    metadata: dict[str, Any]


class ClaimExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: int
    extraction_method: str
    confidence_level: Literal["lower", "model-generated"]
    claims: list[ExtractedFact]
    unsupported_facts: list[dict[str, Any]]


class StructuredProvider(Protocol):
    provider_name: str
    model: str

    def generate_structured(
        self, prompt: str, response_model: type[ProviderFacts]
    ) -> ProviderFacts: ...


def _normalize_quote(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _sentences(text: str) -> list[str]:
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    paragraphs: list[str] = []
    current = ""
    for line in lines:
        if current and re.search(r"[.!?][\"')\]]?$", current):
            paragraphs.append(current)
            current = line
        else:
            current = f"{current} {line}".strip()
    if current:
        paragraphs.append(current)
    fragments = [
        fragment.strip()
        for paragraph in paragraphs
        for fragment in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph)
    ]
    return [
        fragment
        for fragment in fragments
        if 25 <= len(fragment) <= 500
        and any(character.islower() for character in fragment)
    ]


def _rule_based_facts(sections: list[PaperSection]) -> list[ExtractedFact]:
    patterns: tuple[tuple[FactKind, re.Pattern[str]], ...] = (
        (
            "dataset",
            re.compile(
                r"\b(?:MNIST|NIST|CIFAR(?:-\d+)?|ImageNet|dataset|data set|database|corpus|benchmark|training set|test set)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "metric",
            re.compile(
                r"(?:\b\d+(?:\.\d+)?\s?%|\b(?:error rate|accuracy|precision|recall|recognition rate|classification rate)\b)",
                re.IGNORECASE,
            ),
        ),
        (
            "table",
            re.compile(r"\b(?:table|figure|fig\.)\s*(?:[IVXLCDM]+|\d+)\b", re.IGNORECASE),
        ),
        (
            "method",
            re.compile(
                r"\b(?:LeNet|convolutional|gradient|network architecture|trained with|training procedure|backpropagation)\b",
                re.IGNORECASE,
            ),
        ),
    )
    found: set[tuple[str, int, str]] = set()
    facts: list[ExtractedFact] = []
    count_by_kind: dict[str, int] = {}
    for section in sections:
        for sentence in _sentences(section.content):
            for kind, pattern in patterns:
                key = (kind, section.id, sentence)
                if (
                    pattern.search(sentence)
                    and key not in found
                    and count_by_kind.get(kind, 0) < 40
                ):
                    found.add(key)
                    count_by_kind[kind] = count_by_kind.get(kind, 0) + 1
                    facts.append(
                        ExtractedFact(
                            kind=kind,
                            statement=sentence,
                            source_section_id=section.id,
                            evidence_quote=sentence,
                            confidence=0.35,
                        )
                    )
                    if len(facts) >= 120:
                        return facts
    return facts


def _provider_prompt(title: str, sections: list[PaperSection], retry_note: str = "") -> str:
    paper_text = "\n\n".join(
        f"[section_id={section.id}; heading={section.heading}]\n{section.content}"
        for section in sections
    )
    return (
        "Extract only facts explicitly stated in the supplied research paper. "
        "Return JSON matching the required schema. Every fact must include its "
        "source_section_id and an exact, contiguous evidence_quote copied from "
        "that section. Do not infer missing values or add outside knowledge. "
        "Classify facts as dataset, metric, method, result, limitation, table, or other. "
        "A fact without a verbatim quote is not allowed. "
        f"{retry_note}\nPaper title: {title}\n\nPaper sections:\n{paper_text}"
    )


def _extract_provider_facts(
    provider: StructuredProvider,
    title: str,
    sections: list[PaperSection],
) -> ProviderFacts:
    prompt = _provider_prompt(title, sections)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return provider.generate_structured(prompt, ProviderFacts)
        except (ValueError, ValidationError) as error:
            last_error = error
            prompt = _provider_prompt(
                title,
                sections,
                retry_note=(
                    "The previous output was invalid. Return a corrected JSON object "
                    "with schema-valid fields and evidence copied verbatim."
                ),
            )
    assert last_error is not None
    raise ValueError(f"Structured extraction failed after 3 attempts: {last_error}") from last_error


def _validate_facts(
    facts: list[ExtractedFact],
    sections: list[PaperSection],
) -> tuple[list[ExtractedFact], list[dict[str, Any]]]:
    content_by_id = {section.id: section.content for section in sections}
    verified: list[ExtractedFact] = []
    unsupported: list[dict[str, Any]] = []
    for fact in facts:
        section_content = content_by_id.get(fact.source_section_id)
        if section_content is None:
            reason = "source section does not exist in this paper"
        elif _normalize_quote(fact.evidence_quote) not in _normalize_quote(section_content):
            reason = "evidence quote is not an exact excerpt from the source section"
        else:
            verified.append(fact)
            continue
        unsupported.append({"fact": fact.model_dump(), "reason": reason})
    return verified, unsupported


def _load_paper(session_factory: sessionmaker, paper_id: int) -> tuple[Paper, list[PaperSection]]:
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
        return paper, sections


def analyze_paper(
    session_factory: sessionmaker,
    paper_id: int,
    *,
    provider: StructuredProvider | None = None,
) -> dict[str, Any]:
    paper, sections = _load_paper(session_factory, paper_id)
    active_provider = provider if provider is not None else configured_provider()
    if active_provider is None:
        candidate_facts = _rule_based_facts(sections)
        extraction_method = "rule_based"
        confidence_score = 0.35
        provider_model = None
    else:
        result = _extract_provider_facts(active_provider, paper.title, sections)
        candidate_facts = result.facts
        extraction_method = active_provider.provider_name
        confidence_score = 0.75
        provider_model = active_provider.model

    verified_facts, unsupported_facts = _validate_facts(candidate_facts, sections)
    facts_payload = [fact.model_dump() for fact in verified_facts]
    title = paper.title
    authors = json.loads(paper.authors_json)
    metadata = json.loads(paper.metadata_json)
    response = {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "abstract": paper.abstract or "Not identified",
        "extraction_method": extraction_method,
        "model": provider_model,
        "confidence_level": "lower" if extraction_method == "rule_based" else "model-generated",
        "confidence_score": confidence_score,
        "facts": facts_payload,
        "unsupported_facts": unsupported_facts,
        "metadata": metadata,
    }
    response = PaperUnderstanding.model_validate(response).model_dump()
    with session_factory() as session:
        stored_paper = session.get(Paper, paper_id)
        if stored_paper is None:
            raise LookupError(f"Paper {paper_id} was not found")
        stored_paper.understanding_json = json.dumps(response, ensure_ascii=False)
        session.execute(delete(Claim).where(Claim.paper_id == paper_id))
        for fact in verified_facts:
            session.add(
                Claim(
                    paper_id=paper_id,
                    section_id=fact.source_section_id,
                    claim_type=fact.kind,
                    statement=fact.statement,
                    evidence=fact.evidence_quote,
                    confidence=fact.confidence,
                    supported=True,
                    extraction_method=extraction_method,
                )
            )
        session.commit()
    return response


def understand_paper(
    session_factory: sessionmaker,
    paper_id: int,
    *,
    provider: StructuredProvider | None = None,
) -> PaperUnderstanding:
    result = analyze_paper(session_factory, paper_id, provider=provider)
    return PaperUnderstanding.model_validate(result)


def extract_claims(
    session_factory: sessionmaker,
    paper_id: int,
    *,
    provider: StructuredProvider | None = None,
) -> ClaimExtraction:
    result = analyze_paper(session_factory, paper_id, provider=provider)
    return ClaimExtraction(
        paper_id=result["paper_id"],
        extraction_method=result["extraction_method"],
        confidence_level=result["confidence_level"],
        claims=result["facts"],
        unsupported_facts=result["unsupported_facts"],
    )
