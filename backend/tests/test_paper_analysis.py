import json

import fitz
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.models import Base, Claim, Paper
from app.llm_provider import JsonProvider, _inline_schema, configured_provider
from app.paper_analysis import (
    ClaimExtraction,
    ExtractedFact,
    PaperUnderstanding,
    ProviderFacts,
    analyze_paper,
    extract_claims,
    understand_paper,
)
from app.paper_ingestion import ingest_pdf


def make_session_factory() -> sessionmaker:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def make_paper(session_factory: sessionmaker, tmp_path) -> int:
    pdf_path = tmp_path / "analysis-paper.pdf"
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
        "METHODS\n"
        "Layer C1 is a convolutional layer with six feature maps.",
    )
    document.set_metadata(
        {
            "title": "Gradient-based learning applied to document recognition",
            "author": "Yann Lecun, Léon Bottou, Yoshua Bengio, Patrick Haffner",
        }
    )
    document.save(pdf_path)
    document.close()
    return ingest_pdf(session_factory, pdf_path)


class ValidatingFakeProvider:
    provider_name = "gemini"
    model = "test-structured-model"

    def __init__(self, valid_section_id: int) -> None:
        self.calls = 0
        self.valid_section_id = valid_section_id

    def generate_structured(self, prompt, response_model):
        self.calls += 1
        if self.calls == 1:
            raise ValueError("malformed provider JSON")
        return ProviderFacts(
            facts=[
                ExtractedFact(
                    kind="dataset",
                    statement="The paper uses MNIST.",
                    source_section_id=self.valid_section_id,
                    evidence_quote=(
                        "The resulting database was called the modified NIST, or MNIST, dataset."
                    ),
                    confidence=0.9,
                ),
                ExtractedFact(
                    kind="result",
                    statement="The paper reports a separate validation set.",
                    source_section_id=self.valid_section_id,
                    evidence_quote="The paper reports a separate validation set.",
                    confidence=0.95,
                ),
            ]
        )


class FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_rule_fallback_is_labeled_and_extracts_verbatim_facts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PAPER2PRODUCT_LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    session_factory = make_session_factory()
    paper_id = make_paper(session_factory, tmp_path)

    result = analyze_paper(session_factory, paper_id)

    assert result["extraction_method"] == "rule_based"
    assert result["confidence_level"] == "lower"
    assert result["confidence_score"] == 0.35
    assert any(fact["kind"] == "dataset" for fact in result["facts"])
    assert any(fact["kind"] == "metric" for fact in result["facts"])
    assert all(
        fact["confidence"] == 0.35
        and fact["evidence_quote"] in fact["statement"]
        for fact in result["facts"]
    )
    with session_factory() as session:
        paper = session.get(Paper, paper_id)
        claims = list(session.scalars(select(Claim).where(Claim.paper_id == paper_id)))
        assert paper is not None
        persisted = json.loads(paper.understanding_json)
        assert persisted["extraction_method"] == "rule_based"
        assert len(claims) == len(result["facts"])
        assert all(claim.supported for claim in claims)
    understanding = understand_paper(session_factory, paper_id)
    extracted_claims = extract_claims(session_factory, paper_id)
    assert isinstance(understanding, PaperUnderstanding)
    assert isinstance(extracted_claims, ClaimExtraction)
    assert "MNIST" in understanding.abstract
    assert extracted_claims.claims


def test_provider_retries_and_unsupported_facts_are_flagged(tmp_path) -> None:
    session_factory = make_session_factory()
    paper_id = make_paper(session_factory, tmp_path)
    provider = ValidatingFakeProvider(valid_section_id=2)

    result = analyze_paper(session_factory, paper_id, provider=provider)

    assert provider.calls == 2
    assert result["extraction_method"] == "gemini"
    assert result["confidence_level"] == "model-generated"
    assert len(result["facts"]) == 1
    assert result["facts"][0]["evidence_quote"] == (
        "The resulting database was called the modified NIST, or MNIST, dataset."
    )
    assert len(result["unsupported_facts"]) == 1
    assert "not an exact excerpt" in result["unsupported_facts"][0]["reason"]
    with session_factory() as session:
        claims = list(session.scalars(select(Claim).where(Claim.paper_id == paper_id)))
        assert len(claims) == 1
        assert claims[0].statement == "The paper uses MNIST."
        assert claims[0].supported is True


def test_missing_paper_and_empty_sections_are_errors(tmp_path) -> None:
    session_factory = make_session_factory()
    with pytest.raises(LookupError, match="was not found"):
        analyze_paper(session_factory, 999)

    paper_id = make_paper(session_factory, tmp_path)
    with session_factory() as session:
        paper = session.get(Paper, paper_id)
        assert paper is not None
        for section in paper.sections:
            session.delete(section)
        session.commit()
    with pytest.raises(ValueError, match="has no parsed sections"):
        analyze_paper(session_factory, paper_id)


@pytest.mark.parametrize(
    ("provider_name", "key_name", "model_name"),
    [
        ("gemini", "GEMINI_API_KEY", "GEMINI_MODEL"),
        ("openai", "OPENAI_API_KEY", "OPENAI_MODEL"),
        ("anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"),
    ],
)
def test_provider_can_be_selected_from_configuration(
    provider_name, key_name, model_name, monkeypatch
) -> None:
    monkeypatch.setenv("PAPER2PRODUCT_LLM_PROVIDER", provider_name)
    monkeypatch.setenv(key_name, "test-api-key")
    monkeypatch.setenv(model_name, "configured-model")

    provider = configured_provider()

    assert provider is not None
    assert provider.provider_name == provider_name
    assert provider.model == "configured-model"


@pytest.mark.parametrize(
    ("provider_name", "response"),
    [
        (
            "gemini",
            {"candidates": [{"content": {"parts": [{"text": "{\"facts\": []}"}]}}]},
        ),
        (
            "openai",
            {"choices": [{"message": {"content": "{\"facts\": []}"}}]},
        ),
        (
            "anthropic",
            {"content": [{"type": "tool_use", "input": {"facts": []}}]},
        ),
    ],
)
def test_provider_adapters_request_and_parse_json(
    provider_name, response, monkeypatch
) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeHttpResponse(response)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = JsonProvider(provider_name, "test-model", "test-api-key")

    parsed = provider.generate_structured("Use exact source evidence.", ProviderFacts)

    assert parsed.facts == []
    assert captured["timeout"] == 90
    request = captured["request"]
    assert request.get_method() == "POST"
    body = json.loads(request.data)
    if provider_name == "gemini":
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        schema = body["generationConfig"]["responseSchema"]
        assert "$ref" not in json.dumps(schema)
        assert "key=test-api-key" in request.full_url
    elif provider_name == "openai":
        assert body["response_format"]["type"] == "json_schema"
        assert request.get_header("Authorization") == "Bearer test-api-key"
    else:
        assert body["tool_choice"]["type"] == "tool"
        assert request.get_header("X-api-key") == "test-api-key"


def test_provider_json_schema_references_are_inlined() -> None:
    schema = _inline_schema(ProviderFacts.model_json_schema())
    assert "$defs" not in schema
    assert "$ref" not in json.dumps(schema)
