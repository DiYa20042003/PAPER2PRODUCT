"""JSON-mode provider adapters for paper understanding."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel


BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def _read_dotenv_value(name: str) -> str | None:
    env_path = BACKEND_DIRECTORY / ".env"
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            return value.strip().strip("\"'")
    return None


def setting(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or _read_dotenv_value(name) or default


def _inline_schema(schema: dict) -> dict:
    definitions = schema.get("$defs", {})

    def expand(value):
        if isinstance(value, list):
            return [expand(item) for item in value]
        if not isinstance(value, dict):
            return value
        reference = value.get("$ref")
        if reference:
            definition_name = reference.rsplit("/", 1)[-1]
            return expand(definitions[definition_name])
        return {
            key: expand(item)
            for key, item in value.items()
            if key not in {"$defs", "title"}
        }

    return expand(schema)


class JsonProvider:
    """Small stdlib-only adapter for Gemini, OpenAI, or Anthropic JSON output."""

    def __init__(self, provider_name: str, model: str, api_key: str) -> None:
        self.provider_name = provider_name
        self.model = model
        self.api_key = api_key

    def _request(self, url: str, payload: dict, headers: dict[str, str]) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"{self.provider_name} request failed with HTTP {error.code}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"{self.provider_name} request failed: {error.reason}") from error

    def generate_structured(
        self,
        prompt: str,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        schema = response_model.model_json_schema()
        if self.provider_name == "gemini":
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{urllib.parse.quote(self.model, safe='')}:generateContent?"
                f"key={urllib.parse.quote(self.api_key, safe='')}"
            )
            response = self._request(
                url,
                {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": _inline_schema(schema),
                    },
                },
                {},
            )
            try:
                text = response["candidates"][0]["content"]["parts"][0]["text"]
                payload = json.loads(text)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                raise ValueError("Gemini returned malformed structured output") from error
        elif self.provider_name == "openai":
            url = setting("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            response = self._request(
                f"{url}/chat/completions",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_model.__name__.lower(),
                            "strict": True,
                            "schema": schema,
                        },
                    },
                },
                {"Authorization": f"Bearer {self.api_key}"},
            )
            try:
                payload = json.loads(response["choices"][0]["message"]["content"])
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                raise ValueError("OpenAI returned malformed structured output") from error
        elif self.provider_name == "anthropic":
            response = self._request(
                "https://api.anthropic.com/v1/messages",
                {
                    "model": self.model,
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                    "tools": [
                        {
                            "name": "emit_structured_paper_facts",
                            "description": "Return evidence-backed paper facts.",
                            "input_schema": schema,
                        }
                    ],
                    "tool_choice": {
                        "type": "tool",
                        "name": "emit_structured_paper_facts",
                    },
                },
                {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            try:
                block = next(
                    item
                    for item in response["content"]
                    if item.get("type") == "tool_use"
                )
                payload = block["input"]
            except (KeyError, StopIteration, TypeError) as error:
                raise ValueError("Anthropic returned malformed structured output") from error
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider_name}")
        return response_model.model_validate(payload)


def configured_provider() -> JsonProvider | None:
    provider_name = (setting("PAPER2PRODUCT_LLM_PROVIDER", "gemini") or "gemini").lower()
    if provider_name == "gemini":
        api_key = setting("GEMINI_API_KEY")
        model = setting("GEMINI_MODEL", "gemini-2.5-flash")
    elif provider_name == "openai":
        api_key = setting("OPENAI_API_KEY")
        model = setting("OPENAI_MODEL", "gpt-4o-mini")
    elif provider_name == "anthropic":
        api_key = setting("ANTHROPIC_API_KEY")
        model = setting("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
    else:
        raise ValueError(
            "PAPER2PRODUCT_LLM_PROVIDER must be gemini, openai, or anthropic"
        )
    if not api_key:
        return None
    return JsonProvider(provider_name, model or "", api_key)
