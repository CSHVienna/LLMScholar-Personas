"""
Source adapters: one per API/back-end (GPT batch API, Gemini batch API, Ollama).

Each adapter only *extracts* fields from a raw response record into a common
:class:`RawResponse`. Decoding the text and assigning the taxonomy tags are
done elsewhere (``content_parser`` and ``taxonomy``), identically for all sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from utils import constants as cons
from responses.text_signals import contains_refusal


@dataclass(frozen=True)
class RawResponse:
    """Source-independent view of one raw LLM response."""
    text: Optional[str] = None                 # generated text (None if there is none)
    created_at: Optional[str] = None           # ISO-8601 UTC timestamp, if the source provides one
    finish_reason: Optional[str] = None        # e.g. stop, length, MAX_TOKENS
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    duration_s: Optional[float] = None         # total generation time in seconds (Ollama only)
    api_error: Optional[str] = None            # transport / API error message
    api_refusal: Optional[str] = None          # refusal signalled by the API itself


class SourceAdapter(ABC):
    """Base class: converts a raw record of one source into a :class:`RawResponse`."""

    name: str = ''

    def extract(self, record: dict) -> RawResponse:
        """Extract a :class:`RawResponse`; never raises on malformed records."""
        try:
            return self._extract(record or {})
        except Exception as error:  # malformed record: keep going, but keep the evidence
            return RawResponse(api_error=f"malformed record: {type(error).__name__}: {error}")

    @abstractmethod
    def _extract(self, record: dict) -> RawResponse:
        """Source-specific extraction (may raise on unexpected structures)."""


class GptAdapter(SourceAdapter):
    """OpenAI batch API: ``{response: {status_code, body: {created, choices, usage}}, error}``."""

    name = cons.SOURCE_GPT

    def _extract(self, record: dict) -> RawResponse:
        # 1. Transport-level error
        if record.get('error'):
            return RawResponse(api_error=str(record['error']))

        response = record.get('response') or {}
        if response.get('status_code', 200) != 200:
            return RawResponse(api_error=f"status_code={response.get('status_code')}")

        # 2. Message and metadata
        body = response.get('body') or {}
        choice = (body.get('choices') or [{}])[0]
        message = choice.get('message') or {}
        usage = body.get('usage') or {}

        # 3. The `refusal` field sometimes holds the actual answer instead of a refusal:
        #    it only counts as a refusal if the text reads like one.
        text, api_refusal = message.get('content'), None
        refusal = message.get('refusal')
        if refusal:
            if contains_refusal(refusal):
                api_refusal = refusal
            else:
                text = text or refusal

        created = body.get('created')
        return RawResponse(text=text,
                           created_at=_epoch_to_iso(created) if created else None,
                           finish_reason=choice.get('finish_reason'),
                           prompt_tokens=usage.get('prompt_tokens'),
                           completion_tokens=usage.get('completion_tokens'),
                           reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                           api_refusal=api_refusal)


class GeminiAdapter(SourceAdapter):
    """Vertex AI batch API: ``{response: {candidates, usageMetadata, promptFeedback}, key}``."""

    name = cons.SOURCE_GEMINI

    def _extract(self, record: dict) -> RawResponse:
        # 1. Transport-level error
        if 'response' not in record:
            return RawResponse(api_error=str(record.get('error', 'missing response')))

        response = record['response'] or {}
        usage = response.get('usageMetadata') or {}
        tokens = dict(prompt_tokens=usage.get('promptTokenCount'),
                      completion_tokens=usage.get('candidatesTokenCount'),
                      reasoning_tokens=usage.get('thoughtsTokenCount'),
                      created_at=response.get('createTime'))

        # 2. Prompt blocked before generation
        block_reason = (response.get('promptFeedback') or {}).get('blockReason')
        if block_reason:
            return RawResponse(api_refusal=f"blockReason={block_reason}", **tokens)

        # 3. Generated text (thought parts are excluded)
        candidate = (response.get('candidates') or [{}])[0]
        finish_reason = candidate.get('finishReason')
        parts = (candidate.get('content') or {}).get('parts') or []
        text = ''.join(part.get('text', '') for part in parts if not part.get('thought'))

        api_refusal = f"finishReason={finish_reason}" if finish_reason in cons.GEMINI_REFUSAL_FINISH_REASONS else None
        return RawResponse(text=text or None, finish_reason=finish_reason, api_refusal=api_refusal, **tokens)


class OllamaAdapter(SourceAdapter):
    """Ollama chat API: ``{created_at, done_reason, message: {content, thinking}, eval_count, ...}``."""

    name = cons.SOURCE_OLLAMA

    def _extract(self, record: dict) -> RawResponse:
        if record.get('error'):
            return RawResponse(api_error=str(record['error']))

        message = record.get('message') or {}
        total_duration_ns = record.get('total_duration')
        return RawResponse(text=message.get('content'),
                           created_at=record.get('created_at'),
                           finish_reason=record.get('done_reason'),
                           prompt_tokens=record.get('prompt_eval_count'),
                           completion_tokens=record.get('eval_count'),
                           duration_s=total_duration_ns / 1e9 if total_duration_ns else None)


# -----------------------------------------------------------------------------
# Registry and helpers
# -----------------------------------------------------------------------------
ADAPTERS = {adapter.name: adapter for adapter in (GptAdapter(), GeminiAdapter(), OllamaAdapter())}


def get_adapter(source: str) -> SourceAdapter:
    """Return the adapter registered for ``source``."""
    return ADAPTERS[source]


def _epoch_to_iso(seconds: int) -> str:
    """Unix epoch (seconds) -> ISO-8601 UTC string."""
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
