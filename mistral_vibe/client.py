"""Mistral API client with async streaming support and dynamic system prompts."""

from __future__ import annotations

from typing import AsyncGenerator, Optional, TypedDict

try:
    from mistralai import Mistral

    HAS_MISTRAL = True
except ImportError:
    HAS_MISTRAL = False


class Message(TypedDict):
    role: str
    content: str


AVAILABLE_MODELS = [
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest",
    "codestral-latest",
    "open-mistral-7b",
    "open-mixtral-8x7b",
]

SYSTEM_PROMPT = """You are Mistral Vibe, an expert AI coding assistant.
You help developers write, understand, debug, and improve code.
When showing code, always use fenced code blocks with the language identifier.
Be concise but thorough. Prioritize working, idiomatic code.
"""


class MistralClient:
    """Async Mistral API client with streaming and conversation management."""

    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        self.api_key = api_key
        self.model = model
        self._client: "Mistral | None" = None

    def _get_client(self) -> "Mistral":
        if not HAS_MISTRAL:
            raise RuntimeError(
                "mistralai package not installed. Run: pip install mistralai"
            )
        if not self.api_key:
            raise RuntimeError(
                "No API key provided. Set the MISTRAL_API_KEY environment variable."
            )
        if self._client is None:
            self._client = Mistral(api_key=self.api_key)
        return self._client

    async def stream_chat(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat response token by token.

        system_prompt — if provided, overrides / extends the default system
        prompt (used to inject active project instructions + file context).
        """
        client = self._get_client()

        effective_system = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        all_messages = [{"role": "system", "content": effective_system}] + list(messages)

        stream = await client.chat.stream_async(
            model=self.model,
            messages=all_messages,
        )
        async for event in stream:
            delta = event.data.choices[0].delta.content
            if delta:
                yield delta

    async def count_tokens(self, messages: list[Message]) -> int:
        """Estimate token count (rough approximation)."""
        total = sum(len(m["content"].split()) * 1.3 for m in messages)
        return int(total)
