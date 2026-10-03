from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

_ALIASES = {
    "openai": "openai",
    "oai": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "openai_compatible": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "google_genai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",  # common typo
    "claude": "anthropic",
    "ollama": "ollama",
    "openrouter": "openrouter",
    "open-router": "openrouter",
}


@dataclass
class ProviderConfig:
    """Provider settings shared by the agents and the judge model.

    Supported providers: openai, custom (OpenAI-compatible base URL), gemini,
    anthropic, ollama, openrouter.
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`; raise on unknown providers."""

    key = (value or "").strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"Unsupported provider {value!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}")
    return _ALIASES[key]


def build_chat_model(config: ProviderConfig):
    """Instantiate the real LangChain chat model for the selected provider.

    Imports are lazy so offline mode works without any provider SDK installed.
    """

    provider = normalize_provider(config.provider)

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if provider == "custom":
            if not config.base_url:
                raise ValueError("Provider 'custom' requires a base_url (CUSTOM_BASE_URL).")
            kwargs["base_url"] = config.base_url
        elif config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    # openrouter
    from langchain_openrouter import ChatOpenRouter

    kwargs = {"model": config.model_name, "temperature": config.temperature}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return ChatOpenRouter(**kwargs)


def message_text(content) -> str:
    """Flatten LangChain message content (str or list of content blocks) into plain text."""

    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def usage_from_result(result: dict) -> tuple[str, int, int]:
    """Return (final answer, completion tokens, prompt tokens) of one live agent invocation.

    Token counts come from provider `usage_metadata`; both are 0 if the provider reports none
    (callers then fall back to the heuristic estimator).
    """

    messages = result.get("messages", [])
    answer = message_text(messages[-1].content) if messages else ""
    # With a checkpointer the result carries the whole thread; only count this invocation.
    last_user = max((i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"), default=-1)
    completion = prompt = 0
    for message in messages[last_user + 1 :]:
        usage = getattr(message, "usage_metadata", None) or {}
        completion += int(usage.get("output_tokens", 0))
        prompt += int(usage.get("input_tokens", 0))
    return answer, completion, prompt
