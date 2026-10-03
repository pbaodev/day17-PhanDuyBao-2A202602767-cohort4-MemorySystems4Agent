from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider

# Environment variable holding the API key for each provider.
_API_KEY_ENV = {
    "openai": ("OPENAI_API_KEY",),
    "custom": ("CUSTOM_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "ollama": (),
    "openrouter": ("OPENROUTER_API_KEY",),
}

_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - paths: repo root, dataset directory, state directory (`state/profiles/<user>/User.md`)
    - compact memory: token threshold that triggers compaction + number of recent messages kept
    - providers: main chat model and the (optional) judge model
    - use_live: opt-in switch for real LLM calls; offline mode is the deterministic default
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    use_live: bool = False


def _load_dotenv(root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional for offline runs
        return
    load_dotenv(root / ".env", override=False)


def _provider_from_env(prefix: str, default_provider: str = "openai") -> ProviderConfig:
    """Build a ProviderConfig from `<prefix>_PROVIDER` / `<prefix>_MODEL` style variables."""

    provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", default_provider))
    model_name = os.getenv(f"{prefix}_MODEL", _DEFAULT_MODEL[provider])
    temperature = float(os.getenv(f"{prefix}_TEMPERATURE", "0"))

    api_key = None
    for env_name in _API_KEY_ENV[provider]:
        api_key = os.getenv(env_name) or api_key

    base_url = None
    if provider == "custom":
        base_url = os.getenv("CUSTOM_BASE_URL")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "openai":
        base_url = os.getenv("OPENAI_BASE_URL")

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env` + environment variables and return a complete LabConfig."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root)

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_from_env("LLM")
    # The judge defaults to the same provider/model unless JUDGE_* overrides it.
    judge_default = model.provider
    judge_model = _provider_from_env("JUDGE", default_provider=judge_default)
    if "JUDGE_MODEL" not in os.environ and judge_model.provider == model.provider:
        judge_model.model_name = model.model_name

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1000")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
        model=model,
        judge_model=judge_model,
        use_live=os.getenv("LAB_MODE", "offline").strip().lower() == "live",
    )
