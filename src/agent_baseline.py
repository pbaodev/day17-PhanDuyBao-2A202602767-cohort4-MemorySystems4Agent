from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    SYSTEM_PROMPT,
    answer_from_facts,
    estimate_tokens,
    extract_profile_updates,
    merge_facts,
)
from model_provider import build_chat_model, usage_from_result

ACK = "Mình đã ghi nhận."


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session (short-term) memory only.

    - Keeps the full message list per `thread_id` and resends all of it every turn.
    - No `User.md`: a new thread starts from zero, so long-term facts are forgotten.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route to the live agent when available, otherwise the deterministic offline path."""

        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).prompt_tokens_processed

    def memory_file_size(self, user_id: str) -> int:
        return 0  # no persistent memory file

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})

        # The whole thread (and nothing else) is the prompt: cost grows with every turn.
        prompt_tokens = estimate_tokens(SYSTEM_PROMPT) + sum(estimate_tokens(m["content"]) for m in session.messages)

        # Short-term memory: facts are only derivable from *this* thread's user messages.
        thread_facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                thread_facts = merge_facts(thread_facts, extract_profile_updates(item["content"]))
        response = answer_from_facts(message, thread_facts) or ACK

        session.messages.append({"role": "assistant", "content": response})
        agent_tokens = estimate_tokens(response)
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {"response": response, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens, "mode": "offline"}

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        response, completion, prompt = usage_from_result(result)
        session.messages += [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
        agent_tokens = completion or estimate_tokens(response)
        prompt_tokens = prompt or estimate_tokens(SYSTEM_PROMPT) + sum(estimate_tokens(m["content"]) for m in session.messages)
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {"response": response, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens, "mode": "live"}

    def _maybe_build_langchain_agent(self):
        """Build a live LangChain agent (`create_agent` + `InMemorySaver`), or return None.

        Live mode is opt-in (`LAB_MODE=live`) and needs the provider SDK + credentials;
        any problem falls back to the deterministic offline path.
        """

        model_config = self.config.model
        needs_key = model_config.provider not in ("ollama",)
        if not self.config.use_live or (needs_key and not model_config.api_key):
            return None
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver

            return create_agent(
                build_chat_model(model_config),
                tools=[],
                system_prompt=SYSTEM_PROMPT,
                checkpointer=InMemorySaver(),
            )
        except Exception as error:  # missing SDK, bad credentials/base_url, API drift, ...
            warnings.warn(f"Live agent unavailable, falling back to offline mode: {error!r}", stacklevel=2)
            return None
