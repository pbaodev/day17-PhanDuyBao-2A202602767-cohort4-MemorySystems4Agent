from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    SYSTEM_PROMPT,
    CompactMemoryManager,
    UserProfileStore,
    answer_from_facts,
    describe_updates,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model, usage_from_result

try:  # Needed at module level: tool/middleware type hints are resolved from module globals.
    from langchain.agents.middleware import ModelRequest
    from langchain.tools import ToolRuntime
except ImportError:  # offline-only install: the live agent is simply unavailable
    ModelRequest = ToolRuntime = Any

ACK = "Mình đã ghi nhận."


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: three memory layers.

    1. short-term: recent messages of the current thread (`CompactMemoryManager.messages`)
    2. persistent: `User.md` per user, shared across threads/sessions
    3. compact: rolling summary of older messages once the thread passes the token threshold

    Prompt per turn = system prompt + `User.md` + compact summary + recent kept messages.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route to the live agent when available, otherwise the deterministic offline path."""

        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # 1-2. extract stable facts and persist them (layer 2: User.md)
        updates = extract_profile_updates(message)
        changed = self.profile_store.upsert_facts(user_id, updates) if updates else {}

        # 3. short-term + compact memory (layers 1 and 3)
        self.compact_memory.append(thread_id, "user", message)

        # 4. what this turn's prompt carries
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        # 5-6. answer from persisted memory, then record the assistant turn
        response = self._offline_response(user_id, thread_id, message, changed)
        self.compact_memory.append(thread_id, "assistant", response)

        # Writing to User.md costs output tokens too (a live agent emits it as a tool call).
        memory_write_tokens = sum(estimate_tokens(f"- {key}: {value}") for key, value in changed.items())
        agent_tokens = estimate_tokens(response) + memory_write_tokens
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_updates": changed,
            "mode": "offline",
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        recent = sum(estimate_tokens(m["content"]) for m in context["messages"])  # type: ignore[index]
        return (
            estimate_tokens(SYSTEM_PROMPT)
            + estimate_tokens(self.profile_store.read_text(user_id))
            + estimate_tokens(str(context["summary"]))
            + recent
        )

    def _offline_response(
        self, user_id: str, thread_id: str, message: str, changed: dict[str, str] | None = None
    ) -> str:
        """Deterministic answer built from `User.md`.

        Recall questions are answered from persisted facts; statements that wrote new
        facts get a short acknowledgement of what was saved.
        """

        facts = self.profile_store.facts(user_id)
        answer = answer_from_facts(message, facts)
        if answer:
            return answer
        if changed is None:
            changed = extract_profile_updates(message)
        return f"Đã ghi nhớ — {describe_updates(changed)}." if changed else ACK

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        memory_path = str(self.profile_store.path_for(user_id))
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
            context=AgentContext(user_id=user_id, memory_path=memory_path),
        )
        response, completion, prompt = usage_from_result(result)
        # Mirror the turn locally so compaction accounting and prompt estimates stay meaningful.
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = prompt or self._estimate_prompt_context_tokens(user_id, thread_id)
        self.compact_memory.append(thread_id, "assistant", response)
        agent_tokens = completion or estimate_tokens(response)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        return {"response": response, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens, "mode": "live"}

    def _maybe_build_langchain_agent(self):
        """Build the live agent, or return None to stay offline.

        Design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tools to read / write / edit `User.md`
        - dynamic prompt that injects `User.md` into the system prompt on every call
        - `SummarizationMiddleware` for compact memory on long threads
        """

        model_config = self.config.model
        needs_key = model_config.provider not in ("ollama",)
        if not self.config.use_live or (needs_key and not model_config.api_key):
            return None
        try:
            from langchain.agents import create_agent
            from langchain.agents.middleware import SummarizationMiddleware, dynamic_prompt
            from langchain.tools import tool
            from langgraph.checkpoint.memory import InMemorySaver

            store = self.profile_store
            model = build_chat_model(model_config)

            @tool
            def read_user_profile(runtime: ToolRuntime[AgentContext]) -> str:
                """Read the persistent User.md profile of the current user."""

                return store.read_text(runtime.context.user_id)

            @tool
            def write_user_profile(key: str, value: str, runtime: ToolRuntime[AgentContext]) -> str:
                """Save one stable fact (name, location, profession, style, favorite_drink,
                favorite_food, pet, interests) to User.md. A new value replaces the old one."""

                changed = store.upsert_fact(runtime.context.user_id, key, value)
                return "saved" if changed else "unchanged"

            @tool
            def edit_user_profile(search_text: str, replacement: str, runtime: ToolRuntime[AgentContext]) -> str:
                """Replace one piece of text inside User.md (e.g. to correct an outdated fact)."""

                return "edited" if store.edit_text(runtime.context.user_id, search_text, replacement) else "not found"

            @dynamic_prompt
            def profile_prompt(request: ModelRequest) -> str:
                profile = store.read_text(request.runtime.context.user_id)
                return (
                    f"{SYSTEM_PROMPT}\n\nHồ sơ bền vững của người dùng (User.md):\n{profile}\n"
                    "Khi người dùng nêu thông tin ổn định mới hoặc đính chính, hãy ghi vào User.md bằng tool. "
                    "Không lưu câu hỏi, câu đùa hay giả định như một fact."
                )

            return create_agent(
                model,
                tools=[read_user_profile, write_user_profile, edit_user_profile],
                middleware=[
                    profile_prompt,
                    SummarizationMiddleware(
                        model=model,
                        trigger=("tokens", self.config.compact_threshold_tokens),
                        keep=("messages", self.config.compact_keep_messages),
                    ),
                ],
                context_schema=AgentContext,
                checkpointer=InMemorySaver(),
            )
        except Exception as error:  # missing SDK, bad credentials/base_url, API drift, ...
            warnings.warn(f"Live agent unavailable, falling back to offline mode: {error!r}", stacklevel=2)
            return None
