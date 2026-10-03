from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import heuristic_quality, recall_points
from config import load_config
from memory_store import (
    MAX_LIST_ITEMS,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    summarize_messages,
)
from model_provider import normalize_provider

import pytest


def make_config(tmp_path: Path, threshold: int = 150, keep: int = 2):
    """Isolated config: state lives in tmp_path and compaction triggers quickly."""

    config = load_config(tmp_path)
    return replace(config, compact_threshold_tokens=threshold, compact_keep_messages=keep)


def long_turn(i: int) -> str:
    return f"Tin số {i}: " + "đây là một đoạn tin tức khá dài để làm đầy ngữ cảnh hội thoại. " * 4


# --------------------------------------------------------------------------- #
# Required behaviours
# --------------------------------------------------------------------------- #


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    # Unknown user: default profile, nothing on disk yet.
    assert "User Profile" in store.read_text("dungct")
    assert store.file_size("dungct") == 0

    path = store.write_text("dungct", "# User Profile\n\n- name: DũngCT\n- location: Đà Nẵng\n")
    assert path.name == "User.md" and path.exists()
    assert store.read_text("dungct").count("DũngCT") == 1
    assert store.file_size("dungct") == path.stat().st_size > 0

    assert store.edit_text("dungct", "Đà Nẵng", "Huế") is True
    assert "Huế" in store.read_text("dungct") and "Đà Nẵng" not in store.read_text("dungct")
    assert store.edit_text("dungct", "không tồn tại", "x") is False
    assert store.facts("dungct") == {"name": "DũngCT", "location": "Huế"}


def test_compact_trigger(tmp_path: Path) -> None:
    manager = CompactMemoryManager(threshold_tokens=150, keep_messages=2)
    for i in range(10):
        manager.append("t1", "user", long_turn(i))

    context = manager.context("t1")
    assert manager.compaction_count("t1") >= 1
    assert len(context["messages"]) <= 3  # recent messages only
    assert context["summary"]  # older content survived as a summary
    assert context["messages"][-1]["content"] == long_turn(9)  # newest message is kept verbatim

    # Short threads never compact.
    quiet = CompactMemoryManager(threshold_tokens=10_000, keep_messages=2)
    for i in range(5):
        quiet.append("t2", "user", "xin chào")
    assert quiet.compaction_count("t2") == 0 and not quiet.context("t2")["summary"]


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)

    for agent in (advanced, baseline):
        agent.reply("dungct", "session-1", "Chào bạn, mình tên là DũngCT.")
        agent.reply("dungct", "session-1", "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")

    question = "Mình tên gì, hiện ở đâu và làm nghề gì?"
    adv_answer = advanced.reply("dungct", "session-2", question)["response"]
    base_answer = baseline.reply("dungct", "session-2", question)["response"]
    assert all(fact in adv_answer for fact in ("DũngCT", "Đà Nẵng", "backend engineer"))
    assert not any(fact in base_answer for fact in ("DũngCT", "Đà Nẵng", "backend engineer"))

    # The baseline *does* remember inside the same thread (short-term memory only).
    in_thread = baseline.reply("dungct", "session-1", "Mình tên gì?")["response"]
    assert "DũngCT" in in_thread

    # User.md is persistent: a brand-new agent process on the same state dir still recalls.
    restarted = AdvancedAgent(config, force_offline=True)
    assert "DũngCT" in restarted.reply("dungct", "session-3", "Tên mình là gì?")["response"]
    # ...but not for another user.
    assert "DũngCT" not in restarted.reply("someone_else", "session-4", "Tên mình là gì?")["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)

    for i in range(14):
        advanced.reply("dungct", "long", long_turn(i))
        baseline.reply("dungct", "long", long_turn(i))

    assert advanced.compaction_count("long") >= 1
    assert baseline.compaction_count("long") == 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")


def test_compact_costs_more_on_short_threads(tmp_path: Path) -> None:
    """Trade-off: on a short thread User.md is pure overhead, so Advanced is not cheaper."""

    config = make_config(tmp_path, threshold=10_000)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for turn in ("Chào bạn, mình tên là DũngCT.", "Mình ở Huế.", "Mình thích Python."):
        advanced.reply("dungct", "short", turn)
        baseline.reply("dungct", "short", turn)

    assert advanced.compaction_count("short") == 0
    assert advanced.prompt_token_usage("short") > baseline.prompt_token_usage("short")
    assert advanced.token_usage("short") > baseline.token_usage("short")


# --------------------------------------------------------------------------- #
# Guardrails: corrections, noise, confidence
# --------------------------------------------------------------------------- #


def test_correction_replaces_old_fact(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("u", "t1", "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")
    agent.reply("u", "t1", "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.")
    agent.reply("u", "t1", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")

    profile = agent.profile_store.read_text("u")
    assert "location: Huế" in profile and "Đà Nẵng" not in profile
    assert "profession: MLOps engineer" in profile and "backend" not in profile
    answer = agent.reply("u", "t2", "Hiện tại mình làm nghề gì và ở đâu?")["response"]
    assert "MLOps engineer" in answer and "Huế" in answer


@pytest.mark.parametrize(
    "message",
    [
        "Mình ở Huế, bạn nhớ không?",  # question
        "Bạn có biết DũngCT không?",  # question
        "Chắc là mình ở Hà Nội.",  # hedged -> below confidence threshold
        "Có lúc mình đùa rằng hay là chuyển sang product manager.",  # joke
        "Nếu mình ở Hà Nội thì sao.",  # hypothetical
        "Hà Nội chỉ là nơi mình vừa bay ra họp chứ không phải nơi ở hiện tại.",  # noise + negation
    ],
)
def test_noise_questions_and_hedges_are_not_stored(message: str) -> None:
    assert extract_profile_updates(message) == {}


def test_recall_requests_never_pollute_user_md(tmp_path: Path) -> None:
    """Asking the agent to recall something must not write that request into memory."""

    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("u", "t1", "Mình tên là DũngCT. Đồ uống yêu thích là cà phê sữa đá.")
    agent.reply("u", "t1", "Mình muốn bạn trả lời ngắn gọn.")
    before = agent.profile_store.read_text("u")

    for request in (
        "Nhắc lại style trả lời mình thích và đồ uống yêu thích của mình.",
        "Sang thread mới rồi, nhắc lại giúp mình tên và style trả lời mình thích.",
        "Nhắc lại giúp mình: tên, nơi ở hiện tại, đồ uống yêu thích và style trả lời mình thích.",
        "Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.",
    ):
        agent.reply("u", "t2", request)
    assert agent.profile_store.read_text("u") == before
    assert "của mình" not in before


def test_confidence_threshold_is_configurable() -> None:
    assert extract_profile_updates("Chắc là mình ở Hà Nội.") == {}
    assert extract_profile_updates("Chắc là mình ở Hà Nội.", min_confidence=0.3) == {"location": "Hà Nội"}


def test_memory_file_growth_is_bounded(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)
    for i in range(30):
        store.upsert_facts("u", {"interests": f"chủ đề {i}"})
    interests = store.facts("u")["interests"].split("; ")
    assert len(interests) == MAX_LIST_ITEMS
    assert interests[-1] == "chủ đề 29" and "chủ đề 0" not in interests  # oldest decays first

    store.upsert_facts("u", {"interests": "chủ đề 22"})  # re-mention refreshes recency
    assert store.facts("u")["interests"].split("; ")[-1] == "chủ đề 22"


def test_user_id_cannot_escape_profile_root(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    path = store.path_for("../../etc/passwd")
    assert (tmp_path / "profiles") in path.parents


# --------------------------------------------------------------------------- #
# Supporting pieces
# --------------------------------------------------------------------------- #


def test_estimate_tokens_and_summary() -> None:
    assert estimate_tokens("") == 0 and estimate_tokens("   ") == 0
    assert estimate_tokens("abcd") == 1 and estimate_tokens("abcde") == 2

    summary = summarize_messages([{"role": "user", "content": long_turn(i)} for i in range(10)], max_items=3)
    assert len(summary.splitlines()) == 3 and "Tin số 9" in summary


def test_benchmark_scoring() -> None:
    assert recall_points("Tên bạn là DũngCT, thích cà phê sữa đá", ["DũngCT", "cà phê sữa đá"]) == 1.0
    assert recall_points("Tên bạn là DũngCT", ["DũngCT", "cà phê sữa đá"]) == 0.5
    assert recall_points("Mình chưa có thông tin", ["DũngCT"]) == 0.0
    assert recall_points("hai bài toán", ["AI"]) == 0.0  # acronym must not match inside other words
    assert heuristic_quality("Mình chưa có thông tin", ["DũngCT"]) == 0.0
    assert heuristic_quality("Tên bạn là DũngCT", ["DũngCT"]) == 1.0


def test_provider_aliases_and_config(tmp_path: Path, monkeypatch) -> None:
    assert normalize_provider("anthorpic") == "anthropic"
    assert normalize_provider("OpenRouter") == "openrouter"
    with pytest.raises(ValueError):
        normalize_provider("not-a-provider")

    monkeypatch.setenv("LLM_PROVIDER", "anthorpic")
    monkeypatch.delenv("LAB_MODE", raising=False)
    config = load_config(tmp_path)
    assert config.model.provider == "anthropic" and not config.use_live
    assert (tmp_path / "state").is_dir()


def test_offline_mode_is_deterministic(tmp_path: Path) -> None:
    turns = ["Mình tên là DũngCT.", "Mình ở Huế.", "Mình tên gì và ở đâu?"]
    outputs = []
    for run in ("a", "b"):
        agent = AdvancedAgent(make_config(tmp_path / run), force_offline=True)
        outputs.append([agent.reply("u", "t", turn) for turn in turns])
    assert outputs[0] == outputs[1]
