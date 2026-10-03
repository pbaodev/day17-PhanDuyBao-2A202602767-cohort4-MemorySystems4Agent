from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config

CONCISE_MAX_WORDS = 80


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _contains(answer: str, expected: str) -> bool:
    """Whole-word match; case-insensitive except for acronyms (`AI` must not match `hai`)."""

    answer, expected = _normalize(answer), _normalize(expected)
    flags = 0 if expected.isupper() else re.IGNORECASE
    return re.search(rf"(?<!\w){re.escape(expected)}(?!\w)", answer, flags) is not None


def _hits(answer: str, expected: list[str]) -> int:
    return sum(_contains(answer, item) for item in expected)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if only some do, 0 if none."""

    if not expected:
        return 1.0
    hits = _hits(answer, expected)
    if hits == len(expected):
        return 1.0
    return 0.5 if hits else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline quality score in [0, 1]: fact coverage, mildly penalising rambling answers.

    A wrong/empty answer can never score by being short, so quality stays 0 without facts.
    """

    if not expected or not answer.strip():
        return 0.0
    coverage = _hits(answer, expected) / len(expected)
    concise = 1.0 if len(answer.split()) <= CONCISE_MAX_WORDS else 0.5
    return round(coverage * (0.8 + 0.2 * concise), 4)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    Each conversation is fed turn by turn in one thread; every recall question is then
    asked in a *fresh* thread (new session) of the same user.
    """

    users = sorted({conv["user_id"] for conv in conversations})
    size_before = sum(agent.memory_file_size(user) for user in users)
    agent_tokens = prompt_tokens = compactions = 0
    recalls: list[float] = []
    qualities: list[float] = []

    for conv in conversations:
        main_thread = f"{agent_name}-{conv['id']}-main"
        for turn in conv["turns"]:
            agent.reply(conv["user_id"], main_thread, turn)
        agent_tokens += agent.token_usage(main_thread)
        prompt_tokens += agent.prompt_token_usage(main_thread)
        compactions += agent.compaction_count(main_thread)

        for index, item in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{agent_name}-{conv['id']}-recall-{index}"
            answer = agent.reply(conv["user_id"], recall_thread, item["question"])["response"]
            recalls.append(recall_points(answer, item["expected_contains"]))
            qualities.append(heuristic_quality(answer, item["expected_contains"]))
            agent_tokens += agent.token_usage(recall_thread)
            prompt_tokens += agent.prompt_token_usage(recall_thread)

    size_after = sum(agent.memory_file_size(user) for user in users)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(sum(recalls) / len(recalls), 3) if recalls else 0.0,
        response_quality=round(sum(qualities) / len(qualities), 3) if qualities else 0.0,
        memory_growth_bytes=size_after - size_before,
        compactions=compactions,
    )


HEADERS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=HEADERS, tablefmt="github")
    except ImportError:  # tabulate is optional: fall back to a plain markdown table
        lines = ["| " + " | ".join(HEADERS) + " |", "|" + "|".join("---" for _ in HEADERS) + "|"]
        lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in table]
        return "\n".join(lines)


def run_suite(name: str, dataset: Path, config) -> list[BenchmarkRow]:
    """Run both agents on one dataset, each suite in its own freshly wiped state dir."""

    suite_state = config.state_dir / "benchmark" / name
    shutil.rmtree(suite_state, ignore_errors=True)
    suite_config = replace(config, state_dir=suite_state)
    conversations = load_conversations(dataset)

    rows = []
    for agent_name, agent_cls in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)):
        agent = agent_cls(suite_config, force_offline=True)  # deterministic: no API key needed
        rows.append(run_agent_benchmark(agent_name, agent, conversations, suite_config))
    return rows


def _delta(rows: list[BenchmarkRow]) -> str:
    base, adv = rows
    ratio = adv.prompt_tokens_processed / base.prompt_tokens_processed if base.prompt_tokens_processed else 0
    direction = "saves" if ratio < 1 else "costs"
    return (
        f"Advanced {direction} {abs(1 - ratio) * 100:.0f}% prompt tokens vs Baseline "
        f"({adv.prompt_tokens_processed} vs {base.prompt_tokens_processed}); "
        f"recall {base.recall_score:.2f} -> {adv.recall_score:.2f}."
    )


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    suites = (
        ("standard", "Standard Benchmark", config.data_dir / "conversations.json"),
        ("stress", "Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    )
    for key, title, dataset in suites:
        rows = run_suite(key, dataset, config)
        print(f"\n## {title}  ({dataset.name})\n")
        print(format_rows(rows))
        print(f"\n{_delta(rows)}")
    print(
        f"\nCompact threshold: {config.compact_threshold_tokens} tokens, keep last "
        f"{config.compact_keep_messages} messages. Token counts are heuristic (~4 chars/token)."
    )


if __name__ == "__main__":
    main()
