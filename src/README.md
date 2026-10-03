# `src/` — solved lab

All modules are implemented. Offline mode (default) is deterministic and needs no API key.

| File | What it does |
|---|---|
| `model_provider.py` | `ProviderConfig`, `normalize_provider()` (aliases such as `anthorpic`), lazy `build_chat_model()` for openai / custom / gemini / anthropic / ollama / openrouter, live-usage helpers |
| `config.py` | `LabConfig` + `load_config()` (`.env` + env vars, creates `state/`, compact defaults: 1000 tokens / keep 4 messages) |
| `memory_store.py` | `estimate_tokens`, `UserProfileStore` (`User.md`), `extract_profile_updates` (confidence threshold, corrections, noise/question guards), `answer_from_facts`, `summarize_messages`, `CompactMemoryManager` |
| `agent_baseline.py` | Agent A: short-term memory per thread only |
| `agent_advanced.py` | Agent B: short-term + `User.md` + compact memory; live path = tools + dynamic prompt + `SummarizationMiddleware` |
| `benchmark.py` | Standard + long-context stress benchmark, 6 output columns |
| `test_agents.py` | 20 tests: `User.md`, compaction, cross-session recall, prompt load, guardrails |

```bash
pip install -r requirements.txt
python src/benchmark.py
pytest src/test_agents.py -v
```

Live mode is opt-in: `LAB_MODE=live LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini OPENAI_API_KEY=... `. If the SDK or credentials are
missing the agents emit a warning and fall back to offline mode. Other knobs: `COMPACT_THRESHOLD_TOKENS`,
`COMPACT_KEEP_MESSAGES`, `JUDGE_PROVIDER` / `JUDGE_MODEL`.

See `../Analysis.md` for the benchmark results and discussion.
