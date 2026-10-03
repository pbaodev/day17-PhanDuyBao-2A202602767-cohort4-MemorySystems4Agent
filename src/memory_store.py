from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 characters per token)."""

    text = (text or "").strip()
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


# --------------------------------------------------------------------------- #
# Profile facts: representation + merge rules
# --------------------------------------------------------------------------- #

# Facts that accumulate (union, most recently mentioned last). Every other fact is
# a scalar: a newer value *replaces* the old one (this is the conflict handling:
# after a correction `User.md` never keeps the stale value next to the new one).
LIST_KEYS = ("interests", "style")
MAX_LIST_ITEMS = 8  # guardrail against `User.md` growing without bound
LIST_SEPARATOR = "; "

FACT_ORDER = ("name", "location", "profession", "style", "favorite_drink", "favorite_food", "pet", "interests")

DEFAULT_PROFILE = "# User Profile\n\n## Stable facts\n\n_No stable facts yet._\n"
_FACT_LINE = re.compile(r"^- ([a-z_]+): (.+)$")


def split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(LIST_SEPARATOR.strip()) if item.strip()]


def merge_values(key: str, old: str | None, new: str) -> str:
    """Merge a new fact value into an existing one following the key's semantics."""

    if key not in LIST_KEYS or not old:
        return new
    merged = split_list(old)
    for item in split_list(new):
        # Re-mentioned items move to the end, so the oldest ones decay first when capped.
        merged = [existing for existing in merged if existing.casefold() != item.casefold()]
        merged.append(item)
    return LIST_SEPARATOR.join(merged[-MAX_LIST_ITEMS:])


def merge_facts(existing: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    merged = dict(existing)
    for key, value in updates.items():
        merged[key] = merge_values(key, merged.get(key), value)
    return merged


def render_profile(facts: dict[str, str]) -> str:
    if not facts:
        return DEFAULT_PROFILE
    ordered = [key for key in FACT_ORDER if key in facts] + [key for key in facts if key not in FACT_ORDER]
    lines = ["# User Profile", "", "## Stable facts", ""]
    lines += [f"- {key}: {facts[key]}" for key in ordered]
    return "\n".join(lines) + "\n"


def parse_profile(text: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    for line in text.splitlines():
        match = _FACT_LINE.match(line.strip())
        if match:
            facts[match.group(1)] = match.group(2).strip()
    return facts


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` (one markdown file per user)."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        # Sanitize so a hostile user id can never escape `root_dir`.
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", user_id or "").strip("_") or "anonymous"
        return Path(self.root_dir) / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return DEFAULT_PROFILE
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace one occurrence of `search_text`; return whether the file changed."""

        path = self.path_for(user_id)
        if not search_text or not path.exists():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        updated = content.replace(search_text, replacement, 1)
        if updated == content:
            return False
        path.write_text(updated, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return parse_profile(self.read_text(user_id))

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        return bool(self.upsert_facts(user_id, {key: value}))

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> dict[str, str]:
        """Merge facts into `User.md`; return only the facts that actually changed."""

        current = self.facts(user_id)
        merged = merge_facts(current, updates)
        changed = {key: merged[key] for key in updates if current.get(key) != merged[key]}
        if changed:
            self.write_text(user_id, render_profile(merged))
        return changed


# --------------------------------------------------------------------------- #
# Fact extraction (rule based, with confidence + noise/question guardrails)
# --------------------------------------------------------------------------- #

DEFAULT_MIN_CONFIDENCE = 0.6

ROLE_WORDS = r"(?:engineer|developer|scientist|manager|designer|analyst|architect)"
ROLE = rf"((?:[\w+#-]+\s+){{0,2}}?{ROLE_WORDS})\b"
WORD = r"[^\W\d_]+"
DRINK_WORDS = ("cà phê", "trà", "bia", "nước", "sinh tố", "rượu")

NOISE_MARKERS = ("đùa", "hay là", "giả sử", "tưởng tượng")  # jokes / hypotheticals are not facts
HEDGE_MARKERS = ("có lẽ", "chắc là", "hình như", "không chắc", "có thể là", "nghe nói", "đoán")
CORRECTION_MARKERS = ("đính chính", "thực ra", "chứ không", "từ tuần này", "giờ mình", "giờ chuyển", "cập nhật", "chuyển sang")
QUESTION_STARTS = ("bạn có thể", "bạn có biết", "bạn có nhớ", "bạn thử nhớ lại", "bạn thử mô tả")
QUESTION_ENDINGS = re.compile(r"\b(là gì|là ai|ở đâu|không|nào)$")
NEGATION_CLAUSE = re.compile(r"(?:chứ\s+)?không\s+(?:còn|phải|thích)\b[^,;.]*")
STYLE_TRIGGERS = ("trả lời", "giải thích")
STYLE_CUES = ("muốn", "thích", "ưu tiên", "hãy", "giữ", "style", "nên")
STYLE_DESCRIPTORS = (
    ("3 bullet", r"3 bullet"),
    ("bullet", r"(?<!3 )bullet"),
    ("ngắn gọn", r"\b(?:ngắn gọn|ngắn|gọn)\b"),
    ("rõ ý", r"rõ ý"),
    ("có cấu trúc", r"có cấu trúc"),
    ("có ví dụ thực tế", r"ví dụ thực (?:tế|chiến)"),
    ("so sánh trade-off", r"so sánh trade-off|nhấn trade-off|ưu tiên trade-off"),
)
# A clause that *starts* with a recall verb ("Nhắc lại ...", "Sang thread mới rồi, nhắc lại giúp mình ...").
_RECALL_CLAUSE = re.compile(r"(?:^|,\s*)(?:hãy |bạn |bạn thử )?(?:nhắc lại|nhớ lại|tóm tắt)\b")
LIKE_STOPPERS = re.compile(r"\s+(?:vì|để|hơn|nhưng|khi|nên|do)\s+")


@dataclass(frozen=True)
class FactCandidate:
    key: str
    value: str
    confidence: float


def _split_sentences(message: str) -> list[tuple[str, str]]:
    """Split into (sentence, terminator) pairs so question marks are not lost."""

    pairs = []
    for chunk in re.findall(r"[^.!?\n]+[.!?]?", message):
        text = chunk.strip()
        if text:
            pairs.append((text.rstrip(".!?").strip(), chunk.strip()[-1] if chunk.strip()[-1] in ".!?" else ""))
    return pairs


def _is_question(sentence: str, terminator: str) -> bool:
    low = sentence.lower()
    return terminator == "?" or low.startswith(QUESTION_STARTS) or bool(QUESTION_ENDINGS.search(low))


def _cap_words(text: str, max_words: int = 3) -> str:
    """Leading run of capitalised words (proper nouns such as `Đà Nẵng`, `DũngCT Stress`)."""

    head = re.split(r"[,.;:!?\n]", text, maxsplit=1)[0]
    words: list[str] = []
    for word in head.split():
        if not word[0].isupper() or len(words) >= max_words:
            break
        words.append(word)
    return " ".join(words)


def _last_match(pattern: str, text: str, flags: int = re.IGNORECASE) -> re.Match | None:
    matches = list(re.finditer(pattern, text, flags))
    return matches[-1] if matches else None


def _clean_value(value: str) -> str:
    value = re.split(r"[,;.]", value, maxsplit=1)[0]
    value = LIKE_STOPPERS.split(" " + value + " ", maxsplit=1)[0]
    return value.strip()


def _split_items(value: str) -> list[str]:
    value = LIKE_STOPPERS.split(" " + value + " ", maxsplit=1)[0]
    parts = re.split(r",\s*(?:và\s+)?|\s+và\s+", value)
    return [part.strip(" .") for part in parts if part.strip(" .")]


def _identity_candidates(body: str, conf) -> list[FactCandidate]:
    """Facts that identify the user (name/location/profession/...). Skipped for conditionals."""

    out: list[FactCandidate] = []

    match = re.search(r"\btên(?:\s+mình|\s+tôi)?\s+là\s+(.+)", body, re.IGNORECASE)
    if match:
        name = _cap_words(match.group(1))
        if name:
            out.append(FactCandidate("name", name, conf(0.95)))

    # Location: later mentions win ("hiện ở Huế ... mình đang làm việc ở Đà Nẵng").
    location = None
    if (match := re.search(r"nơi ở[^.;]*?\bsang\s+(.+)", body, re.IGNORECASE)):
        location = (_cap_words(match.group(1)), 0.9)
    elif (match := re.search(r"nơi ở hiện tại\s+(?:vẫn\s+)?là\s+(.+)", body, re.IGNORECASE)):
        location = (_cap_words(match.group(1)), 0.9)
    elif (match := _last_match(rf"(?<!\w)ở\s+({WORD}(?:\s+{WORD}){{0,2}})", body)):
        place = _cap_words(match.group(1))
        prefix = body[: match.start()].lower()
        working = prefix.rstrip().endswith("làm việc")
        base = 0.5 if working else (0.85 if re.search(r"(đang|hiện|giờ|vẫn)\s*$", prefix) else 0.75)
        location = (place, base)
    if location and location[0]:
        out.append(FactCandidate("location", location[0], conf(location[1])))

    profession = None
    for pattern, base in (
        (rf"nghề(?:\s+nghiệp)?(?:\s+hiện\s+tại)?(?:\s+của\s+mình)?(?:\s+thì)?\s+(?:vẫn\s+)?là\s+{ROLE}", 0.9),
        (rf"chuyển\s+sang\s+{ROLE}", 0.85),
        (rf"\blàm\s+{ROLE}", 0.8),
        (rf"\bmình\s+là\s+{ROLE}", 0.7),
    ):
        if (match := _last_match(pattern, body)):
            profession = (match.group(1).strip(), base)
            break
    if profession:
        out.append(FactCandidate("profession", profession[0], conf(profession[1])))

    match = re.search(r"đồ uống(?:\s+yêu thích)?(?:\s+của mình)?\s+là\s+(.+)", body, re.IGNORECASE)
    if match and (drink := _clean_value(match.group(1))) and drink.lower() not in ("gì", "nào"):
        out.append(FactCandidate("favorite_drink", drink, conf(0.9)))
    elif (match := re.search(rf"\buống\s+((?:{'|'.join(DRINK_WORDS)})[^,;.]*)", body, re.IGNORECASE)):
        drink = re.split(r"\s+(?:như|nhưng|rồi|vì|để|và)\b", match.group(1))[0].strip()
        if drink:
            out.append(FactCandidate("favorite_drink", drink, conf(0.75)))

    match = re.search(r"món ăn yêu thích(?:\s+của mình)?\s+là\s+(.+)", body, re.IGNORECASE)
    if match and (food := _clean_value(match.group(1))) and food.lower() not in ("gì", "nào"):
        out.append(FactCandidate("favorite_food", food, conf(0.9)))

    match = re.search(rf"\b(?:bé|con)\s+(corgi|chó|mèo|poodle|husky)\s+tên\s+(?:là\s+)?({WORD})", body, re.IGNORECASE)
    if match and match.group(2)[0].isupper():
        out.append(FactCandidate("pet", f"{match.group(1)} tên {match.group(2)}", conf(0.9)))

    return out


def _preference_candidates(body: str, conf) -> list[FactCandidate]:
    """Style + interests. These are allowed even in conditional sentences ("Nếu bạn giải thích, hãy ...")."""

    out: list[FactCandidate] = []
    low = body.lower()

    if any(t in low for t in STYLE_TRIGGERS) and any(re.search(rf"\b{c}\b", low) for c in STYLE_CUES):
        descriptors = [label for label, pattern in STYLE_DESCRIPTORS if re.search(pattern, low)]
        if descriptors:
            out.append(FactCandidate("style", LIST_SEPARATOR.join(descriptors), conf(0.8)))

    interests: list[str] = []
    for pattern in (r"(?<!giải )(?<!yêu )\bthích\s+(.+)", r"quan tâm(?:\s+nhiều)?\s+đến\s+(.+)"):
        for match in re.finditer(pattern, body, re.IGNORECASE):
            for item in _split_items(match.group(1)):
                lowered = item.lower()
                if lowered.startswith(("cách", "kiểu", "câu trả lời", "so sánh")) or lowered.startswith(DRINK_WORDS):
                    continue  # style / drink phrases are handled elsewhere
                if "yêu thích" in lowered or "của mình" in lowered:
                    continue  # a field label ("đồ uống yêu thích của mình"), not an interest
                if re.search(r"\b(?:này|đó|kia)\b", lowered):
                    continue  # deictic ("tin này") = a passing opinion, not a stable interest
                if len(item.split()) <= 6 and len(item) <= 40:
                    interests.append(item)
    if interests:
        out.append(FactCandidate("interests", LIST_SEPARATOR.join(interests), conf(0.75)))

    return out


def extract_candidates(message: str) -> list[FactCandidate]:
    """Every fact candidate in `message`, each with a confidence score."""

    message = unicodedata.normalize("NFC", message)
    candidates: list[FactCandidate] = []
    for sentence, terminator in _split_sentences(message):
        low = sentence.lower()
        if _is_question(sentence, terminator) or any(marker in low for marker in NOISE_MARKERS):
            continue  # questions and jokes never produce facts
        if not low.startswith("nếu ") and _RECALL_CLAUSE.search(low):
            continue  # "Nhắc lại style mình thích ..." asks for memory; it must not be written into it

        hedged = any(marker in low for marker in HEDGE_MARKERS)
        corrected = any(marker in low for marker in CORRECTION_MARKERS)

        def conf(base: float, hedged: bool = hedged, corrected: bool = corrected) -> float:
            return round(min(0.99, base + (0.2 if corrected else 0.0) - (0.35 if hedged else 0.0)), 2)

        body = NEGATION_CLAUSE.sub("", sentence)  # drop "chứ không còn ở X", "không còn làm Y"
        if not low.startswith(("nếu ", "giá như ")):
            candidates += _identity_candidates(body, conf)
        candidates += _preference_candidates(body, conf)
    return candidates


def extract_profile_updates(message: str, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Guardrails: questions, jokes, hypotheticals and negated clauses never become
    facts, and hedged statements ("chắc là mình ở Hà Nội") stay under the
    confidence threshold. Scalar facts keep the last value in the message, list
    facts (interests/style) are unioned.
    """

    updates: dict[str, str] = {}
    for candidate in extract_candidates(message):
        if candidate.confidence < min_confidence:
            continue
        updates[candidate.key] = merge_values(candidate.key, updates.get(candidate.key), candidate.value)
    return updates


# --------------------------------------------------------------------------- #
# Answering recall questions from facts (shared by both offline agents)
# --------------------------------------------------------------------------- #

_FIELD_KEYWORDS = {
    "name": ("tên",),
    "location": ("ở đâu", "nơi ở", "đang ở", "còn ở", "sống ở"),
    "profession": ("nghề", "làm gì", "công việc"),
    "style": ("style", "kiểu trả lời", "cách trả lời", "trả lời mình thích"),
    "favorite_drink": ("đồ uống",),
    "favorite_food": ("món ăn",),
    "pet": ("nuôi", "thú cưng"),
    "interests": ("quan tâm", "sở thích"),
}
_FIELD_LABELS = {
    "name": "tên",
    "location": "nơi ở hiện tại",
    "profession": "nghề nghiệp hiện tại",
    "style": "phong cách trả lời",
    "favorite_drink": "đồ uống yêu thích",
    "favorite_food": "món ăn yêu thích",
    "pet": "thú cưng",
    "interests": "mối quan tâm",
}
_FIELD_TEMPLATES = {
    "name": "tên bạn là {}",
    "location": "hiện bạn ở {}",
    "profession": "nghề nghiệp hiện tại của bạn là {}",
    "style": "bạn thích câu trả lời: {}",
    "favorite_drink": "đồ uống yêu thích của bạn là {}",
    "favorite_food": "món ăn yêu thích của bạn là {}",
    "pet": "bạn nuôi {}",
    "interests": "bạn quan tâm đến {}",
}


def is_recall_request(message: str) -> bool:
    message = unicodedata.normalize("NFC", message)
    for sentence, terminator in _split_sentences(message):
        low = sentence.lower()
        if _is_question(sentence, terminator):
            return True
        if not low.startswith("nếu ") and _RECALL_CLAUSE.search(low):
            return True  # (a conditional like "Nếu bạn nhắc lại ..." is an instruction, not a request)
    return False


def requested_fields(message: str) -> list[str]:
    low = unicodedata.normalize("NFC", message).lower()
    fields = [key for key, words in _FIELD_KEYWORDS.items() if any(word in low for word in words)]
    if not fields and ("tóm tắt" in low or "là ai" in low):
        fields = list(FACT_ORDER)
    return fields


def answer_from_facts(message: str, facts: dict[str, str]) -> str | None:
    """Deterministic answer to a recall question; `None` if `message` is not one."""

    if not is_recall_request(message):
        return None
    fields = requested_fields(message)
    if not fields:
        return None
    known = [_FIELD_TEMPLATES[f].format(facts[f]) for f in FACT_ORDER if f in fields and f in facts]
    missing = [_FIELD_LABELS[f] for f in FACT_ORDER if f in fields and f not in facts]
    parts = []
    if known:
        parts.append("Theo những gì mình nhớ: " + "; ".join(known) + ".")
    if missing:
        parts.append("Mình chưa có thông tin về: " + ", ".join(missing) + ".")
    return " ".join(parts)


def describe_updates(updates: dict[str, str]) -> str:
    """Short acknowledgement of facts that were just written to `User.md`."""

    return "; ".join(f"{_FIELD_LABELS.get(key, key)}: {value}" for key, value in updates.items())


# Same system prompt for both agents so prompt-token comparisons stay fair.
SYSTEM_PROMPT = "Bạn là trợ lý AI hữu ích. Hãy trả lời bằng tiếng Việt, trung thực và đúng trọng tâm."


# --------------------------------------------------------------------------- #
# Compact memory
# --------------------------------------------------------------------------- #

SNIPPET_CHARS = 100
MAX_SUMMARY_LINES = 8


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary: one short line per message, keeping the most recent `max_items`.

    Stable user facts are *not* the summary's job (they live in `User.md`); the
    summary only preserves the gist of the older conversation flow.
    """

    lines = []
    for message in messages[-max_items:]:
        content = " ".join(message.get("content", "").split())
        first_sentence = re.split(r"(?<=[.!?])\s", content, maxsplit=1)[0]
        snippet = first_sentence if len(first_sentence) <= SNIPPET_CHARS else first_sentence[: SNIPPET_CHARS - 1].rstrip() + "…"
        lines.append(f"- {message.get('role', 'user')}: {snippet}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    Keeps the most recent messages in full; when summary + messages exceed
    `threshold_tokens`, everything older than `keep_messages` is folded into a
    rolling summary and the compaction counter is incremented.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        messages.append({"role": role, "content": content})
        if self._total_tokens(thread) > self.threshold_tokens and len(messages) > self.keep_messages:
            self._compact(thread)

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.get(thread_id) or {"messages": [], "summary": "", "compactions": 0}

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])  # type: ignore[arg-type]

    @staticmethod
    def _total_tokens(thread: dict[str, object]) -> int:
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        return estimate_tokens(str(thread["summary"])) + sum(estimate_tokens(m["content"]) for m in messages)

    def _compact(self, thread: dict[str, object]) -> None:
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        keep = max(self.keep_messages, 0)
        old, recent = (messages[:-keep], messages[-keep:]) if keep else (messages, [])
        previous = [line for line in str(thread["summary"]).splitlines() if line]
        fresh = summarize_messages(old).splitlines()
        thread["summary"] = "\n".join((previous + fresh)[-MAX_SUMMARY_LINES:])
        thread["messages"] = recent
        thread["compactions"] = int(thread["compactions"]) + 1  # type: ignore[arg-type]
