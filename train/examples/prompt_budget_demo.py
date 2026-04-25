"""多轮对话 prompt 长度治理 demo。

这个 demo 的目标不是替代当前项目实现，而是单独演示一套更适合未来多轮对话的
prompt 长度控制方案。它刻意不依赖项目里的 `core/` 代码，这样可以：

1. 独立运行，方便先验证思路；
2. 不影响当前项目代码；
3. 给后续正式接入项目时提供可读、可改造的参考实现。

演示内容包含：

1. 统一 token 预算器
2. 历史消息语义评分与筛选
3. 滚动摘要生成
4. 检索证据句子级压缩
5. 结构化 prompt 组装

运行方式：

    python3 train/examples/prompt_budget_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Iterable


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_./:-]+")
_SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])\s*")
_STOPWORDS = {
    "的",
    "了",
    "呢",
    "吗",
    "是",
    "在",
    "和",
    "与",
    "及",
    "怎么",
    "如何",
    "一下",
    "一个",
    "这个",
    "那个",
    "请问",
    "帮我",
    "需要",
    "可以",
}


@dataclass
class HistoryTurn:
    """单轮历史消息。

    这里保留最少字段，避免 demo 过度复杂。
    """

    role: str
    content: str
    turn_id: str


@dataclass
class EvidenceChunk:
    """检索命中的证据块。"""

    chunk_id: str
    title: str
    source: str
    section: str
    content: str
    retrieval_score: float
    authority_level: int = 0
    freshness_level: int = 0


@dataclass
class PromptBudgetConfig:
    """统一预算配置。

    真实项目里建议放进 `settings.py`，这里先直接写成 demo 配置对象。
    """

    context_window_tokens: int = 2400
    reserved_completion_tokens: int = 400
    safety_margin_tokens: int = 120
    system_max_tokens: int = 280
    task_max_tokens: int = 220
    memory_max_tokens: int = 260
    recent_turns_max_tokens: int = 420
    evidence_max_tokens: int = 980
    governance_max_tokens: int = 120
    max_recent_turns: int = 4
    max_selected_evidence: int = 4
    max_sentences_per_evidence: int = 3


@dataclass
class PromptBuildResult:
    """最终构建结果。"""

    prompt: str
    selected_history: list[HistoryTurn] = field(default_factory=list)
    dropped_history: list[HistoryTurn] = field(default_factory=list)
    selected_evidence: list[tuple[EvidenceChunk, str]] = field(default_factory=list)
    rolling_summary: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)
    drop_reasons: list[str] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """对文本做一个轻量 token 估算。

    说明：
    1. 这里不是精确 tokenizer；
    2. demo 的目标是演示“预算驱动”的治理思路；
    3. 真正接项目时，建议替换成和目标模型一致的 tokenizer。

    这个估算器的规则很简单：
    - 每个中文字符按 1 token 估算
    - 连续英文 / 数字 / 路径 / 版本号片段按 1 token 估算
    - 额外加一点结构性开销，避免估算过于乐观
    """

    tokens = _TOKEN_RE.findall(text)
    return len(tokens) + max(1, len(text) // 80)


def normalize_text(text: str) -> str:
    """轻量清洗文本，减少空白噪声。"""

    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    """按中文标点做句子拆分。"""

    text = normalize_text(text)
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [part.strip() for part in parts if part.strip()]


def extract_keywords(text: str) -> set[str]:
    """从问题或历史里提取简单关键词。

    这里故意不做复杂 NLP，只做足够清晰的 demo：
    - 中文单字和英文串都会被切出来
    - 去掉一批常见停用词
    """

    words = {w for w in _TOKEN_RE.findall(normalize_text(text)) if w and w not in _STOPWORDS}
    return words


def overlap_score(left: Iterable[str], right: Iterable[str]) -> float:
    """计算两个词集合的重合度。"""

    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def build_rolling_summary(turns: list[HistoryTurn], max_tokens: int) -> str:
    """把较早历史压成短摘要。

    设计原则：
    1. 不保留原文长段落；
    2. 优先保留主题、约束、未完成任务；
    3. 摘要要短而稳定，便于预算控制。
    """

    if not turns or max_tokens <= 0:
        return ""

    user_constraints: list[str] = []
    facts: list[str] = []
    unresolved: list[str] = []
    topic_hint = ""

    for turn in turns:
        content = normalize_text(turn.content)
        if not content:
            continue
        if not topic_hint and turn.role == "user":
            topic_hint = content[:80]
        if turn.role == "user":
            if any(token in content for token in ("必须", "优先", "不要", "只看", "一定要", "需要")):
                user_constraints.append(content[:80])
        if any(token in content for token in ("结论", "根据", "来自", "入口", "流程", "规范", "安生平台", "二矿", "安环部")):
            facts.append(content[:80])
        if any(token in content for token in ("还没", "还需要", "待确认", "没找到", "不确定")):
            unresolved.append(content[:80])

    summary = {
        "topic": topic_hint or "围绕企业知识问答的持续追问",
        "user_constraints": user_constraints[:2],
        "confirmed_facts": facts[:2],
        "unresolved_tasks": unresolved[:2],
    }
    text = "ACTIVE_SESSION_MEMORY:\n" + json.dumps(summary, ensure_ascii=False, indent=2)
    if estimate_tokens(text) <= max_tokens:
        return text

    # 如果摘要仍偏长，则继续做收缩，先裁剪列表项，再保留单行 JSON。
    summary["user_constraints"] = summary["user_constraints"][:1]
    summary["confirmed_facts"] = summary["confirmed_facts"][:1]
    summary["unresolved_tasks"] = summary["unresolved_tasks"][:1]
    compact = "ACTIVE_SESSION_MEMORY: " + json.dumps(summary, ensure_ascii=False)
    if estimate_tokens(compact) <= max_tokens:
        return compact
    return compact[: max(20, int(max_tokens * 3.5))]


def score_history_turn(
    turn: HistoryTurn,
    current_question: str,
    *,
    recency_rank: int,
) -> float:
    """给历史消息打分。

    分数不是绝对真理，目标是把“更值得保留的消息”排在前面。
    当前采用的策略：
    1. 当前问题语义重合越高，分越高
    2. 越近的消息，分越高
    3. 用户约束、未完成任务、上一轮关键回答有额外加分
    """

    question_keywords = extract_keywords(current_question)
    turn_keywords = extract_keywords(turn.content)
    relevance = overlap_score(question_keywords, turn_keywords)
    recency = 1.0 / (1.0 + recency_rank)
    role_bonus = 0.08 if turn.role == "user" else 0.03
    constraint_bonus = 0.12 if any(token in turn.content for token in ("必须", "优先", "不要", "一定要")) else 0.0
    unresolved_bonus = 0.10 if any(token in turn.content for token in ("待确认", "不确定", "还没", "还需要")) else 0.0
    citation_bonus = 0.08 if "[CHUNK_ID:" in turn.content else 0.0
    score = (
        0.38 * relevance
        + 0.22 * recency
        + role_bonus
        + constraint_bonus
        + unresolved_bonus
        + citation_bonus
    )
    return round(score, 4)


def select_recent_turns(
    turns: list[HistoryTurn],
    current_question: str,
    max_tokens: int,
    max_turns: int,
) -> tuple[list[HistoryTurn], list[HistoryTurn]]:
    """按预算选择最近且最相关的历史原文。

    这里的原则不是“只保留最后 N 条”，而是：
    1. 先按语义重要性打分
    2. 在预算内尽量保留高分消息
    """

    if max_tokens <= 0 or max_turns <= 0:
        return [], turns

    scored: list[tuple[float, int, HistoryTurn]] = []
    total = len(turns)
    for idx, turn in enumerate(turns):
        recency_rank = total - idx - 1
        scored.append((score_history_turn(turn, current_question, recency_rank=recency_rank), idx, turn))

    # 分数高优先；分数相同时，更新近的消息优先。
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected: list[HistoryTurn] = []
    used_tokens = 0
    for _, _, turn in scored:
        turn_block = f"[{turn.role.upper()}] {normalize_text(turn.content)}"
        token_cost = estimate_tokens(turn_block)
        if len(selected) >= max_turns:
            continue
        if selected and used_tokens + token_cost > max_tokens:
            continue
        selected.append(turn)
        used_tokens += token_cost

    # 输出时按原始时间顺序恢复，便于阅读和保持会话连贯。
    selected_ids = {turn.turn_id for turn in selected}
    selected_in_order = [turn for turn in turns if turn.turn_id in selected_ids]
    dropped = [turn for turn in turns if turn.turn_id not in selected_ids]
    return selected_in_order, dropped


def compress_evidence_chunk(
    chunk: EvidenceChunk,
    current_question: str,
    *,
    max_sentences: int,
    max_tokens: int,
) -> str:
    """对证据块做句子级压缩。

    为什么不用简单尾部截断：
    1. 可能把关键条件裁掉
    2. 可能把限制条款裁掉
    3. 可能把结论裁掉

    这里采用“句子打分 + 预算内保留”的方式。
    """

    question_keywords = extract_keywords(current_question)
    sentences = split_sentences(chunk.content)
    if not sentences:
        return chunk.content[:160]

    scored_sentences: list[tuple[float, str]] = []
    for idx, sentence in enumerate(sentences):
        sentence_keywords = extract_keywords(sentence)
        relevance = overlap_score(question_keywords, sentence_keywords)
        position_bonus = 0.08 if idx == 0 else 0.0
        rule_bonus = 0.10 if any(token in sentence for token in ("必须", "禁止", "应当", "不得", "仅限", "例外")) else 0.0
        numeric_bonus = 0.08 if re.search(r"\d", sentence) else 0.0
        entity_bonus = 0.10 if overlap_score(question_keywords, sentence_keywords) > 0 else 0.0
        score = 0.42 * relevance + position_bonus + rule_bonus + numeric_bonus + entity_bonus
        scored_sentences.append((round(score, 4), sentence))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)

    selected_sentences: list[str] = []
    used_tokens = 0
    for _, sentence in scored_sentences:
        if len(selected_sentences) >= max_sentences:
            break
        cost = estimate_tokens(sentence)
        if selected_sentences and used_tokens + cost > max_tokens:
            continue
        selected_sentences.append(sentence)
        used_tokens += cost

    if not selected_sentences:
        selected_sentences = [sentences[0]]

    return " ".join(selected_sentences)


def score_evidence(
    chunk: EvidenceChunk,
    current_question: str,
) -> float:
    """给证据块打综合分。"""

    question_keywords = extract_keywords(current_question)
    chunk_keywords = extract_keywords(chunk.content + " " + chunk.title + " " + chunk.section)
    relevance = overlap_score(question_keywords, chunk_keywords)
    authority_bonus = 0.06 * chunk.authority_level
    freshness_bonus = 0.04 * chunk.freshness_level
    score = 0.50 * chunk.retrieval_score + 0.28 * relevance + authority_bonus + freshness_bonus
    return round(score, 4)


def select_evidence(
    chunks: list[EvidenceChunk],
    current_question: str,
    *,
    max_total_tokens: int,
    max_selected: int,
    max_sentences_per_chunk: int,
) -> list[tuple[EvidenceChunk, str]]:
    """选择最值得进入 prompt 的证据。

    当前策略：
    1. 先按综合分排序
    2. 每个 chunk 先压缩成短摘要
    3. 在总预算内尽量保留更多高价值证据
    """

    ranked = sorted(chunks, key=lambda item: score_evidence(item, current_question), reverse=True)
    selected: list[tuple[EvidenceChunk, str]] = []
    used_tokens = 0

    for chunk in ranked:
        if len(selected) >= max_selected:
            break
        compressed = compress_evidence_chunk(
            chunk,
            current_question,
            max_sentences=max_sentences_per_chunk,
            max_tokens=max(80, max_total_tokens // max_selected),
        )
        block = (
            f"[CHUNK_ID:{chunk.chunk_id}] title={chunk.title} "
            f"source={chunk.source} section={chunk.section}\n{compressed}"
        )
        cost = estimate_tokens(block)
        if selected and used_tokens + cost > max_total_tokens:
            continue
        selected.append((chunk, compressed))
        used_tokens += cost

    return selected


def build_structured_prompt(
    *,
    current_question: str,
    resolved_query: str,
    history_turns: list[HistoryTurn],
    rolling_summary: str,
    evidences: list[tuple[EvidenceChunk, str]],
    governance_notice: str,
) -> str:
    """组装结构化 prompt。

    这里故意用清晰的分区结构，方便后续真正接入项目时：
    1. 独立统计每个区块 token；
    2. 独立对每个区块做裁剪；
    3. 更好地做审计和 explainability。
    """

    parts = [
        "SYSTEM_RULES:\n"
        "- 只能基于证据回答。\n"
        "- 证据不足时必须拒答。\n"
        "- 每个事实结论都要可追溯。\n",
        (
            "CURRENT_TASK:\n"
            f"- question: {normalize_text(current_question)}\n"
            f"- resolved_query: {normalize_text(resolved_query)}\n"
            "- answer_goal: 给出基于证据的可引用回答。\n"
        ),
    ]

    if rolling_summary:
        parts.append(rolling_summary + "\n")

    if history_turns:
        history_lines = ["RECENT_RELEVANT_TURNS:"]
        for turn in history_turns:
            history_lines.append(f"[{turn.role.upper()}] {normalize_text(turn.content)}")
        parts.append("\n".join(history_lines) + "\n")

    if evidences:
        evidence_lines = ["RETRIEVED_EVIDENCE:"]
        for chunk, compressed in evidences:
            evidence_lines.append(
                f"[CHUNK_ID:{chunk.chunk_id}] title={chunk.title} source={chunk.source} "
                f"section={chunk.section}\n{compressed}"
            )
        parts.append("\n\n".join(evidence_lines) + "\n")

    if governance_notice:
        parts.append("GOVERNANCE_NOTICE:\n" + governance_notice + "\n")

    parts.append(
        "OUTPUT_SCHEMA:\n"
        "ANSWER: ...\n"
        "CONFIDENCE: ...\n"
        "REASONING_SUMMARY: ...\n"
        "CITATIONS_JSON: [...]\n"
    )
    return "\n".join(parts)


def build_prompt_with_budget(
    *,
    current_question: str,
    resolved_query: str,
    history_turns: list[HistoryTurn],
    evidence_chunks: list[EvidenceChunk],
    governance_notice: str,
    config: PromptBudgetConfig,
) -> PromptBuildResult:
    """统一预算入口。

    这是整个 demo 里最重要的函数：
    1. 先算总预算；
    2. 再选历史原文；
    3. 再给更早历史做滚动摘要；
    4. 再选证据；
    5. 最后组装结构化 prompt。
    """

    available_input_tokens = (
        config.context_window_tokens
        - config.reserved_completion_tokens
        - config.safety_margin_tokens
    )

    selected_history, dropped_history = select_recent_turns(
        history_turns,
        current_question,
        max_tokens=config.recent_turns_max_tokens,
        max_turns=config.max_recent_turns,
    )
    rolling_summary = build_rolling_summary(dropped_history, max_tokens=config.memory_max_tokens)
    selected_evidence = select_evidence(
        evidence_chunks,
        current_question,
        max_total_tokens=config.evidence_max_tokens,
        max_selected=config.max_selected_evidence,
        max_sentences_per_chunk=config.max_sentences_per_evidence,
    )

    prompt = build_structured_prompt(
        current_question=current_question,
        resolved_query=resolved_query,
        history_turns=selected_history,
        rolling_summary=rolling_summary,
        evidences=selected_evidence,
        governance_notice=governance_notice,
    )

    token_usage = {
        "available_input_tokens": available_input_tokens,
        "prompt_tokens_estimate": estimate_tokens(prompt),
        "recent_turns_tokens_estimate": sum(
            estimate_tokens(f"[{turn.role.upper()}] {turn.content}") for turn in selected_history
        ),
        "memory_tokens_estimate": estimate_tokens(rolling_summary) if rolling_summary else 0,
        "evidence_tokens_estimate": sum(
            estimate_tokens(
                f"[CHUNK_ID:{chunk.chunk_id}] title={chunk.title} source={chunk.source} "
                f"section={chunk.section}\n{compressed}"
            )
            for chunk, compressed in selected_evidence
        ),
    }

    drop_reasons: list[str] = []
    if dropped_history:
        drop_reasons.append("部分较早历史未保留原文，已转入 rolling summary。")
    if len(selected_evidence) < len(evidence_chunks):
        drop_reasons.append("部分低优先级证据未进入 prompt。")
    if token_usage["prompt_tokens_estimate"] > available_input_tokens:
        drop_reasons.append("当前 demo 预算仍偏紧，真实项目中应继续做二次压缩或降级。")

    return PromptBuildResult(
        prompt=prompt,
        selected_history=selected_history,
        dropped_history=dropped_history,
        selected_evidence=selected_evidence,
        rolling_summary=rolling_summary,
        token_usage=token_usage,
        drop_reasons=drop_reasons,
    )


def build_demo_history() -> list[HistoryTurn]:
    """构造一组演示用多轮历史。"""

    return [
        HistoryTurn("user", "我想查一下安环部在二矿安生平台里怎么查看隐患排查记录。", "t1"),
        HistoryTurn("assistant", "我先按安环部、二矿、安生平台这几个锚点理解你的问题。", "t2"),
        HistoryTurn("user", "优先看制度和操作手册，不要给我泛泛解释。", "t3"),
        HistoryTurn("assistant", "明白，我会优先基于制度文件和平台操作手册回答。", "t4"),
        HistoryTurn("user", "如果有菜单路径，也一起给我。", "t5"),
        HistoryTurn("assistant", "还需要确认是查看历史台账，还是查看某次专项排查记录。", "t6"),
        HistoryTurn("user", "看历史台账，最好带出处。", "t7"),
        HistoryTurn("assistant", "好的，当前未确认的点只剩是否区分 PC 端和移动端。", "t8"),
    ]


def build_demo_evidence() -> list[EvidenceChunk]:
    """构造一组演示用证据块。"""

    return [
        EvidenceChunk(
            chunk_id="chunk-101",
            title="安生平台隐患排查操作手册",
            source="manual_v3.pdf",
            section="历史台账查询",
            content=(
                "进入安生平台后，选择“隐患排查”模块。"
                "在左侧菜单进入“排查台账”页面。"
                "用户可按矿区、部门、时间范围查询历史隐患排查记录。"
                "如需导出，必须具备相应导出权限。"
            ),
            retrieval_score=0.95,
            authority_level=2,
            freshness_level=2,
        ),
        EvidenceChunk(
            chunk_id="chunk-208",
            title="安全环保部平台使用规范",
            source="policy_2025.docx",
            section="权限与查询范围",
            content=(
                "安环部人员在职责范围内可以查看本部门和授权矿区的隐患排查台账。"
                "跨部门或跨矿区查询应当经过授权。"
                "未经授权不得导出完整台账。"
            ),
            retrieval_score=0.90,
            authority_level=3,
            freshness_level=1,
        ),
        EvidenceChunk(
            chunk_id="chunk-305",
            title="二矿隐患排查业务流程",
            source="mine2_sop.md",
            section="查询步骤",
            content=(
                "二矿用户在平台首页登录后，应先进入业务导航。"
                "选择安全管理，再进入隐患排查。"
                "在查询页可设置开始时间、结束时间、隐患等级和整改状态。"
                "查询结果支持查看详情，但移动端菜单名称略有不同。"
            ),
            retrieval_score=0.87,
            authority_level=2,
            freshness_level=2,
        ),
        EvidenceChunk(
            chunk_id="chunk-411",
            title="移动端菜单差异说明",
            source="mobile_guide.txt",
            section="移动端入口",
            content=(
                "移动端首页不展示完整左侧菜单。"
                "隐患排查历史记录入口位于“安全”页签内。"
                "如果用户只问 PC 端路径，这条说明优先级较低。"
            ),
            retrieval_score=0.61,
            authority_level=1,
            freshness_level=1,
        ),
        EvidenceChunk(
            chunk_id="chunk-512",
            title="平台登录常见问题",
            source="faq.csv",
            section="账号异常",
            content=(
                "若无法登录，请联系系统管理员处理账号问题。"
                "本段与历史台账查询关系较弱，仅在账号异常场景下相关。"
            ),
            retrieval_score=0.35,
            authority_level=0,
            freshness_level=0,
        ),
    ]


def print_result(result: PromptBuildResult) -> None:
    """把结果格式化打印出来，便于快速理解 demo 行为。"""

    print("=" * 80)
    print("预算统计")
    print("=" * 80)
    for key, value in result.token_usage.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("保留的历史原文")
    print("=" * 80)
    for turn in result.selected_history:
        print(f"- [{turn.role}] {turn.content}")

    print("\n" + "=" * 80)
    print("滚动摘要")
    print("=" * 80)
    print(result.rolling_summary or "无")

    print("\n" + "=" * 80)
    print("保留的证据")
    print("=" * 80)
    for chunk, compressed in result.selected_evidence:
        print(f"- {chunk.chunk_id} | {chunk.title}")
        print(f"  {compressed}")

    print("\n" + "=" * 80)
    print("裁剪说明")
    print("=" * 80)
    for reason in result.drop_reasons or ["无"]:
        print(f"- {reason}")

    print("\n" + "=" * 80)
    print("最终 Prompt")
    print("=" * 80)
    print(result.prompt)


def main() -> None:
    """运行 demo。"""

    history = build_demo_history()
    evidences = build_demo_evidence()
    config = PromptBudgetConfig()

    current_question = "那 PC 端具体菜单路径是什么？如果能看本月的历史台账，顺便告诉我怎么筛选。"
    resolved_query = "安环部在二矿安生平台 PC 端查看本月隐患排查历史台账的菜单路径和筛选方式"
    governance_notice = (
        "- 必须优先依据制度和操作手册回答\n"
        "- 需要给出引用\n"
        "- 如果涉及权限限制，需要显式说明"
    )

    result = build_prompt_with_budget(
        current_question=current_question,
        resolved_query=resolved_query,
        history_turns=history,
        evidence_chunks=evidences,
        governance_notice=governance_notice,
        config=config,
    )
    print_result(result)


if __name__ == "__main__":
    main()
