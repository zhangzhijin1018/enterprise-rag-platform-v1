"""从当前项目 `/eval` 报告自动构造本地生成模型 SFT 数据。

这份脚本解决的是一个很现实的问题：
即使已经决定要做本地生成模型 QLoRA/SFT，很多团队第一步也常常卡在“训练数据怎么来”。

当前项目已经有两类非常值钱的现成资产：
1. 原始评测集：包含 question、ground_truth、contexts、scenario、tags；
2. `/eval` 报告：包含当前系统在这些问题上的治理信号，例如 refusal / conflict / model_route。

所以这份脚本采用一条“先能落地、再持续增强”的路线：
- 上下文优先来自 `/eval` 报告里的 contexts；
- 如果报告里没有 contexts，就回退到评测集里的参考 contexts；
- 答案优先使用报告里的 answer；
- 如果报告里没有 answer，就回退到 ground_truth；
- 最终统一转成 `messages` 格式，供本地生成模型做第一轮监督微调。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "你是新疆能源集团企业知识智能副驾。"
    "必须基于已检索证据回答；证据不足时明确拒答；"
    "不要编造制度、流程、金额和时间。"
)


def load_json(path: str) -> Any:
    """读取 JSON 文件。

    当前主要用于读取 `/eval` 产出的 JSON 报告。
    这类报告和训练集 JSONL 的差别在于：
    - 它更像“整轮评测输出”
    - 顶层通常有 `summary`
    - 逐题样本放在 `rows`
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """读取 JSONL 文件。

    当前默认读取的是原始评测集 `enterprise_eval.jsonl`。
    之所以还要读原始评测集，而不是只看 `/eval` 报告，是因为：
    - 评测集里通常保留了更稳定的 `ground_truth`
    - 评测集里通常有更清晰的 `scenario / tags`
    - 某些报告字段缺失时，可以用评测集补齐
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    """写出 JSONL 文件。

    输出继续采用 JSONL，是为了和训练脚本保持最简单的解耦：
    - 数据整理脚本只负责“产数据”
    - 训练脚本只负责“读 messages 并训练”
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    """做轻量文本归一。

    当前只做最保守的空白折叠，不做激进清洗。
    这是因为：
    - 企业场景里制度号、版本号、路径箭头、字段名都可能很重要；
    - 这里的目标是“保证对齐和判空稳定”，不是“重写文本内容”。
    """
    return " ".join((text or "").strip().split())


def build_reference_map(dataset_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 question 建立评测集索引。

    核心假设：
    - `/eval` 报告里的逐题结果，和原始评测集能通过 `question` 对齐；
    - 对齐后，就可以把评测集看成“较稳定的参考侧”，把报告看成“当前系统运行结果侧”。

    这也是当前脚本的根本思想：
    把“参考侧”和“运行结果侧”拼起来，构造第一版 SFT 训练数据。
    """
    mapping: dict[str, dict[str, Any]] = {}
    for row in dataset_rows:
        question = normalize_text(str(row.get("question", "")))
        if question:
            mapping[question] = row
    return mapping


def resolve_contexts(report_row: dict[str, Any], reference_row: dict[str, Any] | None) -> list[str]:
    """确定训练样本使用的证据文本。

    优先级：
    1. `/eval` 报告里的 contexts
    2. 原始评测集里的 contexts

    这么做的原因是：
    - 如果报告里保留了 contexts，它更接近当前系统真实拿到的证据；
    - 如果报告里没有，再退回到评测集的参考 contexts，保证第一版样本仍可构造。

    这里体现的是一个很重要的训练取舍：
    - 我们希望模型尽量学习“真实运行时拿到的证据”；
    - 但又不能因为报告字段偶尔缺失，就把整条样本丢掉。
    """
    candidates = list(report_row.get("contexts") or [])
    if not candidates and reference_row:
        candidates = list(reference_row.get("contexts") or [])

    results: list[str] = []
    for item in candidates:
        text = normalize_text(str(item))
        if text:
            results.append(text)
    return results


def resolve_answer(report_row: dict[str, Any], reference_row: dict[str, Any] | None) -> str:
    """确定训练样本中的 assistant 目标答案。

    优先级：
    1. `/eval` 报告中的真实 answer
    2. 原始评测集中的 ground_truth
    3. 如果是拒答题但没有标准答案，则根据 refusal_reason 给一个保守模板

    这里的核心取舍是：
    - 第一轮先把训练链路跑起来，不要求每条都是完美人工标注；
    - 但也不能在没有 answer 时直接丢弃所有样本，否则高质量拒答题会大量流失。

    为什么这里优先级是 `answer -> ground_truth -> refusal template`：
    1. `answer` 最接近当前系统真实行为；
    2. `ground_truth` 最接近人工理想答案；
    3. refusal template 是兜底，至少能保住拒答样本不至于完全缺失。
    """
    answer = normalize_text(str(report_row.get("answer", "")))
    if answer:
        return answer

    ground_truth = normalize_text(str((reference_row or {}).get("ground_truth", "")))
    if ground_truth:
        return ground_truth

    if bool(report_row.get("refusal")):
        refusal_reason = normalize_text(str(report_row.get("refusal_reason", "")))
        if refusal_reason:
            return f"当前证据不足或不满足访问条件，无法回答该问题。原因：{refusal_reason}。"
        return "当前证据不足或不满足访问条件，无法回答该问题。"
    return ""


def build_user_content(
    question: str,
    contexts: list[str],
    report_row: dict[str, Any],
) -> str:
    """把 question、contexts 和治理提示拼成 user 消息。

    当前格式故意保持简单稳定：
    - 先给已检索证据
    - 再给治理提示（如果有 conflict）
    - 最后给用户问题

    这样设计的核心原理是：
    - 训练阶段要尽量模拟“RAG 生成前已经拿到上下文”的状态；
    - 但又不要把数据做得过于复杂，导致不同训练样本风格不统一。

    所以当前把证据显式展开进 user message，而不是引入复杂多字段模板。
    """
    lines: list[str] = []
    if contexts:
        lines.append("已检索证据：")
        for index, ctx in enumerate(contexts, start=1):
            lines.append(f"[文档{index}] {ctx}")
        lines.append("")

    if bool(report_row.get("conflict_detected")) and report_row.get("conflict_summary"):
        lines.append(f"治理提示：{report_row.get('conflict_summary')}")
        lines.append("")

    lines.append(f"用户问题：{question}")
    return "\n".join(lines).strip()


def should_skip_row(
    report_row: dict[str, Any],
    *,
    include_refusal: bool,
    include_conflict: bool,
) -> bool:
    """判断当前行是否要跳过。

    默认行为：
    - grounded 样本保留
    - refusal 样本保留
    - conflict 样本保留

    但可以通过参数裁剪，只保留你当前最关心的训练子集。

    为什么默认把 refusal / conflict 也保留：
    - 当前本地生成模型不是只负责“答题”
    - 它还要学会：
      - 证据不足时怎么拒答
      - 多文档冲突时怎么保守说明
    - 如果训练集全是正常回答样本，本地模型很容易在高风险场景里瞎补。
    """
    question = normalize_text(str(report_row.get("question", "")))
    if not question:
        return True

    if bool(report_row.get("refusal")) and not include_refusal:
        return True

    if bool(report_row.get("conflict_detected")) and not include_conflict:
        return True

    return False


def build_messages_row(
    report_row: dict[str, Any],
    reference_row: dict[str, Any] | None,
    *,
    system_prompt: str,
) -> dict[str, Any] | None:
    """把单条评测结果转成一条 SFT `messages` 样本。

    这一步真正完成的是：
    - 从“评测/运行结果数据结构”
    - 转成“生成模型监督微调数据结构”

    这里输出的不只是 `messages`，还保留了：
    - scenario
    - tags
    - refusal
    - conflict_detected
    - model_route

    这些字段当前训练脚本不直接消费，但后续做：
    - 数据分桶
    - 子集采样
    - 错误分析
    非常有价值。
    """
    question = normalize_text(str(report_row.get("question", "")))
    contexts = resolve_contexts(report_row, reference_row)
    answer = resolve_answer(report_row, reference_row)
    if not question or not answer:
        return None

    user_content = build_user_content(question, contexts, report_row)
    scenario = (
        normalize_text(str(report_row.get("scenario", "")))
        or normalize_text(str((reference_row or {}).get("scenario", "")))
    )
    tags = list((reference_row or {}).get("tags") or report_row.get("tags") or [])

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
        "question": question,
        "scenario": scenario,
        "tags": tags,
        "source": "eval_report",
        "refusal": bool(report_row.get("refusal")),
        "refusal_reason": report_row.get("refusal_reason") or "",
        "conflict_detected": bool(report_row.get("conflict_detected")),
        "conflict_summary": report_row.get("conflict_summary") or "",
        "data_classification": report_row.get("data_classification") or "",
        "model_route": report_row.get("model_route") or "",
        "answer_mode": report_row.get("answer_mode") or "",
    }


def build_dataset(
    report_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    *,
    system_prompt: str,
    include_refusal: bool,
    include_conflict: bool,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    """构造本地生成模型第一版 SFT 数据集。

    这一步的目标不是得到“完美黄金训练集”，而是得到：
    - 足够贴当前系统
    - 能覆盖 grounded / refusal / conflict
    - 足够让第一轮 QLoRA 起步

    这是一个非常工程化的思路：
    先让数据链路跑通，再逐轮提升数据质量，而不是一开始就等完美人工标注。
    """
    reference_map = build_reference_map(dataset_rows)
    results: list[dict[str, Any]] = []

    for report_row in report_rows:
        if should_skip_row(
            report_row,
            include_refusal=include_refusal,
            include_conflict=include_conflict,
        ):
            continue

        question = normalize_text(str(report_row.get("question", "")))
        reference_row = reference_map.get(question)
        item = build_messages_row(
            report_row,
            reference_row,
            system_prompt=system_prompt,
        )
        if item is None:
            continue
        results.append(item)
        if max_rows is not None and len(results) >= max_rows:
            break
    return results


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    当前暴露的参数不多，是故意为之。
    这份脚本的目标是第一轮可落地，而不是变成一套庞大的数据工程平台。

    目前最关键的几个开关只有：
    - 输入报告
    - 输入评测集
    - 输出路径
    - 是否保留 refusal
    - 是否保留 conflict
    - 先抽多少行预览
    """
    parser = argparse.ArgumentParser(
        description="从 `/eval` 报告自动构造本地生成模型 SFT 数据。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-report-path",
        required=True,
        help="`/eval` 产生的 JSON 报告路径。",
    )
    parser.add_argument(
        "--dataset-path",
        default="./core/evaluation/datasets/enterprise_eval.jsonl",
        help="原始评测集 JSONL 路径，用于补 ground_truth / contexts / tags / scenario。",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="输出的 SFT JSONL 路径。",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="训练样本统一使用的 system prompt。",
    )
    parser.add_argument(
        "--include-refusal",
        action="store_true",
        default=True,
        help="是否保留拒答样本。第一轮建议保留。",
    )
    parser.add_argument(
        "--include-conflict",
        action="store_true",
        default=True,
        help="是否保留冲突样本。第一轮建议保留。",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="可选，只导出前 N 条样本，适合预览或 smoke test。",
    )
    return parser.parse_args()


def main() -> None:
    """脚本主入口。

    主流程：
    1. 读取 `/eval` JSON 报告；
    2. 读取原始评测集；
    3. 逐题对齐 `question`；
    4. 决定用哪些 contexts、哪些答案；
    5. 生成 `messages` 样本；
    6. 写出新的 JSONL。

    这一步相当于把“评测资产”翻译成“训练资产”。
    """
    args = parse_args()
    report_payload = load_json(args.eval_report_path)
    report_rows = list(report_payload.get("rows") or [])
    if not report_rows:
        raise ValueError("`rows` 为空，当前报告不像有效的 `/eval` JSON 输出。")

    dataset_rows = load_jsonl(args.dataset_path)
    # 这里真正发生的是“监督信号重组”，不是简单的字段复制。
    # 我们把 question / contexts / answer / refusal / conflict 重新组织成
    # 生成模型更容易学到的 `messages` 结构。
    sft_rows = build_dataset(
        report_rows=report_rows,
        dataset_rows=dataset_rows,
        system_prompt=args.system_prompt,
        include_refusal=args.include_refusal,
        include_conflict=args.include_conflict,
        max_rows=args.max_rows,
    )
    if not sft_rows:
        raise ValueError("未构造出任何 SFT 样本，请先检查 eval 报告和评测集是否匹配。")

    write_jsonl(args.output_path, sft_rows)
    print(f"wrote {len(sft_rows)} SFT rows to {args.output_path}")


if __name__ == "__main__":
    main()
